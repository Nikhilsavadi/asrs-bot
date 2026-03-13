"""
spy_extended_correlation.py — Test SPY after-hours/pre-market vs DAX bar 5 outcomes

Checks:
  1. SPY post-market (16:00-20:00 ET previous day) — US after-hours
  2. SPY pre-market (04:00-09:30 ET same day) — before DAX bar 5
  3. SPY futures proxy: last close vs next day open gap
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta, time as dtime
from collections import defaultdict

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SPY_EXT")

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "arL6Kqp4GoBiLF_x97ovrFeHYS7ilN80")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data")


async def fetch_spy_bars(start_date: str, end_date: str, multiplier: int = 1, timespan: str = "hour") -> pd.DataFrame:
    """Fetch SPY intraday bars from Polygon (includes extended hours)."""
    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.error(f"Polygon error: {r.status_code} {r.text}")
            return pd.DataFrame()

        data = r.json()
        count = data.get("resultsCount", 0)
        if count == 0:
            logger.error("No SPY data returned")
            return pd.DataFrame()

        all_results.extend(data["results"])
        logger.info(f"SPY: got {count} bars")

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert("US/Eastern")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("datetime_et")
    return df


async def fetch_spy_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY daily bars for RTH close reference."""
    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if data.get("resultsCount", 0) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
        df = df.set_index("date")
        return df


def analyse_period(spy_hourly: pd.DataFrame, period_start: dtime, period_end: dtime, ref_date) -> dict | None:
    """Get SPY performance for a time window on a given date."""
    try:
        day_bars = spy_hourly[spy_hourly.index.date == ref_date]
        if day_bars.empty:
            return None

        period = day_bars.between_time(period_start.strftime("%H:%M"), period_end.strftime("%H:%M"))
        if period.empty:
            return None

        first_open = period.iloc[0]["open"]
        last_close = period.iloc[-1]["close"]
        pct = round((last_close - first_open) / first_open * 100, 3)
        direction = "UP" if pct > 0.1 else ("DOWN" if pct < -0.1 else "FLAT")

        return {"pct": pct, "direction": direction, "open": first_open, "close": last_close}
    except Exception:
        return None


async def main():
    from backtest import run_backtest_ema, DayResult

    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")

    if not os.path.exists(rth_path):
        logger.error("No cached DAX data.")
        return

    df = pd.read_parquet(rth_path)
    all_df = pd.read_parquet(all_path) if os.path.exists(all_path) else None

    logger.info("Running DAX bar 5 backtest...")
    dax_results = run_backtest_ema(df, all_df, signal_bar=5)

    dates = [r.date for r in dax_results]
    start = (datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    end = max(dates)

    # Fetch SPY hourly (includes extended hours) and daily
    logger.info(f"Fetching SPY hourly bars {start} to {end}...")
    spy_hourly = await fetch_spy_bars(start, end, multiplier=1, timespan="hour")
    if spy_hourly.empty:
        return

    logger.info(f"Fetching SPY daily bars...")
    spy_daily = await fetch_spy_daily(start, end)

    dax_by_date = {r.date: r for r in dax_results if r.triggered}

    rows = []
    for dax_date_str, dax_day in dax_by_date.items():
        dax_date = datetime.strptime(dax_date_str, "%Y-%m-%d").date()

        # Find previous US trading day
        prev_us_date = None
        if not spy_daily.empty:
            for d in sorted(spy_daily.index, reverse=True):
                if d < dax_date:
                    prev_us_date = d
                    break

        if prev_us_date is None:
            continue

        row = {
            "dax_date": dax_date_str,
            "dax_pnl": dax_day.total_pnl,
            "dax_first_dir": dax_day.trades[0].direction if dax_day.trades else "",
        }

        # 1. SPY post-market previous day (16:00-20:00 ET)
        post_mkt = analyse_period(spy_hourly, dtime(16, 0), dtime(19, 59), prev_us_date)
        if post_mkt:
            row["post_mkt_pct"] = post_mkt["pct"]
            row["post_mkt_dir"] = post_mkt["direction"]
        else:
            row["post_mkt_pct"] = None
            row["post_mkt_dir"] = None

        # 2. SPY pre-market same day (04:00-09:30 ET) — this is 09:00-14:30 CET
        #    DAX bar 5 closes 09:25 CET = 03:25 ET, so pre-market barely started
        #    Better to use pre-market of PREVIOUS day or overnight futures
        pre_mkt = analyse_period(spy_hourly, dtime(4, 0), dtime(9, 29), prev_us_date)
        if pre_mkt:
            row["pre_mkt_pct"] = pre_mkt["pct"]
            row["pre_mkt_dir"] = pre_mkt["direction"]
        else:
            row["pre_mkt_pct"] = None
            row["pre_mkt_dir"] = None

        # 3. SPY overnight move: previous RTH close to next day RTH open
        if prev_us_date in spy_daily.index and dax_date in spy_daily.index:
            prev_close = spy_daily.loc[prev_us_date, "close"]
            next_open = spy_daily.loc[dax_date, "open"]
            gap_pct = round((next_open - prev_close) / prev_close * 100, 3)
            row["overnight_gap_pct"] = gap_pct
            row["overnight_gap_dir"] = "UP" if gap_pct > 0.1 else ("DOWN" if gap_pct < -0.1 else "FLAT")
        elif prev_us_date in spy_daily.index:
            # DAX date might not be a US trading day (different holidays)
            row["overnight_gap_pct"] = None
            row["overnight_gap_dir"] = None
        else:
            row["overnight_gap_pct"] = None
            row["overnight_gap_dir"] = None

        # 4. Full extended: RTH close prev day to earliest available next session
        if post_mkt and prev_us_date in spy_daily.index:
            rth_close = spy_daily.loc[prev_us_date, "close"]
            ext_close = post_mkt["close"]
            ext_pct = round((ext_close - rth_close) / rth_close * 100, 3)
            row["extended_pct"] = ext_pct
            row["extended_dir"] = "UP" if ext_pct > 0.1 else ("DOWN" if ext_pct < -0.1 else "FLAT")
        else:
            row["extended_pct"] = None
            row["extended_dir"] = None

        rows.append(row)

    combo = pd.DataFrame(rows)
    logger.info(f"Matched {len(combo)} days")

    print("\n" + "=" * 85)
    print("  SPY EXTENDED HOURS vs DAX BAR 5 — CORRELATION ANALYSIS")
    print("=" * 85)

    def print_analysis(label, dir_col, pct_col):
        valid = combo[combo[dir_col].notna()].copy()
        if valid.empty:
            print(f"\n-- {label}: No data --")
            return

        print(f"\n-- {label} Direction vs DAX P&L --")
        print(f"{'Direction':<10} {'Days':>5} {'DAX P&L':>10} {'Avg P&L':>10} {'Win%':>6}")
        print("-" * 50)

        for d in ["UP", "DOWN", "FLAT"]:
            sub = valid[valid[dir_col] == d]
            if sub.empty:
                continue
            n = len(sub)
            total = sub["dax_pnl"].sum()
            avg = sub["dax_pnl"].mean()
            wins = (sub["dax_pnl"] > 0).sum()
            wr = wins / n * 100
            print(f"{d:<10} {n:>5} {total:>+9.1f} {avg:>+9.1f} {wr:>5.0f}%")

        # Direction confirmation
        confirmed = valid[
            ((valid[dir_col] == "UP") & (valid["dax_first_dir"] == "LONG")) |
            ((valid[dir_col] == "DOWN") & (valid["dax_first_dir"] == "SHORT"))
        ]
        against = valid[
            ((valid[dir_col] == "UP") & (valid["dax_first_dir"] == "SHORT")) |
            ((valid[dir_col] == "DOWN") & (valid["dax_first_dir"] == "LONG"))
        ]

        if len(confirmed) > 0 and len(against) > 0:
            print(f"\n  As direction filter:")
            print(f"  Aligned:  {len(confirmed):>3} trades, P&L: {confirmed['dax_pnl'].sum():>+8.1f}, Avg: {confirmed['dax_pnl'].mean():>+7.1f}")
            print(f"  Against:  {len(against):>3} trades, P&L: {against['dax_pnl'].sum():>+8.1f}, Avg: {against['dax_pnl'].mean():>+7.1f}")

            if confirmed["dax_pnl"].mean() > against["dax_pnl"].mean() * 1.3:
                print(f"  >> USEFUL as direction confirmation")
            else:
                print(f"  >> NOT useful as direction filter")

        # Magnitude buckets
        if pct_col in valid.columns:
            pct_valid = valid[valid[pct_col].notna()]
            if len(pct_valid) > 10:
                print(f"\n  By magnitude:")
                bins = [(-99, -0.3, "< -0.3%"), (-0.3, 0.3, "Flat"), (0.3, 99, "> +0.3%")]
                for lo, hi, lbl in bins:
                    sub = pct_valid[(pct_valid[pct_col] >= lo) & (pct_valid[pct_col] < hi)]
                    if sub.empty:
                        continue
                    n = len(sub)
                    avg = sub["dax_pnl"].mean()
                    print(f"  {lbl:<12} {n:>3} days, Avg DAX: {avg:>+7.1f}")

    print_analysis("1. SPY POST-MARKET (16:00-20:00 ET prev day)", "post_mkt_dir", "post_mkt_pct")
    print_analysis("2. SPY PRE-MARKET (04:00-09:30 ET prev day)", "pre_mkt_dir", "pre_mkt_pct")
    print_analysis("3. SPY OVERNIGHT GAP (prev close -> today open)", "overnight_gap_dir", "overnight_gap_pct")
    print_analysis("4. SPY RTH CLOSE vs POST-MARKET CLOSE", "extended_dir", "extended_pct")

    # Final verdict
    print("\n" + "=" * 85)
    print("  VERDICT")
    print("=" * 85)

    verdicts = []
    for label, dir_col in [("Post-market", "post_mkt_dir"), ("Pre-market", "pre_mkt_dir"),
                            ("Overnight gap", "overnight_gap_dir"), ("Extended", "extended_dir")]:
        valid = combo[combo[dir_col].notna()]
        if valid.empty:
            continue
        confirmed = valid[
            ((valid[dir_col] == "UP") & (valid["dax_first_dir"] == "LONG")) |
            ((valid[dir_col] == "DOWN") & (valid["dax_first_dir"] == "SHORT"))
        ]
        against = valid[
            ((valid[dir_col] == "UP") & (valid["dax_first_dir"] == "SHORT")) |
            ((valid[dir_col] == "DOWN") & (valid["dax_first_dir"] == "LONG"))
        ]
        if len(confirmed) > 0 and len(against) > 0:
            c_avg = confirmed["dax_pnl"].mean()
            a_avg = against["dax_pnl"].mean()
            useful = c_avg > a_avg * 1.3
            verdicts.append((label, c_avg, a_avg, useful))
            icon = "YES" if useful else "NO"
            print(f"  {label:<18} Aligned: {c_avg:>+7.1f}  Against: {a_avg:>+7.1f}  -> {icon}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
