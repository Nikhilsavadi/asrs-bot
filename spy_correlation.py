"""
spy_correlation.py — Test SPY previous-day performance vs DAX bar 5 outcomes
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SPY")

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "arL6Kqp4GoBiLF_x97ovrFeHYS7ilN80")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data")


async def fetch_spy_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY daily bars from Polygon.io"""
    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=15)
        if r.status_code != 200:
            logger.error(f"Polygon error: {r.status_code} {r.text}")
            return pd.DataFrame()

        data = r.json()
        if data.get("resultsCount", 0) == 0:
            logger.error("No SPY data returned")
            return pd.DataFrame()

        results = data["results"]
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["pct_change"] = ((df["close"] - df["open"]) / df["open"] * 100).round(2)
        df["direction"] = df["pct_change"].apply(lambda x: "UP" if x > 0.1 else ("DOWN" if x < -0.1 else "FLAT"))
        df = df.set_index("date")
        logger.info(f"SPY: {len(df)} daily bars from {df.index[0]} to {df.index[-1]}")
        return df


async def main():
    # Load DAX backtest results
    from backtest import run_backtest_ema, DayResult, Trade, _calc_ema_series

    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")

    if not os.path.exists(rth_path):
        logger.error("No cached DAX data. Run backtest.py first.")
        return

    df = pd.read_parquet(rth_path)
    all_df = pd.read_parquet(all_path) if os.path.exists(all_path) else None

    # Run bar 5 backtest
    logger.info("Running DAX bar 5 backtest...")
    dax_results = run_backtest_ema(df, all_df, signal_bar=5)
    logger.info(f"DAX: {len(dax_results)} trading days")

    # Get date range for SPY
    dates = [r.date for r in dax_results]
    start = (datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    end = max(dates)

    # Fetch SPY data
    logger.info(f"Fetching SPY daily data {start} to {end}...")
    spy = await fetch_spy_daily(start, end)
    if spy.empty:
        return

    # Build lookup: for each DAX trading day, get SPY previous day
    dax_by_date = {r.date: r for r in dax_results}

    rows = []
    spy_dates = sorted(spy.index)
    for dax_date_str, dax_day in dax_by_date.items():
        dax_date = datetime.strptime(dax_date_str, "%Y-%m-%d").date()

        # Find previous SPY trading day
        prev_spy = None
        for sd in reversed(spy_dates):
            if sd < dax_date:
                prev_spy = spy.loc[sd]
                break

        if prev_spy is None:
            continue

        rows.append({
            "dax_date": dax_date_str,
            "dax_day": dax_day.day_of_week,
            "dax_pnl": dax_day.total_pnl,
            "dax_triggered": dax_day.triggered,
            "dax_trades": len(dax_day.trades),
            "dax_first_dir": dax_day.trades[0].direction if dax_day.trades else "",
            "spy_prev_date": str(prev_spy.name) if hasattr(prev_spy, 'name') else "",
            "spy_pct": prev_spy["pct_change"],
            "spy_dir": prev_spy["direction"],
            "spy_close": prev_spy["close"],
        })

    combo = pd.DataFrame(rows)
    if combo.empty:
        logger.error("No matching dates")
        return

    logger.info(f"Matched {len(combo)} days with SPY prev-day data")

    # ══════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  SPY PREVIOUS-DAY vs DAX BAR 5 CORRELATION ANALYSIS")
    print("=" * 80)

    triggered = combo[combo["dax_triggered"]]

    # 1. SPY prev-day direction vs DAX P&L
    print("\n-- SPY Prev-Day Direction vs DAX P&L --")
    print(f"{'SPY Dir':<10} {'Days':>5} {'Trades':>7} {'DAX P&L':>10} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 55)

    for spy_dir in ["UP", "DOWN", "FLAT"]:
        subset = triggered[triggered["spy_dir"] == spy_dir]
        if subset.empty:
            continue
        n = len(subset)
        total = subset["dax_pnl"].sum()
        avg = subset["dax_pnl"].mean()
        wins = (subset["dax_pnl"] > 0).sum()
        wr = wins / n * 100
        print(f"{spy_dir:<10} {n:>5} {subset['dax_trades'].sum():>7} {total:>+9.1f} {avg:>+9.1f} {wr:>5.0f}%")

    # 2. SPY direction vs DAX trade direction alignment
    print("\n-- SPY Direction vs DAX First Trade Direction --")
    print(f"{'Combo':<25} {'Days':>5} {'DAX P&L':>10} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 60)

    for spy_dir in ["UP", "DOWN"]:
        for dax_dir in ["LONG", "SHORT"]:
            subset = triggered[(triggered["spy_dir"] == spy_dir) & (triggered["dax_first_dir"] == dax_dir)]
            if subset.empty:
                continue
            n = len(subset)
            total = subset["dax_pnl"].sum()
            avg = subset["dax_pnl"].mean()
            wins = (subset["dax_pnl"] > 0).sum()
            wr = wins / n * 100
            aligned = "ALIGNED" if (spy_dir == "UP" and dax_dir == "LONG") or (spy_dir == "DOWN" and dax_dir == "SHORT") else "AGAINST"
            print(f"SPY {spy_dir} + DAX {dax_dir:<6} {n:>5} {total:>+9.1f} {avg:>+9.1f} {wr:>5.0f}%  {aligned}")

    # 3. SPY magnitude buckets
    print("\n-- SPY Prev-Day Magnitude vs DAX P&L --")
    print(f"{'SPY Move':<20} {'Days':>5} {'DAX P&L':>10} {'Avg P&L':>10} {'Win%':>6}")
    print("-" * 55)

    bins = [(-99, -1.0, "SPY < -1%"), (-1.0, -0.3, "SPY -1% to -0.3%"),
            (-0.3, 0.3, "SPY flat"), (0.3, 1.0, "SPY +0.3% to +1%"),
            (1.0, 99, "SPY > +1%")]

    for lo, hi, label in bins:
        subset = triggered[(triggered["spy_pct"] >= lo) & (triggered["spy_pct"] < hi)]
        if subset.empty:
            continue
        n = len(subset)
        total = subset["dax_pnl"].sum()
        avg = subset["dax_pnl"].mean()
        wins = (subset["dax_pnl"] > 0).sum()
        wr = wins / n * 100
        print(f"{label:<20} {n:>5} {total:>+9.1f} {avg:>+9.1f} {wr:>5.0f}%")

    # 4. Would filtering by SPY improve results?
    print("\n-- Hypothetical: Skip DAX when SPY prev-day was DOWN > 1% --")
    skip_days = triggered[triggered["spy_pct"] < -1.0]
    keep_days = triggered[triggered["spy_pct"] >= -1.0]
    print(f"  Skipped {len(skip_days)} days, P&L of skipped: {skip_days['dax_pnl'].sum():+.1f}")
    print(f"  Kept {len(keep_days)} days, P&L of kept: {keep_days['dax_pnl'].sum():+.1f}")
    print(f"  Original total: {triggered['dax_pnl'].sum():+.1f}")

    print("\n-- Hypothetical: Only trade DAX when SPY prev-day was UP --")
    up_only = triggered[triggered["spy_dir"] == "UP"]
    rest = triggered[triggered["spy_dir"] != "UP"]
    print(f"  UP days: {len(up_only)}, P&L: {up_only['dax_pnl'].sum():+.1f}, Avg: {up_only['dax_pnl'].mean():+.1f}")
    print(f"  Other days: {len(rest)}, P&L: {rest['dax_pnl'].sum():+.1f}, Avg: {rest['dax_pnl'].mean():+.1f}")

    # 5. SPY direction as confirmation for DAX direction
    print("\n-- SPY as Direction Confirmation --")
    print("  If SPY was UP yesterday, only take LONG on DAX (skip SHORT)")
    print("  If SPY was DOWN yesterday, only take SHORT on DAX (skip LONG)")

    confirmed = triggered[
        ((triggered["spy_dir"] == "UP") & (triggered["dax_first_dir"] == "LONG")) |
        ((triggered["spy_dir"] == "DOWN") & (triggered["dax_first_dir"] == "SHORT"))
    ]
    against = triggered[
        ((triggered["spy_dir"] == "UP") & (triggered["dax_first_dir"] == "SHORT")) |
        ((triggered["spy_dir"] == "DOWN") & (triggered["dax_first_dir"] == "LONG"))
    ]
    flat_spy = triggered[triggered["spy_dir"] == "FLAT"]

    print(f"  Confirmed: {len(confirmed)} trades, P&L: {confirmed['dax_pnl'].sum():+.1f}, Avg: {confirmed['dax_pnl'].mean():+.1f}")
    print(f"  Against:   {len(against)} trades, P&L: {against['dax_pnl'].sum():+.1f}, Avg: {against['dax_pnl'].mean():+.1f}")
    print(f"  SPY flat:  {len(flat_spy)} trades, P&L: {flat_spy['dax_pnl'].sum():+.1f}")
    print(f"  All:       {len(triggered)} trades, P&L: {triggered['dax_pnl'].sum():+.1f}")

    print("\n" + "=" * 80)
    print("  VERDICT: Does SPY prev-day add value as a filter?")
    print("=" * 80)

    confirmed_avg = confirmed["dax_pnl"].mean() if len(confirmed) > 0 else 0
    against_avg = against["dax_pnl"].mean() if len(against) > 0 else 0
    all_avg = triggered["dax_pnl"].mean()

    if confirmed_avg > all_avg * 1.2 and against_avg < all_avg * 0.5:
        print("  YES - SPY confirmation adds meaningful edge")
        print(f"  Confirmed avg ({confirmed_avg:+.1f}) >> Against avg ({against_avg:+.1f})")
    elif confirmed_avg > against_avg:
        print("  MARGINAL - Some signal but may not justify filtering out trades")
        print(f"  Confirmed avg ({confirmed_avg:+.1f}) > Against avg ({against_avg:+.1f})")
    else:
        print("  NO - SPY prev-day direction does NOT predict DAX outcomes")
        print(f"  Confirmed avg ({confirmed_avg:+.1f}) vs Against avg ({against_avg:+.1f})")

    print()


if __name__ == "__main__":
    asyncio.run(main())
