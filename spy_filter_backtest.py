"""
spy_filter_backtest.py — Backtest SPY as direction filter for DAX bar 5

Timeline constraint:
  DAX bar 5 signal fires at 08:25 UK (09:25 CET).
  At that time, the ONLY SPY data available is:
    1. Previous-day RTH close (21:00 UK / 16:00 ET)
    2. Previous-day post-market (21:00-00:00 UK / 16:00-19:00 ET)

  SPY pre-market (09:00 UK / 04:00 ET) has barely started and SPY next-day
  open (14:30 UK) is 6 hours away — NOT available for filtering.

Test: Use SPY post-market direction to filter DAX OCA bracket to one side only.
  - SPY post-market UP  → only place BUY stop (skip sell-stop)
  - SPY post-market DOWN → only place SELL stop (skip buy-stop)
  - SPY post-market FLAT → place both (no filter)

Compare vs baseline (always place both stops).
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SPY_FILTER")

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "arL6Kqp4GoBiLF_x97ovrFeHYS7ilN80")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data")


async def fetch_spy_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY hourly bars from Polygon (includes extended hours)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/hour/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.error(f"Polygon error: {r.status_code} {r.text}")
            return pd.DataFrame()
        data = r.json()
        if data.get("resultsCount", 0) == 0:
            logger.error("No SPY data returned")
            return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert("US/Eastern")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("datetime_et")
    logger.info(f"SPY: got {len(df)} hourly bars")
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


def get_post_market_direction(spy_hourly: pd.DataFrame, us_date, spy_daily: pd.DataFrame) -> tuple[str, float]:
    """
    Get SPY post-market direction for a given US trading day.
    Post-market = 16:00-19:59 ET on that day.
    Returns (direction, pct_change).
    """
    try:
        day_bars = spy_hourly[spy_hourly.index.date == us_date]
        if day_bars.empty:
            return "NO_DATA", 0.0

        post = day_bars.between_time("16:00", "19:59")
        if post.empty:
            return "NO_DATA", 0.0

        # Get RTH close as reference
        if us_date in spy_daily.index:
            rth_close = spy_daily.loc[us_date, "close"]
        else:
            return "NO_DATA", 0.0

        post_last = post.iloc[-1]["close"]
        pct = round((post_last - rth_close) / rth_close * 100, 3)
        direction = "UP" if pct > 0.1 else ("DOWN" if pct < -0.1 else "FLAT")
        return direction, pct
    except Exception:
        return "NO_DATA", 0.0


def get_rth_direction(spy_daily: pd.DataFrame, us_date) -> tuple[str, float]:
    """Get SPY RTH day direction (open to close)."""
    if us_date not in spy_daily.index:
        return "NO_DATA", 0.0
    row = spy_daily.loc[us_date]
    pct = round((row["close"] - row["open"]) / row["open"] * 100, 3)
    direction = "UP" if pct > 0.1 else ("DOWN" if pct < -0.1 else "FLAT")
    return direction, pct


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

    logger.info(f"Fetching SPY data {start} to {end}...")
    spy_hourly = await fetch_spy_bars(start, end)
    if spy_hourly.empty:
        return
    spy_daily = await fetch_spy_daily(start, end)
    if spy_daily.empty:
        return

    # ── Build comparison: baseline vs SPY-filtered ──
    dax_by_date = {r.date: r for r in dax_results if r.triggered}

    baseline_pnl = []
    filtered_aligned_pnl = []   # Only take trades aligned with SPY
    filtered_against_pnl = []   # Only take trades against SPY (control)

    # Track per-day for equity curves
    equity_baseline = []
    equity_filtered = []

    match_count = 0
    spy_up_count = 0
    spy_down_count = 0
    spy_flat_count = 0
    spy_nodata_count = 0

    details = []

    for dax_date_str, dax_day in sorted(dax_by_date.items()):
        dax_date = datetime.strptime(dax_date_str, "%Y-%m-%d").date()

        # Find previous US trading day
        prev_us_date = None
        for d in sorted(spy_daily.index, reverse=True):
            if d < dax_date:
                prev_us_date = d
                break
        if prev_us_date is None:
            continue

        # ── Get SPY signals available at 08:20 UK ──
        # 1. SPY RTH direction (previous day)
        rth_dir, rth_pct = get_rth_direction(spy_daily, prev_us_date)

        # 2. SPY post-market direction (previous day, 16:00-20:00 ET = 21:00-00:00 UK)
        pm_dir, pm_pct = get_post_market_direction(spy_hourly, prev_us_date, spy_daily)

        # Use post-market as filter (this is genuinely available at 08:20 UK)
        # If no post-market data, fall back to RTH direction
        filter_dir = pm_dir if pm_dir != "NO_DATA" else rth_dir

        match_count += 1
        if filter_dir == "UP":
            spy_up_count += 1
        elif filter_dir == "DOWN":
            spy_down_count += 1
        elif filter_dir == "FLAT":
            spy_flat_count += 1
        else:
            spy_nodata_count += 1

        # Baseline: all trades as normal
        day_pnl = dax_day.total_pnl
        baseline_pnl.append(day_pnl)

        # Filtered: only take trades aligned with SPY direction
        first_dir = dax_day.trades[0].direction if dax_day.trades else ""

        # Determine if we would have taken this trade under the filter
        if filter_dir == "UP":
            # Only place buy-stop → only LONG trades count
            filtered_day_pnl = sum(t.pnl_pts for t in dax_day.trades if t.direction == "LONG")
            against_day_pnl = sum(t.pnl_pts for t in dax_day.trades if t.direction == "SHORT")
        elif filter_dir == "DOWN":
            # Only place sell-stop → only SHORT trades count
            filtered_day_pnl = sum(t.pnl_pts for t in dax_day.trades if t.direction == "SHORT")
            against_day_pnl = sum(t.pnl_pts for t in dax_day.trades if t.direction == "LONG")
        else:
            # FLAT or NO_DATA → take both (same as baseline)
            filtered_day_pnl = day_pnl
            against_day_pnl = day_pnl

        filtered_aligned_pnl.append(filtered_day_pnl)
        filtered_against_pnl.append(against_day_pnl)

        # Equity curves
        prev_b = equity_baseline[-1] if equity_baseline else 0
        prev_f = equity_filtered[-1] if equity_filtered else 0
        equity_baseline.append(prev_b + day_pnl)
        equity_filtered.append(prev_f + filtered_day_pnl)

        details.append({
            "date": dax_date_str,
            "spy_dir": filter_dir,
            "spy_source": "POST_MKT" if pm_dir != "NO_DATA" else "RTH",
            "spy_pct": pm_pct if pm_dir != "NO_DATA" else rth_pct,
            "first_trade": first_dir,
            "baseline_pnl": day_pnl,
            "filtered_pnl": filtered_day_pnl,
        })

    # ══════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SPY DIRECTION FILTER BACKTEST (using only pre-08:20 UK data)")
    print("=" * 80)

    print(f"\n  Matched days: {match_count}")
    print(f"  SPY signals: UP={spy_up_count}, DOWN={spy_down_count}, FLAT={spy_flat_count}, NO_DATA={spy_nodata_count}")

    b_total = sum(baseline_pnl)
    f_total = sum(filtered_aligned_pnl)
    a_total = sum(filtered_against_pnl)

    b_wins = sum(1 for x in baseline_pnl if x > 0)
    f_wins = sum(1 for x in filtered_aligned_pnl if x > 0)

    b_trades = len(baseline_pnl)
    f_trades = sum(1 for d in details if d["spy_dir"] in ("UP", "DOWN"))
    f_flat = sum(1 for d in details if d["spy_dir"] not in ("UP", "DOWN"))

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'SPY Filter':>12} {'Diff':>10}")
    print("  " + "-" * 65)
    print(f"  {'Total P&L (pts)':<30} {b_total:>+11.1f} {f_total:>+11.1f} {f_total - b_total:>+9.1f}")
    print(f"  {'Avg P&L per day':<30} {np.mean(baseline_pnl):>+11.1f} {np.mean(filtered_aligned_pnl):>+11.1f}")
    print(f"  {'Win rate':<30} {b_wins/b_trades*100:>10.0f}% {f_wins/len(filtered_aligned_pnl)*100:>10.0f}%")
    print(f"  {'Days traded':<30} {b_trades:>12} {b_trades:>12}")
    print(f"  {'Days with direction filter':<30} {'':>12} {f_trades:>12}")
    print(f"  {'Days pass-through (flat/nodata)':<30} {'':>12} {f_flat:>12}")

    # Breakdown by SPY direction
    print(f"\n  -- By SPY Direction --")
    print(f"  {'SPY Dir':<10} {'Days':>5} {'Baseline':>10} {'Filtered':>10} {'Diff':>8} {'Baseline Avg':>12} {'Filt Avg':>10}")
    print("  " + "-" * 70)

    df_details = pd.DataFrame(details)
    for spy_d in ["UP", "DOWN", "FLAT"]:
        sub = df_details[df_details["spy_dir"] == spy_d]
        if sub.empty:
            continue
        n = len(sub)
        b_sum = sub["baseline_pnl"].sum()
        f_sum = sub["filtered_pnl"].sum()
        b_avg = sub["baseline_pnl"].mean()
        f_avg = sub["filtered_pnl"].mean()
        print(f"  {spy_d:<10} {n:>5} {b_sum:>+9.1f} {f_sum:>+9.1f} {f_sum-b_sum:>+7.1f} {b_avg:>+11.1f} {f_avg:>+9.1f}")

    # Test: aligned vs against trades when SPY has direction
    print(f"\n  -- Aligned vs Against (when SPY has clear direction) --")
    directional = df_details[df_details["spy_dir"].isin(["UP", "DOWN"])]
    if not directional.empty:
        aligned_sum = directional["filtered_pnl"].sum()
        against_sum = sum(
            d["baseline_pnl"] - d["filtered_pnl"]
            for _, d in directional.iterrows()
        )
        print(f"  Aligned trades P&L:  {aligned_sum:>+9.1f} ({len(directional)} days)")
        print(f"  Against trades P&L:  {against_sum:>+9.1f} ({len(directional)} days)")
        print(f"  Combined (baseline): {directional['baseline_pnl'].sum():>+9.1f}")

    # Verdict
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)

    improvement = f_total - b_total
    if improvement > 0 and f_total > b_total * 1.1:
        print(f"  SPY filter IMPROVES results by {improvement:+.1f} pts ({improvement/abs(b_total)*100:+.1f}%)")
        print(f"  Consider implementing as optional direction filter.")
    elif improvement > 0:
        print(f"  SPY filter shows marginal improvement: {improvement:+.1f} pts")
        print(f"  Not significant enough to justify reduced trade count.")
    else:
        print(f"  SPY filter HURTS results by {improvement:+.1f} pts")
        print(f"  DO NOT use SPY as direction filter — trade both sides always.")

    # Sample size warning
    if spy_up_count + spy_down_count < 30:
        print(f"\n  WARNING: Only {spy_up_count + spy_down_count} days with directional SPY signal.")
        print(f"  Post-market data is sparse — results are NOT statistically reliable.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
