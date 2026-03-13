"""
backtest_dia.py — ASRS Strategy on DIA via Polygon.io
═══════════════════════════════════════════════════════════════════════════════

Adapts the DAX bar-4 breakout strategy to DIA:
  - RTH: 09:30–16:00 ET (vs 09:00–17:30 CET for DAX)
  - Bar 4 = 09:45–09:50 ET (4th 5-min candle after open)
  - Pre-market range (04:00–09:30 ET) replaces overnight range (00:00–06:00 CET)
  - Thresholds scaled for DIA price level (~$500 vs ~17000 DAX)

Usage:
    python backtest_dia.py                   Full backtest
    python backtest_dia.py --no-overnight    Without pre-market bias filter
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta, date, time as dtime
from dataclasses import dataclass, field

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DIA_BT")

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "arL6Kqp4GoBiLF_x97ovrFeHYS7ilN80")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "dia")

# ── DIA-scaled thresholds ────────────────────────────────────────────────────
# DAX ~17000: BUFFER=2 (0.012%), NARROW=15 (0.09%), WIDE=40 (0.24%)
# DIA ~500:   scale proportionally
BUFFER_PTS    = 0.10     # $0.10 buffer above/below bar 4
NARROW_RANGE  = 0.50     # Bar 4 range < $0.50 = narrow
WIDE_RANGE    = 2.00     # Bar 4 range > $2.00 = wide
MAX_ENTRIES   = 2        # Max entries per day (entry + flip)
GAP_THRESHOLD = 0.50     # $0.50 gap to classify as GAP_UP/DOWN

# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_dia_5min(start_date: str, end_date: str, extended: bool = False) -> pd.DataFrame:
    """
    Fetch DIA 5-min bars from Polygon. Handles pagination.
    If extended=True, includes pre/post-market hours.
    """
    cache_name = f"dia_5min_{'ext' if extended else 'rth'}_{start_date}_{end_date}.parquet"
    cache_path = os.path.join(DATA_DIR, cache_name)
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded cached {'extended' if extended else 'RTH'} data: {len(df)} bars")
        return df

    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/DIA/range/5/minute/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}

    async with httpx.AsyncClient(timeout=60) as client:
        page = 0
        while url:
            page += 1
            logger.info(f"Fetching page {page}...")
            r = await client.get(url, params=params if page == 1 else {"apiKey": POLYGON_KEY})
            if r.status_code == 429:
                logger.warning("Rate limited, waiting 15s...")
                await asyncio.sleep(15)
                continue
            if r.status_code != 200:
                logger.error(f"Polygon error: {r.status_code} {r.text}")
                break
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            logger.info(f"  Got {len(results)} bars (total: {len(all_results)})")
            url = data.get("next_url")
            if url:
                # next_url already has params except apiKey
                url = url + f"&apiKey={POLYGON_KEY}"
                params = {}  # Don't double-send params
                await asyncio.sleep(0.5)  # Be nice to rate limits
            else:
                break

    if not all_results:
        logger.error("No data returned from Polygon")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert("US/Eastern")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df.set_index("datetime_et")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    if not extended:
        df = df.between_time("09:30", "15:59")

    logger.info(f"Total: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"Cached to {cache_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE — Same logic as DAX, adapted for DIA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    date:          str = ""
    direction:     str = ""
    entry_num:     int = 0
    entry_price:   float = 0.0
    exit_price:    float = 0.0
    pnl_pts:       float = 0.0
    mfe_pts:       float = 0.0
    mae_pts:       float = 0.0
    held_bars:     int = 0
    held_to_close: bool = False
    bar_range:     float = 0.0
    range_class:   str = ""
    overnight_bias: str = ""


@dataclass
class DayResult:
    date:           str = ""
    day_of_week:    str = ""
    bar4_high:      float = 0.0
    bar4_low:       float = 0.0
    bar4_range:     float = 0.0
    range_class:    str = ""
    buy_level:      float = 0.0
    sell_level:     float = 0.0
    gap_dir:        str = ""
    gap_size:       float = 0.0
    overnight_bias: str = ""
    trades:         list = field(default_factory=list)
    total_pnl:      float = 0.0
    triggered:      bool = False


def candle_number(timestamp: pd.Timestamp) -> int:
    """Bar number from market open (09:30 ET). Bar 1 = 09:30-09:35."""
    open_time = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def classify_range(rng: float) -> str:
    if rng < NARROW_RANGE:
        return "NARROW"
    elif rng > WIDE_RANGE:
        return "WIDE"
    return "NORMAL"


def classify_gap(prev_close: float, today_open: float) -> tuple[str, float]:
    if prev_close == 0:
        return "FLAT", 0
    gap = round(today_open - prev_close, 2)
    if gap > GAP_THRESHOLD:
        return "GAP_UP", gap
    elif gap < -GAP_THRESHOLD:
        return "GAP_DOWN", gap
    return "FLAT", gap


def calculate_premarket_bias(
    ext_df: pd.DataFrame, trade_date, bar4_high: float, bar4_low: float
) -> str:
    """
    Pre-market range bias (equivalent to V58 overnight range for DAX).
    Pre-market = 04:00-09:30 ET.
    If bar 4 is ABOVE pre-market range → SHORT_ONLY (fade)
    If bar 4 is BELOW pre-market range → LONG_ONLY (fade)
    If bar 4 is INSIDE → STANDARD (both sides)
    """
    try:
        day_data = ext_df[ext_df.index.date == trade_date]
        premarket = day_data.between_time("04:00", "09:29")
        if len(premarket) < 3:
            return "NO_DATA"

        pm_high = premarket["High"].max()
        pm_low = premarket["Low"].min()
        pm_range = pm_high - pm_low

        if pm_range <= 0:
            return "STANDARD"

        bar4_range = bar4_high - bar4_low
        if bar4_range <= 0:
            return "STANDARD"

        if bar4_low >= pm_high:
            return "SHORT_ONLY"
        elif bar4_high <= pm_low:
            return "LONG_ONLY"
        else:
            # Partial overlap — check how much is outside
            if bar4_high > pm_high and bar4_low > pm_low:
                above_pct = (bar4_high - pm_high) / bar4_range
                if above_pct > 0.75:
                    return "SHORT_ONLY"
            elif bar4_low < pm_low and bar4_high < pm_high:
                below_pct = (pm_low - bar4_low) / bar4_range
                if below_pct > 0.75:
                    return "LONG_ONLY"
            return "STANDARD"
    except Exception:
        return "NO_DATA"


def run_backtest(
    rth_df: pd.DataFrame,
    ext_df: pd.DataFrame = None,
    use_overnight: bool = True,
) -> list[DayResult]:
    """Run ASRS backtest on DIA 5-min data. Same logic as DAX."""
    results = []
    prev_close = 0

    for trade_date, day_df in rth_df.groupby(rth_df.index.date):
        if len(day_df) < 10:
            continue

        day_name = trade_date.strftime("%A")

        # Identify bars 1-6
        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx)
            if 1 <= cn <= 6:
                bars[cn] = {
                    "high": row["High"], "low": row["Low"],
                    "open": row["Open"], "close": row["Close"],
                }

        if 4 not in bars:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        bar4 = bars[4]
        bar4_range = round(bar4["high"] - bar4["low"], 2)

        today_open = day_df.iloc[0]["Open"]
        gap_dir, gap_size = classify_gap(prev_close, today_open)
        range_class = classify_range(bar4_range)

        buy_level = round(bar4["high"] + BUFFER_PTS, 2)
        sell_level = round(bar4["low"] - BUFFER_PTS, 2)

        # Pre-market bias (V58 equivalent)
        overnight_bias = "STANDARD"
        if use_overnight and ext_df is not None and not ext_df.empty:
            overnight_bias = calculate_premarket_bias(ext_df, trade_date, bar4["high"], bar4["low"])

        day_result = DayResult(
            date=str(trade_date), day_of_week=day_name,
            bar4_high=round(bar4["high"], 2), bar4_low=round(bar4["low"], 2),
            bar4_range=bar4_range, range_class=range_class,
            buy_level=buy_level, sell_level=sell_level,
            gap_dir=gap_dir, gap_size=round(gap_size, 2),
            overnight_bias=overnight_bias,
        )

        # Post bar-4 candles for simulation
        post_bars = []
        for idx, row in day_df.iterrows():
            if candle_number(idx) > 4:
                post_bars.append((idx, row))

        entries_used = 0
        direction = None
        entry_price = 0.0
        trail_stop = 0.0
        mfe = 0.0
        mae = 0.0
        entry_bar_idx = 0

        for i, (idx, row) in enumerate(post_bars):
            if entries_used >= MAX_ENTRIES and direction is None:
                break

            # ── Entry ──
            if direction is None and entries_used < MAX_ENTRIES:
                can_buy = overnight_bias in ("STANDARD", "LONG_ONLY", "NO_DATA")
                can_sell = overnight_bias in ("STANDARD", "SHORT_ONLY", "NO_DATA")

                if can_buy and row["High"] >= buy_level:
                    direction = "LONG"
                    entry_price = buy_level
                    trail_stop = sell_level
                    entries_used += 1
                    mfe = mae = 0.0
                    entry_bar_idx = i
                elif can_sell and row["Low"] <= sell_level:
                    direction = "SHORT"
                    entry_price = sell_level
                    trail_stop = buy_level
                    entries_used += 1
                    mfe = mae = 0.0
                    entry_bar_idx = i

            # ── Trail & exit ──
            if direction == "LONG":
                mfe = max(mfe, row["High"] - entry_price)
                mae = max(mae, entry_price - row["Low"])

                if i > entry_bar_idx:
                    prev_low = round(post_bars[i - 1][1]["Low"], 2)
                    if prev_low > trail_stop:
                        trail_stop = prev_low

                if row["Low"] <= trail_stop:
                    pnl = round(trail_stop - entry_price, 2)
                    day_result.trades.append(Trade(
                        date=str(trade_date), direction="LONG", entry_num=entries_used,
                        entry_price=entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, mfe_pts=round(mfe, 2), mae_pts=round(mae, 2),
                        held_bars=i - entry_bar_idx, bar_range=bar4_range,
                        range_class=range_class, overnight_bias=overnight_bias,
                    ))
                    direction = None

            elif direction == "SHORT":
                mfe = max(mfe, entry_price - row["Low"])
                mae = max(mae, row["High"] - entry_price)

                if i > entry_bar_idx:
                    prev_high = round(post_bars[i - 1][1]["High"], 2)
                    if prev_high < trail_stop:
                        trail_stop = prev_high

                if row["High"] >= trail_stop:
                    pnl = round(entry_price - trail_stop, 2)
                    day_result.trades.append(Trade(
                        date=str(trade_date), direction="SHORT", entry_num=entries_used,
                        entry_price=entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, mfe_pts=round(mfe, 2), mae_pts=round(mae, 2),
                        held_bars=i - entry_bar_idx, bar_range=bar4_range,
                        range_class=range_class, overnight_bias=overnight_bias,
                    ))
                    direction = None

        # EOD close
        if direction is not None and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 2)
            if direction == "LONG":
                pnl = round(last_price - entry_price, 2)
            else:
                pnl = round(entry_price - last_price, 2)

            day_result.trades.append(Trade(
                date=str(trade_date), direction=direction, entry_num=entries_used,
                entry_price=entry_price, exit_price=last_price,
                pnl_pts=pnl, mfe_pts=round(mfe, 2), mae_pts=round(mae, 2),
                held_bars=len(post_bars) - entry_bar_idx, held_to_close=True,
                bar_range=bar4_range, range_class=range_class,
                overnight_bias=overnight_bias,
            ))

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 2)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)
        prev_close = day_df.iloc[-1]["Close"]

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[DayResult], label: str = ""):
    trades = [t for r in results for t in r.trades]
    if not trades:
        print("No trades!")
        return

    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts < 0]
    total_pnl = sum(t.pnl_pts for t in trades)
    win_sum = sum(t.pnl_pts for t in wins)
    loss_sum = abs(sum(t.pnl_pts for t in losses))
    pf = round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf")
    wr = round(len(wins) / len(trades) * 100, 1)
    avg_win = round(np.mean([t.pnl_pts for t in wins]), 2) if wins else 0
    avg_loss = round(np.mean([t.pnl_pts for t in losses]), 2) if losses else 0

    # Equity curve for drawdown
    equity = []
    running = 0
    for r in sorted(results, key=lambda x: x.date):
        running += r.total_pnl
        equity.append(running)
    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)

    triggered = [r for r in results if r.triggered]

    print(f"\n{'═' * 70}")
    print(f"  ASRS BACKTEST ON DIA — {label}" if label else f"  ASRS BACKTEST ON DIA")
    print(f"{'═' * 70}")
    print(f"  Period:          {results[0].date} to {results[-1].date}")
    print(f"  Trading days:    {len(results)}")
    print(f"  Days triggered:  {len(triggered)} ({len(triggered)/len(results)*100:.0f}%)")
    print(f"  Total trades:    {len(trades)}")
    print(f"  {'─' * 50}")
    print(f"  Total P&L:       ${total_pnl:+,.2f}")
    print(f"  Win rate:        {wr}%")
    print(f"  Profit factor:   {pf}")
    print(f"  Avg winner:      ${avg_win:+.2f}")
    print(f"  Avg loser:       ${avg_loss:+.2f}")
    print(f"  Max drawdown:    ${max_dd:.2f}")
    print(f"  Wins: {len(wins)}  Losses: {len(losses)}  Flat: {len(trades) - len(wins) - len(losses)}")

    # Monthly breakdown
    print(f"\n  {'Month':<12} {'P&L':>10} {'Trades':>8} {'WR':>6}")
    print(f"  {'─' * 40}")
    monthly = {}
    for r in results:
        m = r.date[:7]
        if m not in monthly:
            monthly[m] = {"pnl": 0, "trades": 0, "wins": 0}
        monthly[m]["pnl"] += r.total_pnl
        for t in r.trades:
            monthly[m]["trades"] += 1
            if t.pnl_pts > 0:
                monthly[m]["wins"] += 1

    for m in sorted(monthly):
        d = monthly[m]
        mwr = round(d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0
        print(f"  {m:<12} ${d['pnl']:>+9.2f} {d['trades']:>8} {mwr:>5}%")

    # By overnight bias
    bias_counts = {}
    for r in results:
        if r.triggered:
            b = r.overnight_bias
            if b not in bias_counts:
                bias_counts[b] = {"pnl": 0, "n": 0}
            bias_counts[b]["pnl"] += r.total_pnl
            bias_counts[b]["n"] += 1

    if bias_counts:
        print(f"\n  {'Bias':<15} {'Days':>6} {'P&L':>10} {'Avg':>8}")
        print(f"  {'─' * 42}")
        for b in sorted(bias_counts):
            d = bias_counts[b]
            avg = d["pnl"] / d["n"] if d["n"] > 0 else 0
            print(f"  {b:<15} {d['n']:>6} ${d['pnl']:>+9.2f} ${avg:>+7.2f}")

    # By range class
    range_counts = {}
    for t in trades:
        rc = t.range_class
        if rc not in range_counts:
            range_counts[rc] = {"pnl": 0, "n": 0, "wins": 0}
        range_counts[rc]["pnl"] += t.pnl_pts
        range_counts[rc]["n"] += 1
        if t.pnl_pts > 0:
            range_counts[rc]["wins"] += 1

    print(f"\n  {'Range':<10} {'Trades':>8} {'P&L':>10} {'WR':>6}")
    print(f"  {'─' * 38}")
    for rc in ["NARROW", "NORMAL", "WIDE"]:
        if rc in range_counts:
            d = range_counts[rc]
            wr2 = round(d["wins"] / d["n"] * 100) if d["n"] > 0 else 0
            print(f"  {rc:<10} {d['n']:>8} ${d['pnl']:>+9.2f} {wr2:>5}%")

    print(f"{'═' * 70}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    use_overnight = "--no-overnight" not in sys.argv

    # 2 years of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    logger.info(f"Fetching DIA 5-min data: {start_date} to {end_date}")

    # Fetch RTH and extended hours
    rth_df = await fetch_dia_5min(start_date, end_date, extended=False)
    if rth_df.empty:
        return

    ext_df = None
    if use_overnight:
        ext_df = await fetch_dia_5min(start_date, end_date, extended=True)

    logger.info(f"Running ASRS backtest on DIA ({len(rth_df)} RTH bars)...")

    # Run with pre-market bias
    results = run_backtest(rth_df, ext_df, use_overnight=use_overnight)
    print_results(results, "With Pre-Market Bias" if use_overnight else "No Bias Filter")

    # Also run without bias for comparison
    if use_overnight:
        results_no_bias = run_backtest(rth_df, ext_df, use_overnight=False)
        print_results(results_no_bias, "No Bias Filter (comparison)")


if __name__ == "__main__":
    asyncio.run(main())
