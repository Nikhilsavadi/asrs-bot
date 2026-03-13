"""
backtest_multi.py — ASRS Bar-4 Backtest Across Multiple Instruments
═══════════════════════════════════════════════════════════════════════════════

Tests the bar-4 breakout + candle trail strategy on:
  DAX, SPY, DIA, FTSE

Each instrument has different trading hours, thresholds, and pre-market ranges.
Core logic is identical: bar 4 sets levels → breakout entry → candle trail exit.

Usage:
    python backtest_multi.py              # All instruments
    python backtest_multi.py --spy        # Single instrument
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MULTI_BT")

# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENT CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {
    "DAX": {
        "rth_file": "data/dax/historical_bars.parquet",
        "ext_file": "data/dax/historical_bars_all.parquet",
        "rth_start": "09:00",
        "rth_end": "17:29",
        "open_hour": 9,
        "open_minute": 0,
        "premarket_start": "00:00",
        "premarket_end": "06:00",
        "buffer_pts": 2.0,
        "narrow_range": 15.0,
        "wide_range": 40.0,
        "currency": "pts",
        "eod_hour": 17,
        "eod_minute": 30,
    },
    "SPY": {
        "rth_file": "data/spy/spy_5min_rth_2024-03-10_2026-03-10.parquet",
        "ext_file": "data/spy/spy_5min_ext_2024-03-10_2026-03-10.parquet",
        "rth_start": "09:30",
        "rth_end": "15:59",
        "open_hour": 9,
        "open_minute": 30,
        "premarket_start": "04:00",
        "premarket_end": "09:29",
        "buffer_pts": 0.10,
        "narrow_range": 0.50,
        "wide_range": 2.00,
        "currency": "$",
        "eod_hour": 16,
        "eod_minute": 0,
    },
    "DIA": {
        "rth_file": "data/dia/dia_5min_rth_2024-03-10_2026-03-10.parquet",
        "ext_file": "data/dia/dia_5min_ext_2024-03-10_2026-03-10.parquet",
        "rth_start": "09:30",
        "rth_end": "15:59",
        "open_hour": 9,
        "open_minute": 30,
        "premarket_start": "04:00",
        "premarket_end": "09:29",
        "buffer_pts": 0.10,
        "narrow_range": 0.50,
        "wide_range": 2.00,
        "currency": "$",
        "eod_hour": 16,
        "eod_minute": 0,
    },
    "FTSE": {
        "rth_file": "data/ftse/ftse_rth.parquet",
        "ext_file": "data/ftse/ftse_all.parquet",
        "rth_start": "08:00",
        "rth_end": "16:29",
        "open_hour": 8,
        "open_minute": 0,
        "premarket_start": "00:00",
        "premarket_end": "07:59",
        "buffer_pts": 2.0,
        "narrow_range": 10.0,
        "wide_range": 30.0,
        "currency": "pts",
        "eod_hour": 16,
        "eod_minute": 30,
    },
}

MAX_ENTRIES = 2  # entry + flip


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    date: str = ""
    direction: str = ""
    entry_num: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pts: float = 0.0
    mfe_pts: float = 0.0
    mae_pts: float = 0.0
    held_bars: int = 0
    held_to_close: bool = False
    bar_range: float = 0.0
    range_class: str = ""
    overnight_bias: str = ""
    exit_reason: str = ""


@dataclass
class DayResult:
    date: str = ""
    day_of_week: str = ""
    bar4_high: float = 0.0
    bar4_low: float = 0.0
    bar4_range: float = 0.0
    range_class: str = ""
    buy_level: float = 0.0
    sell_level: float = 0.0
    overnight_bias: str = ""
    trades: list = field(default_factory=list)
    total_pnl: float = 0.0
    triggered: bool = False


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def candle_number(timestamp, open_hour: int, open_minute: int) -> int:
    """Bar number from market open. Bar 1 = first 5-min candle."""
    open_time = timestamp.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    if mins < 0:
        return -1
    return (mins // 5) + 1


def classify_range(rng: float, narrow: float, wide: float) -> str:
    if rng < narrow:
        return "NARROW"
    elif rng > wide:
        return "WIDE"
    return "NORMAL"


def calculate_premarket_bias(ext_df, trade_date, bar4_high, bar4_low, cfg) -> str:
    """Pre-market/overnight range bias. Same V58 logic across instruments."""
    try:
        day_data = ext_df[ext_df.index.date == trade_date]
        premarket = day_data.between_time(cfg["premarket_start"], cfg["premarket_end"])
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
            if bar4_high > pm_high and bar4_low > pm_low:
                if (bar4_high - pm_high) / bar4_range > 0.75:
                    return "SHORT_ONLY"
            elif bar4_low < pm_low and bar4_high < pm_high:
                if (pm_low - bar4_low) / bar4_range > 0.75:
                    return "LONG_ONLY"
            return "STANDARD"
    except Exception:
        return "NO_DATA"


def run_backtest(instrument: str, cfg: dict, use_bias: bool = True) -> list[DayResult]:
    """Run ASRS bar-4 backtest on any instrument."""

    # Load data
    rth_path = cfg["rth_file"]
    ext_path = cfg["ext_file"]

    if not os.path.exists(rth_path):
        logger.error(f"{instrument}: RTH data not found at {rth_path}")
        return []

    rth_df = pd.read_parquet(rth_path)
    logger.info(f"{instrument}: Loaded {len(rth_df)} RTH bars")

    ext_df = None
    if use_bias and os.path.exists(ext_path):
        ext_df = pd.read_parquet(ext_path)
        logger.info(f"{instrument}: Loaded {len(ext_df)} extended bars")

    results = []
    prev_close = 0.0

    for trade_date, day_df in rth_df.groupby(rth_df.index.date):
        # Skip weekends
        if trade_date.weekday() >= 5:
            continue
        if len(day_df) < 10:
            continue

        # Identify bar 4
        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx, cfg["open_hour"], cfg["open_minute"])
            if 1 <= cn <= 6:
                bars[cn] = {"high": row["High"], "low": row["Low"],
                            "open": row["Open"], "close": row["Close"]}

        if 4 not in bars:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        bar4 = bars[4]
        bar4_range = round(bar4["high"] - bar4["low"], 4)
        range_class = classify_range(bar4_range, cfg["narrow_range"], cfg["wide_range"])

        buy_level = round(bar4["high"] + cfg["buffer_pts"], 4)
        sell_level = round(bar4["low"] - cfg["buffer_pts"], 4)

        # Overnight/pre-market bias
        overnight_bias = "STANDARD"
        if use_bias and ext_df is not None:
            overnight_bias = calculate_premarket_bias(
                ext_df, trade_date, bar4["high"], bar4["low"], cfg)

        day_result = DayResult(
            date=str(trade_date),
            day_of_week=trade_date.strftime("%A"),
            bar4_high=round(bar4["high"], 4),
            bar4_low=round(bar4["low"], 4),
            bar4_range=bar4_range,
            range_class=range_class,
            buy_level=buy_level,
            sell_level=sell_level,
            overnight_bias=overnight_bias,
        )

        # Post bar-4 candles
        post_bars = []
        for idx, row in day_df.iterrows():
            if candle_number(idx, cfg["open_hour"], cfg["open_minute"]) > 4:
                post_bars.append((idx, row))

        entries_used = 0
        direction = None
        entry_price = 0.0
        trail_stop = 0.0
        mfe = mae = 0.0
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
                    prev_low = post_bars[i - 1][1]["Low"]
                    if prev_low > trail_stop:
                        trail_stop = prev_low

                if row["Low"] <= trail_stop:
                    pnl = round(trail_stop - entry_price, 4)
                    day_result.trades.append(Trade(
                        date=str(trade_date), direction="LONG", entry_num=entries_used,
                        entry_price=round(entry_price, 4), exit_price=round(trail_stop, 4),
                        pnl_pts=pnl, mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
                        held_bars=i - entry_bar_idx, bar_range=bar4_range,
                        range_class=range_class, overnight_bias=overnight_bias,
                        exit_reason="TRAIL_STOP",
                    ))
                    # Flip: re-enter opposite direction
                    direction = None

            elif direction == "SHORT":
                mfe = max(mfe, entry_price - row["Low"])
                mae = max(mae, row["High"] - entry_price)

                if i > entry_bar_idx:
                    prev_high = post_bars[i - 1][1]["High"]
                    if prev_high < trail_stop:
                        trail_stop = prev_high

                if row["High"] >= trail_stop:
                    pnl = round(entry_price - trail_stop, 4)
                    day_result.trades.append(Trade(
                        date=str(trade_date), direction="SHORT", entry_num=entries_used,
                        entry_price=round(entry_price, 4), exit_price=round(trail_stop, 4),
                        pnl_pts=pnl, mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
                        held_bars=i - entry_bar_idx, bar_range=bar4_range,
                        range_class=range_class, overnight_bias=overnight_bias,
                        exit_reason="TRAIL_STOP",
                    ))
                    direction = None

        # EOD close
        if direction is not None and post_bars:
            last_price = post_bars[-1][1]["Close"]
            if direction == "LONG":
                pnl = round(last_price - entry_price, 4)
            else:
                pnl = round(entry_price - last_price, 4)

            day_result.trades.append(Trade(
                date=str(trade_date), direction=direction, entry_num=entries_used,
                entry_price=round(entry_price, 4), exit_price=round(last_price, 4),
                pnl_pts=pnl, mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
                held_bars=len(post_bars) - entry_bar_idx, held_to_close=True,
                bar_range=bar4_range, range_class=range_class,
                overnight_bias=overnight_bias, exit_reason="EOD_CLOSE",
            ))

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 4)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)
        prev_close = day_df.iloc[-1]["Close"]

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[DayResult], instrument: str, currency: str, label: str = ""):
    trades = [t for r in results for t in r.trades]
    if not trades:
        print(f"\n  {instrument}: No trades!")
        return

    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts < 0]
    flats = [t for t in trades if t.pnl_pts == 0]
    total_pnl = sum(t.pnl_pts for t in trades)
    win_sum = sum(t.pnl_pts for t in wins)
    loss_sum = abs(sum(t.pnl_pts for t in losses))
    pf = round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf")
    wr = round(len(wins) / len(trades) * 100, 1)
    avg_win = round(np.mean([t.pnl_pts for t in wins]), 2) if wins else 0
    avg_loss = round(np.mean([t.pnl_pts for t in losses]), 2) if losses else 0
    avg_mfe = round(np.mean([t.mfe_pts for t in trades]), 2)

    # Equity curve for drawdown
    equity = []
    running = 0.0
    for r in sorted(results, key=lambda x: x.date):
        running += r.total_pnl
        equity.append(running)
    peak = max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    triggered = [r for r in results if r.triggered]

    # Expectancy
    expectancy = round(total_pnl / len(trades), 2)

    print(f"\n{'=' * 70}")
    title = f"  ASRS BAR-4 — {instrument}"
    if label:
        title += f" ({label})"
    print(title)
    print(f"{'=' * 70}")
    print(f"  Period:          {results[0].date} to {results[-1].date}")
    print(f"  Trading days:    {len(results)}")
    print(f"  Days triggered:  {len(triggered)} ({len(triggered)/len(results)*100:.0f}%)")
    print(f"  Total trades:    {len(trades)} (W:{len(wins)} L:{len(losses)} F:{len(flats)})")
    print(f"  {'-' * 50}")
    print(f"  Total P&L:       {total_pnl:+,.2f} {currency}")
    print(f"  Profit factor:   {pf}")
    print(f"  Win rate:        {wr}%")
    print(f"  Expectancy:      {expectancy:+.2f} {currency}/trade")
    print(f"  Avg winner:      {avg_win:+.2f} {currency}")
    print(f"  Avg loser:       {avg_loss:+.2f} {currency}")
    print(f"  Avg MFE:         {avg_mfe:+.2f} {currency}")
    print(f"  Max drawdown:    {max_dd:.2f} {currency}")

    # By bias
    bias_stats = {}
    for t in trades:
        b = t.overnight_bias
        if b not in bias_stats:
            bias_stats[b] = {"pnl": 0, "n": 0, "wins": 0}
        bias_stats[b]["pnl"] += t.pnl_pts
        bias_stats[b]["n"] += 1
        if t.pnl_pts > 0:
            bias_stats[b]["wins"] += 1

    print(f"\n  {'Bias':<15} {'Trades':>7} {'P&L':>12} {'WR':>6} {'Avg':>10}")
    print(f"  {'-' * 52}")
    for b in sorted(bias_stats):
        d = bias_stats[b]
        wr2 = round(d["wins"] / d["n"] * 100) if d["n"] > 0 else 0
        avg = d["pnl"] / d["n"] if d["n"] > 0 else 0
        print(f"  {b:<15} {d['n']:>7} {d['pnl']:>+11.2f} {wr2:>5}% {avg:>+9.2f}")

    # By range class
    range_stats = {}
    for t in trades:
        rc = t.range_class
        if rc not in range_stats:
            range_stats[rc] = {"pnl": 0, "n": 0, "wins": 0}
        range_stats[rc]["pnl"] += t.pnl_pts
        range_stats[rc]["n"] += 1
        if t.pnl_pts > 0:
            range_stats[rc]["wins"] += 1

    print(f"\n  {'Range':<10} {'Trades':>7} {'P&L':>12} {'WR':>6}")
    print(f"  {'-' * 38}")
    for rc in ["NARROW", "NORMAL", "WIDE"]:
        if rc in range_stats:
            d = range_stats[rc]
            wr2 = round(d["wins"] / d["n"] * 100) if d["n"] > 0 else 0
            print(f"  {rc:<10} {d['n']:>7} {d['pnl']:>+11.2f} {wr2:>5}%")

    # Monthly
    print(f"\n  {'Month':<10} {'P&L':>12} {'Trades':>7} {'WR':>6}")
    print(f"  {'-' * 38}")
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
        print(f"  {m:<10} {d['pnl']:>+11.2f} {d['trades']:>7} {mwr:>5}%")

    print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(all_results: dict):
    """Side-by-side comparison of all instruments."""
    print(f"\n{'=' * 80}")
    print(f"  ASRS BAR-4 STRATEGY — CROSS-INSTRUMENT COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Instrument':<12} {'Period':<25} {'Trades':>7} {'P&L':>12} {'PF':>6} {'WR':>6} {'Expect':>10} {'MaxDD':>10}")
    print(f"  {'-' * 78}")

    for inst, (results, cfg) in all_results.items():
        trades = [t for r in results for t in r.trades]
        if not trades:
            print(f"  {inst:<12} {'No data':<25}")
            continue

        wins = [t for t in trades if t.pnl_pts > 0]
        losses = [t for t in trades if t.pnl_pts < 0]
        total_pnl = sum(t.pnl_pts for t in trades)
        win_sum = sum(t.pnl_pts for t in wins)
        loss_sum = abs(sum(t.pnl_pts for t in losses))
        pf = round(win_sum / loss_sum, 2) if loss_sum > 0 else float("inf")
        wr = round(len(wins) / len(trades) * 100, 1)
        expectancy = round(total_pnl / len(trades), 2)

        equity = []
        running = 0.0
        for r in sorted(results, key=lambda x: x.date):
            running += r.total_pnl
            equity.append(running)
        peak = max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)

        period = f"{results[0].date} to {results[-1].date}"
        c = cfg["currency"]
        print(f"  {inst:<12} {period:<25} {len(trades):>7} {total_pnl:>+11.2f} {pf:>6.2f} {wr:>5.1f}% {expectancy:>+9.2f} {max_dd:>9.2f}")

    print(f"{'=' * 80}")
    print(f"  Note: P&L is in instrument points/dollars, not normalized.")
    print(f"  PF > 1.5 = tradeable. PF > 3 = strong edge. PF > 5 = exceptional.")
    print(f"{'=' * 80}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = set(sys.argv[1:])

    # Determine which instruments to run
    if args:
        run_instruments = [k for k in INSTRUMENTS if f"--{k.lower()}" in args]
        if not run_instruments:
            run_instruments = list(INSTRUMENTS.keys())
    else:
        run_instruments = list(INSTRUMENTS.keys())

    all_results = {}

    for inst in run_instruments:
        cfg = INSTRUMENTS[inst]
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Running ASRS backtest on {inst}...")
        logger.info(f"{'=' * 40}")

        results = run_backtest(inst, cfg, use_bias=True)

        if results:
            print_results(results, inst, cfg["currency"], "With Bias")
            all_results[inst] = (results, cfg)
        else:
            logger.warning(f"{inst}: No results")

    # Cross-instrument comparison
    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
