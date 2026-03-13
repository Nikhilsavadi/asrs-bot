"""
backtest_variants.py — Profit-Taking Variant Backtester
========================================================

Compares 5 trailing stop strategies on Bar 5 signals:
  1. BASELINE     — Current 10 EMA trail (3-phase: underwater -> breakeven -> EMA)
  2. PARTIAL       — Close half at +20-25 pts, let rest ride on EMA trail
  3. FAST_EMA      — 8 EMA instead of 10
  4. HYBRID        — Fixed +15 pts trail until EMA catches up, then EMA
  5. TIME_TIGHTEN  — After 30 mins in profit, switch to 5 EMA

Usage:
    python backtest_variants.py
"""

import asyncio
import os
import sys
import logging
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from dax_bot import config
from backtest import (
    Trade, DayResult, candle_number, classify_range, classify_gap,
    analyse_context_bars, _calc_ema_series, RESULTS_DIR,
)
from dax_bot.broker import Broker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VARIANTS")


# ══════════════════════════════════════════════════════════════════════════════
#  Core simulation — parameterised for all variants
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VariantConfig:
    name: str
    ema_period: int = 10
    partial_take_profit: bool = False
    partial_target_pts: float = 22.5     # close half at this many pts profit
    hybrid_fixed_trail: bool = False
    hybrid_trail_pts: float = 15.0       # fixed trail distance
    time_tighten: bool = False
    time_tighten_bars: int = 6           # 30 mins = 6 x 5-min bars
    time_tighten_ema: int = 5            # switch to 5 EMA after timeout


VARIANTS = [
    VariantConfig(name="BASELINE (10 EMA)"),
    VariantConfig(name="PARTIAL TP (+22.5)", partial_take_profit=True),
    VariantConfig(name="FAST EMA (8)", ema_period=8),
    VariantConfig(name="HYBRID (15pt+EMA)", hybrid_fixed_trail=True),
    VariantConfig(name="TIME TIGHTEN (30m->5EMA)", time_tighten=True),
]


def run_variant(
    df: pd.DataFrame,
    all_df: pd.DataFrame | None,
    vcfg: VariantConfig,
    signal_bar: int = 5,
) -> list[DayResult]:
    """Run a single variant backtest over the full dataset."""
    from dax_bot.overnight import calculate_overnight_range

    be_trigger = config.TRAIL_BREAKEVEN_TRIGGER     # 5 pts
    ema_trigger = config.TRAIL_EMA_TRIGGER           # 10 pts
    ema_buffer = config.TRAIL_EMA_BUFFER             # 0.005

    results = []
    prev_close = 0

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        day_name = trade_date.strftime("%A")

        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx)
            if 1 <= cn <= 6:
                bars[cn] = {
                    "high": row["High"], "low": row["Low"],
                    "open": row["Open"], "close": row["Close"],
                }

        if signal_bar not in bars:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        sig = bars[signal_bar]
        sig_range = round(sig["high"] - sig["low"], 1)

        today_open = day_df.iloc[0]["Open"]
        gap_dir, gap_size = classify_gap(prev_close, today_open)
        context_bars = [bars[i] for i in [1, 2, 3] if i in bars]
        context = analyse_context_bars(context_bars)
        range_class = classify_range(sig_range)

        buy_level = round(sig["high"] + config.BUFFER_PTS, 1)
        sell_level = round(sig["low"] - config.BUFFER_PTS, 1)

        prev_day_dir = ""
        if prev_close > 0:
            prev_day_dir = "UP" if today_open > prev_close else "DOWN"

        overnight_bias_str = "NO_DATA"
        bar4_vs_overnight = ""
        if all_df is not None and not all_df.empty:
            try:
                day_all = all_df.loc[str(trade_date)]
                overnight = day_all.between_time("00:00", "06:00")
                if not overnight.empty:
                    ov_result = calculate_overnight_range(
                        overnight, sig["high"], sig["low"]
                    )
                    overnight_bias_str = ov_result.bias.value
                    bar4_vs_overnight = ov_result.bar4_vs_range
            except (KeyError, Exception):
                pass

        day_result = DayResult(
            date=str(trade_date), day_of_week=day_name,
            bar4_high=round(sig["high"], 1), bar4_low=round(sig["low"], 1),
            bar4_range=sig_range, range_class=range_class,
            buy_level=buy_level, sell_level=sell_level,
            gap_dir=gap_dir, gap_size=round(gap_size, 1),
            context=context, overnight_bias=overnight_bias_str,
            bar4_vs_overnight=bar4_vs_overnight,
        )

        # Build post-signal-bar candles
        post_bars = []
        all_closes = []
        for idx, row in day_df.iterrows():
            all_closes.append(row["Close"])
            if candle_number(idx) > signal_bar:
                post_bars.append((idx, row, len(all_closes) - 1))

        # Compute EMA series for primary and time-tighten periods
        ema_series = _calc_ema_series(all_closes, vcfg.ema_period)
        ema_series_tight = None
        if vcfg.time_tighten:
            ema_series_tight = _calc_ema_series(all_closes, vcfg.time_tighten_ema)

        # -- Simulation state --
        direction = ""
        entry_price = 0.0
        trail_stop = 0.0
        initial_stop = 0.0
        mfe = 0.0
        mae = 0.0
        entry_bar_idx = 0
        phase = "UNDERWATER"
        entries_used = 0
        # Partial tracking
        full_qty = 1.0  # 1.0 = full position, 0.5 = half taken
        partial_taken = False
        # Time-tighten tracking
        bars_in_profit = 0
        use_tight_ema = False

        for i, (idx, row, close_idx) in enumerate(post_bars):
            ema_val = ema_series[close_idx]
            ema_tight_val = ema_series_tight[close_idx] if ema_series_tight else None

            # -- Check exit for open position --
            if direction:
                if direction == "LONG":
                    mfe = max(mfe, row["High"] - entry_price)
                    mae = max(mae, entry_price - row["Low"])
                    favour = row["High"] - entry_price
                    above_ema = ema_val is not None and row["Close"] > ema_val
                else:
                    mfe = max(mfe, entry_price - row["Low"])
                    mae = max(mae, row["High"] - entry_price)
                    favour = entry_price - row["Low"]
                    above_ema = ema_val is not None and row["Close"] < ema_val

                # Track time in profit for time-tighten variant
                if vcfg.time_tighten:
                    if favour > 0:
                        bars_in_profit += 1
                    else:
                        bars_in_profit = 0
                    if bars_in_profit >= vcfg.time_tighten_bars:
                        use_tight_ema = True

                # Phase transitions
                old_phase = phase
                active_ema = ema_tight_val if (vcfg.time_tighten and use_tight_ema and ema_tight_val is not None) else ema_val

                if active_ema is not None and favour >= ema_trigger and above_ema:
                    phase = "EMA_TRAIL"
                elif favour >= be_trigger:
                    if phase == "UNDERWATER":
                        phase = "BREAKEVEN"

                # --- PARTIAL TAKE PROFIT ---
                if vcfg.partial_take_profit and not partial_taken and phase != "UNDERWATER":
                    if favour >= vcfg.partial_target_pts:
                        # Record the partial close as a trade
                        if direction == "LONG":
                            partial_exit = round(entry_price + vcfg.partial_target_pts, 1)
                        else:
                            partial_exit = round(entry_price - vcfg.partial_target_pts, 1)
                        partial_pnl = round(vcfg.partial_target_pts * 0.5, 1)  # half qty
                        day_result.trades.append(Trade(
                            date=str(trade_date), day_of_week=day_name,
                            direction=direction, entry_num=i,
                            entry_price=entry_price, exit_price=partial_exit,
                            pnl_pts=partial_pnl, mfe_pts=round(mfe, 1),
                            mae_pts=round(mae, 1), held_bars=i - entry_bar_idx,
                            bar_range=sig_range, range_class=range_class,
                            gap_dir=gap_dir, gap_size=round(gap_size, 1),
                            context=context, bar_bullish=sig["close"] > sig["open"],
                            prev_day_dir=prev_day_dir,
                            overnight_bias=overnight_bias_str,
                            bar4_vs_overnight=bar4_vs_overnight,
                        ))
                        partial_taken = True
                        full_qty = 0.5

                # Update trail stop based on phase
                if phase == "BREAKEVEN" and old_phase == "UNDERWATER":
                    if direction == "LONG":
                        trail_stop = max(trail_stop, entry_price)
                    else:
                        trail_stop = min(trail_stop, entry_price)

                elif phase == "EMA_TRAIL" and active_ema is not None:
                    # --- HYBRID: use fixed trail until EMA catches up ---
                    if vcfg.hybrid_fixed_trail:
                        if direction == "LONG":
                            fixed_trail = round(row["High"] - vcfg.hybrid_trail_pts, 1)
                            ema_trail = round(active_ema * (1 - ema_buffer), 1)
                            ema_trail = max(ema_trail, entry_price)
                            # Use whichever is tighter (higher for LONG)
                            raw = max(fixed_trail, ema_trail)
                            trail_stop = max(trail_stop, raw)
                        else:
                            fixed_trail = round(row["Low"] + vcfg.hybrid_trail_pts, 1)
                            ema_trail = round(active_ema * (1 + ema_buffer), 1)
                            ema_trail = min(ema_trail, entry_price)
                            raw = min(fixed_trail, ema_trail)
                            trail_stop = min(trail_stop, raw)
                    else:
                        # Standard EMA trail
                        if direction == "LONG":
                            raw = round(active_ema * (1 - ema_buffer), 1)
                            raw = max(raw, entry_price)
                            trail_stop = max(trail_stop, raw)
                        else:
                            raw = round(active_ema * (1 + ema_buffer), 1)
                            raw = min(raw, entry_price)
                            trail_stop = min(trail_stop, raw)

                # Check stop hit
                stopped = False
                if phase == "EMA_TRAIL":
                    if direction == "LONG" and row["Close"] < trail_stop:
                        stopped = True
                    elif direction == "SHORT" and row["Close"] > trail_stop:
                        stopped = True
                else:
                    if direction == "LONG" and row["Low"] <= trail_stop:
                        stopped = True
                    elif direction == "SHORT" and row["High"] >= trail_stop:
                        stopped = True

                if stopped:
                    exit_price = trail_stop
                    if direction == "LONG":
                        pnl = round(exit_price - entry_price, 1)
                    else:
                        pnl = round(entry_price - exit_price, 1)

                    # Scale by remaining qty for partial variant
                    pnl = round(pnl * full_qty, 1)

                    day_result.trades.append(Trade(
                        date=str(trade_date), day_of_week=day_name,
                        direction=direction, entry_num=entry_bar_idx,
                        entry_price=entry_price, exit_price=exit_price,
                        pnl_pts=pnl, mfe_pts=round(mfe, 1),
                        mae_pts=round(mae, 1),
                        held_bars=i - entry_bar_idx,
                        bar_range=sig_range, range_class=range_class,
                        gap_dir=gap_dir, gap_size=round(gap_size, 1),
                        context=context, bar_bullish=sig["close"] > sig["open"],
                        prev_day_dir=prev_day_dir,
                        overnight_bias=overnight_bias_str,
                        bar4_vs_overnight=bar4_vs_overnight,
                    ))

                    # Reset for potential flip
                    direction = ""
                    phase = "UNDERWATER"
                    partial_taken = False
                    full_qty = 1.0
                    bars_in_profit = 0
                    use_tight_ema = False
                    continue

            # -- Check for new entry --
            if not direction and entries_used < config.MAX_ENTRIES:
                if row["High"] >= buy_level:
                    direction = "LONG"
                    entry_price = buy_level
                    trail_stop = sell_level
                    initial_stop = sell_level
                    mfe = mae = 0.0
                    entry_bar_idx = i
                    entries_used += 1
                    phase = "UNDERWATER"
                    partial_taken = False
                    full_qty = 1.0
                    bars_in_profit = 0
                    use_tight_ema = False
                elif row["Low"] <= sell_level:
                    direction = "SHORT"
                    entry_price = sell_level
                    trail_stop = buy_level
                    initial_stop = buy_level
                    mfe = mae = 0.0
                    entry_bar_idx = i
                    entries_used += 1
                    phase = "UNDERWATER"
                    partial_taken = False
                    full_qty = 1.0
                    bars_in_profit = 0
                    use_tight_ema = False

        # -- EOD: close open position --
        if direction and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 1)
            if direction == "LONG":
                pnl = round(last_price - entry_price, 1)
            else:
                pnl = round(entry_price - last_price, 1)
            pnl = round(pnl * full_qty, 1)

            day_result.trades.append(Trade(
                date=str(trade_date), day_of_week=day_name,
                direction=direction, entry_num=entry_bar_idx,
                entry_price=entry_price, exit_price=last_price,
                pnl_pts=pnl, mfe_pts=round(mfe, 1),
                mae_pts=round(mae, 1),
                held_bars=len(post_bars) - entry_bar_idx,
                held_to_close=True,
                bar_range=sig_range, range_class=range_class,
                gap_dir=gap_dir, gap_size=round(gap_size, 1),
                context=context, bar_bullish=sig["close"] > sig["open"],
                prev_day_dir=prev_day_dir,
                overnight_bias=overnight_bias_str,
                bar4_vs_overnight=bar4_vs_overnight,
            ))

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)
        prev_close = day_df.iloc[-1]["Close"]

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Stats & Comparison Output
# ══════════════════════════════════════════════════════════════════════════════

def calc_stats(results: list[DayResult]) -> dict:
    trades = [t for r in results for t in r.trades]
    pnl = sum(t.pnl_pts for t in trades)
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts < 0]
    flats = [t for t in trades if t.pnl_pts == 0]
    wr = round(len(wins) / len(trades) * 100) if trades else 0

    avg_w = round(np.mean([t.pnl_pts for t in wins]), 1) if wins else 0
    avg_l = round(np.mean([t.pnl_pts for t in losses]), 1) if losses else 0

    # Profit factor
    gross_w = sum(t.pnl_pts for t in wins)
    gross_l = abs(sum(t.pnl_pts for t in losses))
    pf = round(gross_w / gross_l, 2) if gross_l > 0 else float('inf')

    # Max drawdown on equity curve
    eq = []
    run = 0
    for r in results:
        run += r.total_pnl
        eq.append(run)
    pk = dd = 0
    for e in eq:
        pk = max(pk, e)
        dd = max(dd, pk - e)

    # Average held bars
    avg_held = round(np.mean([t.held_bars for t in trades]), 1) if trades else 0

    # Avg MFE captured
    avg_mfe = round(np.mean([t.mfe_pts for t in trades]), 1) if trades else 0
    avg_mfe_capture = 0
    if trades:
        captures = []
        for t in trades:
            if t.mfe_pts > 0:
                captures.append(t.pnl_pts / t.mfe_pts * 100)
        avg_mfe_capture = round(np.mean(captures), 1) if captures else 0

    return {
        "n": len(trades), "pnl": round(pnl, 1), "wr": wr,
        "wins": len(wins), "losses": len(losses), "flats": len(flats),
        "avg_w": avg_w, "avg_l": avg_l, "pf": pf,
        "dd": round(dd, 1), "eq": eq,
        "avg_held": avg_held, "avg_mfe": avg_mfe,
        "mfe_capture": avg_mfe_capture,
    }


def print_comparison(all_results: dict[str, list[DayResult]]):
    """Print side-by-side comparison of all variants."""
    stats = {name: calc_stats(res) for name, res in all_results.items()}

    # Header
    names = list(stats.keys())
    print("\n" + "=" * 100)
    print("  ASRS PROFIT-TAKING VARIANTS — Bar 5 Comparison")
    print("=" * 100)

    # Table header
    col_w = 14
    header = f"{'Metric':<22}"
    for n in names:
        short = n[:col_w]
        header += f" {short:>{col_w}}"
    print(header)
    print("-" * (22 + (col_w + 1) * len(names)))

    # Rows
    rows = [
        ("Trades", "n", "d"),
        ("Win Rate %", "wr", "d"),
        ("Winners", "wins", "d"),
        ("Losers", "losses", "d"),
        ("Flat", "flats", "d"),
        ("Avg Win (pts)", "avg_w", "+f"),
        ("Avg Loss (pts)", "avg_l", "+f"),
        ("Profit Factor", "pf", "f"),
        ("Total P&L (pts)", "pnl", "+f"),
        ("Max Drawdown", "dd", "f"),
        ("Avg Held (bars)", "avg_held", "f"),
        ("Avg MFE (pts)", "avg_mfe", "f"),
        ("MFE Capture %", "mfe_capture", "f"),
    ]

    for label, key, fmt in rows:
        line = f"  {label:<20}"
        for n in names:
            val = stats[n][key]
            if fmt == "d":
                line += f" {val:>{col_w}}"
            elif fmt == "+f":
                line += f" {val:>+{col_w}.1f}"
            else:
                line += f" {val:>{col_w}.1f}"
        print(line)

    # Delta vs baseline
    base_pnl = stats[names[0]]["pnl"]
    print("-" * (22 + (col_w + 1) * len(names)))
    line = f"  {'vs Baseline':<20}"
    for n in names:
        diff = stats[n]["pnl"] - base_pnl
        line += f" {diff:>+{col_w}.1f}"
    print(line)

    # Best variant
    best_name = max(stats.keys(), key=lambda n: stats[n]["pnl"])
    best_pnl = stats[best_name]["pnl"]
    print(f"\n  BEST: {best_name} with {best_pnl:+.1f} pts total P&L")

    # Monthly breakdown
    print(f"\n{'='*100}")
    print("  MONTHLY BREAKDOWN")
    print(f"{'='*100}")

    monthly = {n: defaultdict(float) for n in names}
    for name, res_list in all_results.items():
        for r in res_list:
            monthly[name][r.date[:7]] += r.total_pnl

    all_months = sorted(set(m for d in monthly.values() for m in d.keys()))
    header = f"  {'Month':<10}"
    for n in names:
        header += f" {n[:10]:>10}"
    header += f" {'Best':>10}"
    print(header)
    print("  " + "-" * (10 + (11) * (len(names) + 1)))

    for m in all_months:
        line = f"  {m:<10}"
        vals = {}
        for n in names:
            v = monthly[n].get(m, 0)
            vals[n] = v
            line += f" {v:>+9.0f}"
        best = max(vals, key=lambda n: vals[n])
        line += f" {best[:10]:>10}"
        print(line)

    # Totals
    line = f"  {'TOTAL':<10}"
    for n in names:
        line += f" {stats[n]['pnl']:>+9.0f}"
    print(line)

    print()


def save_comparison_file(all_results: dict[str, list[DayResult]], filepath: str):
    """Save comparison to a text file."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_comparison(all_results)

    with open(filepath, "w") as f:
        f.write(buf.getvalue())

    logger.info(f"Comparison saved to {filepath}")


def generate_chart(all_results: dict[str, list[DayResult]]):
    """Generate equity curve comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from datetime import datetime

        fig, ax = plt.subplots(figsize=(16, 8))

        colors = ["#9E9E9E", "#FF9800", "#2196F3", "#4CAF50", "#9C27B0"]

        for (name, res_list), color in zip(all_results.items(), colors):
            stats = calc_stats(res_list)
            dates = [datetime.strptime(r.date, "%Y-%m-%d") for r in res_list]
            lw = 2.5 if name == max(all_results.keys(), key=lambda n: calc_stats(all_results[n])["pnl"]) else 1.2
            ax.plot(dates, stats["eq"], color=color, linewidth=lw,
                    label=f"{name} ({stats['pnl']:+.0f} pts)")

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("ASRS Profit-Taking Variants — Equity Curves", fontsize=16, fontweight="bold")
        ax.set_ylabel("Cumulative P&L (points)")
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(RESULTS_DIR, "charts", "variant_comparison.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Chart saved to {path}")
    except Exception as e:
        logger.error(f"Chart error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    # Load cached data
    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")

    if not os.path.exists(rth_path):
        logger.error(f"No cached RTH data at {rth_path}. Run backtest.py first to fetch data.")
        return

    logger.info("Loading cached data...")
    df = pd.read_parquet(rth_path)
    all_df = pd.read_parquet(all_path) if os.path.exists(all_path) else None
    logger.info(f"RTH: {len(df)} bars, All-hours: {len(all_df) if all_df is not None else 0} bars")

    # Run all variants
    all_results = {}
    for vcfg in VARIANTS:
        logger.info(f"Running variant: {vcfg.name}...")
        results = run_variant(df, all_df, vcfg, signal_bar=5)
        all_results[vcfg.name] = results
        s = calc_stats(results)
        logger.info(f"  {vcfg.name}: {s['n']} trades, {s['pnl']:+.1f} pts, WR {s['wr']}%")

    # Print comparison
    print_comparison(all_results)

    # Save to file
    out_path = os.path.join(RESULTS_DIR, "variant_comparison.txt")
    save_comparison_file(all_results, out_path)

    # Generate chart
    generate_chart(all_results)

    print(f"\nFiles saved:")
    print(f"  data/variant_comparison.txt")
    print(f"  data/charts/variant_comparison.png")


if __name__ == "__main__":
    asyncio.run(main())
