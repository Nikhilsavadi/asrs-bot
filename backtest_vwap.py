"""
backtest_vwap.py — VWAP Bounce Backtest
═══════════════════════════════════════════════════════════════════════════════

Simulates the VWAP bounce strategy over 2 years of DAX 5-min bars.

Usage:
    python backtest_vwap.py                    # Summary + trade log
    python backtest_vwap.py --charts           # Generate charts
    python backtest_vwap.py --export           # Export CSV
    python backtest_vwap.py --sweep            # Sweep parameters
    python backtest_vwap.py --compare          # Combined equity vs ASRS
    python backtest_vwap.py --log              # Full trade log
    python backtest_vwap.py --month 2025-06    # Filter by month

Uses cached data from the ASRS backtest (data/historical_bars.parquet).
"""

import asyncio
import os
import sys
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

from dax_bot import config
from strategy_vwap import (
    calculate_vwap, detect_trend, count_vwap_crosses, is_bounce_bar,
    calc_stop, calc_target, VwapBias, VwapPhase,
    TREND_LOOKBACK, TREND_THRESHOLD, CHOP_LOOKBACK, CHOP_MAX_CROSSES,
    STOP_BUFFER_PTS, MAX_STOP_PTS, ENTRY_START, ENTRY_END, FORCE_CLOSE,
    VWAP_MAX_ENTRIES, TRAIL_ACTIVATION, MIN_CLOSE_POSITION,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VwapTrade:
    date:          str = ""
    day_of_week:   str = ""
    direction:     str = ""
    entry_num:     int = 0
    entry_price:   float = 0.0
    entry_time:    str = ""
    exit_price:    float = 0.0
    exit_time:     str = ""
    pnl_pts:       float = 0.0
    mfe_pts:       float = 0.0    # Max favourable excursion
    mae_pts:       float = 0.0    # Max adverse excursion
    held_bars:     int = 0
    exit_reason:   str = ""
    bias:          str = ""       # ABOVE/BELOW
    z_at_entry:    float = 0.0
    vwap_at_entry: float = 0.0
    crosses_at_entry: int = 0


@dataclass
class VwapDayResult:
    date:              str = ""
    day_of_week:       str = ""
    day_range:         float = 0.0
    vwap_final:        float = 0.0
    total_crosses:     int = 0
    trades:            list = field(default_factory=list)
    total_pnl:         float = 0.0
    triggered:         bool = False


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_vwap_backtest(
    df: pd.DataFrame,
    trend_lookback: int = TREND_LOOKBACK,
    trend_threshold: int = TREND_THRESHOLD,
    chop_lookback: int = CHOP_LOOKBACK,
    chop_max_crosses: int = CHOP_MAX_CROSSES,
    stop_buffer: float = STOP_BUFFER_PTS,
    max_stop: float = MAX_STOP_PTS,
    max_entries: int = VWAP_MAX_ENTRIES,
    min_close_pos: float = MIN_CLOSE_POSITION,
) -> list[VwapDayResult]:
    """Run VWAP bounce backtest over all trading days."""
    results = []

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 20:
            continue

        day_name = trade_date.strftime("%A")

        day_result = VwapDayResult(
            date=str(trade_date),
            day_of_week=day_name,
            day_range=round(day_df["High"].max() - day_df["Low"].min(), 1),
        )

        # ── Calculate VWAP for the day ─────────────────────────────────
        vwap_df = calculate_vwap(day_df)
        if vwap_df.empty or len(vwap_df) < 10:
            results.append(day_result)
            continue

        day_result.vwap_final = round(vwap_df["vwap"].iloc[-1], 1)

        # Count total VWAP crosses for the day
        above_list = vwap_df["above_vwap"].tolist()
        day_result.total_crosses = count_vwap_crosses(above_list)

        # ── Simulate trades bar by bar ─────────────────────────────────
        entries_used = 0
        direction = None
        entry_price = 0.0
        entry_time = ""
        stop_price = 0.0
        target_price = 0.0
        trailing = False
        entry_bar_idx = 0
        mfe = 0.0
        mae = 0.0
        entry_bias = ""
        entry_z = 0.0
        entry_vwap = 0.0
        entry_crosses = 0

        rows = list(vwap_df.iterrows())

        for i, (idx, row) in enumerate(rows):
            t = idx.time()
            close = row["Close"]
            high = row["High"]
            low = row["Low"]
            open_ = row["Open"]
            vwap = row["vwap"]
            vwap_std = row["vwap_std"]
            z = row["z_score"]
            upper_1s = row["upper_1s"]
            lower_1s = row["lower_1s"]

            # ── Force close at EOD ─────────────────────────────────────
            if t >= FORCE_CLOSE and direction is not None:
                pnl = round(close - entry_price, 1) if direction == "LONG" \
                    else round(entry_price - close, 1)
                trade = VwapTrade(
                    date=str(trade_date), day_of_week=day_name,
                    direction=direction, entry_num=entries_used,
                    entry_price=entry_price, entry_time=entry_time,
                    exit_price=close, exit_time=str(t),
                    pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                    held_bars=i - entry_bar_idx, exit_reason="EOD close",
                    bias=entry_bias, z_at_entry=entry_z,
                    vwap_at_entry=entry_vwap, crosses_at_entry=entry_crosses,
                )
                day_result.trades.append(trade)
                direction = None
                break

            # ── Manage active position ─────────────────────────────────
            if direction is not None:
                if direction == "LONG":
                    mfe = max(mfe, high - entry_price)
                    mae = max(mae, entry_price - low)

                    # Stop hit?
                    if low <= stop_price:
                        pnl = round(stop_price - entry_price, 1)
                        trade = VwapTrade(
                            date=str(trade_date), day_of_week=day_name,
                            direction="LONG", entry_num=entries_used,
                            entry_price=entry_price, entry_time=entry_time,
                            exit_price=stop_price, exit_time=str(t),
                            pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                            held_bars=i - entry_bar_idx, exit_reason="Stop hit",
                            bias=entry_bias, z_at_entry=entry_z,
                            vwap_at_entry=entry_vwap, crosses_at_entry=entry_crosses,
                        )
                        day_result.trades.append(trade)
                        direction = None
                        continue

                    # Target hit?
                    if high >= target_price:
                        pnl = round(target_price - entry_price, 1)
                        trade = VwapTrade(
                            date=str(trade_date), day_of_week=day_name,
                            direction="LONG", entry_num=entries_used,
                            entry_price=entry_price, entry_time=entry_time,
                            exit_price=target_price, exit_time=str(t),
                            pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                            held_bars=i - entry_bar_idx, exit_reason="Target hit",
                            bias=entry_bias, z_at_entry=entry_z,
                            vwap_at_entry=entry_vwap, crosses_at_entry=entry_crosses,
                        )
                        day_result.trades.append(trade)
                        direction = None
                        continue

                    # Trail: once past upper 1σ, move stop to VWAP
                    if close >= upper_1s and not trailing:
                        trailing = True
                        stop_price = max(stop_price, round(vwap, 1))
                    elif trailing:
                        # Keep tightening to current VWAP
                        new_stop = round(vwap, 1)
                        if new_stop > stop_price:
                            stop_price = new_stop

                elif direction == "SHORT":
                    mfe = max(mfe, entry_price - low)
                    mae = max(mae, high - entry_price)

                    # Stop hit?
                    if high >= stop_price:
                        pnl = round(entry_price - stop_price, 1)
                        trade = VwapTrade(
                            date=str(trade_date), day_of_week=day_name,
                            direction="SHORT", entry_num=entries_used,
                            entry_price=entry_price, entry_time=entry_time,
                            exit_price=stop_price, exit_time=str(t),
                            pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                            held_bars=i - entry_bar_idx, exit_reason="Stop hit",
                            bias=entry_bias, z_at_entry=entry_z,
                            vwap_at_entry=entry_vwap, crosses_at_entry=entry_crosses,
                        )
                        day_result.trades.append(trade)
                        direction = None
                        continue

                    # Target hit?
                    if low <= target_price:
                        pnl = round(entry_price - target_price, 1)
                        trade = VwapTrade(
                            date=str(trade_date), day_of_week=day_name,
                            direction="SHORT", entry_num=entries_used,
                            entry_price=entry_price, entry_time=entry_time,
                            exit_price=target_price, exit_time=str(t),
                            pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                            held_bars=i - entry_bar_idx, exit_reason="Target hit",
                            bias=entry_bias, z_at_entry=entry_z,
                            vwap_at_entry=entry_vwap, crosses_at_entry=entry_crosses,
                        )
                        day_result.trades.append(trade)
                        direction = None
                        continue

                    # Trail: once past lower 1σ, move stop to VWAP
                    if close <= lower_1s and not trailing:
                        trailing = True
                        stop_price = min(stop_price, round(vwap, 1))
                    elif trailing:
                        new_stop = round(vwap, 1)
                        if new_stop < stop_price:
                            stop_price = new_stop

                continue  # Don't look for new entries while in a trade

            # ── Look for new bounce entries ─────────────────────────────
            if entries_used >= max_entries:
                continue

            # Time filter
            if t < ENTRY_START or t > ENTRY_END:
                continue

            # Need enough bars for trend detection
            if i < trend_lookback + 2:
                continue

            # Need non-zero std for target calculation
            if vwap_std <= 0:
                continue

            # Detect trend bias from recent closes
            recent_above = above_list[max(0, i - trend_lookback):i]
            bias = detect_trend(recent_above, trend_lookback, trend_threshold)

            if bias == VwapBias.CHOPPY:
                continue

            # Chop filter — too many VWAP crosses = choppy, skip
            chop_slice = above_list[max(0, i - chop_lookback):i]
            crosses = count_vwap_crosses(chop_slice)
            if crosses > chop_max_crosses:
                continue

            # ── Check for bounce bar ───────────────────────────────────
            bounce = is_bounce_bar(high, low, close, open_, vwap, bias)
            if not bounce:
                continue

            # ── We have a signal! ──────────────────────────────────────
            if bias == VwapBias.ABOVE:
                direction = "LONG"
                entry_price = close
                entry_time = str(t)
                stop_price = calc_stop(vwap, "LONG", close)
                # Find recent swing high for target
                recent_highs = vwap_df["High"].iloc[max(0, i - 12):i]
                swing_high = recent_highs.max() if len(recent_highs) > 0 else 0
                target_price = calc_target(vwap, vwap_std, bias, close, swing_high)
            else:
                direction = "SHORT"
                entry_price = close
                entry_time = str(t)
                stop_price = calc_stop(vwap, "SHORT", close)
                recent_lows = vwap_df["Low"].iloc[max(0, i - 12):i]
                swing_low = recent_lows.min() if len(recent_lows) > 0 else 0
                target_price = calc_target(vwap, vwap_std, bias, close, swing_low)

            trailing = False
            entries_used += 1
            entry_bar_idx = i
            mfe = 0.0
            mae = 0.0
            entry_bias = bias.value
            entry_z = round(z, 2)
            entry_vwap = round(vwap, 1)
            entry_crosses = crosses

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[VwapDayResult]):
    """Print overall VWAP bounce backtest summary."""
    all_trades = [t for r in results for t in r.trades]
    triggered_days = [r for r in results if r.triggered]
    wins = [t for t in all_trades if t.pnl_pts > 0]
    losses = [t for t in all_trades if t.pnl_pts < 0]

    total_pnl = sum(t.pnl_pts for t in all_trades)

    if not all_trades:
        print("\nNo VWAP bounce trades found in the data.")
        return

    avg_win = np.mean([t.pnl_pts for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pts for t in losses]) if losses else 0
    win_rate = len(wins) / len(all_trades) * 100
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
    gross_win = sum(t.pnl_pts for t in wins)
    gross_loss = abs(sum(t.pnl_pts for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    rr = abs(avg_win / avg_loss) if avg_loss else float("inf")

    # MFE/MAE
    avg_mfe = np.mean([t.mfe_pts for t in all_trades])
    avg_mae = np.mean([t.mae_pts for t in all_trades])

    # Drawdown
    equity, running = [], 0
    for r in results:
        running += r.total_pnl
        equity.append(running)
    peak, max_dd = 0, 0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    # Exit reasons
    reasons = defaultdict(int)
    for t in all_trades:
        reasons[t.exit_reason] += 1

    # Bias breakdown
    above_trades = [t for t in all_trades if t.bias == "ABOVE"]
    below_trades = [t for t in all_trades if t.bias == "BELOW"]

    # Avg hold time
    avg_bars = np.mean([t.held_bars for t in all_trades])

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                VWAP BOUNCE BACKTEST                          ║
╠══════════════════════════════════════════════════════════════╣
║  Period:     {results[0].date} → {results[-1].date:<30}║
║  Days:       {len(results)} total, {len(triggered_days)} triggered ({round(len(triggered_days)/len(results)*100)}%)          ║
╠══════════════════════════════════════════════════════════════╣
║  TRADES                                                      ║
║  Total:      {len(all_trades):<47}║
║  Winners:    {len(wins)} ({win_rate:.0f}%){' ' * 40}║
║  Losers:     {len(losses)} ({100 - win_rate:.0f}%){' ' * 40}║
║  Avg win:    {avg_win:+.1f} pts{' ' * 37}║
║  Avg loss:   {avg_loss:+.1f} pts{' ' * 37}║
║  R:R ratio:  {rr:.2f}{' ' * 41}║
║  Expectancy: {expectancy:+.1f} pts/trade{' ' * 31}║
║  Profit fct: {pf:.2f}{' ' * 41}║
╠══════════════════════════════════════════════════════════════╣
║  EFFICIENCY                                                  ║
║  Avg MFE:    {avg_mfe:.1f} pts (how far it goes for you){' ' * 12}║
║  Avg MAE:    {avg_mae:.1f} pts (how far it goes against){' ' * 11}║
║  Avg hold:   {avg_bars:.0f} bars ({avg_bars * 5:.0f} min){' ' * 28}║
╠══════════════════════════════════════════════════════════════╣
║  RESULTS                                                     ║
║  Total P&L:  {total_pnl:+.0f} pts{' ' * 37}║
║  Max DD:     {max_dd:.0f} pts{' ' * 38}║
╠══════════════════════════════════════════════════════════════╣
║  BIAS BREAKDOWN                                              ║
║  ABOVE (long bounces):  {len(above_trades)} trades → {sum(t.pnl_pts for t in above_trades):+.0f} pts{' ' * 15}║
║  BELOW (short bounces): {len(below_trades)} trades → {sum(t.pnl_pts for t in below_trades):+.0f} pts{' ' * 15}║
╠══════════════════════════════════════════════════════════════╣
║  EXIT REASONS                                                ║""")

    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / len(all_trades) * 100
        sub_pnl = sum(t.pnl_pts for t in all_trades if t.exit_reason == reason)
        print(f"║  {reason:<20} {count:>4} ({pct:.0f}%) → {sub_pnl:+.0f} pts{' ' * 17}║")

    print(f"""╠══════════════════════════════════════════════════════════════╣
║  SCALING (2Y total P&L)                                      ║
║  1 Micro  (€1/pt):  €{total_pnl:>+8,.0f}{' ' * 29}║
║  5 Micro  (€5/pt):  €{total_pnl * 5:>+8,.0f}{' ' * 29}║
║  1 Mini   (€5/pt):  €{total_pnl * 5:>+8,.0f}{' ' * 29}║
║  5 Mini   (€25/pt): €{total_pnl * 25:>+8,.0f}{' ' * 29}║
╚══════════════════════════════════════════════════════════════╝""")


def print_trade_log(results: list[VwapDayResult], month_filter: str = ""):
    """Print individual trades."""
    print(f"\n{'Date':<12} {'Dir':<6} {'Bias':<6} {'Entry':>8} {'Exit':>8} "
          f"{'P&L':>7} {'Bars':>5} {'Reason':<15} {'z':>5}")
    print("─" * 85)

    for r in results:
        if month_filter and not r.date.startswith(month_filter):
            continue
        for t in r.trades:
            print(f"{t.date:<12} {t.direction:<6} {t.bias:<6} {t.entry_price:>8.1f} "
                  f"{t.exit_price:>8.1f} {t.pnl_pts:>+6.1f} "
                  f"{t.held_bars:>5} {t.exit_reason:<15} {t.z_at_entry:>+4.1f}")


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETER SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def sweep_params(df: pd.DataFrame):
    """Test different parameter combinations."""
    print(f"\n{'═' * 90}")
    print("PARAMETER SWEEP — VWAP Bounce Strategy")
    print(f"{'═' * 90}")
    print(f"\n{'Lookback':>9} {'Thresh':>7} {'ChopMax':>8} {'StopBuf':>8} "
          f"{'Trades':>7} {'Win%':>6} {'AvgPnL':>8} {'Total':>8} {'PF':>6} {'MaxDD':>7}")
    print("─" * 90)

    best_exp = -999
    best_p = {}

    for lb in [4, 6, 8]:
        for th in [3, 4, 5]:
            if th > lb:
                continue
            for chop in [3, 4, 5, 6]:
                for buf in [3, 5, 8]:
                    results = run_vwap_backtest(
                        df, trend_lookback=lb, trend_threshold=th,
                        chop_max_crosses=chop, stop_buffer=buf,
                    )
                    trades = [t for r in results for t in r.trades]
                    if len(trades) < 20:
                        continue

                    wins = sum(1 for t in trades if t.pnl_pts > 0)
                    wr = wins / len(trades) * 100
                    avg = np.mean([t.pnl_pts for t in trades])
                    total = sum(t.pnl_pts for t in trades)
                    gw = sum(t.pnl_pts for t in trades if t.pnl_pts > 0)
                    gl = abs(sum(t.pnl_pts for t in trades if t.pnl_pts < 0))
                    pf = gw / gl if gl > 0 else 0

                    eq, run = [], 0
                    for r in results:
                        run += r.total_pnl
                        eq.append(run)
                    pk, dd = 0, 0
                    for e in eq:
                        pk = max(pk, e)
                        dd = max(dd, pk - e)

                    flag = " ★" if avg > best_exp else ""
                    if avg > best_exp:
                        best_exp = avg
                        best_p = {"lb": lb, "th": th, "chop": chop, "buf": buf}

                    print(f"{lb:>9} {th:>7} {chop:>8} {buf:>8} {len(trades):>7} "
                          f"{wr:>5.0f}% {avg:>+7.1f} {total:>+7.0f} {pf:>5.2f} "
                          f"{dd:>6.0f}{flag}")

    if best_p:
        print(f"\n★ Best: lookback={best_p['lb']}, threshold={best_p['th']}, "
              f"chop_max={best_p['chop']}, stop_buffer={best_p['buf']} "
              f"→ expectancy={best_exp:+.1f}")


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def generate_vwap_charts(results: list[VwapDayResult], output_dir: str = None):
    """Generate VWAP bounce charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "charts")
    os.makedirs(output_dir, exist_ok=True)

    all_trades = [t for r in results for t in r.trades]
    if not all_trades:
        return []

    wins = [t for t in all_trades if t.pnl_pts > 0]
    losses = [t for t in all_trades if t.pnl_pts < 0]
    saved = []

    # ── 1. Equity Curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    dates, equity = [], []
    running = 0
    for r in results:
        if r.triggered:
            dates.append(datetime.strptime(r.date, "%Y-%m-%d"))
            running += r.total_pnl
            equity.append(running)

    ax.plot(dates, equity, color="#9C27B0", linewidth=1.5, label="VWAP Bounce")
    ax.fill_between(dates, 0, equity,
                    where=[e >= 0 for e in equity], alpha=0.15, color="#4CAF50")
    ax.fill_between(dates, 0, equity,
                    where=[e < 0 for e in equity], alpha=0.15, color="#F44336")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("VWAP Bounce — Equity Curve", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative P&L (points)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "vwap_equity_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ── 2. Win/Loss Panel ──────────────────────────────────────────────
    avg_win = np.mean([t.pnl_pts for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pts for t in losses]) if losses else 0
    win_rate = len(wins) / len(all_trades) * 100
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
    gw = sum(t.pnl_pts for t in wins)
    gl = abs(sum(t.pnl_pts for t in losses))
    pf = gw / gl if gl > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(["Win", "Loss"], [len(wins), len(losses)],
                color=["#4CAF50", "#F44336"], edgecolor="white")
    axes[0].set_title("Count")
    axes[1].bar(["Avg Win", "Avg Loss"], [avg_win, abs(avg_loss)],
                color=["#4CAF50", "#F44336"], edgecolor="white")
    axes[1].set_title("Size (pts)")
    axes[2].axis("off")
    metrics = f"Win Rate:   {win_rate:.1f}%\nR:R Ratio:  {abs(avg_win/avg_loss) if avg_loss else 0:.2f}\nExpectancy: {expectancy:+.1f}\nProfit Fct: {pf:.2f}"
    axes[2].text(0.1, 0.8, metrics, fontsize=13, fontfamily="monospace",
                 transform=axes[2].transAxes, verticalalignment="top")
    fig.suptitle("VWAP Bounce — Win/Loss", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "vwap_win_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ── 3. Time of Day ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    hour_groups = defaultdict(list)
    for t in all_trades:
        try:
            h = int(t.entry_time.split(":")[0])
            hour_groups[h].append(t)
        except (ValueError, IndexError):
            pass
    hours = sorted(hour_groups.keys())
    avg_pnls = [np.mean([t.pnl_pts for t in hour_groups[h]]) for h in hours]
    counts = [len(hour_groups[h]) for h in hours]
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in avg_pnls]
    bars = ax.bar(range(len(hours)), avg_pnls, color=colors, edgecolor="white")
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h}:00" for h in hours])
    ax.set_title("Avg P&L by Entry Hour (CET)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg P&L (pts)")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    for bar, val, n in zip(bars, avg_pnls, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1),
                f"{val:+.1f}\n(n={n})", ha="center", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "vwap_time_of_day.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ── 4. Bias Comparison ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    above = [t for t in all_trades if t.bias == "ABOVE"]
    below = [t for t in all_trades if t.bias == "BELOW"]
    labels = ["ABOVE\n(long bounces)", "BELOW\n(short bounces)"]
    avgs = [np.mean([t.pnl_pts for t in above]) if above else 0,
            np.mean([t.pnl_pts for t in below]) if below else 0]
    ns = [len(above), len(below)]
    colors = ["#4CAF50" if a >= 0 else "#F44336" for a in avgs]
    bars = ax.bar(range(2), avgs, color=colors, edgecolor="white")
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    for bar, val, n in zip(bars, avgs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1),
                f"{val:+.1f}\n(n={n})", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("VWAP Bounce — Long vs Short", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg P&L (pts)")
    plt.tight_layout()
    path = os.path.join(output_dir, "vwap_bias_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ── 5. Monthly P&L ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly = defaultdict(float)
    for r in results:
        monthly[r.date[:7]] += r.total_pnl
    months = sorted(monthly.keys())
    mpnls = [monthly[m] for m in months]
    colors_m = ["#4CAF50" if p >= 0 else "#F44336" for p in mpnls]
    ax.bar(range(len(months)), mpnls, color=colors_m, alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, fontsize=7, ha="right")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("VWAP Bounce — Monthly P&L", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "vwap_monthly.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    logger.info(f"Generated {len(saved)} VWAP charts")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_strategies(asrs_results: list, vwap_results: list[VwapDayResult],
                       output_dir: str = None):
    """Combined equity curve: ASRS + VWAP Bounce."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "charts")
    os.makedirs(output_dir, exist_ok=True)

    # Build equity lines
    asrs_eq, vwap_eq = {}, {}
    r = 0
    for res in asrs_results:
        r += res.total_pnl
        asrs_eq[res.date] = r
    r = 0
    for res in vwap_results:
        r += res.total_pnl
        vwap_eq[res.date] = r

    all_dates = sorted(set(list(asrs_eq.keys()) + list(vwap_eq.keys())))
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates]
    asrs_line, vwap_line, combined = [], [], []
    av, vv = 0, 0
    for d in all_dates:
        av = asrs_eq.get(d, av)
        vv = vwap_eq.get(d, vv)
        asrs_line.append(av)
        vwap_line.append(vv)
        combined.append(av + vv)

    # Correlation
    asrs_daily = [asrs_eq.get(d, 0) for d in all_dates]
    vwap_daily = [vwap_eq.get(d, 0) for d in all_dates]
    asrs_changes = np.diff(asrs_daily) if len(asrs_daily) > 1 else [0]
    vwap_changes = np.diff(vwap_daily) if len(vwap_daily) > 1 else [0]
    min_len = min(len(asrs_changes), len(vwap_changes))
    if min_len > 10:
        corr = np.corrcoef(asrs_changes[:min_len], vwap_changes[:min_len])[0, 1]
    else:
        corr = 0

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(dates, asrs_line, color="#2196F3", linewidth=1.5, label="ASRS (breakout)")
    ax.plot(dates, vwap_line, color="#9C27B0", linewidth=1.5, label="VWAP Bounce")
    ax.plot(dates, combined, color="#4CAF50", linewidth=2.5, label="COMBINED")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    ax.set_title("Strategy Comparison: ASRS + VWAP Bounce",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Cumulative P&L (points)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    stats = (
        f"ASRS:        {asrs_line[-1]:+.0f} pts\n"
        f"VWAP Bounce: {vwap_line[-1]:+.0f} pts\n"
        f"Combined:    {combined[-1]:+.0f} pts\n"
        f"Correlation: {corr:.2f}"
    )
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, "combined_equity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined chart saved: {path}")
    return path


def export_csv(results: list[VwapDayResult]):
    """Export trades to CSV."""
    rows = []
    for r in results:
        for t in r.trades:
            rows.append({
                "date": t.date, "day": t.day_of_week, "direction": t.direction,
                "bias": t.bias, "entry_price": t.entry_price, "entry_time": t.entry_time,
                "exit_price": t.exit_price, "exit_time": t.exit_time,
                "pnl_pts": t.pnl_pts, "mfe": t.mfe_pts, "mae": t.mae_pts,
                "held_bars": t.held_bars, "exit_reason": t.exit_reason,
                "z_score": t.z_at_entry, "vwap": t.vwap_at_entry,
                "crosses": t.crosses_at_entry,
            })
    if rows:
        path = os.path.join(RESULTS_DIR, "vwap_backtest_trades.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\nExported {len(rows)} trades to {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    args = sys.argv[1:]

    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    if not os.path.exists(rth_path):
        print("No cached data found. Run ASRS backtest first to fetch data:")
        print("  python backtest.py")
        return

    df = pd.read_parquet(rth_path)
    logger.info(f"Loaded {len(df)} bars")

    results = run_vwap_backtest(df)
    logger.info(f"Processed {len(results)} trading days")

    # Charts
    if "--charts" in args or "--compare" in args or len(args) == 0:
        generate_vwap_charts(results)
        print(f"\n📊 VWAP charts saved to data/charts/")

    # Sweep
    if "--sweep" in args:
        sweep_params(df)

    # Compare
    if "--compare" in args:
        try:
            from backtest import run_backtest
            asrs_rth = pd.read_parquet(rth_path)
            all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")
            all_df = pd.read_parquet(all_path) if os.path.exists(all_path) else asrs_rth
            asrs_results = run_backtest(asrs_rth, all_df)
            compare_strategies(asrs_results, results)
            print("📊 Combined equity curve saved to data/charts/combined_equity.png")
        except Exception as e:
            print(f"Cannot compare: {e}")

    # Export
    if "--export" in args:
        export_csv(results)

    # Month filter
    month_filter = ""
    if "--month" in args:
        idx = args.index("--month")
        if idx + 1 < len(args):
            month_filter = args[idx + 1]

    # Output
    print_summary(results)
    if "--log" in args or "--trades" in args:
        print_trade_log(results, month_filter)
    elif month_filter:
        filtered = [r for r in results if r.date.startswith(month_filter)]
        if filtered:
            print_summary(filtered)
            print_trade_log(filtered)


if __name__ == "__main__":
    asyncio.run(main())
