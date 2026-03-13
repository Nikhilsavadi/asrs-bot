"""
analyse_volatility.py — Bar 4 vs Bar 5 performance by volatility regime
========================================================================

Calculates rolling 20-day ATR on DAX data, splits days into HIGH / MEDIUM / LOW
volatility terciles, then compares bar 4 vs bar 5 (simple candle trail) across regimes.

Usage:
    python3 analyse_volatility.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from dataclasses import dataclass, field
from collections import defaultdict

logging.disable(logging.CRITICAL)
sys.stderr = open(os.devnull, "w")

from dax_bot import config
from backtest import load_cached_data, candle_number, classify_range, classify_gap, analyse_context_bars

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "dax")


# ── ATR calculation ──────────────────────────────────────────────────────────

def compute_daily_atr(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Compute daily true range and rolling ATR from 5-min bars."""
    daily = []
    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue
        day_high = day_df["High"].max()
        day_low = day_df["Low"].min()
        day_open = day_df.iloc[0]["Open"]
        day_close = day_df.iloc[-1]["Close"]
        daily.append({
            "date": trade_date,
            "high": day_high,
            "low": day_low,
            "open": day_open,
            "close": day_close,
        })

    daily_df = pd.DataFrame(daily)
    daily_df["prev_close"] = daily_df["close"].shift(1)

    # True Range = max(H-L, |H-prevC|, |L-prevC|)
    daily_df["tr"] = daily_df.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        ), axis=1
    )

    daily_df["atr_20"] = daily_df["tr"].rolling(window=period, min_periods=period).mean()
    return daily_df


def classify_volatility(atr_series: pd.Series) -> pd.Series:
    """Split ATR into terciles: LOW / MEDIUM / HIGH."""
    q33 = atr_series.quantile(0.33)
    q66 = atr_series.quantile(0.66)
    return atr_series.apply(
        lambda x: "LOW" if x <= q33 else ("HIGH" if x >= q66 else "MEDIUM")
    )


# ── Backtest engine (simple candle trail, parameterised signal bar) ──────────

@dataclass
class TradeResult:
    date: str
    direction: str
    pnl_pts: float
    signal_bar: int
    vol_regime: str


def run_candle_trail(df: pd.DataFrame, vol_map: dict, signal_bar: int = 4) -> list[TradeResult]:
    """
    Simple candle trail backtest using the specified signal bar.
    vol_map: {date_str: vol_regime}
    """
    results = []
    prev_close = 0

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        date_str = str(trade_date)
        vol_regime = vol_map.get(date_str, None)
        if vol_regime is None:
            prev_close = day_df.iloc[-1]["Close"]
            continue

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
        buy_level = round(sig["high"] + config.BUFFER_PTS, 1)
        sell_level = round(sig["low"] - config.BUFFER_PTS, 1)

        # Post-signal-bar candles
        post_bars = []
        for idx, row in day_df.iterrows():
            if candle_number(idx) > signal_bar:
                post_bars.append((idx, row))

        entries_used = 0
        direction = None
        entry_price = 0
        trail_stop = 0
        entry_bar_idx = 0

        for i, (idx, row) in enumerate(post_bars):
            if entries_used >= config.MAX_ENTRIES and direction is None:
                break

            # Entry
            if direction is None and entries_used < config.MAX_ENTRIES:
                if row["High"] >= buy_level:
                    direction = "LONG"
                    entry_price = buy_level
                    trail_stop = sell_level
                    entries_used += 1
                    entry_bar_idx = i
                elif row["Low"] <= sell_level:
                    direction = "SHORT"
                    entry_price = sell_level
                    trail_stop = buy_level
                    entries_used += 1
                    entry_bar_idx = i

            # Trail + stop
            if direction == "LONG":
                if i > entry_bar_idx:
                    prev_low = round(post_bars[i - 1][1]["Low"], 1)
                    if prev_low > trail_stop:
                        trail_stop = prev_low
                if row["Low"] <= trail_stop:
                    pnl = round(trail_stop - entry_price, 1)
                    results.append(TradeResult(date_str, "LONG", pnl, signal_bar, vol_regime))
                    direction = None

            elif direction == "SHORT":
                if i > entry_bar_idx:
                    prev_high = round(post_bars[i - 1][1]["High"], 1)
                    if prev_high < trail_stop:
                        trail_stop = prev_high
                if row["High"] >= trail_stop:
                    pnl = round(entry_price - trail_stop, 1)
                    results.append(TradeResult(date_str, "SHORT", pnl, signal_bar, vol_regime))
                    direction = None

        # EOD close
        if direction is not None and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 1)
            pnl = round((last_price - entry_price) if direction == "LONG" else (entry_price - last_price), 1)
            results.append(TradeResult(date_str, direction, pnl, signal_bar, vol_regime))
            direction = None

        prev_close = day_df.iloc[-1]["Close"]

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rth_df, all_df = load_cached_data()
    if rth_df is None:
        print("No cached data found in data/dax/historical_bars.parquet")
        return

    # 1. Compute daily ATR
    daily_df = compute_daily_atr(rth_df, period=20)
    valid = daily_df.dropna(subset=["atr_20"]).copy()
    valid["vol_regime"] = classify_volatility(valid["atr_20"])

    # ATR tercile boundaries
    q33 = valid["atr_20"].quantile(0.33)
    q66 = valid["atr_20"].quantile(0.66)

    vol_map = {str(r["date"]): r["vol_regime"] for _, r in valid.iterrows()}

    print("=" * 80)
    print("  BAR 4 vs BAR 5 PERFORMANCE BY VOLATILITY REGIME (20-DAY ATR)")
    print("=" * 80)
    print(f"\n  Data: {len(valid)} trading days with ATR available")
    print(f"  ATR terciles:  LOW < {q33:.1f}  |  MEDIUM {q33:.1f}-{q66:.1f}  |  HIGH > {q66:.1f}")
    print(f"  Distribution:  LOW={len(valid[valid['vol_regime']=='LOW'])}  "
          f"MEDIUM={len(valid[valid['vol_regime']=='MEDIUM'])}  "
          f"HIGH={len(valid[valid['vol_regime']=='HIGH'])}")

    # 2. Run backtests
    bar4_trades = run_candle_trail(rth_df, vol_map, signal_bar=4)
    bar5_trades = run_candle_trail(rth_df, vol_map, signal_bar=5)

    # Also run original (bar 4) for total comparison
    from backtest import run_backtest
    original_results = run_backtest(rth_df, all_df)
    orig_trades = []
    for dr in original_results:
        for t in dr.trades:
            orig_trades.append(t)

    # 3. Build stats per regime
    def calc_stats(trades):
        if not trades:
            return {"count": 0, "wins": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}
        pnls = [t.pnl_pts for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "count": len(pnls),
            "wins": wins,
            "win_rate": round(wins / len(pnls) * 100, 1),
            "total_pnl": round(sum(pnls), 1),
            "avg_pnl": round(np.mean(pnls), 2),
        }

    regimes = ["LOW", "MEDIUM", "HIGH"]

    # Bar 4 stats by regime
    b4_by_regime = {r: calc_stats([t for t in bar4_trades if t.vol_regime == r]) for r in regimes}
    b4_total = calc_stats(bar4_trades)

    # Bar 5 stats by regime
    b5_by_regime = {r: calc_stats([t for t in bar5_trades if t.vol_regime == r]) for r in regimes}
    b5_total = calc_stats(bar5_trades)

    # Original backtest totals (for comparison)
    orig_pnls = [t.pnl_pts for t in orig_trades]
    orig_wins = sum(1 for p in orig_pnls if p > 0)
    orig_total = {
        "count": len(orig_pnls),
        "wins": orig_wins,
        "win_rate": round(orig_wins / len(orig_pnls) * 100, 1) if orig_pnls else 0,
        "total_pnl": round(sum(orig_pnls), 1),
        "avg_pnl": round(np.mean(orig_pnls), 2) if orig_pnls else 0,
    }

    # 4. Print tables
    header = f"{'Regime':<10} {'Trades':>7} {'Wins':>6} {'Win %':>7} {'Total P&L':>11} {'Avg P&L':>9}"
    sep = "-" * 55

    print(f"\n\n{'─' * 55}")
    print(f"  BAR 4 — Simple Candle Trail")
    print(f"{'─' * 55}")
    print(f"  {header}")
    print(f"  {sep}")
    for r in regimes:
        s = b4_by_regime[r]
        pnl_str = f"+{s['total_pnl']}" if s['total_pnl'] > 0 else f"{s['total_pnl']}"
        avg_str = f"+{s['avg_pnl']}" if s['avg_pnl'] > 0 else f"{s['avg_pnl']}"
        print(f"  {r:<10} {s['count']:>7} {s['wins']:>6} {s['win_rate']:>6.1f}% {pnl_str:>11} {avg_str:>9}")
    print(f"  {sep}")
    pnl_str = f"+{b4_total['total_pnl']}" if b4_total['total_pnl'] > 0 else f"{b4_total['total_pnl']}"
    avg_str = f"+{b4_total['avg_pnl']}" if b4_total['avg_pnl'] > 0 else f"{b4_total['avg_pnl']}"
    print(f"  {'TOTAL':<10} {b4_total['count']:>7} {b4_total['wins']:>6} {b4_total['win_rate']:>6.1f}% {pnl_str:>11} {avg_str:>9}")

    print(f"\n\n{'─' * 55}")
    print(f"  BAR 5 — Simple Candle Trail")
    print(f"{'─' * 55}")
    print(f"  {header}")
    print(f"  {sep}")
    for r in regimes:
        s = b5_by_regime[r]
        pnl_str = f"+{s['total_pnl']}" if s['total_pnl'] > 0 else f"{s['total_pnl']}"
        avg_str = f"+{s['avg_pnl']}" if s['avg_pnl'] > 0 else f"{s['avg_pnl']}"
        print(f"  {r:<10} {s['count']:>7} {s['wins']:>6} {s['win_rate']:>6.1f}% {pnl_str:>11} {avg_str:>9}")
    print(f"  {sep}")
    pnl_str = f"+{b5_total['total_pnl']}" if b5_total['total_pnl'] > 0 else f"{b5_total['total_pnl']}"
    avg_str = f"+{b5_total['avg_pnl']}" if b5_total['avg_pnl'] > 0 else f"{b5_total['avg_pnl']}"
    print(f"  {'TOTAL':<10} {b5_total['count']:>7} {b5_total['wins']:>6} {b5_total['win_rate']:>6.1f}% {pnl_str:>11} {avg_str:>9}")

    # 5. Side-by-side comparison
    print(f"\n\n{'=' * 80}")
    print(f"  SIDE-BY-SIDE COMPARISON: BAR 4 vs BAR 5 vs ORIGINAL")
    print(f"{'=' * 80}")

    comp_header = (f"{'Regime':<10} "
                   f"{'BAR4 P&L':>10} {'BAR4 W%':>8} {'BAR4 Avg':>9} "
                   f"{'BAR5 P&L':>10} {'BAR5 W%':>8} {'BAR5 Avg':>9} "
                   f"{'Winner':>8}")
    print(f"\n  {comp_header}")
    print(f"  {'-' * 80}")

    for r in regimes:
        b4 = b4_by_regime[r]
        b5 = b5_by_regime[r]
        winner = "BAR 4" if b4["total_pnl"] >= b5["total_pnl"] else "BAR 5"

        b4_pnl = f"+{b4['total_pnl']}" if b4['total_pnl'] > 0 else f"{b4['total_pnl']}"
        b5_pnl = f"+{b5['total_pnl']}" if b5['total_pnl'] > 0 else f"{b5['total_pnl']}"
        b4_avg = f"+{b4['avg_pnl']}" if b4['avg_pnl'] > 0 else f"{b4['avg_pnl']}"
        b5_avg = f"+{b5['avg_pnl']}" if b5['avg_pnl'] > 0 else f"{b5['avg_pnl']}"

        print(f"  {r:<10} "
              f"{b4_pnl:>10} {b4['win_rate']:>7.1f}% {b4_avg:>9} "
              f"{b5_pnl:>10} {b5['win_rate']:>7.1f}% {b5_avg:>9} "
              f"{'<< ' + winner:>8}")

    print(f"  {'-' * 80}")
    b4_pnl = f"+{b4_total['total_pnl']}" if b4_total['total_pnl'] > 0 else f"{b4_total['total_pnl']}"
    b5_pnl = f"+{b5_total['total_pnl']}" if b5_total['total_pnl'] > 0 else f"{b5_total['total_pnl']}"
    b4_avg = f"+{b4_total['avg_pnl']}" if b4_total['avg_pnl'] > 0 else f"{b4_total['avg_pnl']}"
    b5_avg = f"+{b5_total['avg_pnl']}" if b5_total['avg_pnl'] > 0 else f"{b5_total['avg_pnl']}"
    total_winner = "BAR 4" if b4_total['total_pnl'] >= b5_total['total_pnl'] else "BAR 5"
    print(f"  {'TOTAL':<10} "
          f"{b4_pnl:>10} {b4_total['win_rate']:>7.1f}% {b4_avg:>9} "
          f"{b5_pnl:>10} {b5_total['win_rate']:>7.1f}% {b5_avg:>9} "
          f"{'<< ' + total_winner:>8}")

    # 6. Original backtest comparison
    print(f"\n\n{'─' * 55}")
    print(f"  ORIGINAL BACKTEST (bar 4, full dataset)")
    print(f"{'─' * 55}")
    o_pnl = f"+{orig_total['total_pnl']}" if orig_total['total_pnl'] > 0 else f"{orig_total['total_pnl']}"
    o_avg = f"+{orig_total['avg_pnl']}" if orig_total['avg_pnl'] > 0 else f"{orig_total['avg_pnl']}"
    print(f"  Trades: {orig_total['count']}  |  Win rate: {orig_total['win_rate']}%  "
          f"|  Total P&L: {o_pnl} pts  |  Avg: {o_avg} pts")
    print(f"\n  (Original includes days outside ATR window; vol-filtered has {b4_total['count']} trades)")

    # 7. Key insight
    print(f"\n\n{'=' * 80}")
    print(f"  KEY INSIGHTS")
    print(f"{'=' * 80}")

    best_bar4_regime = max(regimes, key=lambda r: b4_by_regime[r]["avg_pnl"])
    best_bar5_regime = max(regimes, key=lambda r: b5_by_regime[r]["avg_pnl"])
    worst_bar4_regime = min(regimes, key=lambda r: b4_by_regime[r]["avg_pnl"])

    print(f"\n  - Bar 4 best regime:  {best_bar4_regime} vol  "
          f"(avg +{b4_by_regime[best_bar4_regime]['avg_pnl']} pts, "
          f"{b4_by_regime[best_bar4_regime]['win_rate']}% win rate)")
    print(f"  - Bar 5 best regime:  {best_bar5_regime} vol  "
          f"(avg +{b5_by_regime[best_bar5_regime]['avg_pnl']} pts, "
          f"{b5_by_regime[best_bar5_regime]['win_rate']}% win rate)")
    print(f"  - Bar 4 worst regime: {worst_bar4_regime} vol  "
          f"(avg {b4_by_regime[worst_bar4_regime]['avg_pnl']} pts, "
          f"{b4_by_regime[worst_bar4_regime]['win_rate']}% win rate)")

    # Check if bar 5 rescues any regime
    for r in regimes:
        if b4_by_regime[r]["avg_pnl"] < 0 and b5_by_regime[r]["avg_pnl"] > 0:
            print(f"  - ** Bar 5 rescues {r} vol regime: "
                  f"bar4 avg {b4_by_regime[r]['avg_pnl']} -> bar5 avg +{b5_by_regime[r]['avg_pnl']} **")
        elif b5_by_regime[r]["total_pnl"] > b4_by_regime[r]["total_pnl"]:
            diff = round(b5_by_regime[r]["total_pnl"] - b4_by_regime[r]["total_pnl"], 1)
            print(f"  - Bar 5 outperforms in {r} vol by +{diff} pts total")

    # ATR stats
    print(f"\n  ATR stats:  mean={valid['atr_20'].mean():.1f}  "
          f"min={valid['atr_20'].min():.1f}  max={valid['atr_20'].max():.1f}")
    print()


if __name__ == "__main__":
    main()
