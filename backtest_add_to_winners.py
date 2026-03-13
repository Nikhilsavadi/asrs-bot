"""
backtest_add_to_winners.py — Candle Trail with Add-to-Winners
═══════════════════════════════════════════════════════════════

Tests adding to winning positions within the existing bar 4 candle trail strategy.

Logic:
  1. Initial entry: same as baseline (break bar4 high/low + buffer)
  2. Trail stop: previous candle low/high (same as baseline)
  3. ADD trigger: when unrealised P&L >= add_trigger_pts AND current bar
     pulls back toward trail stop but HOLDS (close stays on right side),
     add a new position at the bar's close price.
  4. All positions share the same trail stop — they ALL exit when stop is hit.
  5. Max adds configurable (1-3 extra positions on top of the initial).

Usage:
    python backtest_add_to_winners.py                  # Default: 1 add, 15pt trigger
    python backtest_add_to_winners.py --sweep           # Sweep trigger/max-adds params
"""

import os
import sys
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np

from dax_bot import config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "dax")


# ──────────────────────────────────────────────────────────────────────────────
#  Data helpers (reused from backtest.py)
# ──────────────────────────────────────────────────────────────────────────────

def candle_number(timestamp: pd.Timestamp) -> int:
    open_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def load_data() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    if not os.path.exists(path):
        print(f"ERROR: No data at {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Position tracking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    direction: str = ""
    entry_price: float = 0.0
    entry_bar_idx: int = 0
    is_add: bool = False


@dataclass
class ClosedTrade:
    date: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pts: float = 0.0
    is_add: bool = False
    held_bars: int = 0


@dataclass
class DayResult:
    date: str = ""
    trades: list = field(default_factory=list)
    total_pnl: float = 0.0
    num_adds: int = 0
    triggered: bool = False


# ──────────────────────────────────────────────────────────────────────────────
#  Backtest engine
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, add_trigger_pts: float = 15.0,
                 max_adds: int = 1, add_mode: str = "strength",
                 num_contracts: int = 1, partial_exits: bool = False,
                 tp1_pts: float = 20.0, tp2_pts: float = 50.0) -> list[DayResult]:
    """
    Run candle trail backtest with add-to-winners.

    Args:
        add_trigger_pts: Min unrealised profit (pts) before next add triggers
        max_adds: Max extra positions to add (on top of initial entry)
        add_mode: "strength" = add when price extends +trigger pts from LAST entry
                  "pullback" = add when price pulls back toward stop but holds
        num_contracts: Number of initial contracts (1 or 3)
        partial_exits: If True and num_contracts>=3, C1 exits at TP1, C2 at TP2
        tp1_pts: TP1 distance for contract 1
        tp2_pts: TP2 distance for contract 2
    """
    results = []

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        # ── Identify bar 4 ───────────────────────────────────────────
        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx)
            if 1 <= cn <= 6:
                bars[cn] = {"high": row["High"], "low": row["Low"],
                            "open": row["Open"], "close": row["Close"]}

        if 4 not in bars:
            continue

        bar4 = bars[4]
        buy_level = round(bar4["high"] + config.BUFFER_PTS, 1)
        sell_level = round(bar4["low"] - config.BUFFER_PTS, 1)

        # Post-bar4 candles
        post_bars = [(idx, row) for idx, row in day_df.iterrows()
                     if candle_number(idx) > 4]

        day_result = DayResult(date=str(trade_date))
        positions = []
        adds_used = 0
        direction = None
        trail_stop = 0
        entry_bar_idx = 0
        entries_used = 0  # Allow re-entry up to MAX_ENTRIES (same as original backtest)
        tp1_exited = False
        tp2_exited = False

        for i, (idx, row) in enumerate(post_bars):
            # ── Initial entry (up to MAX_ENTRIES per day) ────────
            if direction is None and not positions and entries_used < config.MAX_ENTRIES:
                if row["High"] >= buy_level:
                    direction = "LONG"
                    for c in range(num_contracts):
                        positions.append(Position(direction="LONG", entry_price=buy_level,
                                                  entry_bar_idx=i))
                    trail_stop = sell_level
                    entry_bar_idx = i
                    entries_used += 1
                    tp1_exited = False
                    tp2_exited = False
                elif row["Low"] <= sell_level:
                    direction = "SHORT"
                    for c in range(num_contracts):
                        positions.append(Position(direction="SHORT", entry_price=sell_level,
                                                  entry_bar_idx=i))
                    trail_stop = buy_level
                    entry_bar_idx = i
                    entries_used += 1
                    tp1_exited = False
                    tp2_exited = False

            if direction is None:
                continue

            # ── Update trail stop (previous candle low/high) ─────
            if i > entry_bar_idx:
                if direction == "LONG":
                    prev_low = round(post_bars[i - 1][1]["Low"], 1)
                    if prev_low > trail_stop:
                        trail_stop = prev_low
                else:
                    prev_high = round(post_bars[i - 1][1]["High"], 1)
                    if prev_high < trail_stop:
                        trail_stop = prev_high

            # ── Partial exits (TP1/TP2) for 3-contract mode ─────
            if partial_exits and positions and len(positions) >= 2:
                entry_p = positions[0].entry_price
                # TP1: exit first non-add contract
                if not tp1_exited:
                    if direction == "LONG" and row["High"] >= entry_p + tp1_pts:
                        tp1_exited = True
                        tp_price = round(entry_p + tp1_pts, 1)
                        # Remove first non-add position
                        for pi, pos in enumerate(positions):
                            if not pos.is_add:
                                day_result.trades.append(ClosedTrade(
                                    date=str(trade_date), direction=direction,
                                    entry_price=pos.entry_price, exit_price=tp_price,
                                    pnl_pts=tp1_pts, is_add=False,
                                    held_bars=i - pos.entry_bar_idx,
                                ))
                                positions.pop(pi)
                                break
                    elif direction == "SHORT" and row["Low"] <= entry_p - tp1_pts:
                        tp1_exited = True
                        tp_price = round(entry_p - tp1_pts, 1)
                        for pi, pos in enumerate(positions):
                            if not pos.is_add:
                                day_result.trades.append(ClosedTrade(
                                    date=str(trade_date), direction=direction,
                                    entry_price=pos.entry_price, exit_price=tp_price,
                                    pnl_pts=tp1_pts, is_add=False,
                                    held_bars=i - pos.entry_bar_idx,
                                ))
                                positions.pop(pi)
                                break

                # TP2: exit second non-add contract
                if tp1_exited and not tp2_exited:
                    if direction == "LONG" and row["High"] >= entry_p + tp2_pts:
                        tp2_exited = True
                        tp_price = round(entry_p + tp2_pts, 1)
                        for pi, pos in enumerate(positions):
                            if not pos.is_add:
                                day_result.trades.append(ClosedTrade(
                                    date=str(trade_date), direction=direction,
                                    entry_price=pos.entry_price, exit_price=tp_price,
                                    pnl_pts=tp2_pts, is_add=False,
                                    held_bars=i - pos.entry_bar_idx,
                                ))
                                positions.pop(pi)
                                break
                    elif direction == "SHORT" and row["Low"] <= entry_p - tp2_pts:
                        tp2_exited = True
                        tp_price = round(entry_p - tp2_pts, 1)
                        for pi, pos in enumerate(positions):
                            if not pos.is_add:
                                day_result.trades.append(ClosedTrade(
                                    date=str(trade_date), direction=direction,
                                    entry_price=pos.entry_price, exit_price=tp_price,
                                    pnl_pts=tp2_pts, is_add=False,
                                    held_bars=i - pos.entry_bar_idx,
                                ))
                                positions.pop(pi)
                                break

            # ── Check stop hit → close ALL positions ─────────────
            stop_hit = False
            if direction == "LONG" and row["Low"] <= trail_stop:
                stop_hit = True
            elif direction == "SHORT" and row["High"] >= trail_stop:
                stop_hit = True

            if stop_hit:
                for pos in positions:
                    if direction == "LONG":
                        pnl = round(trail_stop - pos.entry_price, 1)
                    else:
                        pnl = round(pos.entry_price - trail_stop, 1)
                    day_result.trades.append(ClosedTrade(
                        date=str(trade_date), direction=direction,
                        entry_price=pos.entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, is_add=pos.is_add,
                        held_bars=i - pos.entry_bar_idx,
                    ))
                positions = []
                direction = None
                adds_used = 0
                # Allow re-entry on next bar (up to MAX_ENTRIES per day)

            # ── Add-to-winners check ─────────────────────────────
            if positions and adds_used < max_adds and i > entry_bar_idx:
                last_entry = positions[-1].entry_price

                if add_mode == "strength":
                    # Add into strength: when price extends +trigger from last entry
                    if direction == "LONG":
                        profit_from_last = row["High"] - last_entry
                        if profit_from_last >= add_trigger_pts:
                            add_price = round(last_entry + add_trigger_pts, 1)
                            trigger_ok = True
                        else:
                            trigger_ok = False
                    else:
                        profit_from_last = last_entry - row["Low"]
                        if profit_from_last >= add_trigger_pts:
                            add_price = round(last_entry - add_trigger_pts, 1)
                            trigger_ok = True
                        else:
                            trigger_ok = False
                else:
                    # Pullback mode: price pulls back but holds above trail stop
                    lead = positions[0]
                    if direction == "LONG":
                        unrealised = row["Close"] - lead.entry_price
                        bar_dip = row["Close"] - row["Low"]
                        dist_to_stop = row["Close"] - trail_stop
                        pulled_back = (dist_to_stop > 0 and bar_dip > 0 and
                                       bar_dip / max(1, dist_to_stop + bar_dip) >= 0.25)
                        trigger_ok = unrealised >= add_trigger_pts and pulled_back
                        add_price = round(row["Close"], 1)
                    else:
                        unrealised = lead.entry_price - row["Close"]
                        bar_dip = row["High"] - row["Close"]
                        dist_to_stop = trail_stop - row["Close"]
                        pulled_back = (dist_to_stop > 0 and bar_dip > 0 and
                                       bar_dip / max(1, dist_to_stop + bar_dip) >= 0.25)
                        trigger_ok = unrealised >= add_trigger_pts and pulled_back
                        add_price = round(row["Close"], 1)

                if trigger_ok:
                    positions.append(Position(
                        direction=direction, entry_price=add_price,
                        entry_bar_idx=i, is_add=True,
                    ))
                    adds_used += 1
                    day_result.num_adds += 1

        # ── EOD: close any remaining positions ───────────────────
        if positions and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 1)
            for pos in positions:
                if direction == "LONG":
                    pnl = round(last_price - pos.entry_price, 1)
                else:
                    pnl = round(pos.entry_price - last_price, 1)
                day_result.trades.append(ClosedTrade(
                    date=str(trade_date), direction=direction,
                    entry_price=pos.entry_price, exit_price=last_price,
                    pnl_pts=pnl, is_add=pos.is_add,
                    held_bars=len(post_bars) - pos.entry_bar_idx,
                ))

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)

    return results


def run_baseline(df: pd.DataFrame, num_contracts: int = 1,
                 partial_exits: bool = False) -> list[DayResult]:
    """Run the original candle trail (no adds, with re-entry up to MAX_ENTRIES)."""
    return run_backtest(df, add_trigger_pts=99999, max_adds=0,
                        num_contracts=num_contracts, partial_exits=partial_exits)


# ──────────────────────────────────────────────────────────────────────────────
#  Stats & output
# ──────────────────────────────────────────────────────────────────────────────

def calc_stats(results: list[DayResult], label: str = "") -> dict:
    trades = [t for r in results for t in r.trades]
    initial_trades = [t for t in trades if not t.is_add]
    add_trades = [t for t in trades if t.is_add]
    total_adds = sum(r.num_adds for r in results)

    pnl = sum(t.pnl_pts for t in trades)
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts < 0]
    wr = round(len(wins) / len(trades) * 100, 1) if trades else 0
    avg_w = round(np.mean([t.pnl_pts for t in wins]), 1) if wins else 0
    avg_l = round(np.mean([t.pnl_pts for t in losses]), 1) if losses else 0

    # Equity curve & drawdown
    eq = []
    running = 0
    for r in results:
        running += r.total_pnl
        eq.append(running)
    peak = dd = 0
    for e in eq:
        peak = max(peak, e)
        dd = max(dd, peak - e)

    # Win days
    trig_days = [r for r in results if r.triggered]
    win_days = sum(1 for r in trig_days if r.total_pnl > 0)
    win_day_pct = round(win_days / len(trig_days) * 100, 1) if trig_days else 0

    # Add trade stats
    add_pnl = sum(t.pnl_pts for t in add_trades) if add_trades else 0
    add_wins = sum(1 for t in add_trades if t.pnl_pts > 0)
    add_wr = round(add_wins / len(add_trades) * 100, 1) if add_trades else 0

    return {
        "label": label,
        "total_trades": len(trades),
        "initial_trades": len(initial_trades),
        "add_trades": len(add_trades),
        "total_adds_fired": total_adds,
        "pnl": round(pnl, 1),
        "wr": wr, "avg_w": avg_w, "avg_l": avg_l,
        "max_dd": round(dd, 1),
        "win_days": win_days, "trig_days": len(trig_days),
        "win_day_pct": win_day_pct,
        "eq": eq,
        "add_pnl": round(add_pnl, 1), "add_wr": add_wr,
    }


def print_comparison(baseline_stats: dict, add_stats: dict):
    b, a = baseline_stats, add_stats
    delta_pnl = a["pnl"] - b["pnl"]
    delta_dd = a["max_dd"] - b["max_dd"]

    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║       CANDLE TRAIL: BASELINE vs ADD-TO-WINNERS                   ║
╠════════════════════════════════════════════════════════════════════╣
║                      {'Baseline':>12}  {'Add-to-Win':>12}  {'Delta':>10}  ║
║  Total trades:       {b['total_trades']:>12}  {a['total_trades']:>12}  {a['total_trades']-b['total_trades']:>+10}  ║
║  Initial entries:    {b['initial_trades']:>12}  {a['initial_trades']:>12}               ║
║  Add positions:      {b['add_trades']:>12}  {a['add_trades']:>12}               ║
║  Win rate:           {b['wr']:>11.1f}%  {a['wr']:>11.1f}%  {a['wr']-b['wr']:>+9.1f}%  ║
║  Avg win:            {b['avg_w']:>+11.1f}  {a['avg_w']:>+11.1f}  {a['avg_w']-b['avg_w']:>+10.1f}  ║
║  Avg loss:           {b['avg_l']:>+11.1f}  {a['avg_l']:>+11.1f}  {a['avg_l']-b['avg_l']:>+10.1f}  ║
║  Total P&L:          {b['pnl']:>+11.1f}  {a['pnl']:>+11.1f}  {delta_pnl:>+10.1f}  ║
║  Max drawdown:       {b['max_dd']:>11.1f}  {a['max_dd']:>11.1f}  {delta_dd:>+10.1f}  ║
║  Win days:       {b['win_days']:>4}/{b['trig_days']:<4} ({b['win_day_pct']:>4.0f}%)  {a['win_days']:>4}/{a['trig_days']:<4} ({a['win_day_pct']:>4.0f}%)       ║
╠════════════════════════════════════════════════════════════════════╣
║  ADD POSITION BREAKDOWN                                          ║
║  Add trades fired:   {a['add_trades']:>12}                              ║
║  Add P&L:            {a['add_pnl']:>+11.1f}                              ║
║  Add win rate:       {a['add_wr']:>11.1f}%                              ║
╚════════════════════════════════════════════════════════════════════╝
""")


def print_sweep_results(sweep: list[dict]):
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  PARAMETER SWEEP — Add-to-Winners                                              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Mode      Trigger  MaxAdds  Trades  Adds  Total P&L    Add P&L   Max DD      ║
╠══════════════════════════════════════════════════════════════════════════════════╣""")

    for s in sorted(sweep, key=lambda x: x["pnl"], reverse=True):
        print(f"║  {s['mode']:<9} {s['trigger']:>6.0f}  {s['max_adds']:>7}"
              f"  {s['total_trades']:>6}  {s['add_trades']:>4}"
              f"  {s['pnl']:>+10.1f}  {s['add_pnl']:>+8.1f}  {s['max_dd']:>7.1f}      ║")

    print("╚══════════════════════════════════════════════════════════════════════════════════╝")


def generate_chart(baseline_stats: dict, add_stats_list: list[dict]):
    """Generate equity curve comparison chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        b = baseline_stats
        ax.plot(b["eq"], color="#9E9E9E", linewidth=1.5, alpha=0.8,
                label=f"Baseline ({b['pnl']:+,.0f} pts)")

        colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
        for i, a in enumerate(add_stats_list[:5]):
            c = colors[i % len(colors)]
            ax.plot(a["eq"], color=c, linewidth=1.5,
                    label=f"{a['label']} ({a['pnl']:+,.0f} pts)")

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("ASRS Candle Trail: Baseline vs Add-to-Winners", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative P&L (points)")
        ax.set_xlabel("Trading Days")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(RESULTS_DIR, "charts", "add_to_winners.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nChart saved: {path}")
    except Exception as e:
        print(f"Chart error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  Monthly breakdown
# ──────────────────────────────────────────────────────────────────────────────

def print_monthly(baseline: list[DayResult], adds: list[DayResult], label: str):
    monthly_b = defaultdict(float)
    monthly_a = defaultdict(float)
    for r in baseline:
        monthly_b[r.date[:7]] += r.total_pnl
    for r in adds:
        monthly_a[r.date[:7]] += r.total_pnl

    print(f"\n── Monthly: Baseline vs {label} ─────────────────────────────")
    print(f"{'Month':<10} {'Baseline':>10} {label:>10} {'Delta':>10} {'Better'}")
    print("─" * 55)

    total_b = total_a = 0
    for m in sorted(monthly_b.keys()):
        b, a = round(monthly_b[m], 1), round(monthly_a.get(m, 0), 1)
        total_b += b
        total_a += a
        delta = round(a - b, 1)
        better = "ADD" if a > b else ("BASE" if b > a else "SAME")
        print(f"{m:<10} {b:>+10.1f} {a:>+10.1f} {delta:>+10.1f} {better}")

    print("─" * 55)
    print(f"{'TOTAL':<10} {total_b:>+10.1f} {total_a:>+10.1f} {total_a-total_b:>+10.1f}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, split_date: str = "2025-03-01"):
    """
    Walk-forward test: optimise on data BEFORE split_date, test on data AFTER.
    This validates that add-to-winners isn't just overfit to historical data.
    """
    train_df = df[df.index.date < pd.Timestamp(split_date).date()]
    test_df = df[df.index.date >= pd.Timestamp(split_date).date()]

    train_days = len(train_df.groupby(train_df.index.date))
    test_days = len(test_df.groupby(test_df.index.date))
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Train: {train_df.index[0].date()} → {train_df.index[-1].date()} ({train_days} days)")
    print(f"  Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} ({test_days} days)")
    print(f"{'='*70}")

    # ── Step 1: Find best config on TRAIN set ────────────────────
    print("\n[TRAIN] Running parameter sweep...")
    best_pnl = -999999
    best_cfg = None
    train_sweep = []
    for mode in ["strength"]:  # Only strength since pullback doesn't work
        for trigger in [10, 15, 20, 25, 30, 40, 50]:
            for max_adds in [1, 2, 3]:
                results = run_backtest(train_df, add_trigger_pts=trigger,
                                       max_adds=max_adds, add_mode=mode)
                s = calc_stats(results, f"S{trigger}_A{max_adds}")
                train_sweep.append({
                    "trigger": trigger, "max_adds": max_adds, "mode": mode, **s,
                })
                if s["pnl"] > best_pnl:
                    best_pnl = s["pnl"]
                    best_cfg = (trigger, max_adds, mode)

    # Train baseline
    train_base = run_baseline(train_df)
    train_base_stats = calc_stats(train_base, "Baseline")

    print(f"\n[TRAIN] Baseline: {train_base_stats['pnl']:+.1f} pts")
    print(f"[TRAIN] Top 5 configs:")
    for i, s in enumerate(sorted(train_sweep, key=lambda x: x["pnl"], reverse=True)[:5]):
        print(f"  {i+1}. S{s['trigger']}_A{s['max_adds']}: {s['pnl']:+.1f} pts "
              f"(+{s['pnl']-train_base_stats['pnl']:.0f}), DD={s['max_dd']:.0f}")

    print(f"\n[TRAIN] Best config: Trigger={best_cfg[0]}, MaxAdds={best_cfg[1]}")

    # ── Step 2: Apply best config to TEST set (unseen data) ──────
    print(f"\n[TEST] Applying best config to out-of-sample data...")
    test_base = run_baseline(test_df)
    test_base_stats = calc_stats(test_base, "Baseline")

    test_adds = run_backtest(test_df, add_trigger_pts=best_cfg[0],
                              max_adds=best_cfg[1], add_mode=best_cfg[2])
    test_adds_stats = calc_stats(test_adds, f"S{best_cfg[0]}_A{best_cfg[1]}")

    print(f"\n[TEST] OUT-OF-SAMPLE RESULTS:")
    print_comparison(test_base_stats, test_adds_stats)
    print_monthly(test_base, test_adds, f"S{best_cfg[0]}_A{best_cfg[1]}")

    # ── Step 3: Also test some robust configs on TEST ────────────
    print(f"\n[TEST] Multiple configs on out-of-sample data:")
    print(f"{'Config':<12} {'Train P&L':>10} {'Test P&L':>10} {'Test Add':>10} {'Test DD':>8}")
    print("─" * 55)
    print(f"{'Baseline':<12} {train_base_stats['pnl']:>+10.1f} {test_base_stats['pnl']:>+10.1f} {'':>10} {test_base_stats['max_dd']:>8.1f}")

    for trigger in [15, 20, 25, 30]:
        for max_adds in [1, 2]:
            # Get train stats
            train_r = run_backtest(train_df, add_trigger_pts=trigger,
                                    max_adds=max_adds, add_mode="strength")
            train_s = calc_stats(train_r)
            # Get test stats
            test_r = run_backtest(test_df, add_trigger_pts=trigger,
                                   max_adds=max_adds, add_mode="strength")
            test_s = calc_stats(test_r)
            label = f"S{trigger}_A{max_adds}"
            print(f"{label:<12} {train_s['pnl']:>+10.1f} {test_s['pnl']:>+10.1f} "
                  f"{test_s['add_pnl']:>+10.1f} {test_s['max_dd']:>8.1f}")


def main():
    df = load_data()
    do_sweep = "--sweep" in sys.argv
    do_walk = "--walk" in sys.argv

    # ── Baseline (no adds) ────────────────────────────────────────
    print("\nRunning baseline (no adds)...")
    baseline = run_baseline(df)
    b_stats = calc_stats(baseline, "Baseline")
    triggered = sum(1 for r in baseline if r.triggered)
    total_pnl = sum(r.total_pnl for r in baseline if r.triggered)
    print(f"  Baseline: {triggered} days triggered, {len([t for r in baseline for t in r.trades])} trades, {total_pnl:+.1f} pts")

    if do_walk:
        run_walk_forward(df)
        return

    if do_sweep:
        # ── Parameter sweep ──────────────────────────────────────
        print("\nRunning parameter sweep...")
        sweep = []
        for mode in ["strength", "pullback"]:
            for trigger in [10, 15, 20, 25, 30, 40, 50]:
                for max_adds in [1, 2, 3]:
                    results = run_backtest(df, add_trigger_pts=trigger,
                                           max_adds=max_adds, add_mode=mode)
                    s = calc_stats(results, f"{mode[0].upper()}{trigger}_A{max_adds}")
                    sweep.append({
                        "trigger": trigger, "max_adds": max_adds, "mode": mode,
                        **s,
                    })

        print_sweep_results(sweep)

        # Show top 5
        top5 = sorted(sweep, key=lambda x: x["pnl"], reverse=True)[:5]
        print("\nTop 5 configurations by total P&L:")
        add_stats_for_chart = []
        for i, s in enumerate(top5):
            print(f"  {i+1}. Mode={s['mode']}, Trigger={s['trigger']}, MaxAdds={s['max_adds']} "
                  f"→ {s['pnl']:+.1f} pts "
                  f"(+{s['pnl'] - b_stats['pnl']:.1f} vs baseline), DD={s['max_dd']:.1f}")
            add_stats_for_chart.append(s)

        # Detailed comparison of best config
        best = top5[0]
        print(f"\n── Best config: Mode={best['mode']}, Trigger={best['trigger']}, "
              f"MaxAdds={best['max_adds']} ──")
        best_results = run_backtest(df, add_trigger_pts=best["trigger"],
                                     max_adds=best["max_adds"],
                                     add_mode=best["mode"])
        best_stats = calc_stats(best_results, best["label"])
        print_comparison(b_stats, best_stats)
        print_monthly(baseline, best_results, best["label"])
        generate_chart(b_stats, add_stats_for_chart)

    else:
        # ── Single run with default params ───────────────────────
        print("\nRunning add-to-winners (strength mode, trigger=20, max_adds=1)...")
        adds_results = run_backtest(df, add_trigger_pts=20, max_adds=1, add_mode="strength")
        a_stats = calc_stats(adds_results, "Str(20,1)")

        print_comparison(b_stats, a_stats)
        print_monthly(baseline, adds_results, "Str(20,1)")

        # Also test a few variants for context
        print("\n── Quick variant comparison ─────────────────────────────────")
        variants = [
            (15, 1, "strength", "S15_A1"),
            (20, 1, "strength", "S20_A1"),
            (25, 1, "strength", "S25_A1"),
            (30, 1, "strength", "S30_A1"),
            (20, 2, "strength", "S20_A2"),
            (25, 2, "strength", "S25_A2"),
            (30, 2, "strength", "S30_A2"),
            (15, 1, "pullback", "P15_A1"),
            (20, 1, "pullback", "P20_A1"),
            (20, 2, "pullback", "P20_A2"),
        ]
        chart_stats = []
        print(f"{'Variant':<10} {'Trades':>7} {'Adds':>5} {'P&L':>10} {'Add P&L':>9} {'DD':>8} {'WR':>6}")
        print("─" * 60)
        for trigger, max_a, mode, label in variants:
            r = run_backtest(df, add_trigger_pts=trigger, max_adds=max_a, add_mode=mode)
            s = calc_stats(r, label)
            chart_stats.append(s)
            print(f"{label:<10} {s['total_trades']:>7} {s['add_trades']:>5} "
                  f"{s['pnl']:>+10.1f} {s['add_pnl']:>+9.1f} {s['max_dd']:>8.1f} {s['wr']:>5.1f}%")

        print(f"{'Baseline':<10} {b_stats['total_trades']:>7} {b_stats['add_trades']:>5} "
              f"{b_stats['pnl']:>+10.1f} {b_stats['add_pnl']:>+9.1f} {b_stats['max_dd']:>8.1f} {b_stats['wr']:>5.1f}%")

        generate_chart(b_stats, chart_stats)


if __name__ == "__main__":
    main()
