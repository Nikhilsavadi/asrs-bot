"""
backtest_ftse_atw.py — FTSE 1BN/1BP with Candle Trail + Add-to-Winners
═══════════════════════════════════════════════════════════════════════

Same exit strategy as DAX bot: candle trail stop + S25_A2 add-to-winners.
Uses FTSE 1BN/1BP entry logic (08:00 bar classification).

Usage:
    python backtest_ftse_atw.py              # Default: 3x AllTrail + S25_A2
    python backtest_ftse_atw.py --sweep      # Parameter sweep
    python backtest_ftse_atw.py --walk       # Walk-forward validation
    python backtest_ftse_atw.py --compare    # Compare candle trail vs 3-phase
"""

import os
import sys
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ftse")
RESULTS_DIR = DATA_DIR

BUFFER_PTS = 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    rth_path = os.path.join(DATA_DIR, "ftse_rth.parquet")
    all_path = os.path.join(DATA_DIR, "ftse_all.parquet")

    # Prefer all-hours (includes 08:00 bar)
    for path in [all_path, rth_path]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if df.index.tz is None:
                df.index = df.index.tz_localize("Europe/London")
            dates = pd.Series(df.index.date).unique()
            print(f"Loaded {len(df)} bars ({dates[0]} → {dates[-1]}, {len(dates)} days)")
            return df

    print(f"ERROR: No FTSE data found in {DATA_DIR}")
    sys.exit(1)


def classify_bar(o: float, c: float) -> str:
    if c < o:
        return "1BN"
    elif c > o:
        return "1BP"
    return "DOJI"


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
    bar_type: str = ""


@dataclass
class DayResult:
    date: str = ""
    trades: list = field(default_factory=list)
    total_pnl: float = 0.0
    num_adds: int = 0
    triggered: bool = False
    bar_type: str = ""


# ──────────────────────────────────────────────────────────────────────────────
#  Backtest engine — FTSE 1BN/1BP entry + candle trail + add-to-winners
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, add_trigger_pts: float = 25.0,
                 max_adds: int = 2, num_contracts: int = 3,
                 doji_action: str = "SKIP",
                 max_entries: int = 2) -> list[DayResult]:
    """
    FTSE backtest with candle trail exit + add-to-winners.

    Entry: 1BN/1BP classification of 08:00-08:05 bar
      - 1BN: BUY below bar low - buffer, SELL above bar high + buffer (OCO)
      - 1BP: SELL below bar low - buffer only

    Exit: Previous candle low/high trail (same as DAX)
    Add: Strength mode — add when price extends +trigger pts from last entry
    """
    results = []
    trading_days = sorted(set(df.index.date))

    for trade_date in trading_days:
        day_df = df[df.index.date == trade_date]
        if len(day_df) < 5 or trade_date.weekday() >= 5:
            continue

        # Find 08:00 bar
        first_bar = None
        for idx, row in day_df.iterrows():
            if idx.hour == 8 and idx.minute in (0, 5):
                first_bar = row
                first_bar_ts = idx
                break
            if idx.hour >= 8 and first_bar is None:
                first_bar = row
                first_bar_ts = idx
                break

        if first_bar is None:
            continue

        o = round(first_bar["Open"], 1)
        h = round(first_bar["High"], 1)
        l = round(first_bar["Low"], 1)
        c = round(first_bar["Close"], 1)

        bar_type = classify_bar(o, c)
        resolved_type = bar_type

        if bar_type == "DOJI":
            if doji_action == "SKIP":
                results.append(DayResult(date=str(trade_date), bar_type="DOJI"))
                continue
            elif doji_action == "TREAT_AS_1BN":
                resolved_type = "1BN"
            elif doji_action == "TREAT_AS_1BP":
                resolved_type = "1BP"

        # Entry levels
        buy_level = round(l - BUFFER_PTS, 1)
        sell_level = round(h + BUFFER_PTS, 1)

        # 1BP: only sell below bar low
        if resolved_type == "1BP":
            place_buy = False
            place_sell = True
            sell_level = buy_level  # sell below bar low
        else:  # 1BN: both sides
            place_buy = True
            place_sell = True

        # Post-bar candles (08:05+ until 16:30)
        first_bar_loc = day_df.index.get_loc(first_bar_ts)
        if isinstance(first_bar_loc, slice):
            first_bar_loc = first_bar_loc.start
        post_bars_raw = list(day_df.iloc[first_bar_loc + 1:].iterrows())

        session_bars = []
        for idx, row in post_bars_raw:
            if idx.hour > 16 or (idx.hour == 16 and idx.minute > 25):
                break
            session_bars.append((idx, row))

        day_result = DayResult(date=str(trade_date), bar_type=resolved_type)
        positions = []
        adds_used = 0
        direction = None
        trail_stop = 0.0
        entry_bar_idx = 0
        entries_used = 0

        for i, (idx, row) in enumerate(session_bars):
            # ── Initial entry ──────────────────────────────────────
            if direction is None and not positions and entries_used < max_entries:
                triggered = False

                if place_buy and row["High"] >= buy_level and place_sell and row["Low"] <= sell_level:
                    # Both could trigger — use open direction
                    if abs(row["Open"] - buy_level) < abs(row["Open"] - sell_level):
                        direction, entry_price = "LONG", buy_level
                    else:
                        direction, entry_price = "SHORT", sell_level
                    triggered = True
                elif place_buy and row["High"] >= buy_level:
                    direction, entry_price = "LONG", buy_level
                    triggered = True
                elif place_sell and row["Low"] <= sell_level:
                    direction, entry_price = "SHORT", sell_level
                    triggered = True

                if triggered:
                    for _ in range(num_contracts):
                        positions.append(Position(direction=direction,
                                                  entry_price=entry_price,
                                                  entry_bar_idx=i))
                    # Initial trail stop: bar width below/above entry
                    width = h - l
                    if direction == "LONG":
                        trail_stop = round(entry_price - width, 1)
                    else:
                        trail_stop = round(entry_price + width, 1)
                    entry_bar_idx = i
                    entries_used += 1
                    adds_used = 0
                    continue

            if direction is None:
                continue

            # ── Update trail stop (previous candle low/high) ────────
            if i > entry_bar_idx:
                if direction == "LONG":
                    prev_low = round(session_bars[i - 1][1]["Low"], 1)
                    if prev_low > trail_stop:
                        trail_stop = prev_low
                else:
                    prev_high = round(session_bars[i - 1][1]["High"], 1)
                    if prev_high < trail_stop:
                        trail_stop = prev_high

            # ── Check stop hit → close ALL positions ────────────────
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
                        bar_type=resolved_type,
                    ))
                positions = []
                direction = None
                adds_used = 0
                continue

            # ── Add-to-winners (strength mode) ──────────────────────
            if positions and adds_used < max_adds and i > entry_bar_idx:
                last_entry = positions[-1].entry_price

                if direction == "LONG":
                    profit_from_last = row["High"] - last_entry
                    if profit_from_last >= add_trigger_pts:
                        add_price = round(last_entry + add_trigger_pts, 1)
                        positions.append(Position(
                            direction=direction, entry_price=add_price,
                            entry_bar_idx=i, is_add=True,
                        ))
                        adds_used += 1
                        day_result.num_adds += 1
                else:
                    profit_from_last = last_entry - row["Low"]
                    if profit_from_last >= add_trigger_pts:
                        add_price = round(last_entry - add_trigger_pts, 1)
                        positions.append(Position(
                            direction=direction, entry_price=add_price,
                            entry_bar_idx=i, is_add=True,
                        ))
                        adds_used += 1
                        day_result.num_adds += 1

        # ── EOD: close remaining positions ──────────────────────────
        if positions and session_bars:
            last_price = round(session_bars[-1][1]["Close"], 1)
            for pos in positions:
                if direction == "LONG":
                    pnl = round(last_price - pos.entry_price, 1)
                else:
                    pnl = round(pos.entry_price - last_price, 1)
                day_result.trades.append(ClosedTrade(
                    date=str(trade_date), direction=direction,
                    entry_price=pos.entry_price, exit_price=last_price,
                    pnl_pts=pnl, is_add=pos.is_add,
                    held_bars=len(session_bars) - pos.entry_bar_idx,
                    bar_type=resolved_type,
                ))

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)

    return results


def run_baseline(df: pd.DataFrame, num_contracts: int = 1) -> list[DayResult]:
    """Baseline: single contract, no adds."""
    return run_backtest(df, add_trigger_pts=99999, max_adds=0,
                        num_contracts=num_contracts)


# ──────────────────────────────────────────────────────────────────────────────
#  Stats
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

    eq = []
    running = 0
    for r in results:
        running += r.total_pnl
        eq.append(running)
    peak = dd = 0
    for e in eq:
        peak = max(peak, e)
        dd = max(dd, peak - e)

    trig_days = [r for r in results if r.triggered]
    win_days = sum(1 for r in trig_days if r.total_pnl > 0)
    win_day_pct = round(win_days / len(trig_days) * 100, 1) if trig_days else 0

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


# ──────────────────────────────────────────────────────────────────────────────
#  Output
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison(b: dict, a: dict):
    delta_pnl = a["pnl"] - b["pnl"]
    delta_dd = a["max_dd"] - b["max_dd"]

    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║       FTSE: BASELINE vs ADD-TO-WINNERS                           ║
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
    for m in sorted(set(list(monthly_b.keys()) + list(monthly_a.keys()))):
        b, a = round(monthly_b.get(m, 0), 1), round(monthly_a.get(m, 0), 1)
        total_b += b
        total_a += a
        delta = round(a - b, 1)
        better = "ADD" if a > b else ("BASE" if b > a else "SAME")
        print(f"{m:<10} {b:>+10.1f} {a:>+10.1f} {delta:>+10.1f} {better}")

    print("─" * 55)
    print(f"{'TOTAL':<10} {total_b:>+10.1f} {total_a:>+10.1f} {total_a-total_b:>+10.1f}")


def print_bar_type_breakdown(results: list[DayResult]):
    trades = [t for r in results for t in r.trades]
    bn = [t for t in trades if t.bar_type == "1BN"]
    bp = [t for t in trades if t.bar_type == "1BP"]

    print("\n── Bar Type Breakdown ──────────────────────────────────────")
    for label, group in [("1BN", bn), ("1BP", bp)]:
        if not group:
            print(f"  {label}: no trades")
            continue
        pnl = sum(t.pnl_pts for t in group)
        wins = sum(1 for t in group if t.pnl_pts > 0)
        wr = round(wins / len(group) * 100, 1)
        print(f"  {label}: {len(group)} trades, WR {wr}%, P&L {pnl:+.1f} pts")


# ──────────────────────────────────────────────────────────────────────────────
#  Walk-forward validation
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, split_date: str = "2025-10-01"):
    """Walk-forward: optimise before split, test after."""
    train_df = df[df.index.date < pd.Timestamp(split_date).date()]
    test_df = df[df.index.date >= pd.Timestamp(split_date).date()]

    train_days = len(set(train_df.index.date))
    test_days = len(set(test_df.index.date))

    if train_days < 10 or test_days < 10:
        print(f"Not enough data for walk-forward. Train: {train_days} days, Test: {test_days} days")
        print("Need at least 10 days in each split.")
        return

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Train: {train_df.index[0].date()} → {train_df.index[-1].date()} ({train_days} days)")
    print(f"  Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} ({test_days} days)")
    print(f"{'='*70}")

    # Step 1: Find best config on TRAIN set
    print("\n[TRAIN] Running parameter sweep...")
    best_pnl = -999999
    best_cfg = None
    train_sweep = []

    for trigger in [10, 15, 20, 25, 30, 40, 50]:
        for max_adds in [1, 2, 3]:
            for nc in [1, 3]:
                results = run_backtest(train_df, add_trigger_pts=trigger,
                                       max_adds=max_adds, num_contracts=nc)
                s = calc_stats(results, f"{nc}x_S{trigger}_A{max_adds}")
                train_sweep.append({
                    "trigger": trigger, "max_adds": max_adds,
                    "num_contracts": nc, **s,
                })
                if s["pnl"] > best_pnl:
                    best_pnl = s["pnl"]
                    best_cfg = (trigger, max_adds, nc)

    train_base = run_baseline(train_df, num_contracts=1)
    train_base_stats = calc_stats(train_base, "1x Baseline")

    print(f"\n[TRAIN] 1x Baseline: {train_base_stats['pnl']:+.1f} pts")
    print(f"[TRAIN] Top 5 configs:")
    for i, s in enumerate(sorted(train_sweep, key=lambda x: x["pnl"], reverse=True)[:5]):
        print(f"  {i+1}. {s['label']}: {s['pnl']:+.1f} pts, DD={s['max_dd']:.0f}")

    print(f"\n[TRAIN] Best: Trigger={best_cfg[0]}, MaxAdds={best_cfg[1]}, Contracts={best_cfg[2]}")

    # Step 2: Apply to TEST set
    print(f"\n[TEST] Out-of-sample results:")
    test_base = run_baseline(test_df, num_contracts=1)
    test_base_stats = calc_stats(test_base, "1x Baseline")

    test_adds = run_backtest(test_df, add_trigger_pts=best_cfg[0],
                              max_adds=best_cfg[1], num_contracts=best_cfg[2])
    test_adds_stats = calc_stats(test_adds, f"{best_cfg[2]}x_S{best_cfg[0]}_A{best_cfg[1]}")

    print_comparison(test_base_stats, test_adds_stats)
    print_monthly(test_base, test_adds, test_adds_stats["label"])

    # Step 3: Multiple configs on test
    print(f"\n[TEST] Multiple configs on out-of-sample data:")
    print(f"{'Config':<16} {'Train P&L':>10} {'Test P&L':>10} {'Test Add':>10} {'Test DD':>8}")
    print("─" * 60)
    print(f"{'1x Baseline':<16} {train_base_stats['pnl']:>+10.1f} {test_base_stats['pnl']:>+10.1f} {'':>10} {test_base_stats['max_dd']:>8.1f}")

    for nc in [1, 3]:
        for trigger in [15, 20, 25, 30]:
            for max_adds in [1, 2]:
                train_r = run_backtest(train_df, add_trigger_pts=trigger,
                                        max_adds=max_adds, num_contracts=nc)
                train_s = calc_stats(train_r)
                test_r = run_backtest(test_df, add_trigger_pts=trigger,
                                       max_adds=max_adds, num_contracts=nc)
                test_s = calc_stats(test_r)
                label = f"{nc}x_S{trigger}_A{max_adds}"
                print(f"{label:<16} {train_s['pnl']:>+10.1f} {test_s['pnl']:>+10.1f} "
                      f"{test_s['add_pnl']:>+10.1f} {test_s['max_dd']:>8.1f}")


# ──────────────────────────────────────────────────────────────────────────────
#  Charts
# ──────────────────────────────────────────────────────────────────────────────

def generate_chart(stats_list: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ["#9E9E9E", "#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]

        for i, s in enumerate(stats_list[:6]):
            c = colors[i % len(colors)]
            lw = 2.0 if i == 0 else 1.5
            ax.plot(s["eq"], color=c, linewidth=lw,
                    label=f"{s['label']} ({s['pnl']:+,.0f} pts)")

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("FTSE 1BN/1BP: Candle Trail + Add-to-Winners", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative P&L (points)")
        ax.set_xlabel("Trading Days")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_dir = os.path.join(DATA_DIR, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        path = os.path.join(chart_dir, "ftse_atw.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nChart saved: {path}")
    except Exception as e:
        print(f"Chart error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    if "--walk" in sys.argv:
        # Auto-determine split: use 60% train / 40% test
        dates = sorted(set(df.index.date))
        split_idx = int(len(dates) * 0.6)
        split_date = str(dates[split_idx])
        print(f"Auto split at {split_date} ({split_idx}/{len(dates)} days)")
        run_walk_forward(df, split_date)
        return

    if "--compare" in sys.argv:
        # Compare different configs
        print("\nRunning full comparison...")
        configs = [
            ("1x Baseline", 1, 99999, 0),
            ("3x Baseline", 3, 99999, 0),
            ("1x S25_A2", 1, 25, 2),
            ("3x S25_A2", 3, 25, 2),
            ("3x S20_A2", 3, 20, 2),
            ("3x S30_A2", 3, 30, 2),
            ("3x S25_A1", 3, 25, 1),
            ("3x S25_A3", 3, 25, 3),
        ]

        all_stats = []
        print(f"\n{'Config':<14} {'Trades':>7} {'Adds':>5} {'P&L':>10} {'Add P&L':>9} {'DD':>8} {'WR':>6} {'WinDays':>8}")
        print("─" * 75)
        for label, nc, trigger, max_a in configs:
            r = run_backtest(df, add_trigger_pts=trigger, max_adds=max_a, num_contracts=nc)
            s = calc_stats(r, label)
            all_stats.append(s)
            print(f"{label:<14} {s['total_trades']:>7} {s['add_trades']:>5} "
                  f"{s['pnl']:>+10.1f} {s['add_pnl']:>+9.1f} {s['max_dd']:>8.1f} "
                  f"{s['wr']:>5.1f}% {s['win_days']:>3}/{s['trig_days']}")

        # Detailed comparison: 1x baseline vs best (3x S25_A2)
        base = all_stats[0]
        best = max(all_stats[1:], key=lambda x: x["pnl"])
        print_comparison(base, best)

        base_r = run_baseline(df, num_contracts=1)
        best_r = run_backtest(df, add_trigger_pts=25, max_adds=2, num_contracts=3)
        print_monthly(base_r, best_r, "3x_S25_A2")
        print_bar_type_breakdown(best_r)
        generate_chart(all_stats)
        return

    if "--sweep" in sys.argv:
        print("\nRunning parameter sweep...")
        sweep = []
        for nc in [1, 3]:
            for trigger in [10, 15, 20, 25, 30, 40, 50]:
                for max_adds in [1, 2, 3]:
                    r = run_backtest(df, add_trigger_pts=trigger,
                                     max_adds=max_adds, num_contracts=nc)
                    s = calc_stats(r, f"{nc}x_S{trigger}_A{max_adds}")
                    sweep.append({"trigger": trigger, "max_adds": max_adds,
                                  "nc": nc, **s})

        print(f"\n{'Config':<16} {'Trades':>7} {'Adds':>5} {'P&L':>10} {'Add P&L':>9} {'DD':>8} {'WR':>6}")
        print("─" * 65)
        for s in sorted(sweep, key=lambda x: x["pnl"], reverse=True)[:20]:
            print(f"{s['label']:<16} {s['total_trades']:>7} {s['add_trades']:>5} "
                  f"{s['pnl']:>+10.1f} {s['add_pnl']:>+9.1f} {s['max_dd']:>8.1f} {s['wr']:>5.1f}%")
        return

    # Default: run 3x AllTrail + S25_A2 (same as DAX bot)
    print("\n── FTSE 1BN/1BP: 3x Candle Trail + S25_A2 ──")
    baseline = run_baseline(df, num_contracts=1)
    b_stats = calc_stats(baseline, "1x Baseline")
    print(f"  1x Baseline: {b_stats['total_trades']} trades, {b_stats['pnl']:+.1f} pts, DD {b_stats['max_dd']:.1f}")

    adds = run_backtest(df, add_trigger_pts=25, max_adds=2, num_contracts=3)
    a_stats = calc_stats(adds, "3x S25_A2")

    print_comparison(b_stats, a_stats)
    print_monthly(baseline, adds, "3x_S25_A2")
    print_bar_type_breakdown(adds)
    generate_chart([b_stats, a_stats])


if __name__ == "__main__":
    main()
