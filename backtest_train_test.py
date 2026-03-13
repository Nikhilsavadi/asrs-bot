"""
backtest_train_test.py — Train/Test DAX ASRS backtest
═══════════════════════════════════════════════════════

Compares two modes:
  A) RESTRICTED: Overnight bias filters directions (LONG_ONLY / SHORT_ONLY)
  B) UNRESTRICTED: Always place both BUY + SELL, flips always available

Uses 624 trading days from cached data.
Train = first 70% (~437 days), Test = last 30% (~187 days).
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
BUFFER_PTS = 2
NARROW_RANGE = 15
WIDE_RANGE = 40
MAX_ENTRIES = 2       # Entry 1 + 1 flip
TRAIN_RATIO = 0.70

CSV_PATH = "/root/asrs-bot/data/dax_5min_cache.csv"


# ── Bar helpers ───────────────────────────────────────────────────────────────

def candle_number(ts):
    open_time = ts.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((ts - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def get_bar(day_df, n):
    for idx, row in day_df.iterrows():
        cn = candle_number(idx)
        if cn == n:
            return {
                "high": round(row["High"], 1),
                "low": round(row["Low"], 1),
                "open": round(row["Open"], 1),
                "close": round(row["Close"], 1),
                "range": round(row["High"] - row["Low"], 1),
                "time": idx,
            }
    return None


# ── Overnight bias ────────────────────────────────────────────────────────────

def calculate_overnight_bias(day_df, bar4_high, bar4_low, tolerance_pct=0.25):
    """
    Overnight range = 00:00–06:00 CET.
    Bar4 ABOVE overnight range → SHORT_ONLY (fade overnight up move)
    Bar4 BELOW overnight range → LONG_ONLY (fade overnight down move)
    Bar4 INSIDE → STANDARD (both directions)
    """
    overnight = day_df[(day_df.index.hour >= 0) & (day_df.index.hour < 6)]
    if overnight.empty or len(overnight) < 3:
        return "STANDARD", 0, 0  # No data → treat as standard

    on_high = overnight["High"].max()
    on_low = overnight["Low"].min()
    on_range = on_high - on_low

    if on_range <= 0:
        return "STANDARD", on_high, on_low

    bar4_range = bar4_high - bar4_low
    if bar4_range <= 0:
        return "STANDARD", on_high, on_low

    if bar4_low >= on_high:
        return "SHORT_ONLY", on_high, on_low

    if bar4_high <= on_low:
        return "LONG_ONLY", on_high, on_low

    # Partial overlaps
    if bar4_low > on_low and bar4_high > on_high:
        above_pct = (bar4_high - on_high) / bar4_range
        if above_pct > (1 - tolerance_pct):
            return "SHORT_ONLY", on_high, on_low

    if bar4_high < on_high and bar4_low < on_low:
        below_pct = (on_low - bar4_low) / bar4_range
        if below_pct > (1 - tolerance_pct):
            return "LONG_ONLY", on_high, on_low

    return "STANDARD", on_high, on_low


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_trade(day_df, entry_bar_cn, buy_level, sell_level,
                   allowed_dirs, max_entries=MAX_ENTRIES):
    """
    Simulate trades for one day.
    allowed_dirs: set of allowed directions, e.g. {"LONG","SHORT"} or {"LONG"} or {"SHORT"}
    Returns list of trade dicts.
    """
    trades = []
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    max_adv = 0
    prev_bar = None
    entries_used = 0

    session_end = day_df.index[0].replace(hour=17, minute=30, second=0, microsecond=0)

    # Get bars from regular session only (9:00+)
    session_df = day_df[day_df.index.hour >= 9]

    for idx, row in session_df.iterrows():
        cn = candle_number(idx)
        if cn <= entry_bar_cn:
            if cn >= 1:
                prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        if not direction:
            # Check triggers (only allowed directions)
            triggered = False
            if "LONG" in allowed_dirs and h >= buy_level:
                direction = "LONG"
                entry_price = buy_level
                trailing_stop = sell_level
                max_fav = entry_price
                max_adv = 0
                entries_used += 1
                triggered = True
                trades.append({
                    "num": entries_used, "direction": "LONG",
                    "entry": entry_price, "entry_time": str(idx),
                })
            elif "SHORT" in allowed_dirs and l <= sell_level:
                direction = "SHORT"
                entry_price = sell_level
                trailing_stop = buy_level
                max_fav = entry_price
                max_adv = 0
                entries_used += 1
                triggered = True
                trades.append({
                    "num": entries_used, "direction": "SHORT",
                    "entry": entry_price, "entry_time": str(idx),
                })

            # Check if BOTH levels hit on same bar — take the one matching allowed_dirs
            if not triggered and "LONG" in allowed_dirs and "SHORT" in allowed_dirs:
                if h >= buy_level and l <= sell_level:
                    # Ambiguous — skip (can't determine which hit first on 5-min bar)
                    pass

            if not triggered:
                prev_bar = row
                continue
            prev_bar = row
            continue

        # Position active — check stop
        stopped = False
        if direction == "LONG":
            if l <= trailing_stop:
                pnl = round(trailing_stop - entry_price, 1)
                mfe = round(max_fav - entry_price, 1)
                trades[-1].update({
                    "exit": trailing_stop, "exit_time": str(idx),
                    "exit_reason": "STOPPED", "pnl": pnl, "mfe": mfe, "mae": round(max_adv, 1),
                })
                stopped = True
            else:
                if h > max_fav:
                    max_fav = h
                if entry_price - l > max_adv:
                    max_adv = entry_price - l
                if prev_bar is not None:
                    new_stop = round(prev_bar["Low"], 1)
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop

        elif direction == "SHORT":
            if h >= trailing_stop:
                pnl = round(entry_price - trailing_stop, 1)
                mfe = round(entry_price - max_fav, 1)
                trades[-1].update({
                    "exit": trailing_stop, "exit_time": str(idx),
                    "exit_reason": "STOPPED", "pnl": pnl, "mfe": mfe, "mae": round(max_adv, 1),
                })
                stopped = True
            else:
                if l < max_fav:
                    max_fav = l
                if h - entry_price > max_adv:
                    max_adv = h - entry_price
                if prev_bar is not None:
                    new_stop = round(prev_bar["High"], 1)
                    if new_stop < trailing_stop:
                        trailing_stop = new_stop

        if stopped:
            # Can we flip?
            if entries_used < max_entries:
                # Flip: reverse direction, same levels
                flip_dirs = set()
                if direction == "LONG" and "SHORT" in allowed_dirs:
                    flip_dirs.add("SHORT")
                elif direction == "SHORT" and "LONG" in allowed_dirs:
                    flip_dirs.add("LONG")

                direction = ""
                if flip_dirs:
                    # Continue scanning for flip trigger from next bar
                    prev_bar = row
                    # Set up for flip — allowed_dirs for remaining scan
                    allowed_dirs = flip_dirs
                    continue
                else:
                    break  # Can't flip in restricted mode
            else:
                break  # Max entries reached

        # Session close
        if idx >= session_end:
            if direction:
                if direction == "LONG":
                    pnl = round(c - entry_price, 1)
                    mfe = round(max_fav - entry_price, 1)
                else:
                    pnl = round(entry_price - c, 1)
                    mfe = round(entry_price - max_fav, 1)
                trades[-1].update({
                    "exit": round(c, 1), "exit_time": str(idx),
                    "exit_reason": "SESSION_CLOSE", "pnl": pnl, "mfe": mfe, "mae": round(max_adv, 1),
                })
            break

        prev_bar = row

    # End of data — close open position
    if direction and trades and "exit" not in trades[-1]:
        last_c = session_df["Close"].iloc[-1]
        if direction == "LONG":
            pnl = round(last_c - entry_price, 1)
            mfe = round(max_fav - entry_price, 1)
        else:
            pnl = round(entry_price - last_c, 1)
            mfe = round(entry_price - max_fav, 1)
        trades[-1].update({
            "exit": round(last_c, 1), "exit_time": str(session_df.index[-1]),
            "exit_reason": "EOD", "pnl": pnl, "mfe": mfe, "mae": round(max_adv, 1),
        })

    return trades


def simulate_day(day_df):
    """Run both RESTRICTED and UNRESTRICTED modes for one day."""
    # Get bar 4
    session_df = day_df[day_df.index.hour >= 9]
    bar4 = get_bar(session_df, 4)
    if not bar4:
        return None

    rng = bar4["range"]
    if rng < NARROW_RANGE:
        range_flag = "NARROW"
    elif rng > WIDE_RANGE:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    buy_level = round(bar4["high"] + BUFFER_PTS, 1)
    sell_level = round(bar4["low"] - BUFFER_PTS, 1)

    # Calculate overnight bias
    bias, on_high, on_low = calculate_overnight_bias(
        day_df, bar4["high"], bar4["low"]
    )

    # Determine allowed directions for RESTRICTED mode
    if bias == "LONG_ONLY":
        restricted_dirs = {"LONG"}
    elif bias == "SHORT_ONLY":
        restricted_dirs = {"SHORT"}
    else:
        restricted_dirs = {"LONG", "SHORT"}

    # UNRESTRICTED: always both
    unrestricted_dirs = {"LONG", "SHORT"}

    # Simulate both modes
    trades_restricted = simulate_trade(
        day_df, 4, buy_level, sell_level, restricted_dirs
    )
    trades_unrestricted = simulate_trade(
        day_df, 4, buy_level, sell_level, unrestricted_dirs
    )

    return {
        "date": str(day_df.index[0].date()),
        "bar4": bar4,
        "range_flag": range_flag,
        "buy_level": buy_level,
        "sell_level": sell_level,
        "bias": bias,
        "on_high": round(on_high, 1),
        "on_low": round(on_low, 1),
        "restricted_dirs": restricted_dirs,
        "trades_restricted": trades_restricted,
        "trades_unrestricted": trades_unrestricted,
    }


# ── Stats calculation ────────────────────────────────────────────────────────

def calc_stats(results, mode_key):
    """Calculate stats for a set of results."""
    all_trades = []
    daily_pnl = []

    for r in results:
        if r is None:
            daily_pnl.append(0)
            continue
        trades = r[mode_key]
        day_total = sum(t.get("pnl", 0) for t in trades)
        daily_pnl.append(day_total)
        all_trades.extend(trades)

    completed = [t for t in all_trades if "pnl" in t]
    total_pnl = sum(t["pnl"] for t in completed)
    wins = [t for t in completed if t["pnl"] >= 0]
    losses = [t for t in completed if t["pnl"] < 0]

    # Daily stats
    daily_pnl = np.array(daily_pnl)
    cum_pnl = np.cumsum(daily_pnl)
    max_dd = 0
    peak = 0
    for p in cum_pnl:
        if p > peak:
            peak = p
        dd = peak - p
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Streak analysis
    max_win_streak = 0
    max_loss_streak = 0
    streak = 0
    for t in completed:
        if t["pnl"] >= 0:
            if streak >= 0:
                streak += 1
            else:
                streak = 1
            max_win_streak = max(max_win_streak, streak)
        else:
            if streak <= 0:
                streak -= 1
            else:
                streak = -1
            max_loss_streak = max(max_loss_streak, abs(streak))

    # Average win / loss
    avg_win = round(np.mean([t["pnl"] for t in wins]), 1) if wins else 0
    avg_loss = round(np.mean([t["pnl"] for t in losses]), 1) if losses else 0
    avg_mfe = round(np.mean([t.get("mfe", 0) for t in completed]), 1) if completed else 0

    return {
        "total_trades": len(completed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(completed) * 100, 1) if completed else 0,
        "total_pnl": round(total_pnl, 1),
        "avg_pnl": round(total_pnl / len(completed), 1) if completed else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_mfe": avg_mfe,
        "profit_factor": pf,
        "max_drawdown": round(max_dd, 1),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
    }


def print_stats(label, stats, num_days):
    print(f"\n{'━' * 60}")
    print(f"  {label}")
    print(f"{'━' * 60}")
    print(f"  Days:           {num_days}")
    print(f"  Trades:         {stats['total_trades']}")
    print(f"  Wins:           {stats['wins']}")
    print(f"  Losses:         {stats['losses']}")
    print(f"  Win rate:       {stats['win_rate']}%")
    print(f"  Net P&L:        {'+' if stats['total_pnl'] >= 0 else ''}{stats['total_pnl']} pts")
    print(f"  Avg P&L/trade:  {'+' if stats['avg_pnl'] >= 0 else ''}{stats['avg_pnl']} pts")
    print(f"  Avg winner:     +{stats['avg_win']} pts")
    print(f"  Avg loser:      {stats['avg_loss']} pts")
    print(f"  Avg MFE:        {stats['avg_mfe']} pts")
    print(f"  Profit factor:  {stats['profit_factor']}")
    print(f"  Max drawdown:   {stats['max_drawdown']} pts")
    print(f"  Win streak:     {stats['max_win_streak']}")
    print(f"  Loss streak:    {stats['max_loss_streak']}")
    pnl_eur = round(stats['total_pnl'] * 3, 0)
    print(f"  At €3/pt:       {'+' if pnl_eur >= 0 else ''}€{pnl_eur}")
    per_day = round(stats['total_pnl'] / num_days, 1) if num_days > 0 else 0
    print(f"  Per day avg:    {'+' if per_day >= 0 else ''}{per_day} pts")
    print(f"{'━' * 60}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("Loading cached data...")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    trading_days = sorted(set(df.index.date))
    trading_days = [d for d in trading_days if d.weekday() < 5]
    print(f"Total trading days: {len(trading_days)}")

    # Train/test split
    split_idx = int(len(trading_days) * TRAIN_RATIO)
    train_days = trading_days[:split_idx]
    test_days = trading_days[split_idx:]
    print(f"Train: {len(train_days)} days ({train_days[0]} to {train_days[-1]})")
    print(f"Test:  {len(test_days)} days ({test_days[0]} to {test_days[-1]})")

    # Run simulation
    print(f"\nSimulating all {len(trading_days)} days...")

    train_results = []
    test_results = []
    all_results = []

    bias_counts = defaultdict(int)
    bias_diff_days = []  # Days where restricted != unrestricted

    for i, day in enumerate(trading_days):
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue

        r = simulate_day(day_df)
        all_results.append(r)

        if r is not None:
            bias_counts[r["bias"]] += 1

            # Track days where bias makes a difference
            r_pnl = sum(t.get("pnl", 0) for t in r["trades_restricted"])
            u_pnl = sum(t.get("pnl", 0) for t in r["trades_unrestricted"])
            if abs(r_pnl - u_pnl) > 0.1:
                bias_diff_days.append({
                    "date": r["date"],
                    "bias": r["bias"],
                    "restricted_pnl": round(r_pnl, 1),
                    "unrestricted_pnl": round(u_pnl, 1),
                    "diff": round(u_pnl - r_pnl, 1),
                    "r_trades": len(r["trades_restricted"]),
                    "u_trades": len(r["trades_unrestricted"]),
                })

        if day in train_days:
            train_results.append(r)
        else:
            test_results.append(r)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(trading_days)} days processed")

    print(f"\nDone! {len(all_results)} days simulated.")

    # ── Overnight bias distribution ──────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  OVERNIGHT BIAS DISTRIBUTION")
    print(f"{'═' * 60}")
    for bias, count in sorted(bias_counts.items()):
        pct = round(count / sum(bias_counts.values()) * 100, 1)
        print(f"  {bias:15s}: {count:4d} days ({pct}%)")
    print(f"  Days where bias changes result: {len(bias_diff_days)}")

    # ── Overall results ──────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  ALL DATA — {len(trading_days)} TRADING DAYS")
    print(f"{'═' * 60}")

    all_restricted = calc_stats(all_results, "trades_restricted")
    all_unrestricted = calc_stats(all_results, "trades_unrestricted")

    print_stats("A) RESTRICTED (overnight bias filtering)", all_restricted, len(trading_days))
    print_stats("B) UNRESTRICTED (always both BUY + SELL)", all_unrestricted, len(trading_days))

    # Comparison
    pnl_diff = round(all_unrestricted["total_pnl"] - all_restricted["total_pnl"], 1)
    trade_diff = all_unrestricted["total_trades"] - all_restricted["total_trades"]
    print(f"\n  DELTA (Unrestricted - Restricted):")
    print(f"  P&L:    {'+' if pnl_diff >= 0 else ''}{pnl_diff} pts")
    print(f"  Trades: {'+' if trade_diff >= 0 else ''}{trade_diff}")

    # ── Train results ────────────────────────────────────────────────────────
    print(f"\n\n{'═' * 60}")
    print(f"  TRAIN SET — {len(train_days)} DAYS ({train_days[0]} to {train_days[-1]})")
    print(f"{'═' * 60}")

    train_restricted = calc_stats(train_results, "trades_restricted")
    train_unrestricted = calc_stats(train_results, "trades_unrestricted")

    print_stats("A) RESTRICTED (train)", train_restricted, len(train_days))
    print_stats("B) UNRESTRICTED (train)", train_unrestricted, len(train_days))

    # ── Test results ─────────────────────────────────────────────────────────
    print(f"\n\n{'═' * 60}")
    print(f"  TEST SET — {len(test_days)} DAYS ({test_days[0]} to {test_days[-1]})")
    print(f"{'═' * 60}")

    test_restricted = calc_stats(test_results, "trades_restricted")
    test_unrestricted = calc_stats(test_results, "trades_unrestricted")

    print_stats("A) RESTRICTED (test)", test_restricted, len(test_days))
    print_stats("B) UNRESTRICTED (test)", test_unrestricted, len(test_days))

    # ── Show days where bias made a difference ───────────────────────────────
    if bias_diff_days:
        print(f"\n\n{'═' * 60}")
        print(f"  DAYS WHERE BIAS CHANGED THE OUTCOME (showing first 30)")
        print(f"{'═' * 60}")
        print(f"  {'Date':<12} {'Bias':<13} {'Restricted':>11} {'Unrestricted':>13} {'Diff':>8} {'R#':>3} {'U#':>3}")
        print(f"  {'─'*12} {'─'*13} {'─'*11} {'─'*13} {'─'*8} {'─'*3} {'─'*3}")

        # Sort by absolute diff
        bias_diff_days.sort(key=lambda x: abs(x["diff"]), reverse=True)
        for d in bias_diff_days[:30]:
            ps_r = "+" if d["restricted_pnl"] >= 0 else ""
            ps_u = "+" if d["unrestricted_pnl"] >= 0 else ""
            ps_d = "+" if d["diff"] >= 0 else ""
            print(
                f"  {d['date']:<12} {d['bias']:<13} "
                f"{ps_r}{d['restricted_pnl']:>10} {ps_u}{d['unrestricted_pnl']:>12} "
                f"{ps_d}{d['diff']:>7} {d['r_trades']:>3} {d['u_trades']:>3}"
            )

        # Summary of bias impact
        total_diff_pos = sum(d["diff"] for d in bias_diff_days if d["diff"] > 0)
        total_diff_neg = sum(d["diff"] for d in bias_diff_days if d["diff"] < 0)
        print(f"\n  Unrestricted better: {sum(1 for d in bias_diff_days if d['diff'] > 0)} days ({'+' if total_diff_pos >= 0 else ''}{round(total_diff_pos, 1)} pts)")
        print(f"  Restricted better:  {sum(1 for d in bias_diff_days if d['diff'] < 0)} days ({round(total_diff_neg, 1)} pts)")

    # ── Monthly breakdown ────────────────────────────────────────────────────
    print(f"\n\n{'═' * 60}")
    print(f"  MONTHLY P&L COMPARISON")
    print(f"{'═' * 60}")
    print(f"  {'Month':<10} {'Restricted':>11} {'Unrestricted':>13} {'Diff':>8} {'Set':<6}")
    print(f"  {'─'*10} {'─'*11} {'─'*13} {'─'*8} {'─'*6}")

    monthly_r = defaultdict(float)
    monthly_u = defaultdict(float)
    monthly_set = {}

    for r in all_results:
        if r is None:
            continue
        month = r["date"][:7]  # YYYY-MM
        r_pnl = sum(t.get("pnl", 0) for t in r["trades_restricted"])
        u_pnl = sum(t.get("pnl", 0) for t in r["trades_unrestricted"])
        monthly_r[month] += r_pnl
        monthly_u[month] += u_pnl

    for day in trading_days:
        month = str(day)[:7]
        if day in train_days:
            monthly_set[month] = "TRAIN"
        else:
            monthly_set[month] = "TEST"

    for month in sorted(monthly_r.keys()):
        r_total = round(monthly_r[month], 1)
        u_total = round(monthly_u[month], 1)
        diff = round(u_total - r_total, 1)
        ps_r = "+" if r_total >= 0 else ""
        ps_u = "+" if u_total >= 0 else ""
        ps_d = "+" if diff >= 0 else ""
        set_label = monthly_set.get(month, "?")
        print(f"  {month:<10} {ps_r}{r_total:>10} {ps_u}{u_total:>12} {ps_d}{diff:>7} {set_label:<6}")

    print(f"\n{'═' * 60}")
    print("  CONCLUSION")
    print(f"{'═' * 60}")

    train_diff = round(train_unrestricted["total_pnl"] - train_restricted["total_pnl"], 1)
    test_diff = round(test_unrestricted["total_pnl"] - test_restricted["total_pnl"], 1)

    print(f"  Train delta: {'+' if train_diff >= 0 else ''}{train_diff} pts (unrestricted - restricted)")
    print(f"  Test delta:  {'+' if test_diff >= 0 else ''}{test_diff} pts (unrestricted - restricted)")

    if train_diff > 0 and test_diff > 0:
        print(f"  → Unrestricted BETTER in both train & test ✅")
        print(f"  → Always placing both BUY + SELL orders improves results")
    elif train_diff > 0 and test_diff < 0:
        print(f"  → Unrestricted better in train but WORSE in test ⚠️")
        print(f"  → Overnight bias filtering adds value out-of-sample")
    elif train_diff < 0 and test_diff < 0:
        print(f"  → Restricted BETTER in both train & test ✅")
        print(f"  → Keep overnight bias filtering")
    else:
        print(f"  → Mixed results — needs further analysis")

    print(f"{'═' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
