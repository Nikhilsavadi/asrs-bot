"""
backtest_ftse.py — FTSE 1BN/1BP Backtest (exact match to live bot)
══════════════════════════════════════════════════════════════════════════════

Matches ftse_bot/strategy.py + ftse_bot/config.py exactly:

  Bar classification:
    1BN (close < open): BUY stop below bar low AND SELL stop above bar high
    1BP (close > open): SELL stop below bar low only
    DOJI: SKIP

  Position sizing:
    3 contracts × £1/pt = £3/pt (halved to £0.50/contract if bar > 30pts)

  Initial stop: entry ± bar_width (not opposite level)
  Trail: candle trail (previous bar low/high, only moves in favour)
  Add-to-winners: +1 contract every +25pts (max 2 adds, same stake)
  Flip after stop-out (max 2 entries per day)
  Session: 08:00-16:30 UK
  Slippage: 1pt per fill

Train = first 70%, Test = last 30%.
"""

import asyncio
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

# ── Config (exact match to ftse_bot/config.py) ──────────────────────────────
BUFFER_PTS = 1
BAR_WIDTH_THRESHOLD = 30   # Halve stake if bar > 30pts
MAX_ENTRIES = 2
NUM_CONTRACTS = 3
STAKE_PER_POINT = 1.0      # £1 per point per contract
DOJI_ACTION = "SKIP"

# Add-to-winners
ADD_STRENGTH_ENABLED = True
ADD_STRENGTH_TRIGGER = 25
ADD_STRENGTH_MAX = 2

# Slippage
SLIPPAGE_PER_FILL = 1

TRAIN_RATIO = 0.70
PARQUET_PATH = "/root/asrs-bot/data/ftse/ftse_rth.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ftse_data(path):
    """Load parquet with timezone workaround."""
    t = pq.read_table(path)
    idx = t.schema.get_field_index('date')
    naive = t.column('date').cast(pa.timestamp('us'))
    t = t.set_column(idx, pa.field('date', pa.timestamp('us')), naive)
    df = pd.DataFrame({c: t.column(c).to_pylist() for c in t.column_names})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def get_first_bar(day_df):
    """Get the 08:00-08:05 bar (first 5-min candle at 08:00 UK)."""
    bar = day_df[(day_df.index.hour == 8) & (day_df.index.minute == 0)]
    if bar.empty:
        return None
    first = bar.iloc[0]
    return {
        "open":  round(first["Open"], 1),
        "high":  round(first["High"], 1),
        "low":   round(first["Low"], 1),
        "close": round(first["Close"], 1),
        "range": round(first["High"] - first["Low"], 1),
    }


def classify_bar(bar):
    """1BN, 1BP, or DOJI — matches strategy.py classify_bar."""
    if bar["close"] < bar["open"]:
        return "1BN"
    elif bar["close"] > bar["open"]:
        return "1BP"
    return "DOJI"


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_entry(day_df, start_time, buy_level, sell_level, direction_hint,
                   session_end, stake, bar_width):
    """
    Simulate a single entry — matches strategy.py process_fill + update_candle_trail + process_exit.

    stake = £/pt per contract (1.0 or 0.5 if halved)
    Initial stop = entry ± bar_width
    Trail = previous bar low/high
    Adds = +1 contract at same stake, triggered at +25pts from last entry/add
    P&L = (pts × NUM_CONTRACTS + add_pts) × stake
    """
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    prev_bar = None

    adds_used = 0
    add_entries = []   # list of add entry prices
    last_add_ref = 0

    trade = {
        "direction": "", "entry": 0, "entry_time": "",
        "exit": 0, "exit_time": "", "exit_reason": "",
        "pnl_pts": 0, "pnl_gbp": 0, "mfe": 0,
        "stake": stake, "adds": 0,
        "slippage_pts": 0,
    }

    for idx, row in day_df.iterrows():
        if idx <= start_time:
            prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        # ── Not yet triggered ──
        if not direction:
            if direction_hint in ("LONG", "BOTH") and l <= buy_level:
                direction = "LONG"
                entry_price = buy_level - SLIPPAGE_PER_FILL
                # Initial stop = entry - bar_width (matches strategy.py line 229)
                trailing_stop = round(entry_price - bar_width, 1)
            elif direction_hint in ("SHORT", "BOTH") and h >= sell_level:
                direction = "SHORT"
                entry_price = sell_level + SLIPPAGE_PER_FILL
                # Initial stop = entry + bar_width (matches strategy.py line 232)
                trailing_stop = round(entry_price + bar_width, 1)
            else:
                prev_bar = row
                continue

            max_fav = entry_price
            last_add_ref = entry_price

            trade["direction"] = direction
            trade["entry"] = entry_price
            trade["entry_time"] = str(idx)
            trade["slippage_pts"] = SLIPPAGE_PER_FILL

            prev_bar = row
            continue

        # ── Position active ──
        # Add-to-winners (matches strategy.py check_add_trigger + process_add)
        if ADD_STRENGTH_ENABLED and adds_used < ADD_STRENGTH_MAX:
            if direction == "LONG":
                profit_from_ref = h - last_add_ref
            else:
                profit_from_ref = last_add_ref - l

            if profit_from_ref >= ADD_STRENGTH_TRIGGER:
                adds_used += 1
                if direction == "LONG":
                    add_price = round(last_add_ref + ADD_STRENGTH_TRIGGER + SLIPPAGE_PER_FILL, 1)
                else:
                    add_price = round(last_add_ref - ADD_STRENGTH_TRIGGER - SLIPPAGE_PER_FILL, 1)
                add_entries.append(add_price)
                last_add_ref = add_price
                trade["adds"] = adds_used
                trade["slippage_pts"] += SLIPPAGE_PER_FILL

        # Update MFE (matches strategy.py update_stop)
        if direction == "LONG" and h > max_fav:
            max_fav = h
        elif direction == "SHORT" and l < max_fav:
            max_fav = l

        # Check trailing stop hit
        stopped = False
        if direction == "LONG" and l <= trailing_stop:
            stopped = True
        elif direction == "SHORT" and h >= trailing_stop:
            stopped = True

        if stopped:
            # Exit at stop with slippage
            if direction == "LONG":
                exit_price = trailing_stop - SLIPPAGE_PER_FILL
                pnl_pts_per_contract = round(exit_price - entry_price, 1)
            else:
                exit_price = trailing_stop + SLIPPAGE_PER_FILL
                pnl_pts_per_contract = round(entry_price - exit_price, 1)

            # Add P&L (each add = 1 contract at same stake)
            add_pnl_pts = 0.0
            for ap in add_entries:
                if direction == "LONG":
                    add_pnl_pts += round(exit_price - ap, 1)
                else:
                    add_pnl_pts += round(ap - exit_price, 1)

            # Total P&L: (pts × NUM_CONTRACTS + add_pts) × stake
            total_pnl_pts = round(pnl_pts_per_contract * NUM_CONTRACTS + add_pnl_pts, 1)
            pnl_gbp = round(total_pnl_pts * stake, 2)

            trade["slippage_pts"] += SLIPPAGE_PER_FILL
            if direction == "LONG":
                mfe = round(max_fav - entry_price, 1)
            else:
                mfe = round(entry_price - max_fav, 1)

            trade["exit"] = exit_price
            trade["exit_time"] = str(idx)
            trade["exit_reason"] = "STOPPED"
            trade["pnl_pts"] = total_pnl_pts
            trade["pnl_gbp"] = pnl_gbp
            trade["mfe"] = mfe
            return trade

        # Candle trail (matches strategy.py update_candle_trail)
        if prev_bar is not None:
            if direction == "LONG":
                new_stop = round(prev_bar["Low"], 1)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop
            else:
                new_stop = round(prev_bar["High"], 1)
                if new_stop < trailing_stop:
                    trailing_stop = new_stop

        # Session close (16:30 UK)
        if idx >= session_end:
            if direction == "LONG":
                exit_price = round(c - SLIPPAGE_PER_FILL, 1)
                pnl_pts_per_contract = round(exit_price - entry_price, 1)
            else:
                exit_price = round(c + SLIPPAGE_PER_FILL, 1)
                pnl_pts_per_contract = round(entry_price - exit_price, 1)

            add_pnl_pts = 0.0
            for ap in add_entries:
                if direction == "LONG":
                    add_pnl_pts += round(exit_price - ap, 1)
                else:
                    add_pnl_pts += round(ap - exit_price, 1)

            total_pnl_pts = round(pnl_pts_per_contract * NUM_CONTRACTS + add_pnl_pts, 1)
            pnl_gbp = round(total_pnl_pts * stake, 2)

            trade["slippage_pts"] += SLIPPAGE_PER_FILL
            if direction == "LONG":
                mfe = round(max_fav - entry_price, 1)
            else:
                mfe = round(entry_price - max_fav, 1)

            trade["exit"] = exit_price
            trade["exit_time"] = str(idx)
            trade["exit_reason"] = "SESSION_CLOSE"
            trade["pnl_pts"] = total_pnl_pts
            trade["pnl_gbp"] = pnl_gbp
            trade["mfe"] = mfe
            return trade

        prev_bar = row

    # End of data fallback
    if direction and not trade.get("exit_reason"):
        last_c = day_df["Close"].iloc[-1]
        if direction == "LONG":
            exit_price = round(last_c - SLIPPAGE_PER_FILL, 1)
            pnl_pts_per_contract = round(exit_price - entry_price, 1)
        else:
            exit_price = round(last_c + SLIPPAGE_PER_FILL, 1)
            pnl_pts_per_contract = round(entry_price - exit_price, 1)

        add_pnl_pts = 0.0
        for ap in add_entries:
            if direction == "LONG":
                add_pnl_pts += round(exit_price - ap, 1)
            else:
                add_pnl_pts += round(ap - exit_price, 1)

        total_pnl_pts = round(pnl_pts_per_contract * NUM_CONTRACTS + add_pnl_pts, 1)
        pnl_gbp = round(total_pnl_pts * stake, 2)

        trade["slippage_pts"] += SLIPPAGE_PER_FILL
        trade["exit"] = exit_price
        trade["exit_time"] = str(day_df.index[-1])
        trade["exit_reason"] = "EOD"
        trade["pnl_pts"] = total_pnl_pts
        trade["pnl_gbp"] = pnl_gbp
        if direction == "LONG":
            trade["mfe"] = round(max_fav - entry_price, 1)
        else:
            trade["mfe"] = round(entry_price - max_fav, 1)
        return trade

    if not direction:
        trade["exit_reason"] = "NO_TRIGGER"
    return trade


def simulate_day(day_df):
    """Simulate full FTSE day — exact match to live bot logic."""
    first_bar = get_first_bar(day_df)
    if not first_bar:
        return None

    bar_type = classify_bar(first_bar)
    if bar_type == "DOJI":
        return {
            "date": str(day_df.index[0].date()),
            "bar_type": bar_type, "bar": first_bar,
            "trades": [], "total_pnl_gbp": 0, "skip_reason": "DOJI",
        }

    bar_width = first_bar["range"]

    # Stake: halve if bar > threshold (matches strategy.py line 188-193)
    stake = STAKE_PER_POINT
    stake_halved = False
    if bar_width > BAR_WIDTH_THRESHOLD:
        stake = round(STAKE_PER_POINT / 2, 2)
        stake_halved = True

    # Levels
    buy_level = round(first_bar["low"] - BUFFER_PTS, 1)
    sell_level = round(first_bar["high"] + BUFFER_PTS, 1)

    # Direction hint
    if bar_type == "1BN":
        direction_hint = "BOTH"   # BUY + SELL
    else:  # 1BP: sell stop above bar high
        direction_hint = "SHORT"  # SELL only

    session_end = day_df.index[0].replace(hour=16, minute=30, second=0, microsecond=0)
    bar_time = day_df.index[0].replace(hour=8, minute=0, second=0, microsecond=0)

    trades = []

    # Entry 1
    t1 = simulate_entry(day_df, bar_time, buy_level, sell_level, direction_hint,
                         session_end, stake, bar_width)
    if t1["direction"]:
        trades.append(t1)

        # Flip (Entry 2) — opposite direction only
        if t1["exit_reason"] == "STOPPED" and len(trades) < MAX_ENTRIES:
            flip_dir = "SHORT" if t1["direction"] == "LONG" else "LONG"
            exit_time = pd.Timestamp(t1["exit_time"])

            t2 = simulate_entry(day_df, exit_time, buy_level, sell_level,
                                flip_dir, session_end, stake, bar_width)
            if t2["direction"]:
                trades.append(t2)

    total_pnl_gbp = sum(t.get("pnl_gbp", 0) for t in trades)

    return {
        "date": str(day_df.index[0].date()),
        "bar_type": bar_type,
        "bar": first_bar,
        "bar_width": bar_width,
        "stake": stake,
        "stake_halved": stake_halved,
        "buy_level": buy_level,
        "sell_level": sell_level,
        "trades": trades,
        "total_pnl_gbp": round(total_pnl_gbp, 2),
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def calc_stats(results):
    all_trades = []
    daily_pnl = []

    for r in results:
        if r is None:
            daily_pnl.append(0)
            continue
        day_total = r.get("total_pnl_gbp", 0)
        daily_pnl.append(day_total)
        all_trades.extend(r.get("trades", []))

    completed = [t for t in all_trades if t.get("exit_reason") not in ("", "NO_TRIGGER")]
    total_pnl = sum(t.get("pnl_gbp", 0) for t in completed)
    wins = [t for t in completed if t.get("pnl_gbp", 0) >= 0]
    losses = [t for t in completed if t.get("pnl_gbp", 0) < 0]

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

    gross_profit = sum(t["pnl_gbp"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_gbp"] for t in losses)) if losses else 0
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    avg_win = round(np.mean([t["pnl_gbp"] for t in wins]), 2) if wins else 0
    avg_loss = round(np.mean([t["pnl_gbp"] for t in losses]), 2) if losses else 0

    adds_total = sum(t.get("adds", 0) for t in completed)
    total_slippage = sum(t.get("slippage_pts", 0) for t in completed)

    worst_day = min(daily_pnl) if len(daily_pnl) > 0 else 0
    best_day = max(daily_pnl) if len(daily_pnl) > 0 else 0

    return {
        "total_trades": len(completed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(completed) * 100, 1) if completed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(completed), 2) if completed else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "max_drawdown": round(max_dd, 2),
        "adds_total": adds_total,
        "total_slippage": round(total_slippage, 1),
        "worst_day": round(worst_day, 2),
        "best_day": round(best_day, 2),
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
    }


def print_stats(label, stats, num_days):
    print(f"\n{'━' * 65}")
    print(f"  {label}")
    print(f"{'━' * 65}")
    print(f"  Days:           {num_days}")
    print(f"  Trades:         {stats['total_trades']}")
    print(f"  Wins:           {stats['wins']}")
    print(f"  Losses:         {stats['losses']}")
    print(f"  Win rate:       {stats['win_rate']}%")
    print(f"  ─── P&L (£) ───")
    ps = '+' if stats['total_pnl'] >= 0 else ''
    print(f"  Net P&L:        {ps}£{stats['total_pnl']:,.2f}")
    print(f"  Avg P&L/trade:  {ps}£{stats['avg_pnl']:,.2f}")
    print(f"  Avg winner:     +£{stats['avg_win']:,.2f}")
    print(f"  Avg loser:      £{stats['avg_loss']:,.2f}")
    print(f"  Profit factor:  {stats['profit_factor']}")
    print(f"  Max drawdown:   £{stats['max_drawdown']:,.2f}")
    print(f"  Worst day:      £{stats['worst_day']:,.2f}")
    print(f"  Best day:       +£{stats['best_day']:,.2f}")
    per_day = round(stats['total_pnl'] / num_days, 2) if num_days > 0 else 0
    print(f"  Per day avg:    {'+' if per_day >= 0 else ''}£{per_day:,.2f}")
    print(f"  ─── Detail ───")
    print(f"  Stake:          £{STAKE_PER_POINT}/pt × {NUM_CONTRACTS} contracts = £{NUM_CONTRACTS * STAKE_PER_POINT}/pt")
    print(f"  Add-to-winners: {stats['adds_total']} adds")
    print(f"  Total slippage: {stats['total_slippage']} pts ({SLIPPAGE_PER_FILL} pts/fill)")
    print(f"{'━' * 65}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("Loading FTSE cached data...")
    df = load_ftse_data(PARQUET_PATH)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    trading_days = sorted(set(df.index.date))
    trading_days = [d for d in trading_days if d.weekday() < 5]
    print(f"Total trading days: {len(trading_days)}")

    split_idx = int(len(trading_days) * TRAIN_RATIO)
    train_days = trading_days[:split_idx]
    test_days = trading_days[split_idx:]
    print(f"Train: {len(train_days)} days ({train_days[0]} to {train_days[-1]})")
    print(f"Test:  {len(test_days)} days ({test_days[0]} to {test_days[-1]})")

    print(f"\nConfig (exact match to live bot):")
    print(f"  {NUM_CONTRACTS} contracts × £{STAKE_PER_POINT}/pt = £{NUM_CONTRACTS * STAKE_PER_POINT}/pt")
    print(f"  Stake halved to £{STAKE_PER_POINT/2}/pt if bar > {BAR_WIDTH_THRESHOLD}pts")
    print(f"  Buffer: {BUFFER_PTS}pt | DOJI: {DOJI_ACTION}")
    print(f"  Initial stop: entry ± bar_width")
    print(f"  Trail: candle trail (prev bar low/high)")
    print(f"  Add +1 contract every +{ADD_STRENGTH_TRIGGER}pts (max {ADD_STRENGTH_MAX} adds)")
    print(f"  Slippage: {SLIPPAGE_PER_FILL} pts per fill")
    print(f"  Max entries: {MAX_ENTRIES} (entry + flip)")
    print(f"  1BN: BUY + SELL | 1BP: SELL only")
    print(f"  Session: 08:00-16:30 UK")

    print(f"\nSimulating all {len(trading_days)} days...")

    all_results = []
    train_results = []
    test_results = []
    dojis = 0
    bn_count = 0
    bp_count = 0
    halved_count = 0

    for i, day in enumerate(trading_days):
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue

        r = simulate_day(day_df)
        all_results.append(r)

        if r:
            if r.get("skip_reason") == "DOJI":
                dojis += 1
            if r.get("bar_type") == "1BN":
                bn_count += 1
            elif r.get("bar_type") == "1BP":
                bp_count += 1
            if r.get("stake_halved"):
                halved_count += 1

        if day in train_days:
            train_results.append(r)
        else:
            test_results.append(r)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(trading_days)} days")

    print(f"\nDone!")
    print(f"Bar types: {bn_count} 1BN, {bp_count} 1BP, {dojis} DOJI (skipped)")
    print(f"Stake halved on {halved_count} days (bar > {BAR_WIDTH_THRESHOLD}pts)")

    # ── Bar width distribution ────────────────────────────────────────────
    widths = [r["bar"]["range"] for r in all_results if r and r.get("bar")]
    if widths:
        wa = np.array(widths)
        print(f"\nBar width: mean={np.mean(wa):.1f} median={np.median(wa):.1f} "
              f"min={np.min(wa):.1f} max={np.max(wa):.1f}")
        print(f"  Risk per stopped trade: {NUM_CONTRACTS} × £{STAKE_PER_POINT} × bar_width")
        for bw in [5, 10, 15, 20, 25, 30, 40]:
            s = STAKE_PER_POINT if bw <= BAR_WIDTH_THRESHOLD else STAKE_PER_POINT / 2
            risk = round(NUM_CONTRACTS * s * bw, 2)
            print(f"    Bar {bw:>2}pts → £{s}/pt × {NUM_CONTRACTS} = £{risk} risk/trade")

    # ── Stats ─────────────────────────────────────────────────────────────
    all_stats = calc_stats(all_results)
    print_stats(f"FTSE 1BN/1BP — ALL DATA ({len(trading_days)} days)", all_stats, len(trading_days))

    train_stats = calc_stats(train_results)
    print_stats(f"FTSE 1BN/1BP — TRAIN ({len(train_days)} days: {train_days[0]} to {train_days[-1]})",
                train_stats, len(train_days))

    test_stats = calc_stats(test_results)
    print_stats(f"FTSE 1BN/1BP — TEST ({len(test_days)} days: {test_days[0]} to {test_days[-1]})",
                test_stats, len(test_days))

    # ── Bar type breakdown ────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  P&L BY BAR TYPE")
    print(f"{'═' * 65}")
    for bt in ["1BN", "1BP"]:
        bt_results = [r for r in all_results if r and r.get("bar_type") == bt]
        bt_pnl = sum(r.get("total_pnl_gbp", 0) for r in bt_results)
        bt_trades = sum(len(r.get("trades", [])) for r in bt_results)
        bt_wins = sum(1 for r in bt_results for t in r.get("trades", []) if t.get("pnl_gbp", 0) >= 0)
        wr = round(bt_wins / bt_trades * 100, 1) if bt_trades > 0 else 0
        ps = "+" if bt_pnl >= 0 else ""
        print(f"  {bt}: {len(bt_results)} days, {bt_trades} trades, "
              f"WR={wr}%, P&L={ps}£{bt_pnl:,.2f}")

    # ── Monthly breakdown ────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  MONTHLY P&L (£)")
    print(f"{'═' * 65}")
    print(f"  {'Month':<10} {'P&L (£)':>12} {'Trades':>8} {'WR':>6} {'Set':<6}")
    print(f"  {'─'*10} {'─'*12} {'─'*8} {'─'*6} {'─'*6}")

    monthly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    monthly_set = {}
    for r in all_results:
        if r is None:
            continue
        month = r["date"][:7]
        monthly[month]["pnl"] += r.get("total_pnl_gbp", 0)
        monthly[month]["trades"] += len(r.get("trades", []))
        monthly[month]["wins"] += sum(1 for t in r.get("trades", []) if t.get("pnl_gbp", 0) >= 0)

    for day in trading_days:
        month = str(day)[:7]
        monthly_set[month] = "TRAIN" if day in train_days else "TEST"

    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = round(m["wins"] / m["trades"] * 100, 0) if m["trades"] > 0 else 0
        ps = "+" if m["pnl"] >= 0 else ""
        set_label = monthly_set.get(month, "?")
        print(f"  {month:<10} {ps}£{round(m['pnl'], 2):>10,.2f} {m['trades']:>8} {wr:>5}% {set_label:<6}")

    # ── Top best / worst days ─────────────────────────────────────────────
    valid = [r for r in all_results if r is not None and r.get("trades")]
    valid.sort(key=lambda x: x.get("total_pnl_gbp", 0), reverse=True)

    print(f"\n{'═' * 65}")
    print(f"  TOP 10 BEST DAYS")
    print(f"{'═' * 65}")
    for r in valid[:10]:
        trades_desc = []
        for t in r["trades"]:
            d = t["direction"]
            p = t.get("pnl_gbp", 0)
            ps = "+" if p >= 0 else ""
            trades_desc.append(f"{d}={ps}£{p:.2f}")
        s = r.get("stake", STAKE_PER_POINT)
        bar_info = f"[{r.get('bar_type', '?')} {r.get('bar', {}).get('range', 0):.0f}pt £{s}/pt]"
        print(f"  {r['date']}  +£{r['total_pnl_gbp']:>8,.2f}  "
              f"{bar_info} {' + '.join(trades_desc)}")

    print(f"\n{'═' * 65}")
    print(f"  TOP 10 WORST DAYS")
    print(f"{'═' * 65}")
    for r in valid[-10:]:
        trades_desc = []
        for t in r["trades"]:
            d = t["direction"]
            p = t.get("pnl_gbp", 0)
            ps = "+" if p >= 0 else ""
            trades_desc.append(f"{d}={ps}£{p:.2f}")
        s = r.get("stake", STAKE_PER_POINT)
        bar_info = f"[{r.get('bar_type', '?')} {r.get('bar', {}).get('range', 0):.0f}pt £{s}/pt]"
        print(f"  {r['date']}  £{r['total_pnl_gbp']:>8,.2f}  "
              f"{bar_info} {' + '.join(trades_desc)}")

    print(f"\n{'═' * 65}")


if __name__ == "__main__":
    asyncio.run(main())
