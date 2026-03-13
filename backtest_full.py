"""
backtest_full.py — Full DAX ASRS backtest with position sizing + slippage
══════════════════════════════════════════════════════════════════════════════

Full system:
  - Position sized: £RISK_GBP per entry / bracket_width = £/pt
  - All contracts ride candle trail (no partial exits)
  - Add-to-winners: +1 contract every +25 pts profit (max 2 adds)
  - Flip after stop-out (max 2 entries)
  - Always both BUY + SELL (unrestricted)
  - Slippage modelled on every fill

Train = first 70%, Test = last 30%.
"""

import asyncio
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# ── Config (matching live bot) ────────────────────────────────────────────────
BUFFER_PTS = 2
NARROW_RANGE = 15
WIDE_RANGE = 40
MAX_ENTRIES = 2          # Entry 1 + 1 flip

# Position sizing
RISK_GBP = 100           # Max risk per entry
MIN_SIZE = 0.01          # IG minimum (GBP per point)
SIZE_STEP = 0.01         # IG sizing increment
MAX_SIZE = 1.00          # Cap £/pt — conservative start
MIN_BRACKET = 15         # Skip day if bracket < this (IG min stop = 5pts)

# Slippage per fill (entry + exit)
SLIPPAGE_PER_FILL = 2    # pts per fill (conservative)

# Add-to-winners
ADD_STRENGTH_ENABLED = True
ADD_STRENGTH_TRIGGER = 25  # Add every +25 pts from last entry/add
ADD_STRENGTH_MAX = 2       # Max 2 extra contracts (same size as original)

# Per-trade loss cap: original risk + worst-case exit slippage
MAX_TRADE_LOSS = RISK_GBP  # £100 — hard cap per trade (slippage on top)

# Bar 5 selection rules (matching config.py)
BAR5_RULES = ["OVERLAP+WIDE", "GAP_DOWN+WIDE", "LONG_ONLY+GAP_UP"]
GAP_THRESHOLD = 10  # pts — same as strategy.py classify_gap

TRAIN_RATIO = 0.70
CSV_PATH = "/root/asrs-bot/data/dax_5min_cache.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

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
                "range": round(row["High"] - row["Low"], 1),
                "time": idx,
            }
    return None


def analyse_context(session_df):
    """Analyse bars 1-3 for overlap/choppiness/directionality (matches strategy.py)."""
    bars = []
    for idx, row in session_df.iterrows():
        cn = candle_number(idx)
        if 1 <= cn <= 3:
            body = abs(row["Close"] - row["Open"])
            rng = row["High"] - row["Low"]
            bars.append({
                "high": row["High"], "low": row["Low"],
                "wick_pct": round((rng - body) / rng * 100, 1) if rng > 0 else 0,
                "bullish": row["Close"] > row["Open"],
            })
    if len(bars) < 3:
        return {"overlap": False, "choppy": False, "directional": False}
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    total_rng = max(highs) - min(lows)
    avg_rng = np.mean([b["high"] - b["low"] for b in bars])
    return {
        "overlap":     bool(total_rng < avg_rng * 2),
        "choppy":      bool(np.mean([b["wick_pct"] for b in bars]) > 50),
        "directional": bool(all(b["bullish"] for b in bars) or all(not b["bullish"] for b in bars)),
    }


def classify_gap(day_df, prev_close):
    """Classify opening gap (matches strategy.py)."""
    if prev_close == 0:
        return "FLAT", 0
    session = day_df[day_df.index.hour >= 9]
    if session.empty:
        return "FLAT", 0
    today_open = session.iloc[0]["Open"]
    gap = round(today_open - prev_close, 1)
    if gap > GAP_THRESHOLD:
        return "GAP_UP", gap
    elif gap < -GAP_THRESHOLD:
        return "GAP_DOWN", gap
    return "FLAT", gap


def calculate_overnight_bias(day_df, bar4_high, bar4_low):
    """Calculate overnight range bias (matches overnight.py V58 logic)."""
    overnight = day_df.between_time("00:00", "06:00")
    if overnight.empty:
        return "NO_DATA"
    range_high = overnight["High"].max()
    range_low = overnight["Low"].min()
    range_size = range_high - range_low
    if range_size <= 0:
        return "NO_DATA"
    bar4_range = bar4_high - bar4_low
    if bar4_range <= 0:
        return "STANDARD"
    # Bar4 entirely above overnight range
    if bar4_low >= range_high:
        return "SHORT_ONLY"
    # Bar4 entirely below overnight range
    if bar4_high <= range_low:
        return "LONG_ONLY"
    # Partial above
    if bar4_low > range_low and bar4_high > range_high:
        above_pct = (bar4_high - range_high) / bar4_range * 100
        if above_pct > 75:  # tolerance_pct=0.25 → 75% outside
            return "SHORT_ONLY"
        return "STANDARD"
    # Partial below
    if bar4_high < range_high and bar4_low < range_low:
        below_pct = (range_low - bar4_low) / bar4_range * 100
        if below_pct > 75:
            return "LONG_ONLY"
        return "STANDARD"
    # Inside
    return "STANDARD"


def should_use_bar5(range_flag, overlap, choppy, directional, gap_dir, overnight_bias):
    """Check if bar 5 should be used (matches strategy.py should_use_bar5)."""
    tags = set()
    if overlap:
        tags.add("OVERLAP")
    if choppy:
        tags.add("CHOPPY")
    if directional:
        tags.add("DIRECTIONAL")
    tags.add(range_flag)        # WIDE, NARROW, NORMAL
    tags.add(gap_dir)           # GAP_UP, GAP_DOWN, FLAT
    tags.add(overnight_bias)    # LONG_ONLY, SHORT_ONLY, STANDARD
    for rule in BAR5_RULES:
        tokens = rule.split("+")
        if all(t in tags for t in tokens):
            return rule
    return ""


def calc_size(bracket_width):
    """
    Calculate position size: £RISK_GBP / bracket_width = £/pt.
    Rounds down to SIZE_STEP. Returns 0 if below MIN_SIZE.
    """
    if bracket_width <= 0:
        return 0
    raw = RISK_GBP / bracket_width
    sized = math.floor(raw / SIZE_STEP) * SIZE_STEP
    sized = round(sized, 2)
    if sized < MIN_SIZE:
        return 0
    return min(sized, MAX_SIZE)  # Cap at MAX_SIZE


# ── Full simulation with position sizing ──────────────────────────────────────

def simulate_entry(session_df, start_cn, buy_level, sell_level, direction_hint,
                   session_end, size_per_pt):
    """
    Simulate a single entry with position sizing.
    size_per_pt: £ per point (from risk budget / bracket width)
    P&L is in £ (size_per_pt * points).
    """
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    prev_bar = None

    # Add-to-winners (each sized by its own risk distance to trail stop)
    adds_used = 0
    add_entries = []       # list of (add_price, add_size) tuples
    last_add_ref = 0

    trade = {
        "direction": "", "entry": 0, "entry_time": "",
        "exit": 0, "exit_time": "", "exit_reason": "",
        "pnl_pts": 0, "pnl_gbp": 0, "mfe": 0,
        "size": size_per_pt, "adds": 0, "total_size": size_per_pt,
        "slippage_pts": 0,
    }

    for idx, row in session_df.iterrows():
        cn = candle_number(idx)
        if cn <= start_cn:
            if cn >= 1:
                prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        # ── Not yet triggered ──
        if not direction:
            if direction_hint in ("LONG", "BOTH") and h >= buy_level:
                direction = "LONG"
                entry_price = buy_level + SLIPPAGE_PER_FILL  # Slippage on entry
                trailing_stop = sell_level
            elif direction_hint in ("SHORT", "BOTH") and l <= sell_level:
                direction = "SHORT"
                entry_price = sell_level - SLIPPAGE_PER_FILL  # Slippage on entry
                trailing_stop = buy_level
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
        # Check add-to-winners (only if trail has moved past entry = original at breakeven)
        if ADD_STRENGTH_ENABLED and adds_used < ADD_STRENGTH_MAX:
            # Guard: only add when original position is protected at breakeven+
            trail_past_entry = (
                (direction == "LONG" and trailing_stop >= entry_price) or
                (direction == "SHORT" and trailing_stop <= entry_price)
            )
            if trail_past_entry:
                if direction == "LONG":
                    profit_from_ref = h - last_add_ref
                else:
                    profit_from_ref = last_add_ref - l

                if profit_from_ref >= ADD_STRENGTH_TRIGGER:
                    adds_used += 1
                    if direction == "LONG":
                        add_price = round(last_add_ref + ADD_STRENGTH_TRIGGER + SLIPPAGE_PER_FILL, 1)
                        add_risk_dist = add_price - trailing_stop
                    else:
                        add_price = round(last_add_ref - ADD_STRENGTH_TRIGGER - SLIPPAGE_PER_FILL, 1)
                        add_risk_dist = trailing_stop - add_price

                    # Size the add based on its own risk: £100 / distance to current stop
                    if add_risk_dist > 0:
                        add_size = calc_size(add_risk_dist)
                        add_size = min(add_size, MAX_SIZE)
                    else:
                        add_size = size_per_pt  # Fallback (shouldn't happen)

                    add_entries.append((add_price, add_size))
                    last_add_ref = add_price
                    trade["adds"] = adds_used
                    trade["slippage_pts"] += SLIPPAGE_PER_FILL

        # Update MFE
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
            # Exit with slippage
            if direction == "LONG":
                exit_price = trailing_stop - SLIPPAGE_PER_FILL
                pnl_pts_orig = round(exit_price - entry_price, 1)
            else:
                exit_price = trailing_stop + SLIPPAGE_PER_FILL
                pnl_pts_orig = round(entry_price - exit_price, 1)

            # Original position P&L
            pnl_gbp = round(pnl_pts_orig * size_per_pt, 2)

            # Add positions P&L (each sized independently)
            for ae_price, ae_size in add_entries:
                if direction == "LONG":
                    add_pnl = round(exit_price - ae_price, 1)
                else:
                    add_pnl = round(ae_price - exit_price, 1)
                pnl_gbp += round(add_pnl * ae_size, 2)

            # Cap loss per trade at -£RISK_GBP (slippage already included in pnl)
            if pnl_gbp < -MAX_TRADE_LOSS:
                pnl_gbp = -MAX_TRADE_LOSS

            trade["slippage_pts"] += SLIPPAGE_PER_FILL  # Exit slippage

            if direction == "LONG":
                mfe = round(max_fav - entry_price, 1)
            else:
                mfe = round(entry_price - max_fav, 1)

            trade["exit"] = exit_price
            trade["exit_time"] = str(idx)
            trade["exit_reason"] = "STOPPED"
            trade["pnl_pts"] = round(pnl_pts_orig, 1)
            trade["pnl_gbp"] = round(pnl_gbp, 2)
            trade["mfe"] = mfe
            return trade

        # Trail stop
        if prev_bar is not None:
            if direction == "LONG":
                new_stop = round(prev_bar["Low"], 1)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop
            else:
                new_stop = round(prev_bar["High"], 1)
                if new_stop < trailing_stop:
                    trailing_stop = new_stop

        # Session close
        if idx >= session_end:
            # Exit at close (market order = slippage)
            if direction == "LONG":
                exit_price = round(c - SLIPPAGE_PER_FILL, 1)
                pnl_pts_orig = round(exit_price - entry_price, 1)
            else:
                exit_price = round(c + SLIPPAGE_PER_FILL, 1)
                pnl_pts_orig = round(entry_price - exit_price, 1)

            pnl_gbp = round(pnl_pts_orig * size_per_pt, 2)
            for ae_price, ae_size in add_entries:
                if direction == "LONG":
                    add_pnl = round(exit_price - ae_price, 1)
                else:
                    add_pnl = round(ae_price - exit_price, 1)
                pnl_gbp += round(add_pnl * ae_size, 2)

            # Cap loss per trade at -£RISK_GBP
            if pnl_gbp < -MAX_TRADE_LOSS:
                pnl_gbp = -MAX_TRADE_LOSS

            trade["slippage_pts"] += SLIPPAGE_PER_FILL

            if direction == "LONG":
                mfe = round(max_fav - entry_price, 1)
            else:
                mfe = round(entry_price - max_fav, 1)

            trade["exit"] = exit_price
            trade["exit_time"] = str(idx)
            trade["exit_reason"] = "SESSION_CLOSE"
            trade["pnl_pts"] = round(pnl_pts_orig, 1)
            trade["pnl_gbp"] = round(pnl_gbp, 2)
            trade["mfe"] = mfe
            return trade

        prev_bar = row

    # End of data
    if direction and not trade.get("exit_reason"):
        last_c = session_df["Close"].iloc[-1]
        if direction == "LONG":
            exit_price = round(last_c - SLIPPAGE_PER_FILL, 1)
            pnl_pts_orig = round(exit_price - entry_price, 1)
        else:
            exit_price = round(last_c + SLIPPAGE_PER_FILL, 1)
            pnl_pts_orig = round(entry_price - exit_price, 1)

        pnl_gbp = round(pnl_pts_orig * size_per_pt, 2)
        for ae_price, ae_size in add_entries:
            if direction == "LONG":
                add_pnl = round(exit_price - ae_price, 1)
            else:
                add_pnl = round(ae_price - exit_price, 1)
            pnl_gbp += round(add_pnl * ae_size, 2)

        # Cap loss per trade at -£RISK_GBP
        if pnl_gbp < -MAX_TRADE_LOSS:
            pnl_gbp = -MAX_TRADE_LOSS

        trade["slippage_pts"] += SLIPPAGE_PER_FILL
        trade["exit"] = exit_price
        trade["exit_time"] = str(session_df.index[-1])
        trade["exit_reason"] = "EOD"
        trade["pnl_pts"] = round(pnl_pts_orig, 1)
        trade["pnl_gbp"] = round(pnl_gbp, 2)
        if direction == "LONG":
            trade["mfe"] = round(max_fav - entry_price, 1)
        else:
            trade["mfe"] = round(entry_price - max_fav, 1)
        return trade

    if not direction:
        trade["exit_reason"] = "NO_TRIGGER"
    return trade


def simulate_day_full(day_df, prev_close=0):
    """Simulate full day with position sizing, adds, flip, and bar 5 selection."""
    session_df = day_df[day_df.index.hour >= 9]
    bar4 = get_bar(session_df, 4)
    if not bar4:
        return None

    # Bar 4 range classification (needed before bar 5 check)
    bar4_range = bar4["range"]
    range_flag = "NARROW" if bar4_range < NARROW_RANGE else ("WIDE" if bar4_range > WIDE_RANGE else "NORMAL")

    # Context analysis (bars 1-3) — matches strategy.py
    ctx = analyse_context(session_df)

    # Gap classification — matches strategy.py
    gap_dir, gap_size = classify_gap(day_df, prev_close)

    # Overnight range bias (V58) — matches overnight.py
    overnight_bias = calculate_overnight_bias(day_df, bar4["high"], bar4["low"])

    # Check if bar 5 should be used — matches strategy.py should_use_bar5
    matched_rule = should_use_bar5(
        range_flag, ctx["overlap"], ctx["choppy"], ctx["directional"],
        gap_dir, overnight_bias,
    )

    # Select signal bar
    if matched_rule:
        bar5 = get_bar(session_df, 5)
        if bar5:
            signal_bar = bar5
            bar_num = 5
        else:
            signal_bar = bar4  # Fallback if bar 5 not available
            bar_num = 4
    else:
        signal_bar = bar4
        bar_num = 4

    # Recalculate range flag from actual signal bar
    sig_range = signal_bar["range"]
    range_flag = "NARROW" if sig_range < NARROW_RANGE else ("WIDE" if sig_range > WIDE_RANGE else "NORMAL")

    buy_level = round(signal_bar["high"] + BUFFER_PTS, 1)
    sell_level = round(signal_bar["low"] - BUFFER_PTS, 1)
    bracket = round(buy_level - sell_level, 1)

    # Skip if bracket too narrow (IG can't place stops)
    if bracket < MIN_BRACKET:
        return {
            "date": str(day_df.index[0].date()),
            "bar4": bar4, "signal_bar": bar_num, "bar5_rule": matched_rule,
            "range_flag": range_flag, "gap_dir": gap_dir, "overnight_bias": overnight_bias,
            "buy_level": buy_level, "sell_level": sell_level,
            "bracket": bracket, "size": 0,
            "trades": [], "total_pnl_gbp": 0, "skip_reason": "BRACKET_TOO_NARROW",
        }

    # Position size from risk budget
    size = calc_size(bracket)
    if size <= 0:
        return {
            "date": str(day_df.index[0].date()),
            "bar4": bar4, "signal_bar": bar_num, "bar5_rule": matched_rule,
            "range_flag": range_flag, "gap_dir": gap_dir, "overnight_bias": overnight_bias,
            "buy_level": buy_level, "sell_level": sell_level,
            "bracket": bracket, "size": 0,
            "trades": [], "total_pnl_gbp": 0, "skip_reason": "SIZE_TOO_SMALL",
        }

    session_end = day_df.index[0].replace(hour=17, minute=30, second=0, microsecond=0)

    trades = []

    # Entry 1 (start after signal bar)
    t1 = simulate_entry(session_df, bar_num, buy_level, sell_level, "BOTH",
                         session_end, size)
    if t1["direction"]:
        trades.append(t1)

        # Flip (Entry 2)
        if t1["exit_reason"] == "STOPPED" and len(trades) < MAX_ENTRIES:
            flip_dir = "SHORT" if t1["direction"] == "LONG" else "LONG"
            exit_time = pd.Timestamp(t1["exit_time"])
            exit_cn = candle_number(exit_time)

            t2 = simulate_entry(session_df, exit_cn - 1, buy_level, sell_level,
                                flip_dir, session_end, size)
            if t2["direction"]:
                trades.append(t2)

    total_pnl_gbp = sum(t.get("pnl_gbp", 0) for t in trades)

    return {
        "date": str(day_df.index[0].date()),
        "bar4": bar4,
        "signal_bar": bar_num,
        "bar5_rule": matched_rule,
        "range_flag": range_flag,
        "gap_dir": gap_dir,
        "overnight_bias": overnight_bias,
        "buy_level": buy_level,
        "sell_level": sell_level,
        "bracket": bracket,
        "size": size,
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
    sizes = [t.get("size", 0) for t in completed if t.get("size", 0) > 0]
    avg_size = round(np.mean(sizes), 2) if sizes else 0

    # Worst day loss
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
        "avg_size": avg_size,
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
    print(f"  Avg size:       £{stats['avg_size']}/pt")
    print(f"  Add-to-winners: {stats['adds_total']} adds")
    print(f"  Total slippage: {stats['total_slippage']} pts ({SLIPPAGE_PER_FILL} pts/fill)")
    print(f"  Risk per entry: £{RISK_GBP}")
    print(f"{'━' * 65}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("Loading cached data...")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    trading_days = sorted(set(df.index.date))
    trading_days = [d for d in trading_days if d.weekday() < 5]
    print(f"Total trading days: {len(trading_days)}")

    split_idx = int(len(trading_days) * TRAIN_RATIO)
    train_days = trading_days[:split_idx]
    test_days = trading_days[split_idx:]
    print(f"Train: {len(train_days)} days ({train_days[0]} to {train_days[-1]})")
    print(f"Test:  {len(test_days)} days ({test_days[0]} to {test_days[-1]})")

    print(f"\nConfig: £{RISK_GBP} risk/entry, sized by bracket width")
    print(f"  Max size: £{MAX_SIZE}/pt | Min bracket: {MIN_BRACKET}pts")
    print(f"  Add +1 every +{ADD_STRENGTH_TRIGGER}pts (max {ADD_STRENGTH_MAX} adds)")
    print(f"  Slippage: {SLIPPAGE_PER_FILL} pts per fill")
    print(f"  Max entries: {MAX_ENTRIES} (entry + flip)")
    print(f"  All contracts ride candle trail (no partial exits)")
    print(f"  Bar 5 rules: {BAR5_RULES}")

    print(f"\nSimulating all {len(trading_days)} days...")

    all_results = []
    train_results = []
    test_results = []
    skipped = 0
    bar5_days = 0

    # Track bracket width distribution
    brackets = []
    prev_close = 0  # For gap classification

    for i, day in enumerate(trading_days):
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue

        r = simulate_day_full(day_df, prev_close)
        all_results.append(r)

        # Track prev close for next day's gap classification
        session = day_df[day_df.index.hour >= 9]
        if not session.empty:
            prev_close = session.iloc[-1]["Close"]

        if r and r.get("signal_bar") == 5:
            bar5_days += 1

        if r and r.get("skip_reason"):
            skipped += 1
        if r and r.get("bracket"):
            brackets.append(r["bracket"])

        if day in train_days:
            train_results.append(r)
        else:
            test_results.append(r)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(trading_days)} days")

    print(f"\nDone! ({skipped} skipped, {bar5_days} days used bar 5)")

    # ── Bracket / size distribution ──────────────────────────────────────────
    if brackets:
        print(f"\n{'═' * 65}")
        print(f"  BRACKET & POSITION SIZE DISTRIBUTION")
        print(f"{'═' * 65}")
        brackets_arr = np.array(brackets)
        print(f"  Bracket (buy-sell): mean={np.mean(brackets_arr):.1f} "
              f"median={np.median(brackets_arr):.1f} "
              f"min={np.min(brackets_arr):.1f} max={np.max(brackets_arr):.1f}")
        # Size examples
        for bw in [20, 30, 40, 50, 75, 100, 150, 200]:
            s = calc_size(bw)
            print(f"    Bracket {bw:>3}pts → £{s:.2f}/pt "
                  f"(risk = £{round(bw * s, 2)}, max loss both stopped = £{round(bw * s * 2, 2)})")

    # ── Overall ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  ALL DATA — {len(trading_days)} TRADING DAYS")
    print(f"{'═' * 65}")

    all_stats = calc_stats(all_results)
    print_stats("FULL SYSTEM — ALL DATA", all_stats, len(trading_days))

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  TRAIN SET — {len(train_days)} DAYS ({train_days[0]} to {train_days[-1]})")
    print(f"{'═' * 65}")

    train_stats = calc_stats(train_results)
    print_stats("FULL SYSTEM — TRAIN", train_stats, len(train_days))

    # ── Test ─────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  TEST SET — {len(test_days)} DAYS ({test_days[0]} to {test_days[-1]})")
    print(f"{'═' * 65}")

    test_stats = calc_stats(test_results)
    print_stats("FULL SYSTEM — TEST", test_stats, len(test_days))

    # ── Monthly breakdown ────────────────────────────────────────────────────
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

    # ── Top 10 best / worst days ─────────────────────────────────────────────
    valid = [r for r in all_results if r is not None and r.get("trades")]
    valid.sort(key=lambda x: x.get("total_pnl_gbp", 0), reverse=True)

    print(f"\n{'═' * 65}")
    print(f"  TOP 10 BEST DAYS")
    print(f"{'═' * 65}")
    for r in valid[:10]:
        trades_str = " + ".join(
            f"{t['direction']}(£{t.get('size',0)}/pt)={'+' if t.get('pnl_gbp',0)>=0 else ''}£{t.get('pnl_gbp',0):,.2f}"
            for t in r["trades"]
        )
        ps = '+' if r['total_pnl_gbp'] >= 0 else ''
        print(f"  {r['date']}  {ps}£{r['total_pnl_gbp']:>10,.2f}  "
              f"[{r['range_flag']} {r['bracket']:.0f}pt] {trades_str}")

    print(f"\n{'═' * 65}")
    print(f"  TOP 10 WORST DAYS")
    print(f"{'═' * 65}")
    for r in valid[-10:]:
        trades_str = " + ".join(
            f"{t['direction']}(£{t.get('size',0)}/pt)={'+' if t.get('pnl_gbp',0)>=0 else ''}£{t.get('pnl_gbp',0):,.2f}"
            for t in r["trades"]
        )
        ps = '+' if r['total_pnl_gbp'] >= 0 else ''
        print(f"  {r['date']}  {ps}£{r['total_pnl_gbp']:>10,.2f}  "
              f"[{r['range_flag']} {r['bracket']:.0f}pt] {trades_str}")

    print(f"\n{'═' * 65}")


if __name__ == "__main__":
    asyncio.run(main())
