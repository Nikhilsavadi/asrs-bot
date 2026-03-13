"""
backtest_ema_vs_candle.py — Compare Candle Trail vs EMA Trail
══════════════════════════════════════════════════════════════════════════════

Same system, same sizing, same slippage — only the trailing stop differs:

  CANDLE TRAIL: previous bar low/high (simple, what bot currently uses)
  EMA TRAIL:    3-phase system from strategy.py:
    1. UNDERWATER: initial stop (opposite level)
    2. BREAKEVEN:  stop moves to entry when +5pts in profit
    3. EMA_TRAIL:  10-period EMA with 0.5% buffer (when +10pts & above EMA)
    Stop hit: Low/High in UNDERWATER/BREAKEVEN, Close in EMA phase

Both use: bar 4, £1/pt max, 2pts slippage, adds with breakeven guard, £100 loss cap.
"""

import asyncio
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
BUFFER_PTS = 2
MAX_ENTRIES = 2
RISK_GBP = 100
MIN_SIZE = 0.01
SIZE_STEP = 0.01
MAX_SIZE = 1.00
MIN_BRACKET = 15
SLIPPAGE_PER_FILL = 2
ADD_STRENGTH_ENABLED = True
ADD_STRENGTH_TRIGGER = 25
ADD_STRENGTH_MAX = 2
MAX_TRADE_LOSS = RISK_GBP

# EMA trail config (from config.py)
EMA_PERIOD = 10
BREAKEVEN_TRIGGER = 5     # +5pts → move stop to entry
EMA_TRIGGER = 10          # +10pts & above EMA → EMA trail
EMA_BUFFER = 0.005        # 0.5% buffer below EMA

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
            }
    return None

def calc_size(bracket_width):
    if bracket_width <= 0:
        return 0
    raw = RISK_GBP / bracket_width
    sized = math.floor(raw / SIZE_STEP) * SIZE_STEP
    sized = round(sized, 2)
    if sized < MIN_SIZE:
        return 0
    return min(sized, MAX_SIZE)

def calc_ema_series(closes, period):
    """Calculate EMA for entire series of closes."""
    if len(closes) < period:
        return [None] * len(closes)
    result = [None] * (period - 1)
    sma = sum(closes[:period]) / period
    result.append(sma)
    mult = 2 / (period + 1)
    ema = sma
    for price in closes[period:]:
        ema = (price - ema) * mult + ema
        result.append(round(ema, 1))
    return result


# ── Simulate Entry ────────────────────────────────────────────────────────────

def simulate_entry(session_df, start_cn, buy_level, sell_level, direction_hint,
                   session_end, size_per_pt, trail_mode, all_closes_before=None):
    """
    trail_mode: "CANDLE" or "EMA"
    all_closes_before: list of close prices from start of day (for EMA calculation)
    """
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    prev_bar = None
    ema_phase = "UNDERWATER"  # UNDERWATER, BREAKEVEN, EMA_TRAIL

    adds_used = 0
    add_entries = []
    last_add_ref = 0

    # For EMA: build running close list
    all_closes = list(all_closes_before) if all_closes_before else []

    trade = {
        "direction": "", "entry": 0, "entry_time": "",
        "exit": 0, "exit_time": "", "exit_reason": "",
        "pnl_pts": 0, "pnl_gbp": 0, "mfe": 0,
        "size": size_per_pt, "adds": 0,
        "slippage_pts": 0,
    }

    for idx, row in session_df.iterrows():
        cn = candle_number(idx)
        if cn <= start_cn:
            if cn >= 1:
                prev_bar = row
                all_closes.append(row["Close"])
            continue

        h, l, c = row["High"], row["Low"], row["Close"]
        all_closes.append(c)

        # ── Not yet triggered ──
        if not direction:
            if direction_hint in ("LONG", "BOTH") and h >= buy_level:
                direction = "LONG"
                entry_price = buy_level + SLIPPAGE_PER_FILL
                trailing_stop = sell_level
                ema_phase = "UNDERWATER"
            elif direction_hint in ("SHORT", "BOTH") and l <= sell_level:
                direction = "SHORT"
                entry_price = sell_level - SLIPPAGE_PER_FILL
                trailing_stop = buy_level
                ema_phase = "UNDERWATER"
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

        # Add-to-winners (breakeven guard)
        if ADD_STRENGTH_ENABLED and adds_used < ADD_STRENGTH_MAX:
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
                    if add_risk_dist > 0:
                        add_size = calc_size(add_risk_dist)
                        add_size = min(add_size, MAX_SIZE)
                    else:
                        add_size = size_per_pt
                    add_entries.append((add_price, add_size))
                    last_add_ref = add_price
                    trade["adds"] = adds_used
                    trade["slippage_pts"] += SLIPPAGE_PER_FILL

        # Update MFE
        if direction == "LONG" and h > max_fav:
            max_fav = h
        elif direction == "SHORT" and l < max_fav:
            max_fav = l

        # ── Trail stop update ──
        if trail_mode == "CANDLE":
            # Simple: previous bar low/high
            if prev_bar is not None:
                if direction == "LONG":
                    new_stop = round(prev_bar["Low"], 1)
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop
                else:
                    new_stop = round(prev_bar["High"], 1)
                    if new_stop < trailing_stop:
                        trailing_stop = new_stop

        elif trail_mode == "EMA":
            # 3-phase EMA trail (matches strategy.py)
            ema_val = None
            if len(all_closes) >= EMA_PERIOD:
                # Calculate current EMA
                mult = 2 / (EMA_PERIOD + 1)
                ema = sum(all_closes[:EMA_PERIOD]) / EMA_PERIOD
                for p in all_closes[EMA_PERIOD:]:
                    ema = (p - ema) * mult + ema
                ema_val = round(ema, 1)

            # Determine phase
            if direction == "LONG":
                favour = max_fav - entry_price
                above_ema = ema_val is not None and l > ema_val
            else:
                favour = entry_price - max_fav
                above_ema = ema_val is not None and h < ema_val

            old_phase = ema_phase
            if ema_val is not None and favour >= EMA_TRIGGER and above_ema:
                ema_phase = "EMA_TRAIL"
            elif favour >= BREAKEVEN_TRIGGER:
                if ema_phase == "UNDERWATER":
                    ema_phase = "BREAKEVEN"
            # Never downgrade from EMA_TRAIL

            # Update stop based on phase
            if ema_phase == "BREAKEVEN" and old_phase == "UNDERWATER":
                if direction == "LONG":
                    trailing_stop = max(trailing_stop, entry_price)
                else:
                    trailing_stop = min(trailing_stop, entry_price)
            elif ema_phase == "EMA_TRAIL" and ema_val is not None:
                if direction == "LONG":
                    raw_stop = round(ema_val * (1 - EMA_BUFFER), 1)
                    raw_stop = max(raw_stop, entry_price)
                    trailing_stop = max(trailing_stop, raw_stop)
                else:
                    raw_stop = round(ema_val * (1 + EMA_BUFFER), 1)
                    raw_stop = min(raw_stop, entry_price)
                    trailing_stop = min(trailing_stop, raw_stop)

        # ── Check stop hit ──
        stopped = False
        if trail_mode == "EMA" and ema_phase == "EMA_TRAIL":
            # EMA phase: Close-based stop (more lenient)
            if direction == "LONG" and c < trailing_stop:
                stopped = True
            elif direction == "SHORT" and c > trailing_stop:
                stopped = True
        else:
            # Candle trail / UNDERWATER / BREAKEVEN: Low/High based
            if direction == "LONG" and l <= trailing_stop:
                stopped = True
            elif direction == "SHORT" and h >= trailing_stop:
                stopped = True

        if stopped:
            if direction == "LONG":
                exit_price = trailing_stop - SLIPPAGE_PER_FILL
                pnl_pts = round(exit_price - entry_price, 1)
            else:
                exit_price = trailing_stop + SLIPPAGE_PER_FILL
                pnl_pts = round(entry_price - exit_price, 1)

            pnl_gbp = round(pnl_pts * size_per_pt, 2)
            for ae_price, ae_size in add_entries:
                if direction == "LONG":
                    ap = round(exit_price - ae_price, 1)
                else:
                    ap = round(ae_price - exit_price, 1)
                pnl_gbp += round(ap * ae_size, 2)

            if pnl_gbp < -MAX_TRADE_LOSS:
                pnl_gbp = -MAX_TRADE_LOSS

            trade["slippage_pts"] += SLIPPAGE_PER_FILL
            trade["exit"] = exit_price
            trade["exit_time"] = str(idx)
            trade["exit_reason"] = "STOPPED"
            trade["pnl_pts"] = pnl_pts
            trade["pnl_gbp"] = round(pnl_gbp, 2)
            trade["mfe"] = round(max_fav - entry_price, 1) if direction == "LONG" else round(entry_price - max_fav, 1)
            return trade

        # Session close
        if idx >= session_end:
            if direction == "LONG":
                exit_price = round(c - SLIPPAGE_PER_FILL, 1)
                pnl_pts = round(exit_price - entry_price, 1)
            else:
                exit_price = round(c + SLIPPAGE_PER_FILL, 1)
                pnl_pts = round(entry_price - exit_price, 1)

            pnl_gbp = round(pnl_pts * size_per_pt, 2)
            for ae_price, ae_size in add_entries:
                if direction == "LONG":
                    ap = round(exit_price - ae_price, 1)
                else:
                    ap = round(ae_price - exit_price, 1)
                pnl_gbp += round(ap * ae_size, 2)

            if pnl_gbp < -MAX_TRADE_LOSS:
                pnl_gbp = -MAX_TRADE_LOSS

            trade["slippage_pts"] += SLIPPAGE_PER_FILL
            trade["exit"] = exit_price
            trade["exit_time"] = str(idx)
            trade["exit_reason"] = "SESSION_CLOSE"
            trade["pnl_pts"] = pnl_pts
            trade["pnl_gbp"] = round(pnl_gbp, 2)
            trade["mfe"] = round(max_fav - entry_price, 1) if direction == "LONG" else round(entry_price - max_fav, 1)
            return trade

        prev_bar = row

    if not direction:
        trade["exit_reason"] = "NO_TRIGGER"
    return trade


def simulate_day(day_df, trail_mode):
    session_df = day_df[day_df.index.hour >= 9]
    bar4 = get_bar(session_df, 4)
    if not bar4:
        return None

    buy_level = round(bar4["high"] + BUFFER_PTS, 1)
    sell_level = round(bar4["low"] - BUFFER_PTS, 1)
    bracket = round(buy_level - sell_level, 1)

    if bracket < MIN_BRACKET:
        return {"date": str(day_df.index[0].date()), "trades": [], "total_pnl_gbp": 0, "skip": True}

    size = calc_size(bracket)
    if size <= 0:
        return {"date": str(day_df.index[0].date()), "trades": [], "total_pnl_gbp": 0, "skip": True}

    session_end = day_df.index[0].replace(hour=17, minute=30, second=0, microsecond=0)

    # Pre-collect closes for EMA (bars before signal)
    pre_closes = []
    for idx, row in session_df.iterrows():
        cn = candle_number(idx)
        if cn <= 4:
            pre_closes.append(row["Close"])

    trades = []
    t1 = simulate_entry(session_df, 4, buy_level, sell_level, "BOTH",
                         session_end, size, trail_mode, pre_closes)
    if t1["direction"]:
        trades.append(t1)
        if t1["exit_reason"] == "STOPPED" and len(trades) < MAX_ENTRIES:
            flip_dir = "SHORT" if t1["direction"] == "LONG" else "LONG"
            exit_time = pd.Timestamp(t1["exit_time"])
            exit_cn = candle_number(exit_time)
            # Rebuild closes up to flip point for EMA
            flip_closes = []
            for idx, row in session_df.iterrows():
                if idx <= exit_time:
                    flip_closes.append(row["Close"])
            t2 = simulate_entry(session_df, exit_cn - 1, buy_level, sell_level,
                                flip_dir, session_end, size, trail_mode, flip_closes)
            if t2["direction"]:
                trades.append(t2)

    total = sum(t.get("pnl_gbp", 0) for t in trades)
    return {
        "date": str(day_df.index[0].date()),
        "trades": trades,
        "total_pnl_gbp": round(total, 2),
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def calc_stats(results):
    all_trades = []
    daily_pnl = []
    for r in results:
        if r is None:
            daily_pnl.append(0)
            continue
        daily_pnl.append(r.get("total_pnl_gbp", 0))
        all_trades.extend(r.get("trades", []))

    completed = [t for t in all_trades if t.get("exit_reason") not in ("", "NO_TRIGGER")]
    total_pnl = sum(t.get("pnl_gbp", 0) for t in completed)
    wins = [t for t in completed if t.get("pnl_gbp", 0) >= 0]
    losses = [t for t in completed if t.get("pnl_gbp", 0) < 0]

    daily_pnl = np.array(daily_pnl)
    cum_pnl = np.cumsum(daily_pnl)
    max_dd = peak = 0
    for p in cum_pnl:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd

    gross_profit = sum(t["pnl_gbp"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_gbp"] for t in losses)) if losses else 0
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    avg_win = round(np.mean([t["pnl_gbp"] for t in wins]), 2) if wins else 0
    avg_loss = round(np.mean([t["pnl_gbp"] for t in losses]), 2) if losses else 0
    adds = sum(t.get("adds", 0) for t in completed)
    worst_day = min(daily_pnl) if len(daily_pnl) > 0 else 0
    best_day = max(daily_pnl) if len(daily_pnl) > 0 else 0

    return {
        "trades": len(completed), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins)/len(completed)*100, 1) if completed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "pf": pf, "max_dd": round(max_dd, 2),
        "adds": adds,
        "worst_day": round(worst_day, 2), "best_day": round(best_day, 2),
        "per_day": 0,
    }


def print_comparison(candle_stats, ema_stats, label, num_days):
    cs, es = candle_stats, ema_stats
    cs["per_day"] = round(cs["total_pnl"]/num_days, 2) if num_days > 0 else 0
    es["per_day"] = round(es["total_pnl"]/num_days, 2) if num_days > 0 else 0

    print(f"\n{'━' * 75}")
    print(f"  {label} ({num_days} days)")
    print(f"{'━' * 75}")
    print(f"  {'Metric':<20} {'CANDLE TRAIL':>18} {'EMA TRAIL':>18} {'Diff':>14}")
    print(f"  {'─'*20} {'─'*18} {'─'*18} {'─'*14}")

    rows = [
        ("Net P&L", f"£{cs['total_pnl']:,.2f}", f"£{es['total_pnl']:,.2f}", f"£{es['total_pnl']-cs['total_pnl']:+,.2f}"),
        ("Trades", f"{cs['trades']}", f"{es['trades']}", f"{es['trades']-cs['trades']:+d}"),
        ("Win rate", f"{cs['win_rate']}%", f"{es['win_rate']}%", f"{es['win_rate']-cs['win_rate']:+.1f}%"),
        ("Avg winner", f"£{cs['avg_win']:,.2f}", f"£{es['avg_win']:,.2f}", f"£{es['avg_win']-cs['avg_win']:+,.2f}"),
        ("Avg loser", f"£{cs['avg_loss']:,.2f}", f"£{es['avg_loss']:,.2f}", f"£{es['avg_loss']-cs['avg_loss']:+,.2f}"),
        ("Profit factor", f"{cs['pf']}", f"{es['pf']}", f"{es['pf']-cs['pf']:+.2f}"),
        ("Max drawdown", f"£{cs['max_dd']:,.2f}", f"£{es['max_dd']:,.2f}", f"£{es['max_dd']-cs['max_dd']:+,.2f}"),
        ("Worst day", f"£{cs['worst_day']:,.2f}", f"£{es['worst_day']:,.2f}", ""),
        ("Best day", f"£{cs['best_day']:,.2f}", f"£{es['best_day']:,.2f}", ""),
        ("Per day avg", f"£{cs['per_day']:,.2f}", f"£{es['per_day']:,.2f}", f"£{es['per_day']-cs['per_day']:+,.2f}"),
        ("Adds", f"{cs['adds']}", f"{es['adds']}", f"{es['adds']-cs['adds']:+d}"),
    ]
    for name, cv, ev, diff in rows:
        print(f"  {name:<20} {cv:>18} {ev:>18} {diff:>14}")
    print(f"{'━' * 75}")


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

    print(f"Train: {len(train_days)} days | Test: {len(test_days)} days")
    print(f"\nConfig: £{RISK_GBP} risk, £{MAX_SIZE}/pt max, {SLIPPAGE_PER_FILL}pts slippage")
    print(f"EMA: {EMA_PERIOD}-period, BE trigger={BREAKEVEN_TRIGGER}pts, EMA trigger={EMA_TRIGGER}pts, buffer={EMA_BUFFER}")
    print(f"EMA phase stop: Close-based | Other phases: Low/High-based")

    for mode in ["CANDLE", "EMA"]:
        print(f"\nSimulating {mode} trail...")
        results = {"all": [], "train": [], "test": []}

        for i, day in enumerate(trading_days):
            day_df = df[df.index.date == day]
            if day_df.empty:
                continue
            r = simulate_day(day_df, mode)
            results["all"].append(r)
            if day in train_days:
                results["train"].append(r)
            else:
                results["test"].append(r)
            if (i+1) % 100 == 0:
                print(f"  ... {i+1}/{len(trading_days)}")

        if mode == "CANDLE":
            candle_all = calc_stats(results["all"])
            candle_train = calc_stats(results["train"])
            candle_test = calc_stats(results["test"])
        else:
            ema_all = calc_stats(results["all"])
            ema_train = calc_stats(results["train"])
            ema_test = calc_stats(results["test"])

    print_comparison(candle_all, ema_all, "ALL DATA", len(trading_days))
    print_comparison(candle_train, ema_train, "TRAIN", len(train_days))
    print_comparison(candle_test, ema_test, "TEST", len(test_days))

    # Verdict
    print(f"\n{'═' * 75}")
    print(f"  VERDICT")
    print(f"{'═' * 75}")
    diff = ema_all["total_pnl"] - candle_all["total_pnl"]
    if diff > 0:
        print(f"  EMA TRAIL wins by £{diff:,.2f} overall")
        test_diff = ema_test["total_pnl"] - candle_test["total_pnl"]
        if test_diff > 0:
            print(f"  Also better out-of-sample by £{test_diff:,.2f} — genuine edge")
        else:
            print(f"  But WORSE out-of-sample by £{abs(test_diff):,.2f} — may be overfit to train")
    else:
        print(f"  CANDLE TRAIL wins by £{abs(diff):,.2f} overall")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    asyncio.run(main())
