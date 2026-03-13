"""
backtest_position_sizing.py — Position sizing based on setup quality
════════════════════════════════════════════════════════════════════

Tests: 2x position on STANDARD+NARROW days, 1x on everything else.
Proper 70/30 train/test split.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

BUFFER_PTS = 2
NARROW_RANGE = 15
WIDE_RANGE = 40
MAX_ENTRIES = 2
TRAIN_RATIO = 0.70
BREAKEVEN_PTS = 15

CSV_PATH = "/root/asrs-bot/data/dax_5min_cache.csv"


def candle_number(ts):
    open_time = ts.replace(hour=9, minute=0, second=0, microsecond=0)
    return int((ts - open_time).total_seconds() / 60) // 5 + 1


def get_bar(day_df, n):
    for idx, row in day_df.iterrows():
        if candle_number(idx) == n:
            return {
                "high": round(row["High"], 1), "low": round(row["Low"], 1),
                "open": round(row["Open"], 1), "close": round(row["Close"], 1),
                "range": round(row["High"] - row["Low"], 1), "time": idx,
            }
    return None


def calculate_overnight(day_df, bar4_high, bar4_low, tolerance_pct=0.25):
    overnight = day_df[(day_df.index.hour >= 0) & (day_df.index.hour < 6)]
    if overnight.empty or len(overnight) < 3:
        return "STANDARD", 0, 0

    on_high = overnight["High"].max()
    on_low = overnight["Low"].min()
    on_range = on_high - on_low
    bar4_range = bar4_high - bar4_low

    if on_range <= 0 or bar4_range <= 0:
        return "STANDARD", on_high, on_low
    if bar4_low >= on_high:
        return "SHORT_ONLY", on_high, on_low
    if bar4_high <= on_low:
        return "LONG_ONLY", on_high, on_low

    if bar4_low > on_low and bar4_high > on_high:
        if (bar4_high - on_high) / bar4_range > (1 - tolerance_pct):
            return "SHORT_ONLY", on_high, on_low
    if bar4_high < on_high and bar4_low < on_low:
        if (on_low - bar4_low) / bar4_range > (1 - tolerance_pct):
            return "LONG_ONLY", on_high, on_low

    return "STANDARD", on_high, on_low


def simulate_trade(day_df, entry_bar_cn, buy_level, sell_level,
                   allowed_dirs, max_entries=MAX_ENTRIES,
                   breakeven_pts=BREAKEVEN_PTS):
    """Simulate with candle trail + breakeven. Returns list of trade dicts."""
    trades = []
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    max_adv = 0
    prev_bar = None
    entries_used = 0
    breakeven_hit = False

    session_end = day_df.index[0].replace(hour=17, minute=30, second=0, microsecond=0)
    session_df = day_df[day_df.index.hour >= 9]

    for idx, row in session_df.iterrows():
        cn = candle_number(idx)
        if cn <= entry_bar_cn:
            if cn >= 1:
                prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        if not direction:
            triggered = False
            if "LONG" in allowed_dirs and h >= buy_level:
                direction = "LONG"
                entry_price = buy_level
                trailing_stop = sell_level
                max_fav = entry_price
                max_adv = 0
                entries_used += 1
                breakeven_hit = False
                triggered = True
                trades.append({"num": entries_used, "direction": "LONG",
                               "entry": entry_price, "entry_time": str(idx)})
            elif "SHORT" in allowed_dirs and l <= sell_level:
                direction = "SHORT"
                entry_price = sell_level
                trailing_stop = buy_level
                max_fav = entry_price
                max_adv = 0
                entries_used += 1
                breakeven_hit = False
                triggered = True
                trades.append({"num": entries_used, "direction": "SHORT",
                               "entry": entry_price, "entry_time": str(idx)})
            if not triggered:
                prev_bar = row
                continue
            prev_bar = row
            continue

        stopped = False
        if direction == "LONG":
            if l <= trailing_stop:
                pnl = round(trailing_stop - entry_price, 1)
                mfe = round(max_fav - entry_price, 1)
                trades[-1].update({"exit": trailing_stop, "pnl": pnl, "mfe": mfe,
                                   "mae": round(max_adv, 1), "reason": "STOPPED"})
                stopped = True
            else:
                if h > max_fav: max_fav = h
                if entry_price - l > max_adv: max_adv = entry_price - l
                if not breakeven_hit and c - entry_price >= breakeven_pts:
                    breakeven_hit = True
                    if trailing_stop < entry_price:
                        trailing_stop = entry_price
                if prev_bar is not None:
                    new_stop = round(prev_bar["Low"], 1)
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop

        elif direction == "SHORT":
            if h >= trailing_stop:
                pnl = round(entry_price - trailing_stop, 1)
                mfe = round(entry_price - max_fav, 1)
                trades[-1].update({"exit": trailing_stop, "pnl": pnl, "mfe": mfe,
                                   "mae": round(max_adv, 1), "reason": "STOPPED"})
                stopped = True
            else:
                if l < max_fav: max_fav = l
                if h - entry_price > max_adv: max_adv = h - entry_price
                if not breakeven_hit and entry_price - c >= breakeven_pts:
                    breakeven_hit = True
                    if trailing_stop > entry_price:
                        trailing_stop = entry_price
                if prev_bar is not None:
                    new_stop = round(prev_bar["High"], 1)
                    if new_stop < trailing_stop:
                        trailing_stop = new_stop

        if stopped:
            if entries_used < max_entries:
                flip_dirs = set()
                if direction == "LONG" and "SHORT" in allowed_dirs:
                    flip_dirs.add("SHORT")
                elif direction == "SHORT" and "LONG" in allowed_dirs:
                    flip_dirs.add("LONG")
                direction = ""
                breakeven_hit = False
                if flip_dirs:
                    prev_bar = row
                    allowed_dirs = flip_dirs
                    continue
                else:
                    break
            else:
                break

        if idx >= session_end:
            if direction:
                if direction == "LONG":
                    pnl = round(c - entry_price, 1)
                    mfe = round(max_fav - entry_price, 1)
                else:
                    pnl = round(entry_price - c, 1)
                    mfe = round(entry_price - max_fav, 1)
                trades[-1].update({"exit": round(c, 1), "pnl": pnl, "mfe": mfe,
                                   "mae": round(max_adv, 1), "reason": "EOD"})
            break

        prev_bar = row

    if direction and trades and "exit" not in trades[-1]:
        last_c = session_df["Close"].iloc[-1]
        if direction == "LONG":
            pnl = round(last_c - entry_price, 1)
            mfe = round(max_fav - entry_price, 1)
        else:
            pnl = round(entry_price - last_c, 1)
            mfe = round(entry_price - max_fav, 1)
        trades[-1].update({"exit": round(last_c, 1), "pnl": pnl, "mfe": mfe,
                           "mae": round(max_adv, 1), "reason": "EOD"})

    return trades


def simulate_day(day_df):
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

    bias, on_high, on_low = calculate_overnight(day_df, bar4["high"], bar4["low"])

    if bias == "LONG_ONLY":
        allowed_dirs = {"LONG"}
    elif bias == "SHORT_ONLY":
        allowed_dirs = {"SHORT"}
    else:
        allowed_dirs = {"LONG", "SHORT"}

    trades = simulate_trade(day_df, 4, buy_level, sell_level, allowed_dirs)
    day_pnl = sum(t.get("pnl", 0) for t in trades)

    return {
        "date": str(day_df.index[0].date()),
        "bias": bias,
        "range_flag": range_flag,
        "risk": round(buy_level - sell_level, 1),
        "trades": trades,
        "day_pnl": round(day_pnl, 1),
    }


def calc_stats(daily_pnls):
    total = round(sum(daily_pnls), 1)
    wins = sum(1 for p in daily_pnls if p > 5)
    losses = sum(1 for p in daily_pnls if p < -5)
    flats = sum(1 for p in daily_pnls if -5 <= p <= 5)
    wr = round(wins / len(daily_pnls) * 100, 1) if daily_pnls else 0

    cum = np.cumsum(daily_pnls)
    peak = 0
    max_dd = 0
    for p in cum:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd

    gross_win = sum(p for p in daily_pnls if p > 0)
    gross_loss = abs(sum(p for p in daily_pnls if p < 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    return {
        "total": total, "wins": wins, "losses": losses, "flats": flats,
        "wr": wr, "max_dd": round(max_dd, 1), "pf": pf,
        "avg": round(total / len(daily_pnls), 1) if daily_pnls else 0,
        "days": len(daily_pnls),
    }


def print_comparison(label, baseline, sized):
    print(f"\n  {'─' * 70}")
    print(f"  {label}")
    print(f"  {'─' * 70}")
    print(f"  {'Metric':<25} {'Baseline (1x)':>15} {'Sized (2x NARROW)':>18} {'Delta':>10}")
    print(f"  {'─'*25} {'─'*15} {'─'*18} {'─'*10}")

    def fmt(v):
        ps = "+" if v >= 0 else ""
        return f"{ps}{v}"

    for metric, key in [("Days", "days"), ("Total P&L (pts)", "total"),
                         ("Avg P&L/day", "avg"), ("Win Rate %", "wr"),
                         ("Profit Factor", "pf"), ("Max Drawdown", "max_dd"),
                         ("Winners", "wins"), ("Losers", "losses")]:
        b = baseline[key]
        s = sized[key]
        if isinstance(b, float):
            delta = round(s - b, 1)
            print(f"  {metric:<25} {fmt(b):>15} {fmt(s):>18} {fmt(delta):>10}")
        else:
            delta = s - b
            print(f"  {metric:<25} {b:>15} {s:>18} {delta:>10}")


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    trading_days = sorted(set(df.index.date))
    trading_days = [d for d in trading_days if d.weekday() < 5]

    split_idx = int(len(trading_days) * TRAIN_RATIO)
    train_days = set(trading_days[:split_idx])
    test_days = set(trading_days[split_idx:])
    print(f"Train: {len(train_days)} days ({min(train_days)} to {max(train_days)})")
    print(f"Test:  {len(test_days)} days ({min(test_days)} to {max(test_days)})")

    print(f"\nSimulating {len(trading_days)} days...")

    all_results = []
    for i, day in enumerate(trading_days):
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue
        r = simulate_day(day_df)
        all_results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(trading_days)}")

    valid = [r for r in all_results if r is not None]
    print(f"Valid days: {len(valid)}")

    # ── Test multiple sizing rules ──────────────────────────────────────────
    sizing_rules = [
        ("2x on STANDARD+NARROW",
         lambda r: 2.0 if r["bias"] == "STANDARD" and r["range_flag"] == "NARROW" else 1.0),
        ("2x on NARROW (any bias)",
         lambda r: 2.0 if r["range_flag"] == "NARROW" else 1.0),
        ("2x on STANDARD (any range)",
         lambda r: 2.0 if r["bias"] == "STANDARD" else 1.0),
        ("3x on STANDARD+NARROW",
         lambda r: 3.0 if r["bias"] == "STANDARD" and r["range_flag"] == "NARROW" else 1.0),
        ("2x NARROW, 0.5x WIDE",
         lambda r: 2.0 if r["range_flag"] == "NARROW" else (0.5 if r["range_flag"] == "WIDE" else 1.0)),
        ("2x STANDARD+NARROW, skip biased+WIDE",
         lambda r: 2.0 if r["bias"] == "STANDARD" and r["range_flag"] == "NARROW" else (0.0 if r["bias"] != "STANDARD" and r["range_flag"] == "WIDE" else 1.0)),
    ]

    for set_label, day_set in [("TRAIN", train_days), ("TEST", test_days), ("ALL", set(d for d in trading_days))]:
        set_results = [r for r in valid if r["date"] in {str(d) for d in day_set}]

        print(f"\n{'═' * 75}")
        print(f"  {set_label} SET — {len(set_results)} days")
        print(f"{'═' * 75}")

        # Baseline: all 1x
        baseline_pnls = [r["day_pnl"] for r in set_results]
        baseline = calc_stats(baseline_pnls)

        for rule_name, rule_fn in sizing_rules:
            sized_pnls = [r["day_pnl"] * rule_fn(r) for r in set_results]
            # For skip (0x), remove those days from stats
            active_pnls = [r["day_pnl"] * rule_fn(r) for r in set_results if rule_fn(r) > 0]
            sized = calc_stats(sized_pnls)
            print_comparison(rule_name, baseline, sized)

    # ── Detailed breakdown: which days get 2x ──────────────────────────────
    print(f"\n{'═' * 75}")
    print(f"  STANDARD+NARROW — DETAILED BREAKDOWN")
    print(f"{'═' * 75}")

    for set_label, day_set in [("TRAIN", train_days), ("TEST", test_days)]:
        set_results = [r for r in valid if r["date"] in {str(d) for d in day_set}]

        narrow_std = [r for r in set_results if r["bias"] == "STANDARD" and r["range_flag"] == "NARROW"]
        other = [r for r in set_results if not (r["bias"] == "STANDARD" and r["range_flag"] == "NARROW")]

        narrow_pnls = [r["day_pnl"] for r in narrow_std]
        other_pnls = [r["day_pnl"] for r in other]

        n_wins = sum(1 for p in narrow_pnls if p > 5)
        n_losses = sum(1 for p in narrow_pnls if p < -5)
        o_wins = sum(1 for p in other_pnls if p > 5)
        o_losses = sum(1 for p in other_pnls if p < -5)

        print(f"\n  {set_label}:")
        print(f"    STANDARD+NARROW: {len(narrow_std)} days | "
              f"P&L: +{round(sum(narrow_pnls), 1)} | "
              f"Avg: +{round(np.mean(narrow_pnls), 1) if narrow_pnls else 0} | "
              f"Win%: {round(n_wins/len(narrow_pnls)*100, 1) if narrow_pnls else 0}% | "
              f"W:{n_wins} L:{n_losses}")
        print(f"    OTHER:           {len(other)} days | "
              f"P&L: +{round(sum(other_pnls), 1)} | "
              f"Avg: +{round(np.mean(other_pnls), 1) if other_pnls else 0} | "
              f"Win%: {round(o_wins/len(other_pnls)*100, 1) if other_pnls else 0}% | "
              f"W:{o_wins} L:{o_losses}")

        # Monthly breakdown for STANDARD+NARROW
        monthly = defaultdict(float)
        monthly_count = defaultdict(int)
        for r in narrow_std:
            m = r["date"][:7]
            monthly[m] += r["day_pnl"]
            monthly_count[m] += 1

        if monthly:
            print(f"\n    Monthly (STANDARD+NARROW):")
            print(f"    {'Month':<10} {'Days':>5} {'P&L':>8} {'Avg':>8}")
            for m in sorted(monthly.keys()):
                ps = "+" if monthly[m] >= 0 else ""
                avg = round(monthly[m] / monthly_count[m], 1)
                pa = "+" if avg >= 0 else ""
                print(f"    {m:<10} {monthly_count[m]:>5} {ps}{round(monthly[m], 1):>7} {pa}{avg:>7}")

    print(f"\n{'═' * 75}")
    print(f"  DONE")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    main()
