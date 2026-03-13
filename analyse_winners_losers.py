"""
analyse_winners_losers.py — Feature analysis for ASRS DAX trades
═════════════════════════════════════════════════════════════════

Extracts features for each trading day, classifies outcome (winner/loser/flat),
identifies which features predict big winners vs losers,
and validates with proper train/test split.

Train: first 70% (~437 days), Test: last 30% (~187 days).
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
BUFFER_PTS = 2
NARROW_RANGE = 15
WIDE_RANGE = 40
MAX_ENTRIES = 2
TRAIN_RATIO = 0.70
BREAKEVEN_PTS = 15

CSV_PATH = "/root/asrs-bot/data/dax_5min_cache.csv"


def candle_number(ts):
    open_time = ts.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((ts - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def get_bar(day_df, n):
    for idx, row in day_df.iterrows():
        cn = candle_number(idx)
        if cn == n:
            return {
                "high": round(row["High"], 1), "low": round(row["Low"], 1),
                "open": round(row["Open"], 1), "close": round(row["Close"], 1),
                "range": round(row["High"] - row["Low"], 1), "time": idx,
                "bullish": row["Close"] > row["Open"],
            }
    return None


def calculate_overnight(day_df, bar4_high, bar4_low, tolerance_pct=0.25):
    overnight = day_df[(day_df.index.hour >= 0) & (day_df.index.hour < 6)]
    if overnight.empty or len(overnight) < 3:
        return "STANDARD", 0, 0, 0

    on_high = overnight["High"].max()
    on_low = overnight["Low"].min()
    on_range = on_high - on_low
    bar4_range = bar4_high - bar4_low

    if on_range <= 0 or bar4_range <= 0:
        return "STANDARD", on_high, on_low, on_range

    if bar4_low >= on_high:
        return "SHORT_ONLY", on_high, on_low, on_range
    if bar4_high <= on_low:
        return "LONG_ONLY", on_high, on_low, on_range

    if bar4_low > on_low and bar4_high > on_high:
        above_pct = (bar4_high - on_high) / bar4_range
        if above_pct > (1 - tolerance_pct):
            return "SHORT_ONLY", on_high, on_low, on_range

    if bar4_high < on_high and bar4_low < on_low:
        below_pct = (on_low - bar4_low) / bar4_range
        if below_pct > (1 - tolerance_pct):
            return "LONG_ONLY", on_high, on_low, on_range

    return "STANDARD", on_high, on_low, on_range


def analyse_context(day_df):
    bars = []
    for idx, row in day_df.iterrows():
        cn = candle_number(idx)
        if 1 <= cn <= 3:
            body = abs(row["Close"] - row["Open"])
            rng = row["High"] - row["Low"]
            bars.append({
                "high": row["High"], "low": row["Low"],
                "wick_pct": round((rng - body) / rng * 100, 1) if rng > 0 else 0,
                "bullish": row["Close"] > row["Open"],
                "range": rng,
            })
    if len(bars) < 3:
        return {"overlap": False, "choppy": False, "directional": False,
                "bars13_range": 0, "bars13_avg_range": 0}
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    total_rng = max(highs) - min(lows)
    avg_rng = np.mean([b["range"] for b in bars])
    return {
        "overlap": bool(total_rng < avg_rng * 2),
        "choppy": bool(np.mean([b["wick_pct"] for b in bars]) > 50),
        "directional": bool(all(b["bullish"] for b in bars) or all(not b["bullish"] for b in bars)),
        "bars13_range": round(total_rng, 1),
        "bars13_avg_range": round(avg_rng, 1),
    }


def simulate_day_with_breakeven(day_df, breakeven_pts=BREAKEVEN_PTS):
    """
    Simulate one day with candle trail + breakeven.
    Returns feature dict + trade results.
    """
    session_df = day_df[day_df.index.hour >= 9]
    bar4 = get_bar(session_df, 4)
    if not bar4:
        return None

    # Context
    ctx = analyse_context(session_df)

    # Range
    rng = bar4["range"]
    if rng < NARROW_RANGE:
        range_flag = "NARROW"
    elif rng > WIDE_RANGE:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    buy_level = round(bar4["high"] + BUFFER_PTS, 1)
    sell_level = round(bar4["low"] - BUFFER_PTS, 1)
    risk = round(buy_level - sell_level, 1)

    # Overnight
    bias, on_high, on_low, on_range = calculate_overnight(
        day_df, bar4["high"], bar4["low"]
    )

    # Allowed directions
    if bias == "LONG_ONLY":
        allowed_dirs = {"LONG"}
    elif bias == "SHORT_ONLY":
        allowed_dirs = {"SHORT"}
    else:
        allowed_dirs = {"LONG", "SHORT"}

    # Previous day close (for gap calc)
    prev_close = session_df["Open"].iloc[0] if not session_df.empty else 0

    # Bar 4 characteristics
    bar4_body = abs(bar4["close"] - bar4["open"])
    bar4_upper_wick = bar4["high"] - max(bar4["open"], bar4["close"])
    bar4_lower_wick = min(bar4["open"], bar4["close"]) - bar4["low"]
    bar4_body_pct = round(bar4_body / rng * 100, 1) if rng > 0 else 0

    # Day of week
    dow = day_df.index[0].weekday()

    # Simulate trades with candle trail + breakeven
    trades = simulate_trade_with_breakeven(
        day_df, 4, buy_level, sell_level, allowed_dirs,
        breakeven_pts=breakeven_pts
    )

    day_pnl = sum(t.get("pnl", 0) for t in trades)

    # Classify
    if day_pnl > 5:
        outcome = "WINNER"
    elif day_pnl < -5:
        outcome = "LOSER"
    else:
        outcome = "FLAT"

    features = {
        "date": str(day_df.index[0].date()),
        "dow": dow,
        "dow_name": ["Mon", "Tue", "Wed", "Thu", "Fri"][dow],
        "bias": bias,
        "range_flag": range_flag,
        "risk_pts": risk,
        "bar4_range": rng,
        "bar4_bullish": bar4["bullish"],
        "bar4_body_pct": bar4_body_pct,
        "on_range": round(on_range, 1),
        "context_overlap": ctx["overlap"],
        "context_choppy": ctx["choppy"],
        "context_directional": ctx["directional"],
        "bars13_range": ctx["bars13_range"],
        "bars13_avg_range": ctx["bars13_avg_range"],
        "day_pnl": round(day_pnl, 1),
        "outcome": outcome,
        "trades": trades,
        "num_trades": len(trades),
        "first_dir": trades[0]["direction"] if trades else "",
    }

    # Add per-trade details
    if trades:
        features["trade1_pnl"] = trades[0].get("pnl", 0)
        features["trade1_mfe"] = trades[0].get("mfe", 0)
        features["trade1_dir"] = trades[0].get("direction", "")
        features["trade1_breakeven_hit"] = trades[0].get("breakeven_hit", False)
    if len(trades) > 1:
        features["trade2_pnl"] = trades[1].get("pnl", 0)
        features["trade2_mfe"] = trades[1].get("mfe", 0)
        features["trade2_dir"] = trades[1].get("direction", "")

    return features


def simulate_trade_with_breakeven(day_df, entry_bar_cn, buy_level, sell_level,
                                   allowed_dirs, max_entries=MAX_ENTRIES,
                                   breakeven_pts=BREAKEVEN_PTS):
    """Simulate with candle trail + breakeven stop."""
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
                breakeven_hit = False
                triggered = True
                trades.append({
                    "num": entries_used, "direction": "SHORT",
                    "entry": entry_price, "entry_time": str(idx),
                })

            if not triggered:
                prev_bar = row
                continue
            prev_bar = row
            continue

        # Position active — check stop first
        stopped = False
        if direction == "LONG":
            if l <= trailing_stop:
                pnl = round(trailing_stop - entry_price, 1)
                mfe = round(max_fav - entry_price, 1)
                trades[-1].update({
                    "exit": trailing_stop, "exit_time": str(idx),
                    "exit_reason": "STOPPED", "pnl": pnl, "mfe": mfe,
                    "mae": round(max_adv, 1),
                    "breakeven_hit": breakeven_hit,
                    "bars_held": cn - candle_number(pd.Timestamp(trades[-1]["entry_time"])),
                })
                stopped = True
            else:
                if h > max_fav:
                    max_fav = h
                if entry_price - l > max_adv:
                    max_adv = entry_price - l

                # Breakeven check
                if not breakeven_hit and c - entry_price >= breakeven_pts:
                    breakeven_hit = True
                    if trailing_stop < entry_price:
                        trailing_stop = entry_price

                # Candle trail (use prev completed bar)
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
                    "exit_reason": "STOPPED", "pnl": pnl, "mfe": mfe,
                    "mae": round(max_adv, 1),
                    "breakeven_hit": breakeven_hit,
                    "bars_held": cn - candle_number(pd.Timestamp(trades[-1]["entry_time"])),
                })
                stopped = True
            else:
                if l < max_fav:
                    max_fav = l
                if h - entry_price > max_adv:
                    max_adv = h - entry_price

                # Breakeven check
                if not breakeven_hit and entry_price - c >= breakeven_pts:
                    breakeven_hit = True
                    if trailing_stop > entry_price:
                        trailing_stop = entry_price

                # Candle trail
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
                trades[-1].update({
                    "exit": round(c, 1), "exit_time": str(idx),
                    "exit_reason": "SESSION_CLOSE", "pnl": pnl, "mfe": mfe,
                    "mae": round(max_adv, 1),
                    "breakeven_hit": breakeven_hit,
                    "bars_held": cn - candle_number(pd.Timestamp(trades[-1]["entry_time"])),
                })
            break

        prev_bar = row

    # End of data
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
            "breakeven_hit": breakeven_hit, "bars_held": 0,
        })

    return trades


# ── Analysis ─────────────────────────────────────────────────────────────────

def feature_breakdown(results, feature_name, label):
    """Show P&L breakdown by a categorical feature."""
    groups = defaultdict(list)
    for r in results:
        if r is None:
            continue
        val = r.get(feature_name, "?")
        groups[val].append(r["day_pnl"])

    print(f"\n  {label}:")
    print(f"  {'Value':<20} {'Days':>5} {'Total':>8} {'Avg':>8} {'Win%':>6} {'Winners':>8} {'Losers':>7}")
    print(f"  {'─'*20} {'─'*5} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*7}")

    for val in sorted(groups.keys(), key=lambda v: str(v)):
        pnls = groups[val]
        total = round(sum(pnls), 1)
        avg = round(np.mean(pnls), 1)
        wins = sum(1 for p in pnls if p > 5)
        losses = sum(1 for p in pnls if p < -5)
        win_pct = round(wins / len(pnls) * 100, 1) if pnls else 0
        ps = "+" if total >= 0 else ""
        pa = "+" if avg >= 0 else ""
        print(f"  {str(val):<20} {len(pnls):>5} {ps}{total:>7} {pa}{avg:>7} {win_pct:>5}% {wins:>8} {losses:>7}")

    return groups


def numeric_correlation(results, feature_name, label):
    """Show correlation between a numeric feature and day P&L."""
    vals = []
    pnls = []
    for r in results:
        if r is None:
            continue
        v = r.get(feature_name)
        if v is not None and v > 0:
            vals.append(v)
            pnls.append(r["day_pnl"])

    if len(vals) < 10:
        return

    # Split into terciles
    arr = np.array(vals)
    t1 = np.percentile(arr, 33)
    t2 = np.percentile(arr, 67)

    low = [pnls[i] for i in range(len(vals)) if vals[i] <= t1]
    mid = [pnls[i] for i in range(len(vals)) if t1 < vals[i] <= t2]
    high = [pnls[i] for i in range(len(vals)) if vals[i] > t2]

    corr = round(np.corrcoef(vals, pnls)[0, 1], 3)
    print(f"\n  {label} (corr={corr}):")
    print(f"  {'Tercile':<15} {'Days':>5} {'Total':>8} {'Avg':>8}")
    print(f"  {'─'*15} {'─'*5} {'─'*8} {'─'*8}")
    for name, grp in [("Low", low), ("Mid", mid), ("High", high)]:
        if grp:
            t = round(sum(grp), 1)
            a = round(np.mean(grp), 1)
            ps = "+" if t >= 0 else ""
            pa = "+" if a >= 0 else ""
            print(f"  {name:<15} {len(grp):>5} {ps}{t:>7} {pa}{a:>7}")


def show_top_trades(results, n=15):
    """Show top N winners and losers."""
    completed = [r for r in results if r is not None and abs(r["day_pnl"]) > 0.1]
    completed.sort(key=lambda r: r["day_pnl"], reverse=True)

    print(f"\n  TOP {n} WINNERS:")
    print(f"  {'Date':<12} {'P&L':>8} {'Bias':<12} {'Range':<8} {'Risk':>5} {'ON_Rng':>7} {'Context':<15} {'DOW':<4} {'Dir':<6} {'MFE':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*5} {'─'*7} {'─'*15} {'─'*4} {'─'*6} {'─'*5}")
    for r in completed[:n]:
        ctx_str = []
        if r["context_overlap"]: ctx_str.append("OVR")
        if r["context_choppy"]: ctx_str.append("CHP")
        if r["context_directional"]: ctx_str.append("DIR")
        ctx = "+".join(ctx_str) if ctx_str else "MIXED"
        mfe = r.get("trade1_mfe", 0)
        ps = "+" if r["day_pnl"] >= 0 else ""
        print(f"  {r['date']:<12} {ps}{r['day_pnl']:>7} {r['bias']:<12} {r['range_flag']:<8} {r['risk_pts']:>5} {r['on_range']:>7} {ctx:<15} {r['dow_name']:<4} {r.get('first_dir',''):<6} {mfe:>5}")

    print(f"\n  TOP {n} LOSERS:")
    print(f"  {'Date':<12} {'P&L':>8} {'Bias':<12} {'Range':<8} {'Risk':>5} {'ON_Rng':>7} {'Context':<15} {'DOW':<4} {'Dir':<6} {'MFE':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*5} {'─'*7} {'─'*15} {'─'*4} {'─'*6} {'─'*5}")
    for r in completed[-n:]:
        ctx_str = []
        if r["context_overlap"]: ctx_str.append("OVR")
        if r["context_choppy"]: ctx_str.append("CHP")
        if r["context_directional"]: ctx_str.append("DIR")
        ctx = "+".join(ctx_str) if ctx_str else "MIXED"
        mfe = r.get("trade1_mfe", 0)
        ps = "+" if r["day_pnl"] >= 0 else ""
        print(f"  {r['date']:<12} {ps}{r['day_pnl']:>7} {r['bias']:<12} {r['range_flag']:<8} {r['risk_pts']:>5} {r['on_range']:>7} {ctx:<15} {r['dow_name']:<4} {r.get('first_dir',''):<6} {mfe:>5}")


def test_filter_rule(train_results, test_results, rule_fn, rule_name):
    """
    Test a filter rule: if rule_fn(features) returns False, skip that day.
    Compare filtered vs unfiltered on both train and test.
    """
    for label, results in [("TRAIN", train_results), ("TEST", test_results)]:
        all_pnl = [r["day_pnl"] for r in results if r is not None]
        filtered_pnl = [r["day_pnl"] for r in results if r is not None and rule_fn(r)]
        skipped_pnl = [r["day_pnl"] for r in results if r is not None and not rule_fn(r)]

        if not all_pnl or not filtered_pnl:
            continue

        all_total = round(sum(all_pnl), 1)
        filt_total = round(sum(filtered_pnl), 1)
        skip_total = round(sum(skipped_pnl), 1)

        all_wins = sum(1 for p in all_pnl if p > 5)
        filt_wins = sum(1 for p in filtered_pnl if p > 5)

        all_wr = round(all_wins / len(all_pnl) * 100, 1)
        filt_wr = round(filt_wins / len(filtered_pnl) * 100, 1) if filtered_pnl else 0

        # Max drawdown
        def max_dd(pnls):
            cum = np.cumsum(pnls)
            peak = 0
            dd = 0
            for p in cum:
                if p > peak: peak = p
                d = peak - p
                if d > dd: dd = d
            return round(dd, 1)

        ps_a = "+" if all_total >= 0 else ""
        ps_f = "+" if filt_total >= 0 else ""
        ps_s = "+" if skip_total >= 0 else ""

        print(f"    {label}: {len(all_pnl)} days → {len(filtered_pnl)} traded, {len(skipped_pnl)} skipped | "
              f"P&L: {ps_f}{filt_total} (was {ps_a}{all_total}) | "
              f"Skipped: {ps_s}{skip_total} | "
              f"WR: {filt_wr}% (was {all_wr}%) | "
              f"DD: {max_dd(filtered_pnl)} (was {max_dd(all_pnl)})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    trading_days = sorted(set(df.index.date))
    trading_days = [d for d in trading_days if d.weekday() < 5]

    # Train/test split
    split_idx = int(len(trading_days) * TRAIN_RATIO)
    train_days = set(trading_days[:split_idx])
    test_days = set(trading_days[split_idx:])
    print(f"Train: {len(train_days)} days, Test: {len(test_days)} days")
    print(f"Train: {min(train_days)} to {max(train_days)}")
    print(f"Test:  {min(test_days)} to {max(test_days)}")

    # Simulate all days
    print(f"\nSimulating {len(trading_days)} days...")
    all_results = []
    train_results = []
    test_results = []

    for i, day in enumerate(trading_days):
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue
        r = simulate_day_with_breakeven(day_df)
        all_results.append(r)
        if day in train_days:
            train_results.append(r)
        else:
            test_results.append(r)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(trading_days)}")

    valid = [r for r in all_results if r is not None]
    print(f"\nValid days: {len(valid)}")

    # ── Overall stats ──────────────────────────────────────────────────────
    total_pnl = sum(r["day_pnl"] for r in valid)
    winners = [r for r in valid if r["outcome"] == "WINNER"]
    losers = [r for r in valid if r["outcome"] == "LOSER"]
    flats = [r for r in valid if r["outcome"] == "FLAT"]
    print(f"\n{'═' * 80}")
    print(f"  OVERALL: {len(valid)} days | +{round(total_pnl, 1)} pts | "
          f"W:{len(winners)} L:{len(losers)} F:{len(flats)}")
    print(f"{'═' * 80}")

    # ── Top winners and losers ─────────────────────────────────────────────
    show_top_trades(valid, n=20)

    # ── Feature breakdowns ─────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  FEATURE ANALYSIS (all data)")
    print(f"{'═' * 80}")

    feature_breakdown(valid, "bias", "OVERNIGHT BIAS")
    feature_breakdown(valid, "range_flag", "BAR 4 RANGE")
    feature_breakdown(valid, "dow_name", "DAY OF WEEK")
    feature_breakdown(valid, "context_overlap", "OVERLAP")
    feature_breakdown(valid, "context_choppy", "CHOPPY")
    feature_breakdown(valid, "context_directional", "DIRECTIONAL")

    # Combined context
    for r in valid:
        ctx_parts = []
        if r["context_overlap"]: ctx_parts.append("OVR")
        if r["context_choppy"]: ctx_parts.append("CHP")
        if r["context_directional"]: ctx_parts.append("DIR")
        r["context_combo"] = "+".join(ctx_parts) if ctx_parts else "MIXED"
    feature_breakdown(valid, "context_combo", "CONTEXT COMBO")

    # Bar 4 bullish/bearish
    feature_breakdown(valid, "bar4_bullish", "BAR 4 BULLISH")

    # Bias + range combo
    for r in valid:
        r["bias_range"] = f"{r['bias']}_{r['range_flag']}"
    feature_breakdown(valid, "bias_range", "BIAS + RANGE COMBO")

    # Numeric features
    numeric_correlation(valid, "risk_pts", "RISK (buy-sell spread)")
    numeric_correlation(valid, "on_range", "OVERNIGHT RANGE")
    numeric_correlation(valid, "bar4_range", "BAR 4 RANGE (pts)")
    numeric_correlation(valid, "bars13_range", "BARS 1-3 TOTAL RANGE")
    numeric_correlation(valid, "bar4_body_pct", "BAR 4 BODY %")

    # ── Train-discovered patterns → test validation ────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  FILTER RULES — TRAIN vs TEST VALIDATION")
    print(f"{'═' * 80}")

    train_valid = [r for r in train_results if r is not None]
    test_valid = [r for r in test_results if r is not None]

    # Rule 1: Skip WIDE range days
    print(f"\n  Rule: Skip WIDE range days")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["range_flag"] != "WIDE", "Skip WIDE")

    # Rule 2: Skip NARROW range days
    print(f"\n  Rule: Skip NARROW range days")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["range_flag"] != "NARROW", "Skip NARROW")

    # Rule 3: Only trade STANDARD bias
    print(f"\n  Rule: Only trade STANDARD bias")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["bias"] == "STANDARD", "Only STANDARD")

    # Rule 4: Skip choppy context
    print(f"\n  Rule: Skip CHOPPY context")
    test_filter_rule(train_valid, test_valid,
                     lambda r: not r["context_choppy"], "Skip CHOPPY")

    # Rule 5: Skip directional context
    print(f"\n  Rule: Skip DIRECTIONAL context")
    test_filter_rule(train_valid, test_valid,
                     lambda r: not r["context_directional"], "Skip DIRECTIONAL")

    # Rule 6: Skip overlap context
    print(f"\n  Rule: Skip OVERLAP context")
    test_filter_rule(train_valid, test_valid,
                     lambda r: not r["context_overlap"], "Skip OVERLAP")

    # Rule 7: Only trade Mon-Wed (skip Thu-Fri)
    print(f"\n  Rule: Only trade Mon-Wed")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["dow"] <= 2, "Mon-Wed only")

    # Rule 8: Skip when risk > 50
    print(f"\n  Rule: Skip when risk > 50 pts")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["risk_pts"] <= 50, "Risk <= 50")

    # Rule 9: Skip when overnight range > 200
    print(f"\n  Rule: Skip when overnight range > 200 pts")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["on_range"] <= 200, "ON range <= 200")

    # Rule 10: Skip Fridays
    print(f"\n  Rule: Skip Fridays")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["dow"] != 4, "Skip Friday")

    # Rule 11: Skip WIDE + STANDARD combo (if it's bad)
    print(f"\n  Rule: Skip STANDARD+WIDE combo")
    test_filter_rule(train_valid, test_valid,
                     lambda r: not (r["bias"] == "STANDARD" and r["range_flag"] == "WIDE"),
                     "Skip STD+WIDE")

    # Rule 12: Only trade when bar 4 body > 50% of range
    print(f"\n  Rule: Only trade when bar 4 body > 50% of range")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["bar4_body_pct"] > 50, "Bar4 body > 50%")

    # Rule 13: Combo — skip WIDE + skip risk > 50
    print(f"\n  Rule: COMBO — skip WIDE and skip risk > 50")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["range_flag"] != "WIDE" and r["risk_pts"] <= 50,
                     "No WIDE + Risk<=50")

    # Rule 14: Only LONG_ONLY bias days
    print(f"\n  Rule: Only trade LONG_ONLY bias days")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["bias"] == "LONG_ONLY", "Only LONG_ONLY")

    # Rule 15: Size up on LONG_ONLY days (2x position)
    # Can't size up in this sim, but show the P&L if we only traded these
    print(f"\n  Rule: Only trade biased days (LONG_ONLY or SHORT_ONLY)")
    test_filter_rule(train_valid, test_valid,
                     lambda r: r["bias"] in ("LONG_ONLY", "SHORT_ONLY"),
                     "Only biased")

    print(f"\n{'═' * 80}")
    print(f"  DONE")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
