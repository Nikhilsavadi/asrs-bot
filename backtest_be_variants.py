"""
backtest_be_variants.py — Compare breakeven stop placement strategies.

Tests how different BE rules affect PF on the validated firstrate data.
Train/test split (2008-2019 / 2020-2026) to avoid curve-fitting.

Variants:
  baseline       : current behaviour — when unrealized >= breakeven_pts,
                   move stop to entry exactly. Tick wicks can fire it.
  be_buffer_5    : move stop to entry-5pt (LONG) / entry+5pt (SHORT).
                   Allows tick wicks of up to 5pt without exiting.
  be_buffer_10   : 10pt buffer
  be_close_only  : only update BE on completed-bar close, never intra-bar
                   (matches the 5-min backtest exit behaviour exactly)
  no_be          : disable breakeven entirely (control)
"""
import argparse
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import backtest as bt
from backtest_firstrate import FIRSTRATE_FILES, load_firstrate

# Match live config
for c in bt.INSTRUMENTS.values():
    c["max_entries"] = 3
    c["add_max"] = 0
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["max_risk_gbp"] = 50.0  # current live

BAR5_RULES = ["NORMAL", "WIDE"]


def simulate_session_be(day_bars, hours, minutes, open_h, open_m, eod_h, eod_m, cfg, be_rule):
    """
    Same as bt.simulate_session but with configurable BE rule.

    be_rule:
      "baseline"     — stop = entry on BE
      "be_buffer_5"  — stop = entry ∓ 5pt on BE
      "be_buffer_10" — stop = entry ∓ 10pt on BE
      "be_close_only" — BE check uses bar CLOSE not intrabar low/high
      "no_be"        — no BE move
    """
    n = len(day_bars)
    if n == 0:
        return []

    buffer = cfg["buffer"]
    open_mins = open_h * 60 + open_m
    eod_mins = eod_h * 60 + eod_m

    bar_mins = hours * 60 + minutes
    mins_from_open = bar_mins - open_mins
    bar_nums = mins_from_open // 5 + 1

    bar4_mask = (mins_from_open >= 0) & (bar_nums == 4)
    bar4_idx = np.where(bar4_mask)[0]
    if len(bar4_idx) == 0:
        return []
    b4i = bar4_idx[0]
    bar4_h, bar4_l = day_bars[b4i, 1], day_bars[b4i, 2]
    bar4_range = bar4_h - bar4_l

    if bar4_range < cfg["narrow_range"]: rf = "NARROW"
    elif bar4_range > cfg["wide_range"]: rf = "WIDE"
    else: rf = "NORMAL"

    sig_h, sig_l, bar_num = bar4_h, bar4_l, 4
    if rf in BAR5_RULES:
        bar5_mask = (mins_from_open >= 0) & (bar_nums == 5)
        bar5_idx = np.where(bar5_mask)[0]
        if len(bar5_idx) > 0:
            b5i = bar5_idx[0]
            sig_h, sig_l = day_bars[b5i, 1], day_bars[b5i, 2]
            bar_num = 5

    bar_range = sig_h - sig_l
    if bar_range < cfg["narrow_range"]: rf = "NARROW"
    elif bar_range > cfg["wide_range"]: rf = "WIDE"
    else: rf = "NORMAL"
    if bar_range > cfg["max_bar_range"]:
        return []

    bar_high = round(sig_h, 1)
    bar_low = round(sig_l, 1)
    buy_level = round(bar_high + buffer, 1)
    sell_level = round(bar_low - buffer, 1)

    risk_pts = bar_range + buffer * 2
    if risk_pts > cfg["max_risk_gbp"]:
        max_stop = cfg["max_risk_gbp"]
        bar_high = round(sell_level + max_stop, 1)
        bar_low = round(buy_level - max_stop, 1)
        bar_range = max_stop

    sig_bar_mask = (mins_from_open >= 0) & (bar_nums == bar_num)
    sig_idx = np.where(sig_bar_mask)[0]
    if len(sig_idx) == 0:
        return []
    start_after = sig_idx[0] + 1
    if start_after >= n:
        return []

    rem_open = day_bars[start_after:, 0]
    rem_high = day_bars[start_after:, 1]
    rem_low = day_bars[start_after:, 2]
    rem_close = day_bars[start_after:, 3]
    rem_mins = bar_mins[start_after:]
    rem_n = len(rem_open)

    first_long = -1
    first_short = -1
    for j in range(rem_n):
        if rem_mins[j] >= eod_mins:
            break
        if first_long < 0 and rem_high[j] >= buy_level:
            first_long = j
        if first_short < 0 and rem_low[j] <= sell_level:
            first_short = j
        if first_long >= 0 and first_short >= 0:
            break
    if first_long < 0 and first_short < 0:
        return []

    if first_long >= 0 and first_short >= 0:
        if first_long < first_short:
            direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
        elif first_short < first_long:
            direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
        else:
            if rem_open[first_long] >= buy_level:
                direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
            else:
                direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
    elif first_long >= 0:
        direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
    else:
        direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short

    trades = []
    entries_used = 0
    max_entries = cfg["max_entries"]
    active = True
    breakeven_hit = False
    mfe = 0.0
    waiting = False

    for j in range(start, rem_n):
        bm = rem_mins[j]
        if bm >= eod_mins:
            if active:
                ep = rem_open[j]
                pnl = (ep - entry) if direction == "LONG" else (entry - ep)
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(ep, 1), "pnl_pts": round(pnl, 1),
                               "reason": "EOD"})
            break

        bh = rem_high[j]
        bl = rem_low[j]
        bc = rem_close[j]
        bo = rem_open[j]

        if active:
            # Stop check
            hit = False
            if direction == "LONG" and bl <= stop:
                hit = True
            elif direction == "SHORT" and bh >= stop:
                hit = True
            if hit:
                pnl = (stop - entry) if direction == "LONG" else (entry - stop)
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(stop, 1), "pnl_pts": round(pnl, 1),
                               "reason": "STOP"})
                entries_used += 1
                active = False
                waiting = entries_used < max_entries
                continue

            # MFE
            if direction == "LONG":
                m = bh - entry
                if m > mfe: mfe = m
                unrealized_high = bh - entry
                unrealized_close = bc - entry
            else:
                m = entry - bl
                if m > mfe: mfe = m
                unrealized_high = entry - bl
                unrealized_close = entry - bc

            # Breakeven (varies by rule)
            if be_rule != "no_be" and not breakeven_hit:
                # Use intra-bar high (default) or close (be_close_only)
                trigger_value = unrealized_close if be_rule == "be_close_only" else unrealized_high
                if trigger_value >= cfg["breakeven_pts"]:
                    breakeven_hit = True
                    if be_rule == "baseline":
                        new_stop = entry
                    elif be_rule == "be_buffer_5":
                        new_stop = entry - 5 if direction == "LONG" else entry + 5
                    elif be_rule == "be_buffer_10":
                        new_stop = entry - 10 if direction == "LONG" else entry + 10
                    elif be_rule == "be_buffer_15":
                        new_stop = entry - 15 if direction == "LONG" else entry + 15
                    elif be_rule == "be_buffer_20":
                        new_stop = entry - 20 if direction == "LONG" else entry + 20
                    elif be_rule == "be_buffer_25":
                        new_stop = entry - 25 if direction == "LONG" else entry + 25
                    elif be_rule == "be_close_only":
                        new_stop = entry
                    else:
                        new_stop = entry
                    if direction == "LONG" and new_stop > stop:
                        stop = new_stop
                    elif direction == "SHORT" and new_stop < stop:
                        stop = new_stop

            # Candle trail (uses prev bar)
            if j > start:
                prev_h = rem_high[j - 1]
                prev_l = rem_low[j - 1]
                prev_c = rem_close[j - 1]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop:
                        stop = round(ns, 1)
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop:
                        stop = round(ns, 1)

        elif waiting:
            tl = bh >= buy_level
            ts = bl <= sell_level
            if tl and ts:
                if bo >= buy_level: ts = False
                elif bo <= sell_level: tl = False
                else:
                    tl = bc >= bo
                    ts = not tl
            if tl:
                direction, entry, stop = "LONG", buy_level, sell_level
                active, waiting = True, False
                breakeven_hit = False
                mfe = max(0, bh - entry)
            elif ts:
                direction, entry, stop = "SHORT", sell_level, buy_level
                active, waiting = True, False
                breakeven_hit = False
                mfe = max(0, entry - bl)

    return trades


def run_backtest(file, src_tz, target_tz, cfg, inst_name, be_rule, year_min, year_max):
    df = load_firstrate(file, src_tz, target_tz)
    df = df[(df.index.year >= year_min) & (df.index.year <= year_max)]
    ohlc = df[["Open", "High", "Low", "Close"]].values
    hours = df["_hour"].values
    minutes = df["_minute"].values
    dates = df["_date"].values
    unique_dates = sorted(set(dates))

    sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
    out = []
    for session in sessions:
        oh = cfg[f"s{session}_open_hour"]
        om = cfg[f"s{session}_open_minute"]
        eh = cfg["session_end_hour"]
        em = cfg["session_end_minute"]
        for day in unique_dates:
            mask = dates == day
            trs = simulate_session_be(
                ohlc[mask], hours[mask], minutes[mask],
                oh, om, eh, em, cfg, be_rule,
            )
            for t in trs:
                t["date"] = str(day)
                t["instrument"] = inst_name
                t["signal"] = f"{inst_name}_S{session}"
                out.append(t)
    return out


def stats(trades):
    if not trades:
        return {"trades": 0, "pf": 0, "net": 0, "wr": 0}
    pnl = np.array([t["pnl_pts"] for t in trades])
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    pf = w / l if l else float("inf")
    return {
        "trades": len(trades),
        "pf": pf,
        "net": pnl.sum(),
        "wr": (pnl > 0).mean() * 100,
        "avg_win": pnl[pnl > 0].mean() if (pnl > 0).any() else 0,
        "avg_loss": pnl[pnl < 0].mean() if (pnl < 0).any() else 0,
    }


def main():
    rules = [
        "baseline",
        "be_buffer_5",
        "be_buffer_10",
        "be_buffer_15",
        "be_buffer_20",
        "be_buffer_25",
        "be_close_only",
        "no_be",
    ]

    print("=" * 92)
    print("  BREAKEVEN VARIANT TEST  Train: 2008-2019  Test: 2020-2026")
    print("=" * 92)

    train_results = {}
    test_results = {}
    train_per_inst = {}
    test_per_inst = {}

    for rule in rules:
        train = []
        test = []
        per_inst_train = {}
        per_inst_test = {}
        for inst, fr in FIRSTRATE_FILES.items():
            cfg = bt.INSTRUMENTS[inst]
            inst_train = run_backtest(fr["file"], fr["src_tz"], cfg["timezone"], cfg, inst, rule, 2008, 2019)
            inst_test  = run_backtest(fr["file"], fr["src_tz"], cfg["timezone"], cfg, inst, rule, 2020, 2026)
            per_inst_train[inst] = stats(inst_train)
            per_inst_test[inst]  = stats(inst_test)
            train += inst_train
            test  += inst_test
        train_results[rule] = stats(train)
        test_results[rule]  = stats(test)
        train_per_inst[rule] = per_inst_train
        test_per_inst[rule]  = per_inst_test
        print(f"\n  {rule:<18}")
        print(f"    TRAIN: {train_results[rule]['trades']:>6} trades  PF {train_results[rule]['pf']:.2f}  "
              f"WR {train_results[rule]['wr']:.0f}%  Net {train_results[rule]['net']:+,.0f}  "
              f"avg+{train_results[rule]['avg_win']:.1f}/{train_results[rule]['avg_loss']:.1f}")
        print(f"    TEST:  {test_results[rule]['trades']:>6} trades  PF {test_results[rule]['pf']:.2f}  "
              f"WR {test_results[rule]['wr']:.0f}%  Net {test_results[rule]['net']:+,.0f}  "
              f"avg+{test_results[rule]['avg_win']:.1f}/{test_results[rule]['avg_loss']:.1f}")

    print(f"\n\n{'='*92}")
    print(f"  SUMMARY (TEST set, out-of-sample)")
    print(f"{'='*92}")
    print(f"  {'Rule':<18}{'PF':>7}{'WR':>7}{'Net':>15}{'Δ vs baseline':>17}")
    base_pf = test_results["baseline"]["pf"]
    base_net = test_results["baseline"]["net"]
    for rule in rules:
        r = test_results[rule]
        delta_pf = r["pf"] - base_pf
        delta_net = r["net"] - base_net
        print(f"  {rule:<18}{r['pf']:>7.2f}{r['wr']:>6.0f}%{r['net']:>+15,.0f}"
              f"{delta_pf:>+9.2f}PF /{delta_net:>+10,.0f}")

    # Per-instrument breakdown on TEST
    print(f"\n\n{'='*92}")
    print(f"  PER-INSTRUMENT (TEST set)")
    print(f"{'='*92}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        print(f"\n  {inst}")
        print(f"    {'rule':<18}{'PF':>7}{'Net':>14}{'AvgWin':>10}{'AvgLoss':>10}")
        base_pf = test_per_inst["baseline"][inst]["pf"]
        for rule in rules:
            r = test_per_inst[rule][inst]
            mark = " ⭐" if r["pf"] == max(test_per_inst[rr][inst]["pf"] for rr in rules) else ""
            print(f"    {rule:<18}{r['pf']:>7.2f}{r['net']:>+14,.0f}{r['avg_win']:>10.1f}{r['avg_loss']:>10.1f}{mark}")
        best_inst = max(rules, key=lambda r: test_per_inst[r][inst]["pf"])
        best_inst_pf = test_per_inst[best_inst][inst]["pf"]
        delta = (best_inst_pf - base_pf) / base_pf * 100 if base_pf else 0
        print(f"    BEST: {best_inst} (PF {best_inst_pf:.2f}, {delta:+.1f}% vs baseline)")

    # Pick winner overall
    best_test = max(rules, key=lambda r: test_results[r]["pf"])
    best_train = max(rules, key=lambda r: train_results[r]["pf"])
    print(f"\nOverall best on TEST: {best_test} (PF {test_results[best_test]['pf']:.2f})")
    print(f"Overall best on TRAIN: {best_train} (PF {train_results[best_train]['pf']:.2f})")


if __name__ == "__main__":
    main()
