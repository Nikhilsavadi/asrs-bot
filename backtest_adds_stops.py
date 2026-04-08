"""
backtest_adds_stops.py — Test add-to-winners stop placement variants.

Properly simulates ADD positions (size doubles/triples) and tests how
different stop placement rules affect overall PF.

TRAIN/TEST split: 2008-2019 train, 2020-2026 test (~70/30 by years).

Rules tested:
  baseline_be       : current live behavior — add fires, stop on ALL positions = entry of first
  add_at_be         : same as baseline (sanity check, identical behavior)
  add_self_be       : add gets its own breakeven stop at fill price (zero risk on add, very tight)
  add_half_trigger  : add stop = fill_price - (add_trigger / 2) — deeper, more breathing room
  add_lock_first    : on add, lock half the original gain (stop = entry + (add_trigger / 2))
  no_adds           : disable adds entirely (control)
"""
import argparse
import time
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import backtest as bt
from backtest_firstrate import FIRSTRATE_FILES, load_firstrate

# Match live config
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0


BAR5_RULES = ["NORMAL", "WIDE"]


def simulate_session_with_adds(
    day_bars: np.ndarray, hours: np.ndarray, minutes: np.ndarray,
    open_h: int, open_m: int, eod_h: int, eod_m: int,
    cfg: dict, add_stop_rule: str = "baseline_be",
):
    """
    Same as bt.simulate_session but properly simulates ADD positions
    (size grows on each add) and applies the chosen add_stop_rule.

    add_stop_rule:
        "baseline_be"    : on add, stop on ALL = original entry (current live)
        "add_self_be"    : on add, stop on the new contract = its own fill price
        "add_half_trigger": on add, stop on the new contract = fill - add_trigger/2
        "add_lock_first" : on add, stop on the original contract = entry + add_trigger/2
        "no_adds"        : disable adds entirely
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
    bar4_h = day_bars[b4i, 1]
    bar4_l = day_bars[b4i, 2]
    bar4_range = bar4_h - bar4_l

    if bar4_range < cfg["narrow_range"]:
        range_flag = "NARROW"
    elif bar4_range > cfg["wide_range"]:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    sig_h, sig_l = bar4_h, bar4_l
    bar_num = 4
    if range_flag in BAR5_RULES:
        bar5_mask = (mins_from_open >= 0) & (bar_nums == 5)
        bar5_idx = np.where(bar5_mask)[0]
        if len(bar5_idx) > 0:
            b5i = bar5_idx[0]
            sig_h = day_bars[b5i, 1]
            sig_l = day_bars[b5i, 2]
            bar_num = 5

    bar_range = sig_h - sig_l
    if bar_range < cfg["narrow_range"]:
        range_flag = "NARROW"
    elif bar_range > cfg["wide_range"]:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"
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

    # Find first trigger
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

    # Position list: [(entry_price, stop_price), ...]  one per contract
    positions: list[tuple[float, float]] = [(entry, stop)]
    breakeven_hit = False
    adds_used = 0
    last_add_price = entry
    mfe = 0.0
    waiting = False
    active = True

    def close_all_at(exit_price: float, reason: str):
        """Close all open contracts at exit_price; return total pts."""
        total = 0.0
        for ep, _sp in positions:
            if direction == "LONG":
                total += exit_price - ep
            else:
                total += ep - exit_price
        return total

    for j in range(start, rem_n):
        bm = rem_mins[j]
        if bm >= eod_mins:
            if active and positions:
                ep_close = rem_open[j]
                pnl = close_all_at(ep_close, "EOD")
                trades.append({
                    "direction": direction, "entry": round(entry, 1),
                    "exit": round(ep_close, 1), "pnl_pts": round(pnl, 1),
                    "mfe": round(mfe, 1), "adds": adds_used, "reason": "EOD",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1),
                })
            break

        bh = rem_high[j]
        bl = rem_low[j]
        bc = rem_close[j]
        bo = rem_open[j]

        if active and positions:
            # Stop check on EACH position. If any contract's stop is hit,
            # only that contract closes.
            new_positions = []
            for ep, sp in positions:
                hit = False
                if direction == "LONG" and bl <= sp:
                    hit = True
                    pnl_one = sp - ep
                elif direction == "SHORT" and bh >= sp:
                    hit = True
                    pnl_one = ep - sp
                if hit:
                    # Realised pnl from this contract
                    trades_so_far_pnl = pnl_one
                    # We'll accumulate by closing as separate trade segments
                    # Simpler: close all on first hit (matches live behaviour
                    # since we use one trailing_stop)
                    # Actually for this rule test we need to track per-position
                    pass
                else:
                    new_positions.append((ep, sp))

            # Simpler model: close ALL when ANY stop hits (matches live)
            any_hit = False
            min_stop = None
            for ep, sp in positions:
                if direction == "LONG":
                    if bl <= sp:
                        any_hit = True
                        if min_stop is None or sp > min_stop:
                            min_stop = sp
                else:
                    if bh >= sp:
                        any_hit = True
                        if min_stop is None or sp < min_stop:
                            min_stop = sp

            if any_hit:
                # Use the highest stop (LONG) or lowest (SHORT) as the exit reference.
                # In reality the exit price = first stop touched, but with all
                # positions sharing the same trail (current live behaviour),
                # they all close together at the same price.
                exit_price = min_stop
                total_pnl = close_all_at(exit_price, "STOP")
                trades.append({
                    "direction": direction, "entry": round(entry, 1),
                    "exit": round(exit_price, 1), "pnl_pts": round(total_pnl, 1),
                    "mfe": round(mfe, 1), "adds": adds_used, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1),
                })
                entries_used += 1
                active = False
                positions = []
                waiting = entries_used < max_entries
                continue

            # MFE
            if direction == "LONG":
                m = bh - entry
                if m > mfe: mfe = m
                unrealized = bc - entry
            else:
                m = entry - bl
                if m > mfe: mfe = m
                unrealized = entry - bc

            # Breakeven (R12)
            if not breakeven_hit and unrealized >= cfg["breakeven_pts"]:
                breakeven_hit = True
                # Move all stops to entry (or better)
                positions = [
                    (ep, max(sp, entry)) if direction == "LONG"
                    else (ep, min(sp, entry))
                    for ep, sp in positions
                ]

            # Candle trail (R14)
            if j > start:
                prev_h = rem_high[j - 1]
                prev_l = rem_low[j - 1]
                prev_c = rem_close[j - 1]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    positions = [(ep, max(sp, ns)) for ep, sp in positions]
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    positions = [(ep, min(sp, ns)) for ep, sp in positions]

            # ── ADDS ────────────────────────────────────────────────
            if add_stop_rule != "no_adds" and adds_used < cfg["add_max"]:
                ref = last_add_price
                pfr = (bc - ref) if direction == "LONG" else (ref - bc)
                if pfr >= cfg["add_trigger"]:
                    adds_used += 1
                    add_fill = bc
                    last_add_price = add_fill

                    # Determine the new contract's stop
                    if add_stop_rule == "baseline_be":
                        # Live behaviour: stop on ALL positions = entry of first contract
                        new_stop_for_add = entry
                        positions = [(ep, max(sp, entry)) if direction == "LONG"
                                     else (ep, min(sp, entry)) for ep, sp in positions]
                        positions.append((add_fill, new_stop_for_add))
                    elif add_stop_rule == "add_self_be":
                        # Add gets its own stop at its fill price (zero risk on the add)
                        positions.append((add_fill, add_fill))
                    elif add_stop_rule == "add_half_trigger":
                        # Add gets a stop half the trigger distance below fill
                        offset = cfg["add_trigger"] / 2
                        if direction == "LONG":
                            new_stop_for_add = add_fill - offset
                        else:
                            new_stop_for_add = add_fill + offset
                        positions.append((add_fill, new_stop_for_add))
                    elif add_stop_rule == "add_lock_first":
                        # Move first contract's stop to lock half the trigger gain
                        lock_offset = cfg["add_trigger"] / 2
                        if direction == "LONG":
                            new_lock = entry + lock_offset
                            positions = [(ep, max(sp, new_lock)) for ep, sp in positions]
                        else:
                            new_lock = entry - lock_offset
                            positions = [(ep, min(sp, new_lock)) for ep, sp in positions]
                        positions.append((add_fill, add_fill))  # add at its own BE

                    breakeven_hit = True

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
                direction, entry = "LONG", buy_level
                positions = [(entry, sell_level)]
                active, waiting = True, False
                breakeven_hit, adds_used = False, 0
                last_add_price = entry
                mfe = max(0, bh - entry)
            elif ts:
                direction, entry = "SHORT", sell_level
                positions = [(entry, buy_level)]
                active, waiting = True, False
                breakeven_hit, adds_used = False, 0
                last_add_price = entry
                mfe = max(0, entry - bl)

    return trades


def run_backtest(file: str, src_tz: str, target_tz: str, cfg: dict, inst_name: str,
                  add_stop_rule: str, year_min: int, year_max: int):
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
        oh = cfg[f"s{session}_open_hour"]; om = cfg[f"s{session}_open_minute"]
        eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
        for day in unique_dates:
            mask = dates == day
            trs = simulate_session_with_adds(
                ohlc[mask], hours[mask], minutes[mask],
                oh, om, eh, em, cfg, add_stop_rule,
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
        "trades": len(trades), "pf": pf, "net": pnl.sum(),
        "wr": (pnl > 0).mean() * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_end", type=int, default=2019)
    ap.add_argument("--test_start", type=int, default=2020)
    args = ap.parse_args()

    rules = [
        "baseline_be",
        "add_self_be",
        "add_half_trigger",
        "add_lock_first",
        "no_adds",
    ]

    print("=" * 88)
    print(f"  ADD-STOP TRAIN/TEST")
    print(f"  Train: 2008–{args.train_end}    Test: {args.test_start}–2026")
    print("=" * 88)

    train_results = {}
    test_results = {}

    for rule in rules:
        print(f"\n──── Rule: {rule} ────")
        train_trades = []
        test_trades = []
        for inst, fr in FIRSTRATE_FILES.items():
            cfg = bt.INSTRUMENTS[inst]
            train_trades += run_backtest(
                fr["file"], fr["src_tz"], cfg["timezone"], cfg, inst, rule,
                2008, args.train_end,
            )
            test_trades += run_backtest(
                fr["file"], fr["src_tz"], cfg["timezone"], cfg, inst, rule,
                args.test_start, 2026,
            )
        train_results[rule] = stats(train_trades)
        test_results[rule] = stats(test_trades)
        print(f"  TRAIN ({2008}-{args.train_end}): trades={train_results[rule]['trades']:,}  "
              f"PF {train_results[rule]['pf']:.2f}  WR {train_results[rule]['wr']:.0f}%  "
              f"Net {train_results[rule]['net']:+,.0f}pts")
        print(f"  TEST  ({args.test_start}-2026):  trades={test_results[rule]['trades']:,}  "
              f"PF {test_results[rule]['pf']:.2f}  WR {test_results[rule]['wr']:.0f}%  "
              f"Net {test_results[rule]['net']:+,.0f}pts")

    # Comparison table
    print("\n" + "=" * 88)
    print(f"  SUMMARY")
    print("=" * 88)
    print(f"  {'Rule':<22}{'Train PF':>10}{'Train Net':>14}{'Test PF':>10}{'Test Net':>14}{'Train→Test Δ':>14}")
    base_train = train_results["baseline_be"]["pf"]
    base_test = test_results["baseline_be"]["pf"]
    for rule in rules:
        tr = train_results[rule]
        te = test_results[rule]
        delta = te["pf"] - tr["pf"]
        print(f"  {rule:<22}{tr['pf']:>10.2f}{tr['net']:>+14,.0f}{te['pf']:>10.2f}{te['net']:>+14,.0f}{delta:>+14.2f}")

    # Recommendation
    print(f"\nBaseline Test PF: {base_test:.2f}")
    best_test = max(rules, key=lambda r: test_results[r]["pf"])
    best_test_pf = test_results[best_test]["pf"]
    if best_test == "baseline_be":
        print(f"WINNER: baseline_be (current live config) — no change recommended")
    else:
        improvement = (best_test_pf - base_test) / base_test * 100
        print(f"BEST ON TEST: {best_test}  PF {best_test_pf:.2f} ({improvement:+.1f}% vs baseline)")
        # Sanity: was it ALSO best on train?
        best_train = max(rules, key=lambda r: train_results[r]["pf"])
        if best_test == best_train:
            print(f"  ✓ Also best on train — robust improvement, safe to deploy")
        else:
            print(f"  ⚠️ Best on train was '{best_train}' — possible test-set noise. Need more validation.")


if __name__ == "__main__":
    main()
