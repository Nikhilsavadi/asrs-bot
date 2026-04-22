"""
backtest_v2_adds.py — v2 variant with PROPER adds PnL computation.

The stock v2 tracks adds_used count but never accumulates add_pnl correctly
(add_pnl starts at 0 and is never updated). This fixes that by tracking
individual add entry prices and computing their PnL against the final
exit price when the position closes.

Rule: when profit from last add >= add_trigger, open +1 unit at current
close. On final exit (stop/EOD), each add's pnl = (exit_price - add_entry)
for LONG, or (add_entry - exit_price) for SHORT.
"""
import numpy as np
from backtest_v2 import _find_real_reentry_in_1min


def simulate_session_adds(
    day_5min: np.ndarray, hours_5m, minutes_5m,
    open_h, open_m, eod_h, eod_m, cfg,
    day_1min=None, one_min_hours=None, one_min_minutes=None,
):
    if len(day_5min) == 0: return []

    highs_all = day_5min[:, 1]
    lows_all = day_5min[:, 2]
    closes_all = day_5min[:, 3]
    opens_all = day_5min[:, 0]

    open_mins = open_h * 60 + open_m
    eod_mins = eod_h * 60 + eod_m
    minute_of_day = hours_5m * 60 + minutes_5m

    session_mask = (minute_of_day >= open_mins) & (minute_of_day < eod_mins)
    if not session_mask.any(): return []
    idxs = np.where(session_mask)[0]
    if len(idxs) < 5: return []

    bar4_idx = idxs[3]
    bar4_h = day_5min[bar4_idx, 1]; bar4_l = day_5min[bar4_idx, 2]
    bar_range = bar4_h - bar4_l
    range_flag = "NARROW"; bar_num = 4
    if bar_range < cfg["narrow_range"]:
        sig_h, sig_l = bar4_h, bar4_l
    elif bar_range > cfg["wide_range"]:
        if len(idxs) < 5: return []
        bar5_idx = idxs[4]
        sig_h, sig_l = day_5min[bar5_idx, 1], day_5min[bar5_idx, 2]
        bar_range = sig_h - sig_l
        bar_num = 5
        if bar_range > cfg["max_bar_range"] or bar_range <= 0: return []
        range_flag = "WIDE" if bar_range > cfg["wide_range"] else "NORMAL"
    else:
        sig_h, sig_l = bar4_h, bar4_l; range_flag = "NORMAL"

    buffer_ = cfg["buffer"]
    buy_level = round(sig_h + buffer_, 1)
    sell_level = round(sig_l - buffer_, 1)
    first_scan = bar4_idx + 1 if bar_num == 4 else bar4_idx + 2
    sess_end_idx = idxs[-1]

    first_long = first_short = -1
    for j in range(first_scan, sess_end_idx + 1):
        if minute_of_day[j] >= eod_mins: break
        if day_5min[j, 1] >= buy_level and first_long == -1: first_long = j
        if day_5min[j, 2] <= sell_level and first_short == -1: first_short = j
        if first_long != -1 and first_short != -1: break
    if first_long == -1 and first_short == -1: return []

    if first_long != -1 and first_short != -1:
        if first_long < first_short:
            direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
        elif first_short < first_long:
            direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
        else:
            if day_5min[first_long, 0] >= buy_level:
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
    adds_used = 0
    add_prices = []  # list of add entry prices
    last_add_price = entry
    mfe = 0.0
    waiting = False
    last_stop_min = 0

    def close_position(exit_price, reason, j=None):
        nonlocal active, entries_used, waiting, adds_used, add_prices, last_stop_min, breakeven_hit
        # Initial entry PnL
        base_pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
        # Adds PnL
        add_pnl = 0.0
        for ap in add_prices:
            add_pnl += (exit_price - ap) if direction == "LONG" else (ap - exit_price)
        total = base_pnl + add_pnl
        trades.append({
            "direction": direction, "entry": round(entry, 1),
            "exit": round(exit_price, 1), "pnl_pts": round(total, 1),
            "mfe": round(mfe, 1), "adds": adds_used, "reason": reason,
            "bar_num": bar_num, "range_flag": range_flag,
            "bar_range": round(bar_range, 1),
        })
        entries_used += 1
        active = False
        waiting = entries_used < max_entries
        adds_used = 0; add_prices = []
        breakeven_hit = False
        if j is not None:
            last_stop_min = minute_of_day[j] + 5

    for j in range(start, sess_end_idx + 1):
        bm = minute_of_day[j]
        if bm >= eod_mins:
            if active:
                ep = opens_all[j]
                close_position(ep, "EOD")
            break

        bh = highs_all[j]; bl = lows_all[j]; bc = closes_all[j]

        if active:
            if direction == "LONG" and bl <= stop:
                close_position(stop, "STOP", j=j); continue
            if direction == "SHORT" and bh >= stop:
                close_position(stop, "STOP", j=j); continue

            # MFE
            if direction == "LONG":
                m = bh - entry
                if m > mfe: mfe = m
                unrealized = bc - entry
            else:
                m = entry - bl
                if m > mfe: mfe = m
                unrealized = entry - bc

            # BE
            if not breakeven_hit and unrealized >= cfg["breakeven_pts"]:
                breakeven_hit = True
                if direction == "LONG" and stop < entry: stop = entry
                elif direction == "SHORT" and stop > entry: stop = entry

            # Candle trail
            if j > start:
                prev_h = highs_all[j - 1]; prev_l = lows_all[j - 1]; prev_c = closes_all[j - 1]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop: stop = round(ns, 1)
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop: stop = round(ns, 1)

            # Adds — with proper PnL tracking
            if adds_used < cfg["add_max"]:
                ref = last_add_price
                pfr = (bc - ref) if direction == "LONG" else (ref - bc)
                if pfr >= cfg["add_trigger"]:
                    adds_used += 1
                    add_prices.append(bc)
                    last_add_price = bc
                    breakeven_hit = True
                    # Adds also tighten stop to breakeven of ORIGINAL entry
                    if direction == "LONG" and stop < entry: stop = entry
                    elif direction == "SHORT" and stop > entry: stop = entry

        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(
                    day_1min, one_min_minutes, last_stop_min, eod_mins,
                    buy_level, sell_level,
                )
                if res is None:
                    waiting = False
                else:
                    re_min, re_dir, re_fill = res
                    direction = re_dir; entry = re_fill
                    stop = sell_level if re_dir == "LONG" else buy_level
                    last_add_price = entry
                    add_prices = []; adds_used = 0
                    breakeven_hit = False; mfe = 0.0
                    active = True
                    while j + 1 < len(minute_of_day) and minute_of_day[j + 1] < re_min:
                        j += 1
            else:
                waiting = False

    return trades
