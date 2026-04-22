"""
backtest_v2_1min_trail.py — v2 variant with 1-MIN bar trail (instead of 5-min).

Trail tightens on each 1-min bar close using prev 1-min bar's low (LONG)
or high (SHORT). Otherwise identical to v2 candle trail.

Hypothesis: tighter granularity locks in profit faster on reversals.
Risk: may whipsaw out of winners on 1-min noise.
"""
import numpy as np
from backtest_v2 import _find_real_reentry_in_1min


def simulate_session_1min_trail(
    day_5min: np.ndarray, hours_5m, minutes_5m,
    open_h, open_m, eod_h, eod_m, cfg,
    day_1min=None, one_min_hours=None, one_min_minutes=None,
):
    if len(day_5min) == 0 or day_1min is None: return []

    highs_all = day_5min[:, 1]
    lows_all = day_5min[:, 2]
    closes_all = day_5min[:, 3]
    opens_all = day_5min[:, 0]

    open_mins = open_h * 60 + open_m
    eod_mins = eod_h * 60 + eod_m
    minute_of_day_5m = hours_5m * 60 + minutes_5m
    minute_of_day_1m = one_min_minutes

    session_mask = (minute_of_day_5m >= open_mins) & (minute_of_day_5m < eod_mins)
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
        if minute_of_day_5m[j] >= eod_mins: break
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

    # Find 1-min start index — the 1-min bar at or after the entry's 5-min bar start
    entry_5m_min = minute_of_day_5m[start]
    j1_start = 0
    while j1_start < len(minute_of_day_1m) and minute_of_day_1m[j1_start] < entry_5m_min:
        j1_start += 1

    trades = []
    entries_used = 0
    max_entries = cfg["max_entries"]
    active = True
    breakeven_hit = False
    mfe = 0.0
    waiting = False
    last_stop_min = 0

    j1 = j1_start
    while j1 < len(minute_of_day_1m):
        bm = minute_of_day_1m[j1]
        if bm >= eod_mins:
            if active:
                ep = day_1min[j1, 0]
                pnl = (ep - entry) if direction == "LONG" else (entry - ep)
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(ep, 1), "pnl_pts": round(pnl, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "EOD",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1)})
            break

        bh = day_1min[j1, 1]; bl = day_1min[j1, 2]; bc = day_1min[j1, 3]

        if active:
            # 1-min stop check
            if direction == "LONG" and bl <= stop:
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(stop - entry, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1)})
                entries_used += 1; active = False
                waiting = entries_used < max_entries
                last_stop_min = bm + 1
                j1 += 1; continue
            if direction == "SHORT" and bh >= stop:
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(entry - stop, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1)})
                entries_used += 1; active = False
                waiting = entries_used < max_entries
                last_stop_min = bm + 1
                j1 += 1; continue

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

            # 1-MIN candle trail (every 1-min bar close)
            if j1 > j1_start:
                prev_h = day_1min[j1 - 1, 1]
                prev_l = day_1min[j1 - 1, 2]
                prev_c = day_1min[j1 - 1, 3]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop: stop = round(ns, 1)
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop: stop = round(ns, 1)

        elif waiting:
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
                breakeven_hit = False; mfe = 0.0
                active = True
                while j1 < len(minute_of_day_1m) and minute_of_day_1m[j1] < re_min:
                    j1 += 1
                continue
        j1 += 1

    return trades
