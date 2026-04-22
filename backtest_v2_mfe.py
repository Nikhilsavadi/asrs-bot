"""
backtest_v2_mfe.py — MFE-fraction exit variants.

Exit trigger: once MFE exceeds activation threshold, the trailing stop
sits at (peak_price - retrace_frac × mfe).

Two modes:
  "pure":    replaces candle trail entirely
  "hybrid":  ratchet up — stop = max(candle_trail_stop, mfe_trail_stop)

Activation keeps us from setting a too-tight stop when trade barely moves.
Default: activate once mfe >= 1.0 × initial_stop_distance.
"""
import numpy as np
from backtest_v2 import _find_real_reentry_in_1min


def simulate_session_mfe(
    day_5min: np.ndarray, hours_5m, minutes_5m,
    open_h, open_m, eod_h, eod_m, cfg,
    day_1min=None, one_min_hours=None, one_min_minutes=None,
):
    if len(day_5min) == 0: return []

    retrace_frac = cfg.get("mfe_retrace", 0.5)
    activate_pts_mult = cfg.get("mfe_activate_mult", 1.0)  # × initial stop distance
    mode = cfg.get("mfe_mode", "hybrid")

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
    mfe = 0.0
    waiting = False
    last_stop_min = 0
    initial_stop_dist = abs(entry - stop)
    activate_pts = activate_pts_mult * initial_stop_dist
    peak_price = entry

    for j in range(start, sess_end_idx + 1):
        bm = minute_of_day[j]
        if bm >= eod_mins:
            if active:
                ep = opens_all[j]
                pnl = (ep - entry) if direction == "LONG" else (entry - ep)
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(ep, 1), "pnl_pts": round(pnl, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "EOD",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1)})
            break

        bh = highs_all[j]; bl = lows_all[j]; bc = closes_all[j]

        if active:
            if direction == "LONG" and bl <= stop:
                pnl = stop - entry
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(pnl, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1)})
                entries_used += 1; active = False
                waiting = entries_used < max_entries
                last_stop_min = bm + 5
                continue
            if direction == "SHORT" and bh >= stop:
                pnl = entry - stop
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(pnl, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1)})
                entries_used += 1; active = False
                waiting = entries_used < max_entries
                last_stop_min = bm + 5
                continue

            # MFE + peak tracking
            if direction == "LONG":
                if bh > peak_price: peak_price = bh
                m = bh - entry
                if m > mfe: mfe = m
                unrealized = bc - entry
            else:
                if bl < peak_price: peak_price = bl  # LOW for short
                m = entry - bl
                if m > mfe: mfe = m
                unrealized = entry - bc

            # BE
            if not breakeven_hit and unrealized >= cfg["breakeven_pts"]:
                breakeven_hit = True
                if direction == "LONG" and stop < entry: stop = entry
                elif direction == "SHORT" and stop > entry: stop = entry

            # MFE trail — activate once MFE ≥ activate_pts
            if mfe >= activate_pts:
                if direction == "LONG":
                    mfe_stop = peak_price - retrace_frac * mfe
                    if mfe_stop > stop: stop = round(mfe_stop, 1)
                else:
                    mfe_stop = peak_price + retrace_frac * mfe
                    if mfe_stop < stop: stop = round(mfe_stop, 1)

            # Candle trail (hybrid mode only)
            if mode == "hybrid" and j > start:
                prev_h = highs_all[j - 1]; prev_l = lows_all[j - 1]; prev_c = closes_all[j - 1]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop: stop = round(ns, 1)
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop: stop = round(ns, 1)

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
                    peak_price = entry
                    breakeven_hit = False; mfe = 0.0
                    initial_stop_dist = abs(entry - stop)
                    activate_pts = activate_pts_mult * initial_stop_dist
                    active = True
                    while j + 1 < len(minute_of_day) and minute_of_day[j + 1] < re_min:
                        j += 1
            else:
                waiting = False

    return trades
