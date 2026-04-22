"""
backtest_close_based_variants.py — test two ideas:

  A. Always trail to prev bar's CLOSE (instead of prev bar low for LONG)
  B. Exit only when bar CLOSES below stop (ignore intra-bar wicks)

Both applied via modified v2 simulator.
"""
import time, numpy as np, pandas as pd
from copy import deepcopy
from zoneinfo import ZoneInfo

import backtest as bt
from backtest_v2 import _find_real_reentry_in_1min

DATA_DIR = "data/firstrate"
FR = {
    "DAX": {"5m": f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt",
            "1m": f"{DATA_DIR}/FDAX_full_1min_continuous_ratio_adjusted.txt", "tz": "Europe/Berlin"},
    "US30": {"5m": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",
             "1m": f"{DATA_DIR}/YM_full_1min_continuous_ratio_adjusted.txt", "tz": "America/New_York"},
    "NIKKEI": {"5m": f"{DATA_DIR}/NKD_full_5min_continuous_ratio_adjusted.txt",
               "1m": f"{DATA_DIR}/NKD_full_1min_continuous_ratio_adjusted.txt", "tz": "America/New_York"},
}
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def simulate_v2_custom(day_5min, hours_5m, minutes_5m, open_h, open_m, eod_h, eod_m, cfg,
                       day_1min=None, one_min_hours=None, one_min_minutes=None,
                       trail_to_close: bool = False, close_based_exit: bool = False):
    """v2 engine with optional close-based trail / close-based exit."""
    if len(day_5min) == 0: return []
    highs = day_5min[:, 1]; lows = day_5min[:, 2]; closes = day_5min[:, 3]; opens = day_5min[:, 0]
    open_mins = open_h*60+open_m; eod_mins = eod_h*60+eod_m
    mod = hours_5m*60 + minutes_5m
    sess_mask = (mod >= open_mins) & (mod < eod_mins)
    if not sess_mask.any(): return []
    idxs = np.where(sess_mask)[0]
    if len(idxs) < 5: return []
    bar4_idx = idxs[3]; bar4_h = highs[bar4_idx]; bar4_l = lows[bar4_idx]
    bar_range = bar4_h - bar4_l; range_flag = "NARROW"; bar_num = 4
    if bar_range < cfg["narrow_range"]:
        sig_h, sig_l = bar4_h, bar4_l
    elif bar_range > cfg["wide_range"]:
        if len(idxs) < 5: return []
        bar5_idx = idxs[4]; sig_h = highs[bar5_idx]; sig_l = lows[bar5_idx]
        bar_range = sig_h - sig_l; bar_num = 5
        if bar_range > cfg["max_bar_range"] or bar_range <= 0: return []
        range_flag = "WIDE" if bar_range > cfg["wide_range"] else "NORMAL"
    else:
        sig_h, sig_l = bar4_h, bar4_l; range_flag = "NORMAL"

    buffer_ = cfg["buffer"]
    buy_level = round(sig_h+buffer_, 1); sell_level = round(sig_l-buffer_, 1)
    first_scan = bar4_idx+1 if bar_num == 4 else bar4_idx+2
    sess_end = idxs[-1]
    first_long = first_short = -1
    for j in range(first_scan, sess_end+1):
        if mod[j] >= eod_mins: break
        if highs[j] >= buy_level and first_long == -1: first_long = j
        if lows[j] <= sell_level and first_short == -1: first_short = j
        if first_long != -1 and first_short != -1: break
    if first_long == -1 and first_short == -1: return []
    if first_long != -1 and first_short != -1:
        if first_long < first_short:
            direction,entry,stop,start = "LONG",buy_level,sell_level,first_long
        elif first_short < first_long:
            direction,entry,stop,start = "SHORT",sell_level,buy_level,first_short
        else:
            if opens[first_long] >= buy_level:
                direction,entry,stop,start = "LONG",buy_level,sell_level,first_long
            else:
                direction,entry,stop,start = "SHORT",sell_level,buy_level,first_short
    elif first_long >= 0:
        direction,entry,stop,start = "LONG",buy_level,sell_level,first_long
    else:
        direction,entry,stop,start = "SHORT",sell_level,buy_level,first_short

    trades = []; entries_used = 0; max_entries = cfg["max_entries"]
    active = True; breakeven_hit = False; mfe = 0.0; waiting = False; last_stop_min = 0

    for j in range(start, sess_end+1):
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                pnl = (opens[j]-entry) if direction == "LONG" else (entry-opens[j])
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(opens[j],1),
                               "pnl_pts":round(pnl,1), "mfe":round(mfe,1), "adds":0, "reason":"EOD",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
            break
        bh, bl, bc = highs[j], lows[j], closes[j]
        if active:
            # Stop check
            if close_based_exit:
                hit = (direction == "LONG" and bc <= stop) or (direction == "SHORT" and bc >= stop)
            else:
                hit = (direction == "LONG" and bl <= stop) or (direction == "SHORT" and bh >= stop)
            if hit:
                exit_p = stop  # exit at stop level (same as v2)
                pnl = (exit_p-entry) if direction == "LONG" else (entry-exit_p)
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(exit_p,1),
                               "pnl_pts":round(pnl,1), "mfe":round(mfe,1), "adds":0, "reason":"STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
                entries_used += 1; active = False; waiting = entries_used < max_entries
                last_stop_min = bm + 5
                continue
            # MFE
            if direction == "LONG":
                m = bh-entry
                if m > mfe: mfe = m
                unr = bc-entry
            else:
                m = entry-bl
                if m > mfe: mfe = m
                unr = entry-bc
            # BE
            if not breakeven_hit and unr >= cfg["breakeven_pts"]:
                breakeven_hit = True
                if direction == "LONG" and stop < entry: stop = entry
                elif direction == "SHORT" and stop > entry: stop = entry
            # Trail
            if j > start:
                prev_h, prev_l, prev_c = highs[j-1], lows[j-1], closes[j-1]
                if direction == "LONG":
                    profit = prev_c-entry
                    # Variant A: always use prev_c; else use prev_l unless tight
                    if trail_to_close:
                        ns = prev_c
                    else:
                        ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop: stop = round(ns, 1)
                else:
                    profit = entry-prev_c
                    if trail_to_close:
                        ns = prev_c
                    else:
                        ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop: stop = round(ns, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop_min, eod_mins,
                                                  buy_level, sell_level)
                if res is None:
                    waiting = False
                else:
                    re_min, re_dir, re_fill = res
                    direction = re_dir; entry = re_fill
                    stop = sell_level if re_dir == "LONG" else buy_level
                    breakeven_hit = False; mfe = 0.0; active = True
                    while j+1 < len(mod) and mod[j+1] < re_min:
                        j += 1
            else:
                waiting = False
    return trades


def load(f, tz_src, tz_tgt):
    df = pd.read_csv(f, header=None, names=["dt","Open","High","Low","Close","V"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").tz_localize(ZoneInfo(tz_src)).tz_convert(ZoneInfo(tz_tgt))
    df = df[df.index.dayofweek < 5]
    df["_h"] = df.index.hour; df["_m"] = df.index.minute; df["_d"] = df.index.date
    return df


def build_cache():
    cache = {}
    for inst, meta in FR.items():
        cfg = bt.INSTRUMENTS[inst]
        print(f"Loading {inst} ...")
        d5 = load(meta["5m"], meta["tz"], cfg["timezone"])
        d1 = load(meta["1m"], meta["tz"], cfg["timezone"])
        ohlc5 = d5[["Open","High","Low","Close"]].values
        dates5 = d5["_d"].values
        by5 = {dd: (ohlc5[dates5==dd], d5["_h"].values[dates5==dd], d5["_m"].values[dates5==dd])
               for dd in sorted(set(dates5))}
        ohlc1 = d1[["Open","High","Low","Close"]].values
        dates1 = d1["_d"].values; h1 = d1["_h"].values; m1 = d1["_m"].values
        by1 = {}; cd, cs = None, 0
        for i, dd in enumerate(dates1):
            if dd != cd:
                if cd is not None:
                    s = slice(cs, i)
                    by1[cd] = (ohlc1[s], h1[s], h1[s]*60+m1[s])
                cd, cs = dd, i
        if cd is not None:
            s = slice(cs, len(dates1))
            by1[cd] = (ohlc1[s], h1[s], h1[s]*60+m1[s])
        sessions = [s for s in (1,2,3) if f"s{s}_open_hour" in cfg]
        cache[inst] = {"by5": by5, "by1": by1, "dates": sorted(set(dates5)), "sessions": sessions}
    return cache


def run(cache, trail_to_close, close_based_exit):
    trades = []
    for inst, cached in cache.items():
        cfg = bt.INSTRUMENTS[inst]
        for s in cached["sessions"]:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            name = f"{inst}_S{s}"
            for d in cached["dates"]:
                d5 = cached["by5"].get(d)
                if d5 is None: continue
                o5,h5,m5 = d5
                d1 = cached["by1"].get(d)
                o1,h1,min1 = d1 if d1 else (None,None,None)
                ts = simulate_v2_custom(o5,h5,m5, oh,om,eh,em, cfg, day_1min=o1,
                                        one_min_hours=h1, one_min_minutes=min1,
                                        trail_to_close=trail_to_close,
                                        close_based_exit=close_based_exit)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def filter_combined(df):
    df = df.sort_values(["date","signal","entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date","signal"]).cumcount()+1
    first = df[df["tn"]==1][["date","signal","direction","pnl_pts"]].rename(
        columns={"direction":"fd","pnl_pts":"fp"})
    df = df.merge(first, on=["date","signal"], how="left")
    df["first_won"] = df["fp"]>0
    df["is_re"] = df["tn"]>1
    df["same"] = df["direction"]==df["fd"]
    drop = df["is_re"] & ~df["same"] & (
        df["instrument"].isin(["DAX","US30"])
        | ((df["instrument"]=="NIKKEI") & df["first_won"]))
    return df[~drop].copy()


def stats(d, label):
    if len(d)==0: return
    w=d[d["pnl_pts"]>0]; l=d[d["pnl_pts"]<0]
    pf = w.pnl_pts.sum()/max(abs(l.pnl_pts.sum()),0.001)
    print(f"  {label:<40} n={len(d):>6}  PF={pf:>5.2f}  net={d.pnl_pts.sum():>+9,.0f}  WR={len(w)/len(d)*100:.1f}%")


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n")
    variants = [
        ("A_baseline (v2 candle)",       False, False),
        ("B_trail_to_close_always",       True, False),
        ("C_close_based_exit",            False, True),
        ("D_both",                         True, True),
    ]
    for lbl, tc, ce in variants:
        t1 = time.time()
        print(f"--- {lbl} ---")
        trades = run(cache, tc, ce)
        df = pd.DataFrame(trades)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        filt = filter_combined(df)
        print(f"  {lbl}: {len(trades):,} → {len(filt):,} ({time.time()-t1:.0f}s)")
        print(f"\n  {lbl} RESULTS:")
        stats(filt, "    FULL")
        stats(filt[filt["year"]<=2017], "    TRAIN")
        stats(filt[filt["year"]>=2018], "    TEST")
        for inst in ["DAX","US30","NIKKEI"]:
            stats(filt[filt["instrument"]==inst], f"      {inst}")
        print()


if __name__ == "__main__":
    main()
