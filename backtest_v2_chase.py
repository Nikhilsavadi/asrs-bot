"""
backtest_v2_chase.py — 18yr test of momentum chase after trail-stop winners.

Extends backtest_v2_atr to also simulate a chase trade after each TRAIL_STOP
winner. Chase = open SAME direction at winner's exit, fixed target + stop,
walk 5-min bars until hit.

Reports baseline PF vs baseline+chase PF, across all 3 instruments.
"""
import os, sys, time, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, stats
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def sim_with_chase(day_5min, hours_5m, minutes_5m, open_h, open_m, eod_h, eod_m, cfg,
                    day_1min=None, one_min_hours=None, one_min_minutes=None,
                    chase_stop=0, chase_target=0):
    """Run standard sim + for each TRAIL_STOP winner, simulate a chase trade."""
    if len(day_5min) == 0: return []
    highs = day_5min[:, 1]; lows = day_5min[:, 2]; closes = day_5min[:, 3]; opens = day_5min[:, 0]
    open_mins = open_h*60+open_m; eod_mins = eod_h*60+eod_m
    mod = hours_5m*60 + minutes_5m
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return []
    idxs = np.where(sess)[0]
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
    buy_level = round(sig_h+cfg["buffer"], 1); sell_level = round(sig_l-cfg["buffer"], 1)
    first_scan = bar4_idx+1 if bar_num==4 else bar4_idx+2
    sess_end = idxs[-1]
    fl = fs = -1
    for j in range(first_scan, sess_end+1):
        if mod[j] >= eod_mins: break
        if highs[j] >= buy_level and fl == -1: fl = j
        if lows[j] <= sell_level and fs == -1: fs = j
        if fl != -1 and fs != -1: break
    if fl == -1 and fs == -1: return []
    if fl != -1 and fs != -1:
        if fl < fs: direction,entry,stop,start = "LONG",buy_level,sell_level,fl
        elif fs < fl: direction,entry,stop,start = "SHORT",sell_level,buy_level,fs
        else:
            if opens[fl] >= buy_level: direction,entry,stop,start = "LONG",buy_level,sell_level,fl
            else: direction,entry,stop,start = "SHORT",sell_level,buy_level,fs
    elif fl >= 0: direction,entry,stop,start = "LONG",buy_level,sell_level,fl
    else: direction,entry,stop,start = "SHORT",sell_level,buy_level,fs

    trades = []; entries_used = 0; max_e = cfg["max_entries"]
    active=True; be=False; mfe=0.0; waiting=False; last_stop=0

    for j in range(start, sess_end+1):
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                pnl = (opens[j]-entry) if direction=="LONG" else (entry-opens[j])
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(opens[j],1),
                               "pnl_pts":round(pnl,1), "mfe":round(mfe,1), "adds":0, "reason":"EOD",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1),
                               "exit_idx": j})
            break
        bh,bl,bc = highs[j], lows[j], closes[j]
        if active:
            if (direction=="LONG" and bl<=stop) or (direction=="SHORT" and bh>=stop):
                pnl = (stop-entry) if direction=="LONG" else (entry-stop)
                trail_win = pnl > 0
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(stop,1),
                               "pnl_pts":round(pnl,1), "mfe":round(mfe,1), "adds":0,
                               "reason": "TRAIL_WIN" if trail_win else "STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1),
                               "exit_idx": j})
                entries_used += 1; active=False; waiting = entries_used<max_e
                last_stop = bm + 5
                continue
            if direction=="LONG":
                m = bh-entry
                if m>mfe: mfe=m
                unr = bc-entry
            else:
                m = entry-bl
                if m>mfe: mfe=m
                unr = entry-bc
            if not be and unr >= cfg["breakeven_pts"]:
                be = True
                if direction=="LONG" and stop<entry: stop=entry
                elif direction=="SHORT" and stop>entry: stop=entry
            if j > start:
                prev_c = closes[j-1]
                if direction=="LONG":
                    if prev_c > stop: stop = round(prev_c, 1)
                else:
                    if prev_c < stop: stop = round(prev_c, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop, eod_mins, buy_level, sell_level)
                if res is None: waiting=False
                else:
                    re_min, re_dir, re_fill = res
                    direction=re_dir; entry=re_fill
                    stop = sell_level if re_dir=="LONG" else buy_level
                    be=False; mfe=0.0; active=True
                    while j+1 < len(mod) and mod[j+1] < re_min:
                        j += 1
            else: waiting=False

    # Chase: simulate same-direction add after each TRAIL_WIN
    chase_trades = []
    if chase_stop > 0 and chase_target > 0:
        for t in trades:
            if t["reason"] != "TRAIL_WIN": continue
            e_idx = t["exit_idx"]
            c_entry = t["exit"]
            c_dir = t["direction"]
            if c_dir == "LONG":
                c_target = c_entry + chase_target
                c_stop = c_entry - chase_stop
            else:
                c_target = c_entry - chase_target
                c_stop = c_entry + chase_stop
            c_reason = "EOD"; c_exit = c_entry
            found = False
            for k in range(e_idx + 1, sess_end + 1):
                if mod[k] >= eod_mins:
                    c_exit = opens[k]; c_reason = "EOD"; found = True; break
                bh2, bl2 = highs[k], lows[k]
                if c_dir == "LONG":
                    if bh2 >= c_target: c_exit = c_target; c_reason = "TARGET"; found = True; break
                    if bl2 <= c_stop: c_exit = c_stop; c_reason = "STOP"; found = True; break
                else:
                    if bl2 <= c_target: c_exit = c_target; c_reason = "TARGET"; found = True; break
                    if bh2 >= c_stop: c_exit = c_stop; c_reason = "STOP"; found = True; break
            if not found:
                c_exit = closes[sess_end] if sess_end < len(closes) else c_entry
            c_pnl = (c_exit - c_entry) if c_dir == "LONG" else (c_entry - c_exit)
            chase_trades.append({
                "direction": c_dir, "entry": round(c_entry, 1),
                "exit": round(c_exit, 1), "pnl_pts": round(c_pnl, 1),
                "reason": f"CHASE_{c_reason}", "bar_range": t["bar_range"],
                "bar_num": t["bar_num"], "range_flag": t["range_flag"],
                "is_chase": True,
            })
    return trades + chase_trades


def build_cache():
    cache = {}
    for inst, meta in FR.items():
        cfg = bt.INSTRUMENTS[inst]
        print(f"Loading {inst} ...", flush=True)
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
                    s = slice(cs, i); by1[cd] = (ohlc1[s], h1[s], h1[s]*60+m1[s])
                cd, cs = dd, i
        if cd is not None:
            s = slice(cs, len(dates1)); by1[cd] = (ohlc1[s], h1[s], h1[s]*60+m1[s])
        sessions = [s for s in (1,2,3) if f"s{s}_open_hour" in cfg]
        cache[inst] = {"by5": by5, "by1": by1, "dates": sorted(set(dates5)), "sessions": sessions}
    return cache


def run(cache, chase_stop, chase_target):
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
                o5, h5, m5 = d5
                d1 = cached["by1"].get(d)
                o1, h1, min1 = d1 if d1 else (None, None, None)
                ts = sim_with_chase(o5, h5, m5, oh, om, eh, em, cfg,
                                     day_1min=o1, one_min_hours=h1, one_min_minutes=min1,
                                     chase_stop=chase_stop, chase_target=chase_target)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    print("--- BASELINE (Variant B, no chase) ---", flush=True)
    base = filter_combined(pd.DataFrame(run(cache, 0, 0)))
    base["year"] = pd.to_datetime(base["date"]).dt.year

    variants = [
        ("CHASE 30/100", 30, 100),
        ("CHASE 30/150", 30, 150),
        ("CHASE 50/100", 50, 100),
        ("CHASE 50/50",  50, 50),
        ("CHASE 75/50",  75, 50),
    ]

    print(f"\n{'='*100}\n  BASELINE RESULTS\n{'='*100}")
    stats(base, "  FULL")
    stats(base[base.year<=2017], "  TRAIN 2008-17")
    stats(base[base.year>=2018], "  TEST 2018-26")
    for inst in ["DAX", "US30", "NIKKEI"]:
        stats(base[base.instrument==inst], f"    {inst}")

    for lbl, cs, ct in variants:
        t1 = time.time()
        print(f"\n--- {lbl} ---", flush=True)
        all_trades = filter_combined(pd.DataFrame(run(cache, cs, ct)))
        all_trades["year"] = pd.to_datetime(all_trades["date"]).dt.year
        has_chase = "is_chase" in all_trades.columns
        if has_chase:
            all_trades["is_chase"] = all_trades["is_chase"].fillna(False)
            chase_only = all_trades[all_trades["is_chase"] == True]
            base_only = all_trades[all_trades["is_chase"] != True]
        else:
            chase_only = pd.DataFrame()
            base_only = all_trades

        print(f"{lbl}: total {len(all_trades):,} ({len(chase_only):,} chase) ({time.time()-t1:.0f}s)")
        stats(all_trades, "  FULL (base+chase)")
        if len(chase_only):
            stats(chase_only, "  CHASE only")
        stats(base_only, "  BASE only (sanity)")
        stats(all_trades[all_trades.year<=2017], "  TRAIN 2008-17")
        stats(all_trades[all_trades.year>=2018], "  TEST 2018-26")
        for inst in ["DAX", "US30", "NIKKEI"]:
            d = all_trades[all_trades.instrument == inst]
            stats(d, f"    {inst} full")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
