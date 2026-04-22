"""
18yr test: 1MIN drill-down gate vs BAR5 (tick-like) gate on re-entries.

Isolates the "1-min drill-down" contribution to backtest PF. If the current
backtest PF 4.17 drops when we remove the 1-min filter, that's the measure
of how optimistic the drill-down is.

Both runs use Variant B trail. Only re-entry gate differs.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import FR, load, filter_combined, stats, build_cache
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def sim(day_5m, h5, m5, open_h, open_m, eod_h, eod_m, cfg,
        day_1min=None, h1m=None, m1m_abs=None, gate_mode="1MIN"):
    if len(day_5m) == 0: return []
    highs = day_5m[:, 1]; lows = day_5m[:, 2]; closes = day_5m[:, 3]; opens = day_5m[:, 0]
    open_mins = open_h*60+open_m; eod_mins = eod_h*60+eod_m
    mod = h5*60 + m5
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
    active=True; be=False; mfe=0.0; waiting=False; last_stop_min=0

    for j in range(start, sess_end+1):
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                pnl = (opens[j]-entry) if direction=="LONG" else (entry-opens[j])
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(opens[j],1),
                               "pnl_pts":round(pnl,1), "reason":"EOD",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
            break
        bh,bl,bc = highs[j], lows[j], closes[j]
        if active:
            if (direction=="LONG" and bl<=stop) or (direction=="SHORT" and bh>=stop):
                pnl = (stop-entry) if direction=="LONG" else (entry-stop)
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(stop,1),
                               "pnl_pts":round(pnl,1), "reason":"STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
                entries_used += 1; active=False; waiting = entries_used<max_e
                last_stop_min = bm + 5
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
            if gate_mode == "1MIN" and day_1min is not None and m1m_abs is not None:
                res = _find_real_reentry_in_1min(day_1min, m1m_abs, last_stop_min, eod_mins,
                                                  buy_level, sell_level)
                if res is None: waiting=False
                else:
                    re_min, re_dir, re_fill = res
                    direction=re_dir; entry=re_fill
                    stop = sell_level if re_dir=="LONG" else buy_level
                    be=False; mfe=0.0; active=True
                    while j+1 < len(mod) and mod[j+1] < re_min:
                        j += 1
            elif gate_mode == "BAR5":
                # Re-entry on NEXT 5-min bar whose H/L crosses level
                # (simulates tick-level gate — no 1-min filter)
                if bm >= last_stop_min:
                    cl = bh >= buy_level
                    cs = bl <= sell_level
                    if cl and cs:
                        if abs(opens[j] - buy_level) <= abs(opens[j] - sell_level):
                            direction = "LONG"; entry = buy_level; stop = sell_level
                        else:
                            direction = "SHORT"; entry = sell_level; stop = buy_level
                        be=False; mfe=0.0; active=True
                    elif cl:
                        direction = "LONG"; entry = buy_level; stop = sell_level
                        be=False; mfe=0.0; active=True
                    elif cs:
                        direction = "SHORT"; entry = sell_level; stop = buy_level
                        be=False; mfe=0.0; active=True
            else:
                waiting=False
    return trades


def run(cache, gate_mode):
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
                ts = sim(o5, h5, m5, oh, om, eh, em, cfg,
                         day_1min=o1, h1m=h1, m1m_abs=min1, gate_mode=gate_mode)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    for lbl, mode in [("1MIN drill-down (current backtest)", "1MIN"),
                       ("BAR5 tick-like gate (no 1-min filter)", "BAR5")]:
        t1 = time.time()
        print(f"--- {lbl} ---", flush=True)
        trades = filter_combined(pd.DataFrame(run(cache, mode)))
        trades["year"] = pd.to_datetime(trades["date"]).dt.year
        print(f"  n trades: {len(trades):,}  ({time.time()-t1:.0f}s)")
        stats(trades, "  FULL")
        stats(trades[trades.year<=2017], "  TRAIN 2008-17")
        stats(trades[trades.year>=2018], "  TEST 2018-26")
        for inst in ["DAX", "US30", "NIKKEI"]:
            stats(trades[trades.instrument==inst], f"    {inst}")
        print()

    print(f"  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
