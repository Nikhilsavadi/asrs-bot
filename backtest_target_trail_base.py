"""
Test TARGET+TRAIL on BASE trades.
  A. BASELINE — Variant B (prev_close trail + BE)
  B. MILESTONE_LOCK — at profit ≥ N pts, lock stop at (profit - lock_giveback)
  C. AGGRESSIVE_TRAIL — at profit ≥ N pts, switch to prev_mid instead of prev_close
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import filter_combined, build_cache
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3
SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def sim(day_5m, h5, m5, open_h, open_m, eod_h, eod_m, cfg,
        day_1min=None, h1m=None, m1m_abs=None,
        mode="BASELINE", target_pts=50, lock_giveback=20):
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
    milestone_hit = False; milestone_lock = None

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
                               "pnl_pts":round(pnl,1), "reason": "TRAIL_WIN" if pnl>0 else "STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
                entries_used += 1; active=False; waiting = entries_used<max_e
                last_stop_min = bm + 5
                milestone_hit = False; milestone_lock = None
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

            if mode in ("MILESTONE_LOCK", "AGGRESSIVE_TRAIL") and not milestone_hit:
                if mfe >= target_pts:
                    milestone_hit = True
                    if mode == "MILESTONE_LOCK":
                        if direction == "LONG":
                            milestone_lock = entry + mfe - lock_giveback
                            if milestone_lock > stop: stop = round(milestone_lock, 1)
                        else:
                            milestone_lock = entry - mfe + lock_giveback
                            if milestone_lock < stop: stop = round(milestone_lock, 1)

            if j > start:
                prev_h, prev_l, prev_c = highs[j-1], lows[j-1], closes[j-1]
                prev_mid = (prev_h + prev_l) / 2
                if mode == "AGGRESSIVE_TRAIL" and milestone_hit:
                    trail_level = prev_mid
                else:
                    trail_level = prev_c
                if direction=="LONG":
                    if trail_level > stop: stop = round(trail_level, 1)
                else:
                    if trail_level < stop: stop = round(trail_level, 1)

                if mode == "MILESTONE_LOCK" and milestone_lock is not None:
                    if direction == "LONG" and stop < milestone_lock:
                        stop = round(milestone_lock, 1)
                    elif direction == "SHORT" and stop > milestone_lock:
                        stop = round(milestone_lock, 1)
        elif waiting:
            if day_1min is not None and m1m_abs is not None:
                res = _find_real_reentry_in_1min(day_1min, m1m_abs, last_stop_min, eod_mins,
                                                  buy_level, sell_level)
                if res is None: waiting=False
                else:
                    re_min, re_dir, re_fill = res
                    direction=re_dir; entry=re_fill
                    stop = sell_level if re_dir=="LONG" else buy_level
                    be=False; mfe=0.0; active=True
                    milestone_hit = False; milestone_lock = None
                    while j+1 < len(mod) and mod[j+1] < re_min:
                        j += 1
            else: waiting=False
    return trades


def run(cache, mode, target_pts=50, lock_giveback=20):
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
                         day_1min=o1, h1m=h1, m1m_abs=min1,
                         mode=mode, target_pts=target_pts, lock_giveback=lock_giveback)
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst; t["signal"] = name
                    trades.append(t)
    return trades


def analyse(trades_list, label):
    df = pd.DataFrame(trades_list)
    df_f = filter_combined(df)
    df_f["spread_cost"] = df_f["instrument"].map(SPREAD)
    df_f["slip_cost"] = np.where(df_f["pnl_pts"] < 0, df_f["instrument"].map(SLIP), 0)
    df_f["pnl_net"] = df_f["pnl_pts"] - df_f["spread_cost"] - df_f["slip_cost"]
    df_f["year"] = pd.to_datetime(df_f["date"]).dt.year
    tr = df_f[df_f.year <= 2017]; te = df_f[df_f.year >= 2018]
    print(f"  {label}:")
    print(f"    n={len(df_f):,}  PF={pf(df_f['pnl_net']):.2f}  net={df_f['pnl_net'].sum():+,.0f}"
          f"  TR={pf(tr['pnl_net']):.2f}  TE={pf(te['pnl_net']):.2f}")


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    print("--- BASELINE ---")
    trades = run(cache, "BASELINE")
    analyse(trades, "A. BASELINE (Variant B)")

    for target in [30, 50, 75, 100]:
        for giveback in [10, 20, 30]:
            print(f"\n--- MILESTONE_LOCK target={target} giveback={giveback} ---")
            trades = run(cache, "MILESTONE_LOCK", target_pts=target, lock_giveback=giveback)
            analyse(trades, f"B. LOCK at {target}pt, giveback {giveback}")
        print(f"\n--- AGGRESSIVE_TRAIL target={target} ---")
        trades = run(cache, "AGGRESSIVE_TRAIL", target_pts=target)
        analyse(trades, f"C. AGGRESSIVE_TRAIL at {target}pt (→ prev_mid)")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
