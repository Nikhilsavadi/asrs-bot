"""
backtest_signal_bar_variants.py — does it matter which bar we use for signal?

4 variants, all with Variant B trail + COMBINED re-entry filter:
  A: current hybrid (bar 4 unless range > wide, then bar 5)
  B: always bar 4
  C: always bar 5
  D: narrower of bar 4 and bar 5
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, stats
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def pick_signal_bar(idxs, highs, lows, cfg, mode):
    """
    mode: "HYBRID" | "BAR2" | "BAR3" | "BAR4" | "BAR5" | "NARROWER"
    Returns (sig_h, sig_l, bar_num, start_idx) or None if skipped.
    """
    if len(idxs) < 5: return None

    def bar(i):
        idx = idxs[i - 1]
        return highs[idx], lows[idx], idx

    b2_h, b2_l, b2_idx = bar(2)
    b3_h, b3_l, b3_idx = bar(3)
    b4_h, b4_l, b4_idx = bar(4)
    b5_h, b5_l, b5_idx = bar(5)
    b2_r = b2_h - b2_l; b3_r = b3_h - b3_l
    b4_r = b4_h - b4_l; b5_r = b5_h - b5_l

    if mode == "HYBRID":
        if b4_r < cfg["narrow_range"]:
            return b4_h, b4_l, 4, b4_idx + 1
        elif b4_r > cfg["wide_range"]:
            if b5_r > cfg["max_bar_range"] or b5_r <= 0: return None
            return b5_h, b5_l, 5, b5_idx + 1
        else:
            return b4_h, b4_l, 4, b4_idx + 1
    if mode == "BAR2":
        if b2_r > cfg["max_bar_range"] or b2_r <= 0: return None
        return b2_h, b2_l, 2, b2_idx + 1
    if mode == "BAR3":
        if b3_r > cfg["max_bar_range"] or b3_r <= 0: return None
        return b3_h, b3_l, 3, b3_idx + 1
    if mode == "BAR4":
        if b4_r > cfg["max_bar_range"] or b4_r <= 0: return None
        return b4_h, b4_l, 4, b4_idx + 1
    if mode == "BAR5":
        if b5_r > cfg["max_bar_range"] or b5_r <= 0: return None
        return b5_h, b5_l, 5, b5_idx + 1
    if mode == "NARROWER":
        if b4_r <= 0 and b5_r <= 0: return None
        if b4_r <= 0: return b5_h, b5_l, 5, b5_idx + 1
        if b5_r <= 0: return b4_h, b4_l, 4, b4_idx + 1
        if b4_r <= b5_r:
            if b4_r > cfg["max_bar_range"]: return None
            return b4_h, b4_l, 4, b4_idx + 1
        else:
            if b5_r > cfg["max_bar_range"]: return None
            return b5_h, b5_l, 5, b5_idx + 1
    return None


def sim(day_5m, hours, mins, open_h, open_m, eod_h, eod_m, cfg, mode,
        day_1min=None, one_min_hours=None, one_min_minutes=None):
    if len(day_5m) == 0: return []
    highs = day_5m[:, 1]; lows = day_5m[:, 2]; closes = day_5m[:, 3]; opens = day_5m[:, 0]
    open_mins = open_h*60 + open_m; eod_mins = eod_h*60 + eod_m
    mod = hours*60 + mins
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return []
    idxs = np.where(sess)[0]
    pick = pick_signal_bar(idxs, highs, lows, cfg, mode)
    if pick is None: return []
    sig_h, sig_l, bar_num, first_scan = pick
    bar_range = sig_h - sig_l
    range_flag = "NARROW" if bar_range < cfg["narrow_range"] else ("WIDE" if bar_range > cfg["wide_range"] else "NORMAL")

    buy_level = round(sig_h + cfg["buffer"], 1)
    sell_level = round(sig_l - cfg["buffer"], 1)
    sess_end = idxs[-1]

    fl = fs = -1
    for j in range(first_scan, sess_end + 1):
        if mod[j] >= eod_mins: break
        if highs[j] >= buy_level and fl == -1: fl = j
        if lows[j] <= sell_level and fs == -1: fs = j
        if fl != -1 and fs != -1: break
    if fl == -1 and fs == -1: return []
    if fl != -1 and fs != -1:
        if fl < fs: direction, entry, stop, start = "LONG", buy_level, sell_level, fl
        elif fs < fl: direction, entry, stop, start = "SHORT", sell_level, buy_level, fs
        else:
            if opens[fl] >= buy_level: direction, entry, stop, start = "LONG", buy_level, sell_level, fl
            else: direction, entry, stop, start = "SHORT", sell_level, buy_level, fs
    elif fl >= 0: direction, entry, stop, start = "LONG", buy_level, sell_level, fl
    else: direction, entry, stop, start = "SHORT", sell_level, buy_level, fs

    trades = []; entries_used = 0; max_e = cfg["max_entries"]
    active=True; be=False; mfe=0.0; waiting=False; last_stop=0

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
                last_stop = bm + 5
                continue
            unr = (bc-entry) if direction=="LONG" else (entry-bc)
            if not be and unr >= cfg["breakeven_pts"]:
                be = True
                if direction=="LONG" and stop<entry: stop=entry
                elif direction=="SHORT" and stop>entry: stop=entry
            # Variant B trail
            if j > start:
                prev_c = closes[j-1]
                if direction=="LONG":
                    if prev_c > stop: stop = round(prev_c, 1)
                else:
                    if prev_c < stop: stop = round(prev_c, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop, eod_mins,
                                                  buy_level, sell_level)
                if res is None: waiting=False
                else:
                    re_min, re_dir, re_fill = res
                    direction=re_dir; entry=re_fill
                    stop = sell_level if re_dir=="LONG" else buy_level
                    be=False; mfe=0.0; active=True
                    while j+1 < len(mod) and mod[j+1] < re_min:
                        j += 1
            else: waiting=False
    return trades


def build_cache():
    cache = {}
    for inst, meta in FR.items():
        cfg = bt.INSTRUMENTS[inst]
        print(f"Loading {inst} ...", flush=True)
        d5 = load(meta["5m"], meta["tz"], cfg["timezone"])
        d1 = load(meta["1m"], meta["tz"], cfg["timezone"])
        ohlc5 = d5[["Open","High","Low","Close"]].values
        dates5 = d5["_d"].values; h5 = d5["_h"].values; m5 = d5["_m"].values
        by5 = {dd: (ohlc5[dates5==dd], h5[dates5==dd], m5[dates5==dd])
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


def run(cache, mode):
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
                ts = sim(o5, h5, m5, oh, om, eh, em, cfg, mode,
                         day_1min=o1, one_min_hours=h1, one_min_minutes=min1)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    variants = [("A HYBRID (current)", "HYBRID"),
                ("B always bar 2",     "BAR2"),
                ("C always bar 3",     "BAR3"),
                ("D always bar 4",     "BAR4"),
                ("E always bar 5",     "BAR5"),
                ("F narrower of 4/5",  "NARROWER")]
    results = {}
    for lbl, m in variants:
        t1 = time.time()
        print(f"--- {lbl} ---", flush=True)
        trades = run(cache, m)
        df = pd.DataFrame(trades)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        filt = filter_combined(df)
        results[lbl] = filt
        print(f"  {lbl}: {len(trades):,} → {len(filt):,} ({time.time()-t1:.0f}s)", flush=True)

    print(f"\n{'='*95}\n  RESULTS (with Variant B trail + COMBINED filter)\n{'='*95}")
    for lbl, _ in variants:
        d = results[lbl]
        print(f"\n  {lbl}")
        stats(d, "    FULL")
        stats(d[d.year<=2017], "    TRAIN 2008-17")
        stats(d[d.year>=2018], "    TEST 2018-26")
        for inst in ["DAX","US30","NIKKEI"]:
            stats(d[d.instrument==inst], f"      {inst}")

    # Bar-number mix for HYBRID
    d_h = results["A HYBRID (current)"]
    if "bar_num" in d_h.columns:
        mix = d_h.groupby(["instrument", "bar_num"]).size().unstack(fill_value=0)
        print(f"\n  HYBRID bar-num distribution:\n{mix}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
