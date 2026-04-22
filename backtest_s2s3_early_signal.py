"""
backtest_s2s3_early_signal.py — test earlier signal bars for S2/S3 only.

User question: for S2/S3 (mid-day sessions with no 'market open' inflection),
do we really need to wait 15-20 min for bar 4/5? Or can we use the bar
immediately before / at session open?

Variants for S2/S3 (S1 stays on HYBRID always):
  A: HYBRID  (current — bar 4 unless wide, then bar 5)
  B: BAR_PRE (5-min bar ending AT session open, e.g. 12:55-13:00 for DAX S2)
  C: BAR1    (5-min bar starting at session open, e.g. 13:00-13:05)
  D: BAR2    (second bar, e.g. 13:05-13:10)

All use Variant B trail, COMBINED re-entry filter, 18yr firstrate futures data.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, stats
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def pick_signal_bar_s2s3(highs, lows, mod, open_mins, cfg, mode):
    """Pick signal bar for S2/S3. Returns (sig_h, sig_l, bar_num, start_idx)
    or None. For BAR_PRE the bar ENDS at session open; start_idx is first bar
    at/after session open."""
    open_idx_arr = np.where(mod >= open_mins)[0]
    if len(open_idx_arr) == 0: return None
    open_idx = open_idx_arr[0]          # first bar on or after session open
    # Bar 1 starts at open. Bars indexed by session offset:
    # open_idx   → bar 1 (open-open+5)
    # open_idx-1 → bar 0 = BAR_PRE (open-5 to open)
    # open_idx+1 → bar 2
    # open_idx+3 → bar 4
    # open_idx+4 → bar 5

    def _bar(idx_offset):
        i = open_idx + idx_offset
        if i < 0 or i >= len(highs): return None
        r = highs[i] - lows[i]
        if r <= 0 or r > cfg["max_bar_range"]: return None
        return highs[i], lows[i], i

    if mode == "HYBRID":
        b4 = _bar(3)
        if b4 is None: return None
        b4_r = b4[0] - b4[1]
        if b4_r > cfg["wide_range"]:
            b5 = _bar(4)
            if b5 is None: return None
            return b5[0], b5[1], 5, b5[2] + 1
        return b4[0], b4[1], 4, b4[2] + 1
    if mode == "BAR_PRE":
        b0 = _bar(-1)
        if b0 is None: return None
        return b0[0], b0[1], 0, open_idx    # start scanning AT session open
    if mode == "BAR1":
        b1 = _bar(0)
        if b1 is None: return None
        return b1[0], b1[1], 1, b1[2] + 1
    if mode == "BAR2":
        b2 = _bar(1)
        if b2 is None: return None
        return b2[0], b2[1], 2, b2[2] + 1
    return None


def sim(day_5m, hours, mins, open_h, open_m, eod_h, eod_m, cfg, mode,
        session_num=1, day_1min=None, one_min_hours=None, one_min_minutes=None):
    """Simulate one session. S1 always uses HYBRID (ignore mode).
    S2/S3 use the requested mode."""
    if len(day_5m) == 0: return []
    highs = day_5m[:, 1]; lows = day_5m[:, 2]; closes = day_5m[:, 3]; opens = day_5m[:, 0]
    open_mins = open_h*60 + open_m; eod_mins = eod_h*60 + eod_m
    mod = hours*60 + mins

    pick_mode = "HYBRID" if session_num == 1 else mode
    pick = pick_signal_bar_s2s3(highs, lows, mod, open_mins, cfg, pick_mode)
    if pick is None: return []
    sig_h, sig_l, bar_num, first_scan = pick
    bar_range = sig_h - sig_l
    range_flag = "NARROW" if bar_range < cfg["narrow_range"] else ("WIDE" if bar_range > cfg["wide_range"] else "NORMAL")
    buy_level = round(sig_h + cfg["buffer"], 1)
    sell_level = round(sig_l - cfg["buffer"], 1)

    # Scan range for triggers: from first_scan up through session end
    sess_end_candidates = np.where(mod < eod_mins)[0]
    if len(sess_end_candidates) == 0: return []
    sess_end = sess_end_candidates[-1]

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
    active = True; be = False; mfe = 0.0; waiting = False; last_stop = 0

    j = start
    while j <= sess_end:
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                pnl = (opens[j]-entry) if direction=="LONG" else (entry-opens[j])
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(opens[j],1),
                               "pnl_pts":round(pnl,1), "reason":"EOD",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
            break
        bh, bl, bc = highs[j], lows[j], closes[j]
        if active:
            if (direction=="LONG" and bl<=stop) or (direction=="SHORT" and bh>=stop):
                pnl = (stop-entry) if direction=="LONG" else (entry-stop)
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(stop,1),
                               "pnl_pts":round(pnl,1), "reason":"STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1)})
                entries_used += 1; active = False; waiting = entries_used < max_e
                last_stop = bm + 5
                j += 1; continue
            unr = (bc-entry) if direction=="LONG" else (entry-bc)
            if not be and unr >= cfg["breakeven_pts"]:
                be = True
                if direction=="LONG" and stop<entry: stop = entry
                elif direction=="SHORT" and stop>entry: stop = entry
            if j > start:
                prev_c = closes[j-1]
                if direction=="LONG" and prev_c > stop: stop = round(prev_c, 1)
                elif direction=="SHORT" and prev_c < stop: stop = round(prev_c, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop, eod_mins,
                                                  buy_level, sell_level)
                if res is None: waiting = False
                else:
                    re_min, re_dir, re_fill = res
                    direction = re_dir; entry = re_fill
                    stop = sell_level if re_dir=="LONG" else buy_level
                    be = False; mfe = 0.0; active = True
                    while j+1 < len(mod) and mod[j+1] < re_min: j += 1
            else: waiting = False
        j += 1
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
                         session_num=s,
                         day_1min=o1, one_min_hours=h1, one_min_minutes=min1)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    t["session"] = s
                    trades.append(t)
    return trades


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache built in {time.time()-t0:.0f}s\n", flush=True)

    variants = [
        ("A HYBRID everywhere (baseline)", "HYBRID"),
        ("B BAR_PRE for S2/S3",            "BAR_PRE"),
        ("C BAR1 for S2/S3",               "BAR1"),
        ("D BAR2 for S2/S3",               "BAR2"),
    ]
    results = {}
    for lbl, m in variants:
        t1 = time.time()
        print(f"--- {lbl} ---", flush=True)
        trades = run(cache, m)
        df = pd.DataFrame(trades)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        filt = filter_combined(df)
        results[lbl] = filt
        print(f"  {lbl}: {len(trades):,} → {len(filt):,} filtered ({time.time()-t1:.0f}s)", flush=True)

    print(f"\n{'='*100}")
    print(f"  RESULTS (Variant B trail + COMBINED reverse-reentry filter, S1 always HYBRID)")
    print(f"{'='*100}")
    for lbl, _ in variants:
        d = results[lbl]
        print(f"\n  {lbl}")
        stats(d, "    FULL 18yr")
        stats(d[d.year<=2017], "    TRAIN 2008-17")
        stats(d[d.year>=2018], "    TEST 2018-26")

    # Per-instrument, per-session breakdown for the baseline vs each variant
    print(f"\n{'='*100}")
    print(f"  PER-INSTRUMENT × PER-SESSION PF")
    print(f"{'='*100}")
    print(f"  {'variant':<35} {'inst':<8} {'S1 PF':>8} {'S2 PF':>8} {'S3 PF':>8}  "
          f"{'S2 n':>6} {'S3 n':>6}")
    for lbl, _ in variants:
        d = results[lbl]
        for inst in ["DAX", "US30", "NIKKEI"]:
            row_pf = {}
            row_n = {}
            for s in [1,2,3]:
                sub = d[(d.instrument==inst) & (d.session==s)]
                if len(sub) == 0:
                    row_pf[s] = "—"; row_n[s] = 0; continue
                w = sub[sub.pnl_pts>0].pnl_pts.sum()
                l = abs(sub[sub.pnl_pts<0].pnl_pts.sum())
                pf = w/l if l>0 else float("inf")
                row_pf[s] = f"{pf:.2f}"
                row_n[s] = len(sub)
            print(f"  {lbl:<35} {inst:<8} {row_pf[1]:>8} {row_pf[2]:>8} {row_pf[3]:>8}  "
                  f"{row_n[2]:>6} {row_n[3]:>6}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
