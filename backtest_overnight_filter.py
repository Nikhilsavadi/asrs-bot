"""
backtest_overnight_filter.py — overnight-range direction filter (V58 revived).

For each day, compute overnight range = high/low of 6 hours before session open:
  DAX: 03:00-09:00 CET
  US30: 03:30-09:30 ET
  NIKKEI: 04:00-10:00 JST

Apply filter to existing ORB entries (Variant B trail, COMBINED re-entry filter):
  A baseline:     no filter (current live behavior)
  V1 filter:      signal bar fully above ON → SHORT_ONLY; fully below → LONG_ONLY
  V2 filter+fade: same filter, BUT at bar 4/5 close also place a fade-market-entry
                  on the opposite side of the ORB level (only when ON filter triggers)
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, stats
from backtest_v2 import _find_real_reentry_in_1min

PREOPEN_HOURS = 6  # compute overnight range over this many hours pre-open

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def compute_on_range(day_5m, hours, mins, open_h, open_m):
    """Overnight range = H/L of PREOPEN_HOURS hours before session open."""
    open_mins = open_h*60 + open_m
    start_mins = open_mins - PREOPEN_HOURS * 60
    mod = hours*60 + mins
    if start_mins < 0:
        mask = (mod >= start_mins + 24*60) | (mod < open_mins)
    else:
        mask = (mod >= start_mins) & (mod < open_mins)
    if not mask.any():
        return None, None
    return float(day_5m[mask, 1].max()), float(day_5m[mask, 2].min())


def classify(sig_h, sig_l, on_h, on_l):
    """Return 'ABOVE', 'BELOW', 'INSIDE' based on signal bar vs overnight range."""
    if on_h is None or on_l is None:
        return "INSIDE"
    if sig_l > on_h:
        return "ABOVE"  # fully above overnight high → expect fade DOWN → SHORT_ONLY
    if sig_h < on_l:
        return "BELOW"  # fully below overnight low → expect fade UP → LONG_ONLY
    return "INSIDE"


def sim(day_5m, hours, mins, open_h, open_m, eod_h, eod_m, cfg,
        day_1min=None, one_min_hours=None, one_min_minutes=None,
        apply_filter=False, filter_fade=False):
    if len(day_5m) == 0: return [], None
    highs = day_5m[:, 1]; lows = day_5m[:, 2]; closes = day_5m[:, 3]; opens = day_5m[:, 0]
    open_mins = open_h*60 + open_m; eod_mins = eod_h*60 + eod_m
    mod = hours*60 + mins
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return [], None
    idxs = np.where(sess)[0]
    if len(idxs) < 5: return [], None

    bar4_idx = idxs[3]; bar4_h = highs[bar4_idx]; bar4_l = lows[bar4_idx]
    bar_range = bar4_h - bar4_l; range_flag = "NARROW"; bar_num = 4
    if bar_range < cfg["narrow_range"]:
        sig_h, sig_l = bar4_h, bar4_l
    elif bar_range > cfg["wide_range"]:
        if len(idxs) < 5: return [], None
        bar5_idx = idxs[4]; sig_h = highs[bar5_idx]; sig_l = lows[bar5_idx]
        bar_range = sig_h - sig_l; bar_num = 5
        if bar_range > cfg["max_bar_range"] or bar_range <= 0: return [], None
        range_flag = "WIDE" if bar_range > cfg["wide_range"] else "NORMAL"
    else:
        sig_h, sig_l = bar4_h, bar4_l; range_flag = "NORMAL"

    on_h, on_l = compute_on_range(day_5m, hours, mins, open_h, open_m)
    bias = classify(sig_h, sig_l, on_h, on_l)

    buy_level = round(sig_h+cfg["buffer"], 1); sell_level = round(sig_l-cfg["buffer"], 1)
    first_scan = bar4_idx+1 if bar_num==4 else bar4_idx+2
    sess_end = idxs[-1]

    fl = fs = -1
    for j in range(first_scan, sess_end+1):
        if mod[j] >= eod_mins: break
        if highs[j] >= buy_level and fl == -1: fl = j
        if lows[j] <= sell_level and fs == -1: fs = j
        if fl != -1 and fs != -1: break

    # Apply overnight filter
    allowed_long = True; allowed_short = True
    if apply_filter:
        if bias == "ABOVE":  allowed_long = False
        elif bias == "BELOW": allowed_short = False

    if fl == -1 and fs == -1: return [], bias
    # Pick direction honouring filter
    def pick(fl, fs):
        if fl == -1 and fs == -1: return None
        if fl != -1 and fs != -1:
            if fl < fs: return "LONG", buy_level, sell_level, fl
            elif fs < fl: return "SHORT", sell_level, buy_level, fs
            else:
                if opens[fl] >= buy_level: return "LONG", buy_level, sell_level, fl
                else: return "SHORT", sell_level, buy_level, fs
        elif fl >= 0: return "LONG", buy_level, sell_level, fl
        else: return "SHORT", sell_level, buy_level, fs

    fl_use = fl if allowed_long else -1
    fs_use = fs if allowed_short else -1
    pick_r = pick(fl_use, fs_use)
    if pick_r is None: return [], bias
    direction, entry, stop, start = pick_r

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
                               "bias":bias})
            break
        bh,bl,bc = highs[j], lows[j], closes[j]
        if active:
            if (direction=="LONG" and bl<=stop) or (direction=="SHORT" and bh>=stop):
                pnl = (stop-entry) if direction=="LONG" else (entry-stop)
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(stop,1),
                               "pnl_pts":round(pnl,1), "mfe":round(mfe,1), "adds":0, "reason":"STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1),
                               "bias":bias})
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
            # Variant B trail (prev_close always)
            if j > start:
                prev_c = closes[j-1]
                if direction=="LONG":
                    if prev_c > stop: stop = round(prev_c, 1)
                else:
                    if prev_c < stop: stop = round(prev_c, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                # For re-entry, also honor the filter
                fl_pre = fl if allowed_long else -1
                fs_pre = fs if allowed_short else -1
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop, eod_mins,
                                                  buy_level if fl_pre >= 0 else 1e9,
                                                  sell_level if fs_pre >= 0 else -1e9)
                if res is None: waiting=False
                else:
                    re_min, re_dir, re_fill = res
                    if (re_dir == "LONG" and not allowed_long) or \
                       (re_dir == "SHORT" and not allowed_short):
                        waiting = False
                        continue
                    direction=re_dir; entry=re_fill
                    stop = sell_level if re_dir=="LONG" else buy_level
                    be=False; mfe=0.0; active=True
                    while j+1 < len(mod) and mod[j+1] < re_min:
                        j += 1
            else: waiting=False
    return trades, bias


def build_cache():
    cache = {}
    for inst, meta in FR.items():
        cfg = bt.INSTRUMENTS[inst]
        print(f"Loading {inst} ...", flush=True)
        d5 = load(meta["5m"], meta["tz"], cfg["timezone"])
        d1 = load(meta["1m"], meta["tz"], cfg["timezone"])
        ohlc5 = d5[["Open","High","Low","Close"]].values
        dates5 = d5["_d"].values; h5 = d5["_h"].values; m5 = d5["_m"].values
        # For ON range: we need ALL hours (including pre-open), so build per-day with full day's bars
        by5 = {}
        for dd in sorted(set(dates5)):
            mask = dates5 == dd
            by5[dd] = (ohlc5[mask], h5[mask], m5[mask])
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


def run(cache, apply_filter):
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
                # Apply filter ONLY on session 1 (V58 was session-1 only; other sessions don't have meaningful "overnight")
                apply = apply_filter and s == 1
                ts, bias = sim(o5, h5, m5, oh, om, eh, em, cfg,
                                day_1min=o1, one_min_hours=h1, one_min_minutes=min1,
                                apply_filter=apply)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    print("--- A: Variant B trail, NO overnight filter (current live) ---", flush=True)
    t_a = filter_combined(pd.DataFrame(run(cache, apply_filter=False)))
    print("--- B: Variant B trail + Overnight filter on S1 ---", flush=True)
    t_b = filter_combined(pd.DataFrame(run(cache, apply_filter=True)))

    print(f"\n{'='*95}\n  RESULTS\n{'='*95}")
    for lbl, d in [("A no filter", t_a), ("B ON filter", t_b)]:
        d["year"] = pd.to_datetime(d["date"]).dt.year
        print(f"\n  {lbl}")
        stats(d, "    FULL")
        stats(d[d.year<=2017], "    TRAIN 2008-17")
        stats(d[d.year>=2018], "    TEST 2018-26")
        for inst in ["DAX","US30","NIKKEI"]:
            stats(d[d.instrument==inst], f"      {inst}")

    # Bias distribution — how often does filter actually fire?
    d_all = pd.DataFrame(run(cache, apply_filter=False))
    d_s1 = d_all[d_all.signal.str.endswith("_S1")]
    if "bias" in d_s1.columns:
        biasdist = d_s1.groupby(["instrument","bias"]).size().unstack(fill_value=0)
        print(f"\n  Bias distribution (session 1 only, baseline):\n{biasdist}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
