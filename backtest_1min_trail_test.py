"""
Test fade with TRAILING exit (mini-B-trail) vs fixed-target exit.
Hypothesis: fade can capture MORE than bar 4/5 extreme if reversal continues.
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


def sim_fade_trailing(day_5m, h5, m5, open_h, open_m, eod_h, eod_m, cfg,
                      day_1min=None, h1m=None, m1m_abs=None,
                      fade_stop_pts=50.0, fade_exit_mode="FIXED_TARGET"):
    """
    fade_exit_mode:
      FIXED_TARGET — exit at bar 4/5 extreme (current prod)
      TRAIL_TARGET — exit at bar 4/5 extreme, BUT once hit, trail prev_close from there
      TRAIL_ONLY   — no fixed target, trail prev_close from entry
    """
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
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1),
                               "exit_idx": j})
            break
        bh,bl,bc = highs[j], lows[j], closes[j]
        if active:
            if (direction=="LONG" and bl<=stop) or (direction=="SHORT" and bh>=stop):
                pnl = (stop-entry) if direction=="LONG" else (entry-stop)
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(stop,1),
                               "pnl_pts":round(pnl,1), "reason": "TRAIL_WIN" if pnl>0 else "STOP",
                               "bar_num":bar_num, "range_flag":range_flag, "bar_range":round(bar_range,1),
                               "exit_idx": j})
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
            if day_1min is not None and m1m_abs is not None:
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
            else: waiting=False

    # FADE with configurable exit mode
    fade_trades = []
    for t in trades:
        if t["reason"] != "TRAIL_WIN" or t["pnl_pts"] < 20: continue  # ≥20pt filter
        e_idx = t["exit_idx"]; f_entry = t["exit"]; orig_dir = t["direction"]
        f_dir = "SHORT" if orig_dir == "LONG" else "LONG"
        f_target = sig_l if f_dir == "LONG" else sig_h
        if f_dir == "LONG":
            f_stop = f_entry - fade_stop_pts
        else:
            f_stop = f_entry + fade_stop_pts

        f_exit = f_entry; f_reason = "EOD"; found = False
        target_hit = False

        for k in range(e_idx + 1, sess_end + 1):
            if mod[k] >= eod_mins:
                f_exit = opens[k]; f_reason = "EOD"; found = True; break
            bh2, bl2, bc2 = highs[k], lows[k], closes[k]

            # Trail stop once target has been hit (for TRAIL_TARGET mode)
            # OR trail from entry (for TRAIL_ONLY mode)
            if fade_exit_mode == "TRAIL_ONLY" or (fade_exit_mode == "TRAIL_TARGET" and target_hit):
                if k > e_idx + 1:
                    prev_c = closes[k-1]
                    if f_dir == "LONG":
                        if prev_c > f_stop: f_stop = round(prev_c, 1)
                    else:
                        if prev_c < f_stop: f_stop = round(prev_c, 1)

            # Check target hit (FIXED_TARGET exits immediately; TRAIL_TARGET flags then trails)
            if not target_hit:
                if f_dir == "LONG" and bh2 >= f_target:
                    target_hit = True
                    if fade_exit_mode == "FIXED_TARGET":
                        f_exit = f_target; f_reason = "TARGET"; found = True; break
                elif f_dir == "SHORT" and bl2 <= f_target:
                    target_hit = True
                    if fade_exit_mode == "FIXED_TARGET":
                        f_exit = f_target; f_reason = "TARGET"; found = True; break

            # Check stop (always active)
            if f_dir == "LONG":
                if bl2 <= f_stop: f_exit = f_stop; f_reason = "STOP"; found = True; break
            else:
                if bh2 >= f_stop: f_exit = f_stop; f_reason = "STOP"; found = True; break

        if not found:
            f_exit = closes[sess_end] if sess_end < len(closes) else f_entry
        f_pnl = (f_exit - f_entry) if f_dir == "LONG" else (f_entry - f_exit)
        fade_trades.append({
            "direction": f_dir, "entry": round(f_entry, 1), "exit": round(f_exit, 1),
            "pnl_pts": round(f_pnl, 1), "reason": f"FADE_{f_reason}",
            "bar_range": t["bar_range"], "bar_num": t["bar_num"],
            "range_flag": t["range_flag"], "is_fade": True,
        })
    return trades + fade_trades


def run(cache, fade_exit_mode):
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
                ts = sim_fade_trailing(o5, h5, m5, oh, om, eh, em, cfg,
                                        day_1min=o1, h1m=h1, m1m_abs=min1,
                                        fade_stop_pts=50, fade_exit_mode=fade_exit_mode)
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst
                    t["signal"] = f"{name}_FADE" if t.get("is_fade") else name
                    trades.append(t)
    return trades


def analyse(trades_list, label):
    df = pd.DataFrame(trades_list)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    base = df[~df.is_fade].copy()
    fade = df[df.is_fade].copy()
    base_f = filter_combined(base)
    base_f["spread_cost"] = base_f["instrument"].map(SPREAD)
    base_f["slip_cost"] = np.where(base_f["pnl_pts"] < 0, base_f["instrument"].map(SLIP), 0)
    base_f["pnl_net"] = base_f["pnl_pts"] - base_f["spread_cost"] - base_f["slip_cost"]
    fade["spread_cost"] = fade["instrument"].map(SPREAD)
    fade["slip_cost"] = np.where(fade["pnl_pts"] < 0, fade["instrument"].map(SLIP), 0)
    fade["pnl_net"] = fade["pnl_pts"] - fade["spread_cost"] - fade["slip_cost"]
    combined = pd.concat([base_f, fade], ignore_index=True)
    combined["year"] = pd.to_datetime(combined["date"]).dt.year

    tr = combined[combined.year <= 2017]; te = combined[combined.year >= 2018]
    print(f"\n  {label}:")
    print(f"    FADE only: n={len(fade):,} PF={pf(fade['pnl_net']):.2f} net={fade['pnl_net'].sum():+,.0f}")
    print(f"    COMBINED:  PF={pf(combined['pnl_net']):.2f} net={combined['pnl_net'].sum():+,.0f}  "
          f"TRAIN PF={pf(tr['pnl_net']):.2f}  TEST PF={pf(te['pnl_net']):.2f}")


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    for mode in ["FIXED_TARGET", "TRAIL_ONLY", "TRAIL_TARGET"]:
        t1 = time.time()
        trades = run(cache, mode)
        analyse(trades, f"{mode}")
        print(f"    ({time.time()-t1:.0f}s)")
    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
