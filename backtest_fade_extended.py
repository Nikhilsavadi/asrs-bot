"""
Extend fade hypothesis: test fading after SMALL LOSSES (<20pt) and
after ANY small exit. 18yr + TRAIN/TEST validation.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, build_cache
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}


def sim(day_5m, h5, m5, open_h, open_m, eod_h, eod_m, cfg,
        day_1min=None, h1m=None, m1m_abs=None,
        fade_stop_pts=50.0, trigger_on="WIN_GE_20", min_abs=20.0):
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
                trail_win = pnl > 0
                trades.append({"direction":direction, "entry":round(entry,1), "exit":round(stop,1),
                               "pnl_pts":round(pnl,1), "reason": "TRAIL_WIN" if trail_win else "STOP",
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

    # FADE (extended trigger)
    fade_trades = []
    for t in trades:
        reason = t["reason"]; pnl = t["pnl_pts"]
        is_win = reason == "TRAIL_WIN" and pnl > 0
        is_loss = reason == "STOP" and pnl < 0
        trigger = False
        use_same_dir = False
        if trigger_on == "WIN_GE_20":
            trigger = is_win and pnl >= min_abs
        elif trigger_on == "LOSS_SMALL":
            trigger = is_loss and abs(pnl) < min_abs
        elif trigger_on == "LOSS_SMALL_SAME_DIR":
            trigger = is_loss and abs(pnl) < min_abs
            use_same_dir = True
        elif trigger_on == "LOSS_BIG_OPPOSITE":
            # Big loss, OPPOSITE direction (momentum continuation bet)
            trigger = is_loss and abs(pnl) >= min_abs
            use_same_dir = False
        elif trigger_on == "LOSS_BIG_SAME":
            # Big loss, SAME direction (rescue / range-return bet)
            trigger = is_loss and abs(pnl) >= min_abs
            use_same_dir = True
        if not trigger: continue

        e_idx = t["exit_idx"]; f_entry = t["exit"]; orig_dir = t["direction"]
        if use_same_dir:
            f_dir = orig_dir
        else:
            f_dir = "SHORT" if orig_dir == "LONG" else "LONG"
        # Target logic depends on trigger type:
        #   Winner fade (post-trail): target = opposite bar extreme (mean-reversion)
        #   Small-loss variants:      same logic (bar extreme)
        #   Big-loss SAME direction:  target = OPPOSITE bar extreme (return-to-range)
        #     e.g. LONG lost at sig_l → re-LONG, target sig_h (back to range top)
        #   Big-loss OPPOSITE:        target = fixed 50pt continuation
        #     e.g. LONG lost at sig_l → SHORT, target = sig_l - 50 (continue down)
        if trigger_on == "LOSS_BIG_OPPOSITE":
            # Continuation target
            if f_dir == "LONG":
                f_target = f_entry + 50.0
            else:
                f_target = f_entry - 50.0
        elif trigger_on == "LOSS_BIG_SAME":
            # Return-to-range target: for re-entered LONG (was loss LONG), target sig_h
            if f_dir == "LONG":
                f_target = sig_h
            else:
                f_target = sig_l
        else:
            # Winner-fade (default): opposite bar extreme
            f_target = sig_l if f_dir == "LONG" else sig_h
        if f_dir == "LONG":
            f_stop = f_entry - fade_stop_pts
        else:
            f_stop = f_entry + fade_stop_pts
        f_exit = f_entry; f_reason = "EOD"; found = False
        for k in range(e_idx + 1, sess_end + 1):
            if mod[k] >= eod_mins:
                f_exit = opens[k]; f_reason = "EOD"; found = True; break
            bh2, bl2 = highs[k], lows[k]
            if f_dir == "LONG":
                if bh2 >= f_target: f_exit = f_target; f_reason = "TARGET"; found = True; break
                if bl2 <= f_stop: f_exit = f_stop; f_reason = "STOP"; found = True; break
            else:
                if bl2 <= f_target: f_exit = f_target; f_reason = "TARGET"; found = True; break
                if bh2 >= f_stop: f_exit = f_stop; f_reason = "STOP"; found = True; break
        if not found:
            f_exit = closes[sess_end] if sess_end < len(closes) else f_entry
        f_pnl = (f_exit - f_entry) if f_dir == "LONG" else (f_entry - f_exit)
        fade_trades.append({
            "direction": f_dir, "entry": round(f_entry, 1), "exit": round(f_exit, 1),
            "pnl_pts": round(f_pnl, 1), "reason": f"FADE_{f_reason}",
            "bar_range": t["bar_range"], "bar_num": t["bar_num"],
            "range_flag": t["range_flag"], "is_fade": True,
            "orig_dir": orig_dir, "orig_pnl": pnl, "orig_reason": reason,
        })
    return trades + fade_trades


def run(cache, trigger_on, min_abs=20.0):
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
                         trigger_on=trigger_on, min_abs=min_abs)
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst
                    t["signal"] = f"{name}_FADE" if t.get("is_fade") else name
                    trades.append(t)
    return trades


def pf(pnl):
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    scenarios = [
        ("A. WIN only ≥20pt (baseline — known good)", "WIN_GE_20", 20),
        ("F1. LOSS ≥30pt — OPPOSITE dir (momentum continuation)", "LOSS_BIG_OPPOSITE", 30),
        ("F2. LOSS ≥50pt — OPPOSITE dir (momentum continuation)", "LOSS_BIG_OPPOSITE", 50),
        ("F3. LOSS ≥75pt — OPPOSITE dir (tail continuation)", "LOSS_BIG_OPPOSITE", 75),
        ("G1. LOSS ≥30pt — SAME dir (rescue/return to range)", "LOSS_BIG_SAME", 30),
        ("G2. LOSS ≥50pt — SAME dir (rescue/return to range)", "LOSS_BIG_SAME", 50),
        ("G3. LOSS ≥75pt — SAME dir (deep rescue)", "LOSS_BIG_SAME", 75),
    ]

    for lbl, trigger, min_abs in scenarios:
        t1 = time.time()
        print(f"--- {lbl} ---", flush=True)
        all_trades = pd.DataFrame(run(cache, trigger, min_abs))
        if "is_fade" not in all_trades.columns:
            all_trades["is_fade"] = False
        all_trades["is_fade"] = all_trades["is_fade"].fillna(False).astype(bool)
        base = all_trades[~all_trades.is_fade].copy()
        fade = all_trades[all_trades.is_fade].copy()
        base_f = filter_combined(base)
        base_f["pnl_pts_net"] = base_f["pnl_pts"] - base_f["instrument"].map(SPREAD)
        fade["pnl_pts_net"] = fade["pnl_pts"] - fade["instrument"].map(SPREAD)
        combined = pd.concat([base_f, fade], ignore_index=True)
        combined["year"] = pd.to_datetime(combined["date"]).dt.year

        print(f"  fade n={len(fade):,}  ({time.time()-t1:.0f}s)")
        if len(fade):
            print(f"  FADE only   PF={pf(fade['pnl_pts_net']):.2f}  "
                  f"net={fade['pnl_pts_net'].sum():+,.0f}  "
                  f"WR={(fade['pnl_pts_net']>0).mean()*100:.1f}%")
        print(f"  COMBINED    PF={pf(combined['pnl_pts_net']):.2f}  "
              f"net={combined['pnl_pts_net'].sum():+,.0f}")
        tr = combined[combined.year <= 2017]; te = combined[combined.year >= 2018]
        print(f"  TRAIN PF={pf(tr['pnl_pts_net']):.2f}  TEST PF={pf(te['pnl_pts_net']):.2f}")
        if len(fade) and "orig_reason" in fade.columns:
            for r in ["TRAIL_WIN", "STOP"]:
                f_sub = fade[fade.orig_reason == r]
                if len(f_sub):
                    print(f"    after {r}: n={len(f_sub):,} "
                          f"PF={pf(f_sub['pnl_pts_net']):.2f} "
                          f"net={f_sub['pnl_pts_net'].sum():+,.0f}")
        print()

    print(f"  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
