"""
backtest_partial_and_milestone.py — test both partial profit + milestone tuning.

Runs 18yr backtest (Variant B + fade) with two variables:

#3 Partial profit-taking (PP):
  - close 50% of position at MFE = TP1 multiplier × initial risk
  - remaining 50% continues on Variant B trail
  - variants: TP1 disabled (baseline), 1.0x, 1.5x, 2.0x, 2.5x risk

#4 Milestone_lock tuning:
  - at MFE >= milestone_pts, lock stop at (MFE - giveback_pts)
  - current: 30/10
  - variants: 20/5, 20/10, 30/10 (baseline), 40/10, 40/15, 50/20

Tests run on cached bars (one cache build, multiple simulations).
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_fade import build_cache
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["session_end_hour"] = 15
bt.INSTRUMENTS["NIKKEI"]["session_end_minute"] = 0

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
UPLIFT = 1.06


def sim_tunable(day_5min, hours_5m, minutes_5m, open_h, open_m, eod_h, eod_m,
                cfg, day_1min=None, one_min_hours=None, one_min_minutes=None,
                milestone_pts=30.0, giveback_pts=10.0,
                tp1_mult=0.0):
    """Extended sim with configurable milestone + partial-profit TP1.
    tp1_mult=0 disables partial profit."""
    if len(day_5min) == 0: return []
    highs = day_5min[:, 1]; lows = day_5min[:, 2]; closes = day_5min[:, 3]; opens = day_5min[:, 0]
    open_mins = open_h*60+open_m; eod_mins = eod_h*60+eod_m
    mod = hours_5m*60 + minutes_5m
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return []
    idxs = np.where(sess)[0]
    if len(idxs) < 5: return []

    bar4_idx = idxs[3]; bar4_h = highs[bar4_idx]; bar4_l = lows[bar4_idx]
    r4 = bar4_h - bar4_l
    if r4 < cfg["narrow_range"]:
        sig_h, sig_l = bar4_h, bar4_l; bar_num = 4
    elif r4 > cfg["wide_range"]:
        if len(idxs) < 5: return []
        bar5_idx = idxs[4]; sig_h = highs[bar5_idx]; sig_l = lows[bar5_idx]; bar_num = 5
        if sig_h - sig_l > cfg["max_bar_range"] or sig_h - sig_l <= 0: return []
    else:
        sig_h, sig_l = bar4_h, bar4_l; bar_num = 4

    bar_range = sig_h - sig_l
    buy_level = round(sig_h + cfg["buffer"], 1)
    sell_level = round(sig_l - cfg["buffer"], 1)
    first_scan = bar4_idx+1 if bar_num == 4 else bar4_idx+2
    sess_end = idxs[-1]

    fl = fs = -1
    for j in range(first_scan, sess_end+1):
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

    initial_risk = bar_range + cfg["buffer"] * 2
    if initial_risk > cfg["max_risk_gbp"]:
        initial_risk = cfg["max_risk_gbp"]
    tp1_price = None
    if tp1_mult > 0:
        if direction == "LONG":
            tp1_price = entry + tp1_mult * initial_risk
        else:
            tp1_price = entry - tp1_mult * initial_risk

    trades = []; entries_used = 0; max_e = cfg["max_entries"]
    active = True; be = False; mfe = 0.0; milestone_hit = False
    tp1_hit = False; tp1_pnl = 0.0
    waiting = False; last_stop = 0

    j = start
    while j <= sess_end:
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                # EOD close remaining half (or full if no TP1)
                size_remaining = 0.5 if tp1_hit else 1.0
                remaining_pnl = (opens[j]-entry) if direction == "LONG" else (entry-opens[j])
                total_pnl = tp1_pnl + remaining_pnl * size_remaining
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(opens[j], 1), "pnl_pts": round(total_pnl, 1),
                               "reason": "EOD", "bar_num": bar_num,
                               "tp1_hit": tp1_hit, "mfe": round(mfe, 1)})
            break
        bh, bl, bc = highs[j], lows[j], closes[j]
        if active:
            # Update MFE
            if direction == "LONG":
                bar_mfe = max(0.0, bh - entry); mfe = max(mfe, bar_mfe)
            else:
                bar_mfe = max(0.0, entry - bl); mfe = max(mfe, bar_mfe)

            # Check TP1 hit (partial profit)
            if tp1_price is not None and not tp1_hit:
                if direction == "LONG" and bh >= tp1_price:
                    tp1_hit = True
                    tp1_pnl = (tp1_price - entry) * 0.5
                elif direction == "SHORT" and bl <= tp1_price:
                    tp1_hit = True
                    tp1_pnl = (entry - tp1_price) * 0.5

            # Check stop
            if (direction == "LONG" and bl <= stop) or (direction == "SHORT" and bh >= stop):
                size_remaining = 0.5 if tp1_hit else 1.0
                stop_pnl_full = (stop - entry) if direction == "LONG" else (entry - stop)
                total_pnl = tp1_pnl + stop_pnl_full * size_remaining
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(stop, 1), "pnl_pts": round(total_pnl, 1),
                               "reason": "STOP", "bar_num": bar_num,
                               "tp1_hit": tp1_hit, "mfe": round(mfe, 1)})
                if tp1_hit and stop_pnl_full > 0:
                    # Trail-stop winner with TP1 hit → treat as TRAIL_WIN for fade
                    trades[-1]["reason"] = "TRAIL_WIN"
                entries_used += 1; active = False
                waiting = entries_used < max_e
                last_stop = bm + 5
                be = False; mfe = 0.0; tp1_hit = False; tp1_pnl = 0.0; milestone_hit = False
                j += 1; continue

            unr = (bc - entry) if direction == "LONG" else (entry - bc)
            # Breakeven
            if not be and unr >= cfg.get("breakeven_pts", 15):
                be = True
                if direction == "LONG" and stop < entry: stop = entry
                elif direction == "SHORT" and stop > entry: stop = entry
            # Milestone lock
            if not milestone_hit and mfe >= milestone_pts:
                milestone_hit = True
                if direction == "LONG":
                    lock = entry + (mfe - giveback_pts)
                    if lock > stop: stop = round(lock, 1)
                else:
                    lock = entry - (mfe - giveback_pts)
                    if lock < stop: stop = round(lock, 1)
            # Re-apply milestone trail as MFE grows
            if milestone_hit:
                if direction == "LONG":
                    lock = entry + (mfe - giveback_pts)
                    if lock > stop: stop = round(lock, 1)
                else:
                    lock = entry - (mfe - giveback_pts)
                    if lock < stop: stop = round(lock, 1)
            # Variant B trail (prev close)
            if j > start:
                prev_c = closes[j-1]
                if direction == "LONG" and prev_c > stop: stop = round(prev_c, 1)
                elif direction == "SHORT" and prev_c < stop: stop = round(prev_c, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop, eod_mins,
                                                  buy_level, sell_level)
                if res is None: waiting = False
                else:
                    re_min, re_dir, re_fill = res
                    direction = re_dir; entry = re_fill
                    stop = sell_level if re_dir == "LONG" else buy_level
                    if tp1_mult > 0:
                        tp1_price = (entry + tp1_mult * initial_risk) if direction == "LONG" \
                                    else (entry - tp1_mult * initial_risk)
                    else:
                        tp1_price = None
                    active = True
                    while j+1 < len(mod) and mod[j+1] < re_min: j += 1
            else: waiting = False
        j += 1
    return trades


def run(cache, milestone_pts, giveback_pts, tp1_mult):
    trades = []
    for inst, cached in cache.items():
        cfg = bt.INSTRUMENTS[inst]
        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        for s in sessions:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            for d in cached["dates"]:
                d5 = cached["by5"].get(d)
                if d5 is None: continue
                o5, h5, m5 = d5
                d1 = cached["by1"].get(d)
                o1, h1, min1 = d1 if d1 else (None, None, None)
                ts = sim_tunable(o5, h5, m5, oh, om, eh, em, cfg,
                                  day_1min=o1, one_min_hours=h1, one_min_minutes=min1,
                                  milestone_pts=milestone_pts, giveback_pts=giveback_pts,
                                  tp1_mult=tp1_mult)
                for t in ts:
                    t["date"] = str(d); t["signal"] = f"{inst}_S{s}"; t["instrument"] = inst
                    trades.append(t)
    return trades


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def filter_rr(df):
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"})
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    opp = df["is_re"] & ~df["same"]
    drop = opp & (df["instrument"].isin(["DAX", "US30"])
                  | ((df["instrument"] == "NIKKEI") & df["first_won"]))
    return df[~drop].copy()


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def summarise(df, label):
    p = pf(df.pnl_net)
    wr = (df.pnl_net > 0).mean() * 100
    avg = df.pnl_net.mean()
    print(f"  {label:<30} n={len(df):>6,} PF={p:.2f} net={df.pnl_net.sum():>+10,.0f} avg={avg:>+6.2f} WR={wr:.1f}%")


def main():
    t0 = time.time()
    print("Building cache ...", flush=True)
    cache = build_cache()
    print(f"  cache in {time.time()-t0:.0f}s\n", flush=True)

    # #4 Milestone tuning (keep TP1 off)
    print("="*100)
    print("  #4 MILESTONE_LOCK TUNING (TP1 off, Variant B + fade baseline)")
    print("="*100)
    milestone_variants = [
        ("A milestone 20/5",   20.0,  5.0),
        ("B milestone 20/10",  20.0, 10.0),
        ("C milestone 30/10 (current)", 30.0, 10.0),
        ("D milestone 40/10",  40.0, 10.0),
        ("E milestone 40/15",  40.0, 15.0),
        ("F milestone 50/20",  50.0, 20.0),
    ]
    results_m = {}
    for lbl, m, g in milestone_variants:
        t1 = time.time()
        trades = run(cache, milestone_pts=m, giveback_pts=g, tp1_mult=0.0)
        df = pd.DataFrame(trades)
        df = filter_rr(df)
        df = apply_friction(df)
        results_m[lbl] = df
        print(f"  {lbl}: {len(trades):,} raw → {len(df):,} filtered  ({time.time()-t1:.0f}s)",
              flush=True)

    print(f"\n  {'Variant':<32} {'n':>7} {'PF':>6} {'net':>12} {'avg':>8} {'WR':>6}  {'vs C':>10}")
    baseline_net = results_m["C milestone 30/10 (current)"].pnl_net.sum()
    for lbl, _, _ in milestone_variants:
        d = results_m[lbl]
        p = pf(d.pnl_net); wr = (d.pnl_net > 0).mean() * 100
        delta = d.pnl_net.sum() - baseline_net
        print(f"  {lbl:<32} {len(d):>7,} {p:>6.2f} {d.pnl_net.sum():>+12,.0f} "
              f"{d.pnl_net.mean():>+8.2f} {wr:>5.1f}%  {delta:>+10,.0f}")

    # #3 Partial profit-taking (keep best milestone from #4, or 30/10)
    print("\n" + "="*100)
    print("  #3 PARTIAL PROFIT-TAKING (close 50% at TP1 × risk, milestone 30/10 kept)")
    print("="*100)
    tp_variants = [
        ("A no TP1 (current)",  0.0),
        ("B TP1 at 1.0× risk",  1.0),
        ("C TP1 at 1.5× risk",  1.5),
        ("D TP1 at 2.0× risk",  2.0),
        ("E TP1 at 2.5× risk",  2.5),
    ]
    results_t = {}
    for lbl, mult in tp_variants:
        t1 = time.time()
        trades = run(cache, milestone_pts=30.0, giveback_pts=10.0, tp1_mult=mult)
        df = pd.DataFrame(trades)
        df = filter_rr(df)
        df = apply_friction(df)
        results_t[lbl] = df
        print(f"  {lbl}: {len(trades):,} raw → {len(df):,} filtered  ({time.time()-t1:.0f}s)",
              flush=True)

    print(f"\n  {'Variant':<32} {'n':>7} {'PF':>6} {'net':>12} {'avg':>8} {'WR':>6}  {'vs A':>10}")
    baseline_net = results_t["A no TP1 (current)"].pnl_net.sum()
    for lbl, _ in tp_variants:
        d = results_t[lbl]
        p = pf(d.pnl_net); wr = (d.pnl_net > 0).mean() * 100
        delta = d.pnl_net.sum() - baseline_net
        print(f"  {lbl:<32} {len(d):>7,} {p:>6.2f} {d.pnl_net.sum():>+12,.0f} "
              f"{d.pnl_net.mean():>+8.2f} {wr:>5.1f}%  {delta:>+10,.0f}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
