"""
backtest_s3_full.py — run S3 on 18yr firstrate data for US30 + NIKKEI.

Adds S3 session config (missing from bt.INSTRUMENTS) so the loop picks it up.
S1/S2 use HYBRID (current default). S3 uses HYBRID.

Reports per-instrument per-session PF so we can see S3's standalone edge.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, stats
from backtest_v2 import _find_real_reentry_in_1min
from backtest_s2s3_early_signal import sim, build_cache

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

# Add S3 config (not present in backtest.py's INSTRUMENTS)
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
# NIKKEI session_end = 15:00 in asrs config; backtest.py currently sets it
bt.INSTRUMENTS["NIKKEI"]["session_end_hour"] = 15
bt.INSTRUMENTS["NIKKEI"]["session_end_minute"] = 0


def run(cache):
    trades = []
    for inst, cached in cache.items():
        cfg = bt.INSTRUMENTS[inst]
        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        print(f"  {inst}: sessions {sessions}", flush=True)
        for s in sessions:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            name = f"{inst}_S{s}"
            for d in cached["dates"]:
                d5 = cached["by5"].get(d)
                if d5 is None: continue
                o5, h5, m5 = d5
                d1 = cached["by1"].get(d)
                o1, h1, min1 = d1 if d1 else (None, None, None)
                ts = sim(o5, h5, m5, oh, om, eh, em, cfg, mode="HYBRID",
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

    print(f"--- FULL backtest S1+S2+S3 (HYBRID all sessions) ---", flush=True)
    t1 = time.time()
    trades = run(cache)
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    filt = filter_combined(df)
    print(f"  trades: {len(trades):,} → {len(filt):,} filtered ({time.time()-t1:.0f}s)", flush=True)

    print(f"\n{'='*100}")
    print(f"  OVERALL (S1+S2+S3 HYBRID)")
    print(f"{'='*100}")
    stats(filt, "    FULL 18yr     ")
    stats(filt[filt.year<=2017], "    TRAIN 2008-17")
    stats(filt[filt.year>=2018], "    TEST  2018-26")

    print(f"\n{'='*100}")
    print(f"  PER-INSTRUMENT × PER-SESSION")
    print(f"{'='*100}")
    print(f"  {'inst':<8} {'S':<3} {'n_raw':>7} {'n_filt':>7} {'PF':>6} {'net pts':>12} {'WR%':>5}  {'S3 avg/yr':>10}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        for s in [1, 2, 3]:
            sub = filt[(filt.instrument==inst) & (filt.session==s)]
            raw_sub = df[(df.instrument==inst) & (df.session==s)]
            if len(sub) == 0:
                print(f"  {inst:<8} S{s:<2} {len(raw_sub):>7} {0:>7} {'—':>6} {'—':>12} {'—':>5}  {'—':>10}")
                continue
            w = sub[sub.pnl_pts>0].pnl_pts.sum()
            l = abs(sub[sub.pnl_pts<0].pnl_pts.sum())
            pf = w/l if l>0 else float("inf")
            wins = (sub.pnl_pts>0).sum()
            losses = (sub.pnl_pts<0).sum()
            wr = wins/(wins+losses)*100 if (wins+losses)>0 else 0
            per_yr = len(sub) / sub["year"].nunique() if sub["year"].nunique() > 0 else 0
            print(f"  {inst:<8} S{s:<2} {len(raw_sub):>7} {len(sub):>7} {pf:>6.2f} {sub.pnl_pts.sum():>+12,.0f} {wr:>5.1f}  {per_yr:>10.0f}")

    # Aggregate contribution: S3 net/year
    print(f"\n{'='*100}")
    print(f"  AGGREGATE CONTRIBUTION")
    print(f"{'='*100}")
    total_net = filt.pnl_pts.sum()
    for inst in ["DAX", "US30", "NIKKEI"]:
        base = filt[(filt.instrument==inst) & (filt.session.isin([1,2]))]
        s3 = filt[(filt.instrument==inst) & (filt.session==3)]
        if len(s3) > 0:
            lift_pct = (s3.pnl_pts.sum() / base.pnl_pts.sum() * 100) if base.pnl_pts.sum() > 0 else 0
            print(f"  {inst}: S1+S2 net {base.pnl_pts.sum():>+10,.0f}  |  S3 net {s3.pnl_pts.sum():>+10,.0f}  |  S3 lift = {lift_pct:+.1f}%")
        else:
            print(f"  {inst}: S3 not configured (DAX only has 2 sessions)")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
