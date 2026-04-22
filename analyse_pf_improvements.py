"""
analyse_pf_improvements.py — spread-adjusted PF for base vs fade vs combined.
"""
import os, sys, time, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 3.0, "US30": 4.0, "NIKKEI": 15.0}

from backtest_v2_fade import build_cache, sim_with_fade
import backtest as bt
from backtest_v2_atr import filter_combined

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def pf(pnl):
    if len(pnl) == 0: return 0
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def apply_spread(df, spread_map):
    df = df.copy()
    df["pnl_pts_net"] = df["pnl_pts"] - df["instrument"].map(spread_map)
    return df


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    all_trades = []
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
                ts = sim_with_fade(o5, h5, m5, oh, om, eh, em, cfg,
                                    day_1min=o1, one_min_hours=h1, one_min_minutes=min1,
                                    fade_stop=50, fade_target_mode="extreme")
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst
                    t["signal"] = f"{name}_FADE" if t.get("is_fade") else name
                    all_trades.append(t)
    df = pd.DataFrame(all_trades)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False)
    base = df[~df["is_fade"]].copy()
    fade = df[df["is_fade"]].copy()
    base_f = filter_combined(base)
    combined = pd.concat([base_f, fade], ignore_index=True)

    print(f"{'='*100}")
    print(f"  SPREAD-ADJUSTED PF  (DAX {SPREAD['DAX']} / US30 {SPREAD['US30']} / NIKKEI {SPREAD['NIKKEI']} pt round-trip)")
    print(f"{'='*100}")

    for name_lbl, ds in [("BASE alone", base_f), ("FADE alone", fade), ("COMBINED", combined)]:
        print(f"\n  {name_lbl}")
        print(f"    {'inst':<8} {'n':<7} {'raw PF':<9} {'raw net':<12} {'net PF':<9} {'net net':<12} {'Δ PF':<6}")
        for inst in ["DAX", "US30", "NIKKEI", "ALL"]:
            d = ds if inst == "ALL" else ds[ds.instrument == inst]
            if len(d) == 0: continue
            d_adj = apply_spread(d, SPREAD)
            raw_pf = pf(d["pnl_pts"]); net_pf = pf(d_adj["pnl_pts_net"])
            raw_net = d["pnl_pts"].sum(); net_net = d_adj["pnl_pts_net"].sum()
            print(f"    {inst:<8} {len(d):<7,} {raw_pf:<6.2f}    "
                  f"{raw_net:>+9,.0f}    {net_pf:<6.2f}    "
                  f"{net_net:>+9,.0f}   -{raw_pf-net_pf:.2f}")

    print(f"\n{'='*100}\n  AVG WIN/LOSS PER TRADE (spread sensitivity)\n{'='*100}")
    for lbl, ds in [("BASE", base_f), ("FADE", fade)]:
        print(f"\n  {lbl}:")
        for inst in ["DAX", "US30", "NIKKEI"]:
            d = ds[ds.instrument == inst]
            if len(d) == 0: continue
            w = d[d["pnl_pts"] > 0]["pnl_pts"]
            l = d[d["pnl_pts"] < 0]["pnl_pts"]
            s = SPREAD[inst]
            pct_w = (s/w.mean()*100) if len(w) else 0
            pct_l = (s/abs(l.mean())*100) if len(l) else 0
            print(f"    {inst:<8}  avg_W={w.mean():>+6.1f}pt  avg_L={l.mean():>+6.1f}pt  "
                  f"spread={s}pt  ({pct_w:.0f}% of W, {pct_l:.0f}% of L)")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
