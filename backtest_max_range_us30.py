"""
Test US30 max_bar_range tightening — cap to reduce WIDE-bar tail losses.
Top 10 TEST losses were all WIDE bars 130-225pt. Tightening US30 max from 300
should trim big losses at cost of fewer trades.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import filter_combined, build_cache
from backtest_v2_fade import sim_with_fade

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3
SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def run(cache, us30_max):
    orig = bt.INSTRUMENTS["US30"]["max_bar_range"]
    bt.INSTRUMENTS["US30"]["max_bar_range"] = us30_max
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
                ts = sim_with_fade(o5, h5, m5, oh, om, eh, em, cfg,
                                    day_1min=o1, one_min_hours=h1, one_min_minutes=min1,
                                    fade_stop=50, fade_target_mode="extreme")
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst
                    t["signal"] = f"{name}_FADE" if t.get("is_fade") else name
                    trades.append(t)
    bt.INSTRUMENTS["US30"]["max_bar_range"] = orig
    return trades


def analyse(trades_list, label):
    df = pd.DataFrame(trades_list)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    df = df.reset_index(drop=True)
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        fades = group[group.is_fade]
        for i, (idx, _) in enumerate(fades.iterrows()):
            if i < len(q): orig_win[idx] = q[i]
    fade = df[df.is_fade].copy()
    fade["orig_win"] = fade.index.map(orig_win)
    fade_f = fade[fade["orig_win"].fillna(0) >= 20].copy()
    base = df[~df.is_fade].copy()
    base_f = filter_combined(base)
    combined = pd.concat([base_f, fade_f], ignore_index=True)
    combined["spread_cost"] = combined["instrument"].map(SPREAD)
    combined["slip_cost"] = np.where(combined["pnl_pts"] < 0,
                                       combined["instrument"].map(SLIP), 0)
    combined["pnl_net"] = combined["pnl_pts"] - combined["spread_cost"] - combined["slip_cost"]
    combined["year"] = pd.to_datetime(combined["date"]).dt.year

    print(f"\n  {label}")
    tot = combined['pnl_net']
    print(f"    Combined: n={len(combined):,} PF={pf(tot):.2f} net={tot.sum():+,.0f}")
    us30 = combined[combined.instrument == "US30"]
    big = us30[us30.pnl_net < -100]
    print(f"    US30:     n={len(us30):,} PF={pf(us30['pnl_net']):.2f} net={us30['pnl_net'].sum():+,.0f}  "
          f"big-loss(>-100): n={len(big)} sum={big['pnl_net'].sum():+,.0f}")
    tr = combined[combined.year <= 2017]; te = combined[combined.year >= 2018]
    print(f"    TRAIN PF={pf(tr['pnl_net']):.2f}  TEST PF={pf(te['pnl_net']):.2f}")


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)
    for max_range in [300, 250, 200, 150, 100]:
        t1 = time.time()
        print(f"\n{'='*80}\n  US30 max_bar_range = {max_range}\n{'='*80}")
        trades = run(cache, max_range)
        analyse(trades, f"max_range={max_range}")
        print(f"    ({time.time()-t1:.0f}s)")
    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
