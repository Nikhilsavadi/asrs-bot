"""
Full spread-adjusted analysis using REAL IG spreads measured from tick data.
Spreads: DAX 1.5pt, US30 2.75pt, NIKKEI 7.0pt.

Produces: base/fade/combined PF + MC + stress, all post-spread.
Caches trades to parquet for faster reruns.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"

import backtest as bt
from backtest_v2_atr import filter_combined, stats
from backtest_v2_fade import build_cache, sim_with_fade

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def pf(pnl):
    if len(pnl) == 0: return 0
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def apply_spread(df):
    df = df.copy()
    df["pnl_pts_net"] = df["pnl_pts"] - df["instrument"].map(SPREAD)
    return df


def mc_sim(daily, rng, runs=3000, days=1008, start=5000, stake=0.5):
    vals = daily.values
    eqs = np.empty((runs, days))
    for i in range(runs):
        sample = rng.choice(vals, size=days, replace=True)
        eqs[i] = start + np.cumsum(sample * stake)
    final = eqs[:, -1]
    runmax = np.maximum.accumulate(eqs, axis=1)
    dd = (runmax - eqs) / np.maximum(runmax, 1e-9) * 100
    return final, dd.max(axis=1)


def mc_report(label, final, dd, start=5000):
    prob_loss = (final < start).mean() * 100
    print(f"  {label:<25} p5=£{np.percentile(final,5):>8,.0f} "
          f"p50=£{np.percentile(final,50):>8,.0f} "
          f"p95=£{np.percentile(final,95):>8,.0f}  "
          f"DD p50={np.percentile(dd,50):>5.2f}% p95={np.percentile(dd,95):>5.2f}%  "
          f"P(loss)={prob_loss:.1f}%")


def outlier_sweep(d, col, label):
    print(f"\n  {label}:")
    print(f"    baseline: n={len(d):,} PF={pf(d[col]):.2f} net={d[col].sum():+,.0f}")
    for p_pct in [1, 5, 10]:
        k = int(len(d) * p_pct / 100)
        ds = d.sort_values(col, ascending=False)
        d_top = ds.iloc[k:]
        d_both = ds.iloc[k:-k] if k > 0 else ds
        print(f"    -{p_pct}% top:  PF={pf(d_top[col]):>5.2f}  net={d_top[col].sum():>+9,.0f}"
              f"   |  -{p_pct}% both: PF={pf(d_both[col]):>5.2f}  net={d_both[col].sum():>+9,.0f}")


def win_flip(d, col, label):
    print(f"\n  {label} win-flip (200 runs):")
    rng = np.random.default_rng(42)
    for flip_pct in [5, 10, 20]:
        pfs = []
        for _ in range(200):
            d2 = d.copy()
            winners = d2.index[d2[col] > 0].tolist()
            k = int(len(winners) * flip_pct / 100)
            if k > 0:
                flip_idx = rng.choice(winners, size=k, replace=False)
                d2.loc[flip_idx, col] = -d2.loc[flip_idx, col]
            pfs.append(pf(d2[col]))
        pfs = np.array(pfs)
        print(f"    flip {flip_pct:>3}%: p5 PF={np.percentile(pfs,5):.2f}  "
              f"p50 PF={np.percentile(pfs,50):.2f}  p95 PF={np.percentile(pfs,95):.2f}")


def per_year_table(d, col, label):
    print(f"\n  {label} per-year (post-spread):")
    print(f"    {'year':<6} {'n':<7} {'PF':<7} {'net':<10} {'WR':<8}")
    d = d.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    losing = 0
    for y in sorted(d["year"].unique()):
        dy = d[d["year"] == y]
        wr = (dy[col] > 0).mean() * 100
        net = dy[col].sum()
        if net < 0: losing += 1
        print(f"    {y:<6} {len(dy):<7,} {pf(dy[col]):.2f}   "
              f"{net:>+8,.0f}   {wr:>5.1f}%")
    print(f"    → Losing years: {losing}/{len(d['year'].unique())}")


def build_or_load_trades():
    if os.path.exists(TRADES_CACHE):
        print(f"Loading trades from cache: {TRADES_CACHE}", flush=True)
        df = pd.read_parquet(TRADES_CACHE)
        return df
    print("Building trades (cache miss) — this takes ~11 min...", flush=True)
    t0 = time.time()
    cache = build_cache()
    print(f"Cache built in {time.time()-t0:.0f}s. Simulating...", flush=True)
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
    df.to_parquet(TRADES_CACHE)
    print(f"Cached {len(df):,} trades → {TRADES_CACHE}")
    return df


def main():
    t0 = time.time()
    df = build_or_load_trades()

    base = df[~df["is_fade"]].copy()
    fade = df[df["is_fade"]].copy()
    base_f = filter_combined(base)
    combined = pd.concat([base_f, fade], ignore_index=True)

    # Apply spread
    base_f = apply_spread(base_f)
    fade = apply_spread(fade)
    combined = apply_spread(combined)

    print(f"\n{'='*100}")
    print(f"  SPREAD-ADJUSTED HEADLINE  (DAX 1.5 / US30 2.75 / NIKKEI 7.0 pt)")
    print(f"{'='*100}")
    print(f"  {'dataset':<20} {'n':<8} {'raw PF':<9} {'raw net':<12} {'net PF':<9} {'net net':<12}")
    for lbl, d in [("BASE alone", base_f), ("FADE alone", fade), ("COMBINED", combined)]:
        raw = pf(d["pnl_pts"]); adj = pf(d["pnl_pts_net"])
        print(f"  {lbl:<20} {len(d):<8,} {raw:<6.2f}    "
              f"{d['pnl_pts'].sum():>+9,.0f}    {adj:<6.2f}    "
              f"{d['pnl_pts_net'].sum():>+9,.0f}")

    print(f"\n  Per-instrument (post-spread):")
    for inst in ["DAX", "US30", "NIKKEI"]:
        db = base_f[base_f.instrument == inst]
        dfd = fade[fade.instrument == inst]
        dc = combined[combined.instrument == inst]
        print(f"    {inst:<8}  BASE n={len(db):>5,} PF={pf(db['pnl_pts_net']):.2f} net={db['pnl_pts_net'].sum():>+8,.0f}"
              f"  |  FADE n={len(dfd):>5,} PF={pf(dfd['pnl_pts_net']):.2f} net={dfd['pnl_pts_net'].sum():>+8,.0f}"
              f"  |  COMBINED PF={pf(dc['pnl_pts_net']):.2f} net={dc['pnl_pts_net'].sum():>+8,.0f}")

    # Train/Test
    combined["year"] = pd.to_datetime(combined["date"]).dt.year
    print(f"\n  Train/Test (post-spread):")
    train = combined[combined.year <= 2017]
    test = combined[combined.year >= 2018]
    for lbl, d in [("TRAIN 2008-17", train), ("TEST 2018-26", test)]:
        print(f"    {lbl:<18}  n={len(d):<7,} PF={pf(d['pnl_pts_net']):.2f} "
              f"net={d['pnl_pts_net'].sum():>+9,.0f} "
              f"WR={(d['pnl_pts_net']>0).mean()*100:.1f}%")

    # MC
    print(f"\n{'='*100}\n  MC DRAWDOWN (3000 × 4yr × £0.50/pt, POST-SPREAD)\n{'='*100}")
    rng = np.random.default_rng(42)
    for lbl, d in [("BASE alone", base_f), ("FADE alone", fade), ("COMBINED", combined)]:
        daily = d.groupby(pd.to_datetime(d["date"]).dt.date)["pnl_pts_net"].sum()
        final, dd = mc_sim(daily, rng)
        mc_report(lbl, final, dd)

    # Stress
    print(f"\n{'='*100}\n  STRESS (combined, post-spread)\n{'='*100}")
    outlier_sweep(combined, "pnl_pts_net", "Outlier concentration")
    win_flip(combined, "pnl_pts_net", "Win-flip")
    per_year_table(combined, "pnl_pts_net", "Combined per-year")

    # Canary per-instrument fade (post-spread)
    print(f"\n{'='*100}\n  FADE per-instrument POST-SPREAD (canary data)\n{'='*100}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = fade[fade.instrument == inst].copy()
        if len(d) == 0: continue
        d["year"] = pd.to_datetime(d["date"]).dt.year
        print(f"\n  {inst} fade:")
        print(f"    FULL       n={len(d):<6,} PF={pf(d['pnl_pts_net']):.2f} "
              f"net={d['pnl_pts_net'].sum():>+8,.0f}")
        tr = d[d.year <= 2017]; te = d[d.year >= 2018]
        print(f"    TRAIN 08-17 n={len(tr):<6,} PF={pf(tr['pnl_pts_net']):.2f} "
              f"net={tr['pnl_pts_net'].sum():>+8,.0f}")
        print(f"    TEST  18-26 n={len(te):<6,} PF={pf(te['pnl_pts_net']):.2f} "
              f"net={te['pnl_pts_net'].sum():>+8,.0f}")
        losing = sum(1 for y in sorted(d["year"].unique())
                      if d[d["year"] == y]["pnl_pts_net"].sum() < 0)
        print(f"    Losing years: {losing}/{len(d['year'].unique())}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
