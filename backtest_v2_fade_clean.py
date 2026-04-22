"""
backtest_v2_fade_clean.py — clean fade test + MC + stress.

1. Base strategy (Variant B + COMBINED) — anchor
2. Fade tagged on separate signal so filter_combined doesn't drop either
3. Combined stats
4. MC drawdown: base / fade / combined
5. Stress: outlier concentration, win-flip, per-year, regime splits
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR, filter_combined, stats
from backtest_v2 import _find_real_reentry_in_1min
from backtest_v2_fade import sim_with_fade, build_cache

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

FADE_STOP = 50
FADE_TARGET = "extreme"


def pf(pnl):
    if len(pnl) == 0: return 0
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def run_split(cache, fade_stop, fade_target_mode):
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
                                    fade_stop=fade_stop, fade_target_mode=fade_target_mode)
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst
                    if t.get("is_fade"):
                        t["signal"] = f"{name}_FADE"
                    else:
                        t["signal"] = name
                    all_trades.append(t)
    df = pd.DataFrame(all_trades)
    if "is_fade" not in df.columns:
        df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False)
    base = df[~df["is_fade"]].copy()
    fade = df[df["is_fade"]].copy()
    base_f = filter_combined(base)
    fade["year"] = pd.to_datetime(fade["date"]).dt.year
    base_f["year"] = pd.to_datetime(base_f["date"]).dt.year
    return base_f, fade


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


def mc_report(label, final, max_dd, start=5000):
    prob_loss = (final < start).mean() * 100
    print(f"  {label:<25} p5=£{np.percentile(final,5):>8,.0f} "
          f"p50=£{np.percentile(final,50):>8,.0f} "
          f"p95=£{np.percentile(final,95):>8,.0f}  "
          f"DD p50={np.percentile(max_dd,50):>5.2f}% p95={np.percentile(max_dd,95):>5.2f}%  "
          f"P(loss)={prob_loss:.1f}%")


def outlier_sweep(d, label):
    print(f"\n  {label} outlier concentration (remove top X% wins):")
    print(f"    baseline: n={len(d):,} PF={pf(d['pnl_pts']):.2f} net={d.pnl_pts.sum():+,.0f}")
    for p_pct in [1, 5, 10]:
        k = int(len(d) * p_pct / 100)
        d_sorted = d.sort_values("pnl_pts", ascending=False)
        d_no_top = d_sorted.iloc[k:]
        d_both = d_sorted.iloc[k:-k] if k > 0 else d_sorted
        print(f"    -{p_pct}% top:  PF={pf(d_no_top['pnl_pts']):>5.2f}  net={d_no_top.pnl_pts.sum():>+9,.0f}"
              f"   |  -{p_pct}% both: PF={pf(d_both['pnl_pts']):>5.2f}  net={d_both.pnl_pts.sum():>+9,.0f}")


def win_flip(d, label):
    print(f"\n  {label} win-flip robustness (200 runs):")
    rng = np.random.default_rng(42)
    for flip_pct in [5, 10, 20]:
        pfs = []
        for _ in range(200):
            d2 = d.copy()
            winners = d2.index[d2["pnl_pts"] > 0].tolist()
            k = int(len(winners) * flip_pct / 100)
            if k > 0:
                flip_idx = rng.choice(winners, size=k, replace=False)
                d2.loc[flip_idx, "pnl_pts"] = -d2.loc[flip_idx, "pnl_pts"]
            pfs.append(pf(d2["pnl_pts"]))
        pfs = np.array(pfs)
        print(f"    flip {flip_pct:>3}%: p5 PF={np.percentile(pfs,5):.2f}  "
              f"p50 PF={np.percentile(pfs,50):.2f}  p95 PF={np.percentile(pfs,95):.2f}")


def per_year(d, label):
    print(f"\n  {label} per-year consistency:")
    print(f"    {'year':<6} {'n':<7} {'PF':<7} {'net':<10} {'WR':<8}")
    losing = 0
    for y in sorted(d["year"].unique()):
        dy = d[d["year"] == y]
        wr = (dy["pnl_pts"] > 0).mean() * 100
        net = dy.pnl_pts.sum()
        if net < 0: losing += 1
        print(f"    {y:<6} {len(dy):<7,} {pf(dy['pnl_pts']):.2f}   "
              f"{net:>+8,.0f}   {wr:>5.1f}%")
    print(f"    → Losing years: {losing}/{len(d['year'].unique())}")


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n", flush=True)

    print(f"--- FADE {FADE_STOP}/{FADE_TARGET} (clean run) ---", flush=True)
    t1 = time.time()
    base_f, fade = run_split(cache, FADE_STOP, FADE_TARGET)
    print(f"  base: {len(base_f):,} | fade: {len(fade):,}  ({time.time()-t1:.0f}s)")

    combined = pd.concat([base_f, fade], ignore_index=True)

    print(f"\n{'='*100}\n  HEADLINE: BASE vs FADE vs COMBINED\n{'='*100}")
    stats(base_f,   "  BASE alone")
    stats(fade,     "  FADE alone")
    stats(combined, "  COMBINED")

    print(f"\n  Per-instrument:")
    for inst in ["DAX", "US30", "NIKKEI"]:
        db = base_f[base_f.instrument == inst]
        df_ = fade[fade.instrument == inst]
        dc = combined[combined.instrument == inst]
        print(f"    {inst:<8}  base n={len(db):>5} PF={pf(db.pnl_pts):.2f} net={db.pnl_pts.sum():>+8,.0f}"
              f"  |  fade n={len(df_):>5} PF={pf(df_.pnl_pts):.2f} net={df_.pnl_pts.sum():>+8,.0f}"
              f"  |  combined PF={pf(dc.pnl_pts):.2f} net={dc.pnl_pts.sum():>+8,.0f}")

    print(f"\n  Train/Test:")
    stats(combined[combined.year <= 2017], "    TRAIN 2008-17")
    stats(combined[combined.year >= 2018], "    TEST  2018-26")

    print(f"\n{'='*100}\n  MC DRAWDOWN (3000 × 4yr × £0.50/pt flat)\n{'='*100}")
    rng = np.random.default_rng(42)
    b_daily = base_f.groupby(pd.to_datetime(base_f["date"]).dt.date)["pnl_pts"].sum()
    f_daily = fade.groupby(pd.to_datetime(fade["date"]).dt.date)["pnl_pts"].sum()
    c_daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_pts"].sum()
    for lbl, daily in [("BASE alone", b_daily), ("FADE alone", f_daily), ("COMBINED", c_daily)]:
        final, maxdd = mc_sim(daily, rng)
        mc_report(lbl, final, maxdd)

    print(f"\n{'='*100}\n  STRESS TESTS (on COMBINED system)\n{'='*100}")
    outlier_sweep(combined, "COMBINED")
    win_flip(combined, "COMBINED")
    per_year(combined, "COMBINED")

    print(f"\n{'='*100}\n  CANARY DATA: FADE alone per-instrument\n{'='*100}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = fade[fade.instrument == inst]
        if len(d) == 0: continue
        print(f"\n  {inst} fade:")
        stats(d, "    FULL")
        stats(d[d.year <= 2017], "    TRAIN 2008-17")
        stats(d[d.year >= 2018], "    TEST  2018-26")
        losing = sum(1 for y in sorted(d["year"].unique())
                      if d[d["year"] == y].pnl_pts.sum() < 0)
        print(f"    Losing years: {losing}/{len(d['year'].unique())}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
