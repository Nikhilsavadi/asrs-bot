"""
stress_test_variant_b.py — comprehensive robustness sweep on LIVE config.

LIVE config = Variant B trail + COMBINED re-entry filter + hybrid bar 4/5.

Tests:
  1. Outlier concentration (remove top N%, bottom N% of trades)
  2. Win-flip stress (random X% winners → losers)
  3. Win-rate haircut + loss amplifier
  4. Per-year PF consistency
  5. Longest losing streak + max trade-level DD
  6. MC bootstrap (3000 × 4yr)
  7. Regime splits (5-year blocks)
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import build_cache, run, filter_combined

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def pf(d):
    if len(d) == 0: return 0
    w = d[d["pnl_pts"] > 0]["pnl_pts"].sum()
    l = abs(d[d["pnl_pts"] < 0]["pnl_pts"].sum())
    return w / max(l, 0.001)


def run_and_save():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s", flush=True)
    trades = filter_combined(pd.DataFrame(run(cache, trail_close=True, mfe_mode=False)))
    trades["year"] = pd.to_datetime(trades["date"]).dt.year
    trades["date_dt"] = pd.to_datetime(trades["date"])
    return trades


def outlier_concentration(d):
    print(f"\n{'='*80}\n  1. OUTLIER CONCENTRATION\n{'='*80}")
    base_pf = pf(d); base_net = d["pnl_pts"].sum()
    print(f"  Baseline: n={len(d):,} PF={base_pf:.2f} net={base_net:+,.0f}")
    print(f"\n  {'pct removed':<15} {'top-wins-only':<22} {'bottom-losses-only':<22} {'both tails':<22}")
    for pct in [0.5, 1, 2, 5, 10]:
        k = int(len(d) * pct / 100)
        # Remove top K winners
        d_sorted = d.sort_values("pnl_pts", ascending=False)
        d_no_top = d_sorted.iloc[k:]
        # Remove bottom K losers (biggest losses)
        d_no_bot = d_sorted.iloc[:-k] if k > 0 else d_sorted
        # Remove both tails
        d_both = d_sorted.iloc[k:-k] if k > 0 else d_sorted
        print(f"  {pct:>3.1f}% ({k:>4}):    "
              f"PF={pf(d_no_top):>5.2f} net={d_no_top.pnl_pts.sum():>+8,.0f}  "
              f"PF={pf(d_no_bot):>5.2f} net={d_no_bot.pnl_pts.sum():>+8,.0f}  "
              f"PF={pf(d_both):>5.2f} net={d_both.pnl_pts.sum():>+8,.0f}")


def flip_wins(d):
    print(f"\n{'='*80}\n  2. WIN-FLIP STRESS (winner → mirrored loser)\n{'='*80}")
    rng = np.random.default_rng(42)
    print(f"  {'flip %':<10} {'runs':<5} {'p5 PF':<10} {'p50 PF':<10} {'p95 PF':<10} {'p50 net':<12}")
    for flip_pct in [0, 2, 5, 10, 20]:
        pfs = []; nets = []
        runs = 200
        for _ in range(runs):
            d2 = d.copy()
            winners = d2.index[d2["pnl_pts"] > 0].tolist()
            k = int(len(winners) * flip_pct / 100)
            flip_idx = rng.choice(winners, size=k, replace=False) if k > 0 else []
            d2.loc[flip_idx, "pnl_pts"] = -d2.loc[flip_idx, "pnl_pts"]
            pfs.append(pf(d2)); nets.append(d2.pnl_pts.sum())
        pfs = np.array(pfs); nets = np.array(nets)
        print(f"  {flip_pct:>3}%        {runs:<5} "
              f"{np.percentile(pfs,5):>5.2f}     {np.percentile(pfs,50):>5.2f}     "
              f"{np.percentile(pfs,95):>5.2f}     {np.percentile(nets,50):>+8,.0f}")


def degradation(d):
    print(f"\n{'='*80}\n  3. WIN-RATE HAIRCUT + LOSS AMPLIFIER\n{'='*80}")
    rng = np.random.default_rng(42)
    print(f"  {'haircut%':<12} {'loss_mult':<12} {'PF':<8} {'net':<12} {'WR':<8}")
    for haircut in [0, 5, 10, 15, 20]:
        for mult in [1.0, 1.1, 1.2, 1.3]:
            d2 = d.copy()
            # Randomly turn X% of winners into 0 (haircut)
            winners = d2.index[d2["pnl_pts"] > 0].tolist()
            k = int(len(winners) * haircut / 100)
            if k > 0:
                zero_idx = rng.choice(winners, size=k, replace=False)
                d2.loc[zero_idx, "pnl_pts"] = 0
            # Amplify losses
            loss_mask = d2["pnl_pts"] < 0
            d2.loc[loss_mask, "pnl_pts"] *= mult
            wr = (d2["pnl_pts"] > 0).mean() * 100
            print(f"  {haircut:>3}%         {mult:>3.1f}x         "
                  f"{pf(d2):>5.2f}   {d2.pnl_pts.sum():>+8,.0f}   {wr:>5.1f}%")


def per_year(d):
    print(f"\n{'='*80}\n  4. PER-YEAR CONSISTENCY\n{'='*80}")
    print(f"  {'year':<6} {'n':<7} {'PF':<8} {'net':<12} {'WR':<8} {'best':<8} {'worst':<8}")
    losing_years = 0
    for y in sorted(d["year"].unique()):
        dy = d[d["year"] == y]
        wr = (dy["pnl_pts"] > 0).mean() * 100
        net = dy.pnl_pts.sum()
        if net < 0: losing_years += 1
        print(f"  {y:<6} {len(dy):<7,} {pf(dy):<5.2f}    "
              f"{net:>+8,.0f}    {wr:>5.1f}%   "
              f"{dy.pnl_pts.max():>+6.1f}  {dy.pnl_pts.min():>+6.1f}")
    print(f"  → Losing years: {losing_years}/{len(d['year'].unique())}")


def streak_and_dd(d):
    print(f"\n{'='*80}\n  5. LOSING STREAK + TRADE-LEVEL MAX DD\n{'='*80}")
    # Sort by date+trade order
    d2 = d.sort_values(["date_dt", "signal"]).reset_index(drop=True)
    streak = 0; max_streak = 0
    for v in d2["pnl_pts"]:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        elif v > 0:
            streak = 0
    # Equity curve
    eq = d2["pnl_pts"].cumsum().values
    runmax = np.maximum.accumulate(eq)
    dd = runmax - eq
    max_dd_pts = dd.max()
    max_dd_idx = dd.argmax()
    peak_idx = runmax[:max_dd_idx].argmax() if max_dd_idx > 0 else 0
    recovery_idx = np.where(eq[max_dd_idx:] >= runmax[max_dd_idx])[0]
    recovery_bars = recovery_idx[0] if len(recovery_idx) > 0 else None

    print(f"  Longest losing streak: {max_streak} trades")
    print(f"  Max DD (points): {max_dd_pts:,.0f}pt")
    print(f"  DD peak date: {d2.iloc[peak_idx]['date']}")
    print(f"  DD trough date: {d2.iloc[max_dd_idx]['date']}")
    if recovery_bars is not None:
        print(f"  Trades to recover: {recovery_bars}")
    else:
        print(f"  Trades to recover: never fully recovered in sample")

    # Worst N consecutive days
    daily = d2.groupby(d2["date_dt"].dt.date)["pnl_pts"].sum()
    print(f"\n  Worst single day: {daily.min():+,.0f}pt on {daily.idxmin()}")
    print(f"  Worst 5-day window: {daily.rolling(5).sum().min():+,.0f}pt")
    print(f"  Worst 20-day window: {daily.rolling(20).sum().min():+,.0f}pt")
    print(f"  Worst 60-day window: {daily.rolling(60).sum().min():+,.0f}pt")


def mc_bootstrap(d, stake=0.5):
    print(f"\n{'='*80}\n  6. MC BOOTSTRAP (3000 × 4yr × £{stake}/pt)\n{'='*80}")
    daily = d.groupby(d["date_dt"].dt.date)["pnl_pts"].sum()
    rng = np.random.default_rng(42)
    runs = 3000; days = 1008; start = 5000
    eqs = np.empty((runs, days))
    for i in range(runs):
        sample = rng.choice(daily.values, size=days, replace=True)
        eqs[i] = start + np.cumsum(sample * stake)
    final = eqs[:, -1]
    runmax = np.maximum.accumulate(eqs, axis=1)
    dd_pct = ((runmax - eqs) / runmax).max(axis=1) * 100
    print(f"  {'p':<4} {'final £':<12} {'max DD %':<10}")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  p{p:<3} £{np.percentile(final,p):>9,.0f}   {np.percentile(dd_pct,p):>5.2f}%")
    # Probability of loss / big DD
    prob_loss = (final < start).mean() * 100
    prob_dd_10 = (dd_pct > 10).mean() * 100
    prob_dd_25 = (dd_pct > 25).mean() * 100
    print(f"  P(final < start): {prob_loss:.1f}%")
    print(f"  P(max DD > 10%): {prob_dd_10:.1f}%")
    print(f"  P(max DD > 25%): {prob_dd_25:.1f}%")


def regime_splits(d):
    print(f"\n{'='*80}\n  7. REGIME SPLITS (5-year blocks)\n{'='*80}")
    blocks = [(2008, 2012, "2008-12 GFC+recovery"),
              (2013, 2017, "2013-17 bull"),
              (2018, 2022, "2018-22 COVID+war"),
              (2023, 2026, "2023-26 AI+rates")]
    for lo, hi, lbl in blocks:
        db = d[(d["year"] >= lo) & (d["year"] <= hi)]
        if len(db) == 0: continue
        wr = (db["pnl_pts"] > 0).mean() * 100
        print(f"  {lbl:<25} n={len(db):>6,} PF={pf(db):>5.2f} net={db.pnl_pts.sum():>+8,.0f} WR={wr:>5.1f}%")


def main():
    t0 = time.time()
    print("Loading + running Variant B (live config) once...")
    d = run_and_save()
    print(f"\n  BASE: n={len(d):,}  PF={pf(d):.2f}  net={d.pnl_pts.sum():+,.0f}  WR={(d.pnl_pts>0).mean()*100:.1f}%")

    outlier_concentration(d)
    flip_wins(d)
    degradation(d)
    per_year(d)
    streak_and_dd(d)
    mc_bootstrap(d, stake=0.5)
    regime_splits(d)

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
