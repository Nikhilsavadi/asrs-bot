"""
mc_filter_mfe_compare.py — MC drawdown comparison across 4 variants:
  A. Baseline candle
  B. Daily trend filter (bias_midhl) only
  C. MFE hybrid only
  D. MFE hybrid + daily trend filter (combined)

Save trade CSVs from existing backtest data + MFE simulation, then MC on each.
"""
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from pathlib import Path

V2 = "/root/asrs-bot/data/mfe_mc_baseline.csv"
MFE = "/root/asrs-bot/data/mfe_mc_hybrid_50_0.5.csv"
FIRSTRATE = {
    "DAX":    ("/root/asrs-bot/data/firstrate/FDAX_full_5min_continuous_ratio_adjusted.txt", "Europe/Berlin"),
    "US30":   ("/root/asrs-bot/data/firstrate/YM_full_5min_continuous_ratio_adjusted.txt",   "America/New_York"),
    "NIKKEI": ("/root/asrs-bot/data/firstrate/NKD_full_5min_continuous_ratio_adjusted.txt",  "America/New_York"),
}


def daily_bias_midhl(path, tz):
    df = pd.read_csv(path, header=None, names=["dt","O","H","L","C","V"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").tz_localize(ZoneInfo(tz))
    df = df[df.index.dayofweek < 5]
    df["_d"] = df.index.date
    g = df.groupby("_d").agg(H=("H","max"), L=("L","min"), C=("C","last"))
    g["prev_H"] = g["H"].shift(1)
    g["prev_L"] = g["L"].shift(1)
    g["prev_C"] = g["C"].shift(1)
    g["bias"] = np.where(g["prev_C"] > (g["prev_H"]+g["prev_L"])/2, "LONG", "SHORT")
    return g["bias"]


def apply_filter(trades_csv):
    df = pd.read_csv(trades_csv)
    df["date"] = pd.to_datetime(df["date"])
    df["date_d"] = df["date"].dt.date
    # Build bias maps
    bias_maps = {inst: daily_bias_midhl(path, tz) for inst, (path, tz) in FIRSTRATE.items()}
    # Attach bias per trade (vectorised)
    def lookup(row):
        bm = bias_maps.get(row["instrument"])
        if bm is None: return None
        return bm.get(row["date_d"])
    df["bias"] = df.apply(lookup, axis=1)
    # Keep only aligned trades
    aligned = df[df["direction"] == df["bias"]].copy()
    return aligned


def to_daily_pnl(df):
    return df.groupby(pd.to_datetime(df["date"]).dt.date)["pnl_pts"].sum()


def sim(daily_pts, rng, runs=3000, days=252*4, starting=5000, stake=0.5):
    vals = daily_pts.values
    eqs = []
    for _ in range(runs):
        sampled = rng.choice(vals, size=days, replace=True)
        eq = starting + np.cumsum(sampled * stake)
        eqs.append(eq)
    return np.array(eqs)


def report(eq, label):
    final = eq[:, -1]
    runmax = np.maximum.accumulate(eq, axis=1)
    dd = (runmax - eq) / runmax
    maxdd = dd.max(axis=1) * 100
    cagr = ((final / 5000) ** (1/4) - 1) * 100
    print(f"\n  {label}")
    print(f"    Final £:  p5={np.percentile(final,5):>9,.0f}  p50={np.percentile(final,50):>9,.0f}  p95={np.percentile(final,95):>9,.0f}")
    print(f"    CAGR %:   p5={np.percentile(cagr,5):>5.1f}  p50={np.percentile(cagr,50):>5.1f}  p95={np.percentile(cagr,95):>5.1f}")
    print(f"    Max DD%:  p5={np.percentile(maxdd,5):>5.1f}  p50={np.percentile(maxdd,50):>5.1f}  p95={np.percentile(maxdd,95):>5.1f}")
    return final, maxdd


def main():
    print("Loading and applying filter...")
    bl = pd.read_csv(V2)
    bl["date"] = pd.to_datetime(bl["date"])
    mfe = pd.read_csv(MFE)
    mfe["date"] = pd.to_datetime(mfe["date"])
    bl_filt = apply_filter(V2)
    mfe_filt = apply_filter(MFE)
    print(f"  baseline: {len(bl):,} → filtered: {len(bl_filt):,}")
    print(f"  MFE:      {len(mfe):,} → filtered: {len(mfe_filt):,}")

    d_bl     = to_daily_pnl(bl)
    d_mfe    = to_daily_pnl(mfe)
    d_bl_f   = to_daily_pnl(bl_filt)
    d_mfe_f  = to_daily_pnl(mfe_filt)

    print(f"\n  Daily pnl samples: baseline={len(d_bl)}, MFE={len(d_mfe)}, "
          f"bl_filt={len(d_bl_f)}, mfe_filt={len(d_mfe_f)}")

    rng = np.random.default_rng(42)
    print("\nRunning 3000 MC × 4yr × £0.50/pt flat stake...")
    eq_a = sim(d_bl,    rng)
    eq_b = sim(d_bl_f,  rng)
    eq_c = sim(d_mfe,   rng)
    eq_d = sim(d_mfe_f, rng)

    f_a, d_a = report(eq_a, "A. CANDLE baseline (current live)")
    f_b, d_b = report(eq_b, "B. CANDLE + trend filter")
    f_c, d_c = report(eq_c, "C. MFE hybrid (no filter)")
    f_d, d_d = report(eq_d, "D. MFE hybrid + trend filter (combined)")

    print("\n  HEAD-TO-HEAD vs A (baseline):")
    for lbl, f, d in [("B", f_b, d_b), ("C", f_c, d_c), ("D", f_d, d_d)]:
        f50 = np.percentile(f, 50); a50 = np.percentile(f_a, 50)
        d50 = np.percentile(d, 50); da50 = np.percentile(d_a, 50)
        print(f"    {lbl}: Final £ Δ = £{f50 - a50:>+8,.0f} ({(f50/a50-1)*100:+5.1f}%)  "
              f"DD Δ = {d50 - da50:+5.2f}pp")


if __name__ == "__main__":
    main()
