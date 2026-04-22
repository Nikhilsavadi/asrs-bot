"""
Re-run monthly + annual MC using expanded parquet (S1+S2+S3).
Compares vs S1+S2-only to show S3 uplift clearly.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
UPLIFT = 1.06

START = 2000
MIN_STAKE = 0.50
MAX_STAKE = 35.0
RUNS = 5000

PARQUETS = {
    "S1+S2 only (old)": "/root/asrs-bot/data/variant_b_fade_trades.parquet",
    "S1+S2+S3 (new)": "/root/asrs-bot/data/variant_b_fade_trades_s3.parquet",
}


def fmt(x):
    if x < 1000: return f"£{x:,.0f}"
    if x < 1_000_000: return f"£{x/1000:.1f}k"
    return f"£{x/1_000_000:.2f}M"


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def build_daily(path):
    df = pd.read_parquet(path)
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
    base_f = apply_friction(base_f)
    fade_f = apply_friction(fade_f)
    combined = pd.concat([base_f, fade_f], ignore_index=True)
    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_net"].sum().values
    return daily, combined


def sim(daily, seed, days, stress=1.0, start=START):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily, size=days, replace=True) * stress
    eq = np.empty(days + 1)
    eq[0] = start
    for i in range(days):
        stake = max(MIN_STAKE, min(MAX_STAKE, eq[i] * 1e-4))
        eq[i+1] = eq[i] + sample[i] * stake
        if eq[i+1] < 0:
            eq[i+1:] = 0; break
    return eq[1:]


def mc(daily, days, stress=1.0, start=START):
    eqs = np.empty((RUNS, days))
    for r in range(RUNS):
        eqs[r] = sim(daily, seed=r, days=days, stress=stress, start=start)
    return eqs


def main():
    for label, path in PARQUETS.items():
        daily, combined = build_daily(path)
        pf = combined[combined.pnl_net > 0].pnl_net.sum() / abs(combined[combined.pnl_net < 0].pnl_net.sum())
        print(f"\n{'='*100}")
        print(f"  {label}")
        print(f"{'='*100}")
        print(f"  post-friction: n={len(combined):,} PF={pf:.2f} "
              f"daily mean={daily.mean():+.1f}pt worst={daily.min():+.0f} best={daily.max():+.0f}")

        # Year 1 monthly (unstressed + stressed)
        print(f"\n  YEAR 1 MONTHLY (single account, compound 1e-4)")
        print(f"  {'Month':<6} {'WORST p5':<11} {'MEDIAN p50':<11} {'BEST p95':<11} "
              f"|| {'S-WORST p5':<11} {'S-MED p50':<11} {'S-BEST p95':<11}")
        eqs_u = mc(daily, 252, stress=1.0)
        eqs_s = mc(daily, 252, stress=0.5)
        for m in [1, 3, 6, 9, 12]:
            d = m * 21 - 1
            u = eqs_u[:, d]; s = eqs_s[:, d]
            print(f"  M{m:<5} {fmt(np.percentile(u,5)):<11} {fmt(np.percentile(u,50)):<11} {fmt(np.percentile(u,95)):<11} "
                  f"|| {fmt(np.percentile(s,5)):<11} {fmt(np.percentile(s,50)):<11} {fmt(np.percentile(s,95)):<11}")

        # 4yr single + dual (compound 1e-4)
        eqs_4y_u = mc(daily, 252*4, stress=1.0)
        eqs_4y_s = mc(daily, 252*4, stress=0.5)
        print(f"\n  4YEAR SINGLE (compound 1e-4)")
        print(f"  {'Stress':<6} {'Y1 p50':<10} {'Y2 p50':<10} {'Y3 p50':<10} {'Y4 p50':<10} {'Y4 p5':<10}")
        print(f"  {'none':<6} {fmt(np.percentile(eqs_4y_u[:,251],50)):<10} {fmt(np.percentile(eqs_4y_u[:,503],50)):<10} "
              f"{fmt(np.percentile(eqs_4y_u[:,755],50)):<10} {fmt(np.percentile(eqs_4y_u[:,-1],50)):<10} "
              f"{fmt(np.percentile(eqs_4y_u[:,-1],5)):<10}")
        print(f"  {'50%':<6} {fmt(np.percentile(eqs_4y_s[:,251],50)):<10} {fmt(np.percentile(eqs_4y_s[:,503],50)):<10} "
              f"{fmt(np.percentile(eqs_4y_s[:,755],50)):<10} {fmt(np.percentile(eqs_4y_s[:,-1],50)):<10} "
              f"{fmt(np.percentile(eqs_4y_s[:,-1],5)):<10}")

        # Dual: main £2k + wife £10k seed at Y2, 50% stress
        def sim_dual(daily, seed, days, stress, seed_day=252, seed_amt=10000):
            rng = np.random.default_rng(seed)
            samp_m = rng.choice(daily, size=days, replace=True) * stress
            rng2 = np.random.default_rng(seed + 100_000)
            samp_w = rng2.choice(daily, size=days, replace=True) * stress
            m_eq = np.empty(days); w_eq = np.empty(days)
            main_, wife = START, 0.0
            for i in range(days):
                if i == seed_day:
                    main_ = max(0, main_ - seed_amt); wife = seed_amt
                if main_ > 0:
                    stk = max(MIN_STAKE, min(MAX_STAKE, main_ * 1e-4))
                    main_ = max(0, main_ + samp_m[i] * stk)
                if i >= seed_day and wife > 0:
                    stk = max(MIN_STAKE, min(MAX_STAKE, wife * 1e-4))
                    wife = max(0, wife + samp_w[i] * stk)
                m_eq[i] = main_; w_eq[i] = wife
            return m_eq + w_eq

        total_s = np.array([sim_dual(daily, r, 252*4, 0.5) for r in range(RUNS)])
        print(f"\n  4YEAR DUAL (main £2k + wife £10k Y2, 50% stress)")
        print(f"  Y1 p50={fmt(np.percentile(total_s[:,251],50))} "
              f"Y2 p50={fmt(np.percentile(total_s[:,503],50))} "
              f"Y3 p50={fmt(np.percentile(total_s[:,755],50))} "
              f"Y4 p50={fmt(np.percentile(total_s[:,-1],50))} "
              f"Y4 p5={fmt(np.percentile(total_s[:,-1],5))}")


if __name__ == "__main__":
    main()
