"""
MC from £2k start — per-year p5/p50/p95 equity bands.
Single account, all 3 scaling modes. Uses same post-friction combined parquet.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"
UPLIFT = 1.06

START = 2000
MIN_STAKE = 0.50
MAX_STAKE = 35.0
DAYS_PER_YEAR = 252
YEARS = 4
DAYS = YEARS * DAYS_PER_YEAR
RUNS = 5000


def fmt(x):
    if x <= 0: return "£0"
    if x < 1000: return f"£{x:.0f}"
    if x < 1_000_000: return f"£{x/1000:.1f}k"
    return f"£{x/1_000_000:.2f}M"


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def simulate(daily_pts, mode, factor, seed):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily_pts, size=DAYS, replace=True)
    eq = np.empty(DAYS + 1)
    eq[0] = START
    for i in range(DAYS):
        stake = MIN_STAKE if mode == "flat" else max(MIN_STAKE, min(MAX_STAKE, eq[i] * factor))
        eq[i+1] = eq[i] + sample[i] * stake
        if eq[i+1] < 0:
            eq[i+1:] = 0; break
    return eq[1:]


def build_daily():
    df = pd.read_parquet(TRADES_CACHE)
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


def mc(daily, mode, factor):
    eqs = np.empty((RUNS, DAYS))
    for r in range(RUNS):
        eqs[r] = simulate(daily, mode, factor, seed=r)
    return eqs


def main():
    daily, combined = build_daily()
    pf_net = combined[combined.pnl_net > 0].pnl_net.sum() / abs(combined[combined.pnl_net < 0].pnl_net.sum())
    print(f"Post-friction combined: n={len(combined):,}  PF={pf_net:.2f}  daily mean={daily.mean():+.1f}pt  worst={daily.min():+.0f}pt  best={daily.max():+.0f}pt\n")

    scenarios = [
        ("A flat £0.50/pt (no scale)",  "flat",     None),
        ("B compound 1e-4 (default)",   "compound", 1e-4),
        ("C conservative 5e-5",         "compound", 5e-5),
    ]

    print(f"{'='*108}")
    print(f"  START £{START:,} — per-year equity bands (p5 WORST / p50 MEDIAN / p95 BEST), {RUNS} runs, cap £{MAX_STAKE}/pt")
    print(f"{'='*108}\n")

    for lbl, mode, factor in scenarios:
        eqs = mc(daily, mode, factor)
        print(f"  {lbl}")
        print(f"  {'Year':<6} {'WORST p5':<14} {'MEDIAN p50':<14} {'BEST p95':<14} {'p50 max-DD':<12}  {'P(below start)':<14}")
        for y in range(1, YEARS + 1):
            d_idx = y * DAYS_PER_YEAR - 1
            slice_eq = eqs[:, :d_idx+1]
            eq_at_year = eqs[:, d_idx]
            runmax = np.maximum.accumulate(slice_eq, axis=1)
            dd_max = ((runmax - slice_eq) / np.maximum(runmax, 1e-9) * 100).max(axis=1)
            p_below = (eq_at_year < START).mean() * 100
            print(f"  Y{y:<5} {fmt(np.percentile(eq_at_year,5)):<14} "
                  f"{fmt(np.percentile(eq_at_year,50)):<14} "
                  f"{fmt(np.percentile(eq_at_year,95)):<14} "
                  f"{np.percentile(dd_max,50):>5.1f}%      "
                  f"{p_below:>5.1f}%")
        final = eqs[:, -1]
        p_ruin = (final <= 500).mean() * 100
        print(f"  → 4yr P(below £500)={p_ruin:.2f}%\n")


if __name__ == "__main__":
    main()
