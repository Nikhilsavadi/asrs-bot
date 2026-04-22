"""
Year 1 month-by-month: p5/p50/p95 equity from £2k, single account.
Shows both unstressed and 50%-haircut stressed scenarios.
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
DAYS_PER_MONTH = 21
MONTHS = 12
DAYS = MONTHS * DAYS_PER_MONTH
RUNS = 5000


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


def simulate(daily_pts, seed, stress=1.0):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily_pts, size=DAYS, replace=True) * stress
    eq = np.empty(DAYS + 1)
    eq[0] = START
    stakes = np.empty(DAYS)
    for i in range(DAYS):
        stake = max(MIN_STAKE, min(MAX_STAKE, eq[i] * 1e-4))
        stakes[i] = stake
        eq[i+1] = eq[i] + sample[i] * stake
        if eq[i+1] < 0:
            eq[i+1:] = 0; break
    return eq[1:], stakes


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
    return daily


def run(stress_label, stress):
    daily = build_daily()
    print(f"\n  {stress_label} — daily post-friction mean={daily.mean()*stress:+.1f}pt, "
          f"worst={daily.min()*stress:+.0f}pt, best={daily.max()*stress:+.0f}pt")

    eqs = np.empty((RUNS, DAYS))
    stakes = np.empty((RUNS, DAYS))
    for r in range(RUNS):
        eqs[r], stakes[r] = simulate(daily, seed=r, stress=stress)

    print(f"  {'Month':<6} {'days':<6} {'WORST p5':<12} {'MEDIAN p50':<12} {'BEST p95':<12} "
          f"{'stake p50':<10} {'gain/mo p50':<12}")
    last_p50 = START
    for m in range(1, MONTHS + 1):
        d_idx = m * DAYS_PER_MONTH - 1
        eq_at = eqs[:, d_idx]
        p5 = np.percentile(eq_at, 5); p50 = np.percentile(eq_at, 50); p95 = np.percentile(eq_at, 95)
        stake_p50 = np.percentile(stakes[:, d_idx], 50)
        gain_p50 = p50 - last_p50
        print(f"  M{m:<5} {d_idx+1:<6} {fmt(p5):<12} {fmt(p50):<12} {fmt(p95):<12} "
              f"£{stake_p50:<8.2f} £{gain_p50:+,.0f}")
        last_p50 = p50


def main():
    print("="*100)
    print(f"  START £{START:,}  |  compound 1e-4 (shipped)  |  {RUNS} runs  |  21 trading days/mo")
    print("="*100)
    run("UNSTRESSED (18yr backtest repeats)", 1.0)
    run("REALISTIC STRESS (50% daily haircut)", 0.5)

    print("\n  NOTE: first 2-3 months are on MIN_STAKE £0.50/pt (compound formula still "
          "below floor at equity < £5k).\n  Growth is LINEAR during this phase. Compounding "
          "kicks in once equity exceeds ~£5-10k.")


if __name__ == "__main__":
    main()
