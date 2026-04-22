"""
MC dual account: main £2k start + wife £10k seeded from main's profits at start of Y2.
Per-year household equity bands.
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

MAIN_START = 2000
WIFE_SEED = 10_000          # transferred from main at start of Y2
WIFE_TRIGGER_DAY = 252      # day 252 = start of Y2

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


def simulate_path(daily_pts, start, mode, factor, seed,
                  seed_out_day=None, seed_out_amount=0,
                  seed_in_day=None, seed_in_amount=0,
                  start_trading_day=0):
    """Simulate one account path with optional transfer out (main) or in (wife).
    start_trading_day: days before which equity stays flat (wife waits for seed)."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily_pts, size=DAYS, replace=True)
    eq = np.empty(DAYS + 1)
    eq[0] = start
    for i in range(DAYS):
        if seed_out_day is not None and i == seed_out_day:
            eq[i] = max(0, eq[i] - seed_out_amount)
        if seed_in_day is not None and i == seed_in_day:
            eq[i] = eq[i] + seed_in_amount
        if i < start_trading_day:
            eq[i+1] = eq[i]
            continue
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
    return daily


def mc_dual(daily, mode, factor):
    main_eqs = np.empty((RUNS, DAYS))
    wife_eqs = np.empty((RUNS, DAYS))
    for r in range(RUNS):
        main_eqs[r] = simulate_path(
            daily, MAIN_START, mode, factor, seed=r,
            seed_out_day=WIFE_TRIGGER_DAY, seed_out_amount=WIFE_SEED
        )
        wife_eqs[r] = simulate_path(
            daily, 0, mode, factor, seed=r + 100_000,
            seed_in_day=WIFE_TRIGGER_DAY, seed_in_amount=WIFE_SEED,
            start_trading_day=WIFE_TRIGGER_DAY,
        )
    return main_eqs + wife_eqs, main_eqs, wife_eqs


def main():
    daily = build_daily()
    print(f"Daily samples: {len(daily)}  mean={daily.mean():+.1f}pt  worst={daily.min():+.0f}  best={daily.max():+.0f}\n")

    scenarios = [
        ("A flat £0.50/pt",            "flat",     None),
        ("B compound 1e-4 (shipped)",  "compound", 1e-4),
        ("C conservative 5e-5",        "compound", 5e-5),
    ]

    print(f"{'='*108}")
    print(f"  DUAL ACCOUNT — main £{MAIN_START:,} + wife £{WIFE_SEED:,} seed at start of Y2 (from main's profits)")
    print(f"  {RUNS} runs, cap £{MAX_STAKE}/pt each")
    print(f"{'='*108}\n")

    # Also run with 50% daily haircut
    daily_stressed = daily * 0.5

    for lbl, mode, factor in scenarios:
        total, m_eqs, w_eqs = mc_dual(daily, mode, factor)
        print(f"  {lbl}  (household combined equity)")
        print(f"  {'Year':<6} {'WORST p5':<14} {'MEDIAN p50':<14} {'BEST p95':<14} {'main p50':<12} {'wife p50':<12}")
        for y in range(1, YEARS + 1):
            d_idx = y * DAYS_PER_YEAR - 1
            tot = total[:, d_idx]
            m = m_eqs[:, d_idx]
            w = w_eqs[:, d_idx]
            print(f"  Y{y:<5} {fmt(np.percentile(tot,5)):<14} "
                  f"{fmt(np.percentile(tot,50)):<14} "
                  f"{fmt(np.percentile(tot,95)):<14} "
                  f"{fmt(np.percentile(m,50)):<12} "
                  f"{fmt(np.percentile(w,50)):<12}")
        final = total[:, -1]
        p_ruin = (final <= 1000).mean() * 100
        print(f"  → 4yr P(household < £1k)={p_ruin:.2f}%\n")

    print(f"{'='*108}")
    print(f"  DUAL + 50% DAILY HAIRCUT — unforeseen stress (worst of: regime shift, platform drag, missed days)")
    print(f"{'='*108}\n")
    for lbl, mode, factor in scenarios:
        # Re-run with halved daily samples
        global daily_pts_override
        total_s = np.empty((RUNS, DAYS))
        for r in range(RUNS):
            m = simulate_path(daily_stressed, MAIN_START, mode, factor, seed=r,
                              seed_out_day=WIFE_TRIGGER_DAY, seed_out_amount=WIFE_SEED)
            w = simulate_path(daily_stressed, 0, mode, factor, seed=r + 100_000,
                              seed_in_day=WIFE_TRIGGER_DAY, seed_in_amount=WIFE_SEED,
                              start_trading_day=WIFE_TRIGGER_DAY)
            total_s[r] = m + w
        print(f"  {lbl}  (stressed household)")
        print(f"  {'Year':<6} {'WORST p5':<14} {'MEDIAN p50':<14} {'BEST p95':<14}")
        for y in range(1, YEARS + 1):
            d_idx = y * DAYS_PER_YEAR - 1
            tot = total_s[:, d_idx]
            print(f"  Y{y:<5} {fmt(np.percentile(tot,5)):<14} "
                  f"{fmt(np.percentile(tot,50)):<14} "
                  f"{fmt(np.percentile(tot,95)):<14}")
        p_ruin = (total_s[:, -1] <= 1000).mean() * 100
        print(f"  → 4yr P(household < £1k)={p_ruin:.2f}%\n")


if __name__ == "__main__":
    main()
