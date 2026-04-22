"""
Can £2k → dual account sustain £150k/yr withdrawal for 10+ years under 50% stress?
Models different withdrawal-start triggers and reports survival probability.
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
WIFE_SEED = 10_000
WIFE_TRIGGER_DAY = 252

MIN_STAKE = 0.50
MAX_STAKE = 35.0
DAYS_PER_YEAR = 252
YEARS = 15
DAYS = YEARS * DAYS_PER_YEAR
RUNS = 3000

WITHDRAW_PER_YEAR = 150_000
# Withdraw on the last trading day of each year
WITHDRAW_DAYS = set(y * DAYS_PER_YEAR - 1 for y in range(1, YEARS + 1))

STRESS_FACTOR = 0.5
COMPOUND = 1e-4


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
    return daily * STRESS_FACTOR  # apply 50% haircut


def simulate_household(daily, seed, withdraw_trigger_equity):
    """Simulate dual account. Withdraw £150k at end of each year once household
    equity first hits `withdraw_trigger_equity`."""
    rng = np.random.default_rng(seed)
    sample_m = rng.choice(daily, size=DAYS, replace=True)
    sample_w = rng.choice(daily, size=DAYS, replace=True)
    main = MAIN_START
    wife = 0.0
    m_eq = np.empty(DAYS); w_eq = np.empty(DAYS)
    withdrawals = np.zeros(DAYS)
    withdrawing = False
    cum_withdrawn = 0.0
    for i in range(DAYS):
        # Seed wife from main at start of Y2
        if i == WIFE_TRIGGER_DAY:
            main = max(0, main - WIFE_SEED)
            wife = WIFE_SEED
        # Main always trades (once funded)
        if main > 0:
            stake_m = max(MIN_STAKE, min(MAX_STAKE, main * COMPOUND))
            main = main + sample_m[i] * stake_m
            main = max(0, main)
        # Wife trades only after seeded
        if i >= WIFE_TRIGGER_DAY and wife > 0:
            stake_w = max(MIN_STAKE, min(MAX_STAKE, wife * COMPOUND))
            wife = wife + sample_w[i] * stake_w
            wife = max(0, wife)
        # At end of year, check withdraw
        if i in WITHDRAW_DAYS:
            household = main + wife
            if household >= withdraw_trigger_equity:
                withdrawing = True
            if withdrawing and household > WITHDRAW_PER_YEAR * 1.5:  # keep reserve
                w = WITHDRAW_PER_YEAR
                # split withdrawal pro-rata from each account
                if (main + wife) > 0:
                    frac_m = main / (main + wife)
                    main -= w * frac_m
                    wife -= w * (1 - frac_m)
                main = max(0, main); wife = max(0, wife)
                withdrawals[i] = w
                cum_withdrawn += w
        m_eq[i] = main
        w_eq[i] = wife
    return m_eq, w_eq, withdrawals, cum_withdrawn


def run_scenario(daily, trigger_label, trigger_eq):
    totals = np.empty((RUNS, DAYS))
    cum_wd = np.zeros(RUNS)
    years_withdrawn = np.zeros(RUNS)
    for r in range(RUNS):
        m, w, wd, cw = simulate_household(daily, seed=r, withdraw_trigger_equity=trigger_eq)
        totals[r] = m + w
        cum_wd[r] = cw
        years_withdrawn[r] = (wd > 0).sum()

    print(f"\n  Trigger: start withdrawing {fmt(WITHDRAW_PER_YEAR)}/yr once household ≥ {trigger_label}")
    print(f"  {'Year':<6} {'WORST p5':<14} {'MEDIAN p50':<14} {'BEST p95':<14}")
    for y in [1, 2, 3, 4, 5, 7, 10, 15]:
        if y > YEARS: continue
        d_idx = y * DAYS_PER_YEAR - 1
        tot = totals[:, d_idx]
        print(f"  Y{y:<5} {fmt(np.percentile(tot,5)):<14} "
              f"{fmt(np.percentile(tot,50)):<14} "
              f"{fmt(np.percentile(tot,95)):<14}")
    # Survival analysis: final equity > 0 AND cumulative withdrawals hit target
    target_withdraws_10y = 8 * WITHDRAW_PER_YEAR   # e.g. 8 clean years in 10 (start yr3)
    target_withdraws_15y = 13 * WITHDRAW_PER_YEAR  # 13 years in 15
    survived_10y = ((totals[:, 10 * DAYS_PER_YEAR - 1] > 0) & (cum_wd >= target_withdraws_10y)).mean() * 100 \
                    if YEARS >= 10 else None
    survived_15y = ((totals[:, -1] > 0) & (cum_wd >= target_withdraws_15y)).mean() * 100
    pct_no_wd = (cum_wd == 0).mean() * 100
    med_yrs = np.median(years_withdrawn)
    print(f"  median years actually withdrawn: {med_yrs:.0f}/{YEARS}")
    print(f"  cumulative withdrawn p5 / p50 / p95: "
          f"{fmt(np.percentile(cum_wd,5))} / {fmt(np.percentile(cum_wd,50))} / {fmt(np.percentile(cum_wd,95))}")
    if survived_10y is not None:
        print(f"  P(survived with ≥8yr of withdrawals by Y10) = {survived_10y:.1f}%")
    print(f"  P(survived with ≥13yr of withdrawals by Y15) = {survived_15y:.1f}%")


def main():
    daily = build_daily()
    print(f"Daily (50% stressed): mean={daily.mean():+.1f}pt  worst={daily.min():+.0f}")
    print(f"Household: main £2k + wife £10k seed Y2; £150k/yr withdrawal trigger")

    for lbl, trig in [("£500k", 500_000), ("£1M", 1_000_000), ("£1.5M", 1_500_000)]:
        run_scenario(daily, lbl, trig)


if __name__ == "__main__":
    main()
