"""
MC simulation: main account + wife account + annual withdrawals.
  Year 0: main starts £5k, compound 1e-4, cap £35/pt
  End Year 1: £10k of profits moved to wife's account (wife starts year 2)
  Year 2+: withdraw £300k/year total, split pro-rata between accounts
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"
UPLIFT = 1.06

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

# Params
MAIN_START = 5000
WIFE_SEED_AMOUNT = 10000   # moved from main at end of year 1
WIFE_START_DAY = 252       # year 2 trading day 1 (252 trading days/yr)
ANNUAL_WITHDRAWAL = 300_000
WITHDRAW_PER_DAY = ANNUAL_WITHDRAWAL / 252
WITHDRAW_START_DAY = 252    # withdrawals start year 2 too
DAYS = 252 * 4              # 4 years
RUNS = 3000
MIN_STAKE = 0.50
MAX_STAKE = 35.0
FACTOR = 1e-4


def fmt_gbp(x, _=None):
    if x <= 0: return "£0"
    if abs(x) < 1000: return f"£{x:.0f}"
    if abs(x) < 1_000_000: return f"£{x/1000:.0f}k"
    if abs(x) < 1_000_000_000: return f"£{x/1_000_000:.2f}M"
    return f"£{x/1_000_000_000:.2f}B"


def simulate_household(daily_pts_main, daily_pts_wife, seed):
    """Simulate main + wife household with withdrawals."""
    rng = np.random.default_rng(seed)
    sample_main = rng.choice(daily_pts_main, size=DAYS, replace=True)
    sample_wife = rng.choice(daily_pts_wife, size=DAYS, replace=True)

    main_eq = np.zeros(DAYS + 1)
    wife_eq = np.zeros(DAYS + 1)
    cum_withdrawn = np.zeros(DAYS + 1)
    main_eq[0] = MAIN_START

    for i in range(DAYS):
        # Trading pnl for main
        stake_main = MIN_STAKE if main_eq[i] < 0.01 else max(MIN_STAKE, min(MAX_STAKE, main_eq[i] * FACTOR))
        main_eq[i+1] = main_eq[i] + sample_main[i] * stake_main

        # Wife's trading (only after wife start day)
        if i >= WIFE_START_DAY and wife_eq[i] > 0:
            stake_wife = max(MIN_STAKE, min(MAX_STAKE, wife_eq[i] * FACTOR))
            wife_eq[i+1] = wife_eq[i] + sample_wife[i] * stake_wife
        else:
            wife_eq[i+1] = wife_eq[i]

        cum_withdrawn[i+1] = cum_withdrawn[i]

        # End of year 1: seed wife account from main
        if i + 1 == WIFE_START_DAY:
            if main_eq[i+1] >= WIFE_SEED_AMOUNT:
                main_eq[i+1] -= WIFE_SEED_AMOUNT
                wife_eq[i+1] = WIFE_SEED_AMOUNT

        # Withdrawals from year 2 day 1 onwards, pro-rata
        if i + 1 >= WITHDRAW_START_DAY:
            total = main_eq[i+1] + wife_eq[i+1]
            if total > WITHDRAW_PER_DAY:
                main_share = main_eq[i+1] / total if total > 0 else 0.5
                withdraw_main = WITHDRAW_PER_DAY * main_share
                withdraw_wife = WITHDRAW_PER_DAY - withdraw_main
                main_eq[i+1] -= withdraw_main
                wife_eq[i+1] -= withdraw_wife
                cum_withdrawn[i+1] += WITHDRAW_PER_DAY

        if main_eq[i+1] < 0: main_eq[i+1] = 0
        if wife_eq[i+1] < 0: wife_eq[i+1] = 0

    return main_eq, wife_eq, cum_withdrawn


def main():
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

    def apply_friction(d):
        d = d.copy()
        d["spread_cost"] = d["instrument"].map(SPREAD)
        d["slip_cost"] = np.where(d["pnl_pts"] < 0, d["instrument"].map(SLIP), 0)
        d["pnl_net"] = (d["pnl_pts"] - d["spread_cost"] - d["slip_cost"]) * UPLIFT
        return d
    base_f = apply_friction(base_f); fade_f = apply_friction(fade_f)
    combined = pd.concat([base_f, fade_f], ignore_index=True)
    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_net"].sum().values
    print(f"Daily samples: {len(daily)}  mean={daily.mean():+.1f}pt")

    # MC
    mains = np.empty((RUNS, DAYS + 1))
    wives = np.empty((RUNS, DAYS + 1))
    withdraws = np.empty((RUNS, DAYS + 1))
    for r in range(RUNS):
        m, w, wd = simulate_household(daily, daily, seed=r)
        mains[r] = m; wives[r] = w; withdraws[r] = wd
    total_eq = mains + wives
    net_worth = total_eq + withdraws   # equity plus cumulative cash taken out

    print(f"\n{'='*100}")
    print(f"  4-YEAR HOUSEHOLD PLAN — compound 1e-4, cap £35/pt")
    print(f"  Year 1: main only (£5k start)")
    print(f"  Year 2 start: seed wife £10k from main")
    print(f"  Year 2+: withdraw £{ANNUAL_WITHDRAWAL:,}/year (£{WITHDRAW_PER_DAY:.0f}/day) pro-rata")
    print(f"{'='*100}")

    checkpoints = [252, 504, 756, 1008]  # end of year 1, 2, 3, 4
    for i, dy in enumerate(checkpoints, start=1):
        eq_p50 = np.percentile(total_eq[:, dy], 50)
        eq_p5 = np.percentile(total_eq[:, dy], 5)
        eq_p95 = np.percentile(total_eq[:, dy], 95)
        wd_p50 = np.percentile(withdraws[:, dy], 50)
        nw_p50 = np.percentile(net_worth[:, dy], 50)
        nw_p5 = np.percentile(net_worth[:, dy], 5)
        nw_p95 = np.percentile(net_worth[:, dy], 95)
        main_p50 = np.percentile(mains[:, dy], 50)
        wife_p50 = np.percentile(wives[:, dy], 50)
        print(f"\n  End Year {i} (day {dy}):")
        print(f"    MAIN equity p50:     {fmt_gbp(main_p50)}")
        print(f"    WIFE equity p50:     {fmt_gbp(wife_p50)}")
        print(f"    Total equity p50:    {fmt_gbp(eq_p50)}  (p5 {fmt_gbp(eq_p5)} — p95 {fmt_gbp(eq_p95)})")
        print(f"    Cumulative withdrawn {fmt_gbp(wd_p50)}")
        print(f"    NET WORTH p50:       {fmt_gbp(nw_p50)}  (p5 {fmt_gbp(nw_p5)} — p95 {fmt_gbp(nw_p95)})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    days_axis = np.arange(DAYS + 1) / 252

    # Top-left: total equity bands
    ax = axes[0, 0]
    for p, c, lbl in [(5, "red", "WORST(p5)"), (50, "black", "MEDIAN(p50)"), (95, "blue", "BEST(p95)")]:
        band = np.percentile(total_eq, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
    ax.fill_between(days_axis,
                    np.percentile(total_eq, 5, axis=0),
                    np.percentile(total_eq, 95, axis=0),
                    color="blue", alpha=0.1)
    ax.axvline(1.0, color="orange", linestyle="--", alpha=0.7, label="Wife seed + withdrawals start")
    ax.set_title("Total equity (main + wife) — withdrawals reduce visible equity")
    ax.set_xlabel("Years"); ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    # Top-right: cumulative withdrawals
    ax = axes[0, 1]
    for p, c, lbl in [(5, "red", "WORST(p5)"), (50, "black", "MEDIAN"), (95, "blue", "BEST(p95)")]:
        band = np.percentile(withdraws, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
    ax.set_title("Cumulative withdrawals to your pocket")
    ax.set_xlabel("Years"); ax.set_ylabel("£ withdrawn")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Bottom-left: net worth (equity + cash taken out)
    ax = axes[1, 0]
    for p, c, lbl in [(5, "red", "WORST(p5)"), (50, "black", "MEDIAN"), (95, "blue", "BEST(p95)")]:
        band = np.percentile(net_worth, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
    ax.fill_between(days_axis,
                    np.percentile(net_worth, 5, axis=0),
                    np.percentile(net_worth, 95, axis=0),
                    color="blue", alpha=0.1)
    ax.set_title("NET WORTH = trading equity + cumulative withdrawals (true total)")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    # Bottom-right: main vs wife split (median)
    ax = axes[1, 1]
    ax.plot(days_axis, np.percentile(mains, 50, axis=0), label="Main (median)", linewidth=2, color="navy")
    ax.plot(days_axis, np.percentile(wives, 50, axis=0), label="Wife (median)", linewidth=2, color="purple")
    ax.axvline(1.0, color="orange", linestyle="--", alpha=0.7, label="Wife £10k seed at year 2")
    ax.set_title("Main vs Wife — median equity paths")
    ax.set_xlabel("Years"); ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(f"Household wealth plan — main £5k + wife £10k (yr 2) + £{ANNUAL_WITHDRAWAL//1000}k/yr withdrawal",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_withdraw_plan.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
