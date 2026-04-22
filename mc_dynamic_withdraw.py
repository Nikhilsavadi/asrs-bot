"""
Dynamic withdrawal variant: 10% of equity/year instead of fixed £300k/year.
Compare vs fixed £300k under BASE and worst-realistic stress.
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

MAIN_START = 5000
WIFE_SEED = 10000
WIFE_DAY = 252
WITHDRAW_START = 252
DAYS = 252 * 4
RUNS = 3000
MIN_STAKE = 0.50
MAX_STAKE = 35.0
FACTOR = 1e-4


def fmt_gbp(x, _=None):
    if x <= 0 or (isinstance(x, float) and np.isnan(x)): return "£0"
    if abs(x) < 1000: return f"£{x:.0f}"
    if abs(x) < 1_000_000: return f"£{x/1000:.0f}k"
    return f"£{x/1_000_000:.2f}M"


def simulate(daily_pts, seed, stress=None, withdraw_mode="FIXED", withdraw_rate=0.10,
              fixed_annual=300_000, floor_equity=0):
    """
    withdraw_mode:
      FIXED — fixed £X/year regardless of equity
      DYNAMIC — withdraw rate × equity/year (capped daily)
      DYNAMIC_FLOOR — dynamic but only withdraw if equity > floor_equity
    """
    stress = stress or {}
    rng = np.random.default_rng(seed)
    sample_m = rng.choice(daily_pts, size=DAYS, replace=True).astype(float)
    sample_w = rng.choice(daily_pts, size=DAYS, replace=True).astype(float)

    if stress.get("pf_haircut", 0) > 0:
        hc = stress["pf_haircut"]
        sample_m = np.where(sample_m > 0, sample_m * (1 - hc), sample_m)
        sample_w = np.where(sample_w > 0, sample_w * (1 - hc), sample_w)
    if stress.get("loss_amplify", 0) > 0:
        amp = stress["loss_amplify"]
        sample_m = np.where(sample_m < 0, sample_m * (1 + amp), sample_m)
        sample_w = np.where(sample_w < 0, sample_w * (1 + amp), sample_w)
    if stress.get("miss_day_prob", 0) > 0:
        sample_m[rng.random(DAYS) < stress["miss_day_prob"]] = 0
        sample_w[rng.random(DAYS) < stress["miss_day_prob"]] = 0

    fixed_daily = fixed_annual / 252

    main = np.zeros(DAYS + 1); wife = np.zeros(DAYS + 1); wd = np.zeros(DAYS + 1)
    main[0] = MAIN_START
    for i in range(DAYS):
        stake_m = max(MIN_STAKE, min(MAX_STAKE, main[i] * FACTOR)) if main[i] > 0 else 0
        main[i+1] = main[i] + sample_m[i] * stake_m if stake_m > 0 else main[i]
        if i >= WIFE_DAY and wife[i] > 0:
            stake_w = max(MIN_STAKE, min(MAX_STAKE, wife[i] * FACTOR))
            wife[i+1] = wife[i] + sample_w[i] * stake_w
        else:
            wife[i+1] = wife[i]
        wd[i+1] = wd[i]

        if i + 1 == WIFE_DAY:
            if main[i+1] >= WIFE_SEED:
                main[i+1] -= WIFE_SEED
                wife[i+1] = WIFE_SEED

        if i + 1 >= WITHDRAW_START:
            total = main[i+1] + wife[i+1]
            if withdraw_mode == "FIXED":
                withdraw_today = fixed_daily if total > fixed_daily else 0
            elif withdraw_mode == "DYNAMIC":
                withdraw_today = (total * withdraw_rate) / 252 if total > 0 else 0
            elif withdraw_mode == "DYNAMIC_FLOOR":
                if total > floor_equity:
                    withdraw_today = (total * withdraw_rate) / 252
                else:
                    withdraw_today = 0
            else:
                withdraw_today = 0

            if withdraw_today > total:
                withdraw_today = max(0, total - 100)
            if withdraw_today > 0 and total > 0:
                share_m = main[i+1] / total
                main[i+1] -= withdraw_today * share_m
                wife[i+1] -= withdraw_today * (1 - share_m)
                wd[i+1] += withdraw_today

        main[i+1] = max(0, main[i+1])
        wife[i+1] = max(0, wife[i+1])
    return main, wife, wd


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
    def af(d):
        d = d.copy()
        d["spread_cost"] = d["instrument"].map(SPREAD)
        d["slip_cost"] = np.where(d["pnl_pts"] < 0, d["instrument"].map(SLIP), 0)
        d["pnl_net"] = (d["pnl_pts"] - d["spread_cost"] - d["slip_cost"]) * UPLIFT
        return d
    base_f = af(base_f); fade_f = af(fade_f)
    combined = pd.concat([base_f, fade_f], ignore_index=True)
    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_net"].sum().values

    withdraw_plans = [
        ("FIXED £300k/yr", {"withdraw_mode": "FIXED", "fixed_annual": 300_000}),
        ("DYNAMIC 10%/yr", {"withdraw_mode": "DYNAMIC", "withdraw_rate": 0.10}),
        ("DYNAMIC 15%/yr", {"withdraw_mode": "DYNAMIC", "withdraw_rate": 0.15}),
        ("DYNAMIC 10% floor@£500k", {"withdraw_mode": "DYNAMIC_FLOOR", "withdraw_rate": 0.10, "floor_equity": 500_000}),
    ]
    stress_scenarios = [
        ("BASE", {}),
        ("STRESS (hairct+loss+miss)", {"pf_haircut": 0.20, "loss_amplify": 0.5, "miss_day_prob": 0.10}),
    ]

    results = {}
    print(f"\n{'='*120}")
    print(f"  WITHDRAWAL STRATEGY COMPARISON — 4yr compound 1e-4, cap £35/pt, main+wife")
    print(f"{'='*120}")
    print(f"  {'scenario/plan':<40} {'NW p5':<10} {'NW p50':<10} {'Eq p50':<10} {'Cum-WD p50':<12} {'P(ruin)':<10}")

    for stress_lbl, stress in stress_scenarios:
        for plan_lbl, plan_kwargs in withdraw_plans:
            mains = np.empty((RUNS, DAYS + 1))
            wives = np.empty((RUNS, DAYS + 1))
            wds = np.empty((RUNS, DAYS + 1))
            for r in range(RUNS):
                m, w, wd = simulate(daily, seed=r, stress=stress, **plan_kwargs)
                mains[r] = m; wives[r] = w; wds[r] = wd
            nw = mains + wives + wds
            eq = mains + wives
            final_nw = nw[:, -1]
            final_eq = eq[:, -1]
            final_wd = wds[:, -1]
            ruin = (final_eq < 5000).mean() * 100
            key = (stress_lbl, plan_lbl)
            results[key] = (mains, wives, wds, nw)
            lbl = f"{stress_lbl} | {plan_lbl}"
            print(f"  {lbl:<40} {fmt_gbp(np.percentile(final_nw,5)):<10} "
                  f"{fmt_gbp(np.percentile(final_nw,50)):<10} "
                  f"{fmt_gbp(np.percentile(final_eq,50)):<10} "
                  f"{fmt_gbp(np.percentile(final_wd,50)):<12} "
                  f"{ruin:.1f}%")

    # Plot — 4 panels: BASE withdraw strategies + STRESS withdraw strategies + bands
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    days_axis = np.arange(DAYS + 1) / 252

    # Top-left: BASE — all withdraw plans, net worth p50
    ax = axes[0, 0]
    for plan_lbl, _ in withdraw_plans:
        _, _, _, nw = results[("BASE", plan_lbl)]
        p50 = np.percentile(nw, 50, axis=0)
        ax.plot(days_axis, p50, label=f"{plan_lbl}: {fmt_gbp(p50[-1])}", linewidth=2)
    ax.set_title("BASE (no stress) — net worth by withdrawal plan")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    # Top-right: STRESS — all withdraw plans
    ax = axes[0, 1]
    for plan_lbl, _ in withdraw_plans:
        _, _, _, nw = results[("STRESS (hairct+loss+miss)", plan_lbl)]
        p50 = np.percentile(nw, 50, axis=0)
        ax.plot(days_axis, p50, label=f"{plan_lbl}: {fmt_gbp(p50[-1])}", linewidth=2)
    ax.set_title("STRESS (20% haircut + 50% loss amp + 10% miss) — net worth p50")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    # Bottom-left: DYNAMIC 10% under BASE — bands
    ax = axes[1, 0]
    _, _, _, nw = results[("BASE", "DYNAMIC 10%/yr")]
    for p, c, lbl in [(5, "red", "p5"), (50, "black", "p50"), (95, "blue", "p95")]:
        band = np.percentile(nw, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
    ax.fill_between(days_axis,
                    np.percentile(nw, 5, axis=0),
                    np.percentile(nw, 95, axis=0),
                    color="blue", alpha=0.1)
    ax.set_title("DYNAMIC 10%/yr + BASE — bands")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    # Bottom-right: Cumulative withdrawals — FIXED vs DYNAMIC under BASE
    ax = axes[1, 1]
    for plan_lbl, _ in withdraw_plans:
        _, _, wds, _ = results[("BASE", plan_lbl)]
        p50 = np.percentile(wds, 50, axis=0)
        ax.plot(days_axis, p50, label=f"{plan_lbl}: {fmt_gbp(p50[-1])}/yr4 cum", linewidth=2)
    ax.set_title("Cumulative withdrawals under BASE")
    ax.set_xlabel("Years"); ax.set_ylabel("£ withdrawn")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("Dynamic withdrawal strategies — main + wife household, compound 1e-4",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_dynamic_withdraw.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
