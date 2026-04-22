"""
Worst-case MC stress: household plan under realistic bad-luck scenarios.
  BASE — no stress (what we showed earlier)
  STRESS A — 30% PF haircut (strategy underperforms backtest)
  STRESS B — 10% of trading days missed (operational bugs)
  STRESS C — losses 2× amplified (worse slippage than modeled)
  STRESS D — worst 3-month quarter replayed at start of year 2
  CATASTROPHIC — ALL above stacked
Outputs: net worth at year 4 per scenario, MC distribution + graphs.
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
ANNUAL_WITHDRAW = 300_000
WD_PER_DAY = ANNUAL_WITHDRAW / 252
DAYS = 252 * 4
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


def simulate(daily_pts, seed, stress):
    """
    stress dict keys:
      pf_haircut     — multiply winning days by (1 - haircut)  [0.0 to 0.5]
      miss_day_prob  — probability each day yields 0 pnl       [0.0 to 0.3]
      loss_amplify   — multiply losing days by (1 + amp)        [0.0 to 1.5]
      bad_quarter    — inject worst 63-day streak at start of year 2 [bool]
    """
    rng = np.random.default_rng(seed)
    sample_m = rng.choice(daily_pts, size=DAYS, replace=True)
    sample_w = rng.choice(daily_pts, size=DAYS, replace=True)

    def apply_stress(sample):
        s = sample.copy().astype(float)
        if stress.get("pf_haircut", 0) > 0:
            hc = stress["pf_haircut"]
            s = np.where(s > 0, s * (1 - hc), s)
        if stress.get("loss_amplify", 0) > 0:
            amp = stress["loss_amplify"]
            s = np.where(s < 0, s * (1 + amp), s)
        if stress.get("miss_day_prob", 0) > 0:
            mask = rng.random(len(s)) < stress["miss_day_prob"]
            s[mask] = 0
        return s

    sample_m = apply_stress(sample_m)
    sample_w = apply_stress(sample_w)

    if stress.get("bad_quarter", False):
        # Find the worst 63-day window in the historical daily pts
        rolling = pd.Series(daily_pts).rolling(63).sum()
        worst_start = rolling.idxmin()
        worst_seq = daily_pts[worst_start - 62:worst_start + 1]  # 63 days
        if len(worst_seq) == 63:
            # Inject at start of year 2 (days 252-315) for MAIN
            s_apply = apply_stress(worst_seq.copy().astype(float))
            sample_m[252:252+63] = s_apply

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
            if total > WD_PER_DAY:
                share_m = main[i+1] / total if total > 0 else 0.5
                main[i+1] -= WD_PER_DAY * share_m
                wife[i+1] -= WD_PER_DAY * (1 - share_m)
                wd[i+1] += WD_PER_DAY
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

    scenarios = [
        ("BASE (no stress)",         {}),
        ("A: PF haircut 30%",        {"pf_haircut": 0.30}),
        ("B: Miss 10% of days",      {"miss_day_prob": 0.10}),
        ("C: Losses 2× amplified",   {"loss_amplify": 1.0}),
        ("D: Worst quarter @yr2",    {"bad_quarter": True}),
        ("CATASTROPHIC (A+B+C+D)",   {"pf_haircut": 0.30, "miss_day_prob": 0.10,
                                       "loss_amplify": 1.0, "bad_quarter": True}),
    ]

    results = {}
    print(f"\n{'='*110}")
    print(f"  HOUSEHOLD PLAN UNDER STRESS — 4yr, compound 1e-4, cap £35, wife £10k @yr2, £300k/yr withdraw")
    print(f"{'='*110}")
    print(f"  {'scenario':<32} {'NW p5':<12} {'NW p25':<12} {'NW p50':<12} {'NW p95':<12} {'P(ruin)':<10}")
    for lbl, stress in scenarios:
        mains = np.empty((RUNS, DAYS + 1))
        wives = np.empty((RUNS, DAYS + 1))
        wds = np.empty((RUNS, DAYS + 1))
        for r in range(RUNS):
            m, w, wd = simulate(daily, seed=r, stress=stress)
            mains[r] = m; wives[r] = w; wds[r] = wd
        net_worth = mains + wives + wds
        final_nw = net_worth[:, -1]
        # "Ruin" = equity < £5k (original seed) at year 4
        equity_final = mains[:, -1] + wives[:, -1]
        ruin_rate = (equity_final < 5000).mean() * 100
        results[lbl] = (mains, wives, wds, net_worth)
        print(f"  {lbl:<32} {fmt_gbp(np.percentile(final_nw,5)):<12} "
              f"{fmt_gbp(np.percentile(final_nw,25)):<12} "
              f"{fmt_gbp(np.percentile(final_nw,50)):<12} "
              f"{fmt_gbp(np.percentile(final_nw,95)):<12} "
              f"{ruin_rate:.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    days_axis = np.arange(DAYS + 1) / 252

    # Top-left: net worth p50 across scenarios
    ax = axes[0, 0]
    for lbl, _ in scenarios:
        _, _, _, nw = results[lbl]
        p50 = np.percentile(nw, 50, axis=0)
        ax.plot(days_axis, p50, label=f"{lbl}: {fmt_gbp(p50[-1])}", linewidth=2)
    ax.set_title("Net worth p50 (median) — all stress scenarios")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)

    # Top-right: worst-case (p5) across scenarios
    ax = axes[0, 1]
    for lbl, _ in scenarios:
        _, _, _, nw = results[lbl]
        p5 = np.percentile(nw, 5, axis=0)
        ax.plot(days_axis, p5, label=f"{lbl}: {fmt_gbp(p5[-1])}", linewidth=2)
    ax.set_title("Net worth p5 (worst-case 5th percentile)")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth p5")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)

    # Bottom-left: CATASTROPHIC scenario bands
    ax = axes[1, 0]
    _, _, _, nw_c = results["CATASTROPHIC (A+B+C+D)"]
    for p, c, lbl in [(5, "red", "WORST(p5)"), (25, "orange", "p25"), (50, "black", "MEDIAN"),
                      (75, "green", "p75"), (95, "blue", "BEST(p95)")]:
        band = np.percentile(nw_c, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=1.8)
    ax.fill_between(days_axis,
                    np.percentile(nw_c, 5, axis=0),
                    np.percentile(nw_c, 95, axis=0),
                    color="blue", alpha=0.1)
    ax.set_title("CATASTROPHIC — worst-realistic scenario all stacked")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)

    # Bottom-right: Base vs Catastrophic — worst + median
    ax = axes[1, 1]
    _, _, _, nw_b = results["BASE (no stress)"]
    _, _, _, nw_c = results["CATASTROPHIC (A+B+C+D)"]
    for label, nw, colors in [("BASE", nw_b, ("blue", "navy")), ("CATASTROPHIC", nw_c, ("red", "darkred"))]:
        p5 = np.percentile(nw, 5, axis=0)
        p50 = np.percentile(nw, 50, axis=0)
        ax.plot(days_axis, p50, label=f"{label} p50: {fmt_gbp(p50[-1])}", color=colors[1], linewidth=2)
        ax.plot(days_axis, p5, label=f"{label} p5: {fmt_gbp(p5[-1])}", color=colors[0], linewidth=1.5, linestyle="--")
    ax.set_title("BASE vs CATASTROPHIC — worst + median")
    ax.set_xlabel("Years"); ax.set_ylabel("Net worth")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("Household stress test — main + wife + £300k/yr withdrawal under bad-luck scenarios",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_bad_luck.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
