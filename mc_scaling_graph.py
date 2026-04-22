"""
Updated MC: realistic friction + £35/pt cap + single vs dual account.
Uses post-friction combined trades from the cached parquet.
Note: this parquet reflects BAR5_RULES=WIDE already (bar 4 for NORMAL in backtest).
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

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

START_EQUITY = 5000
DAYS = 1008
RUNS = 3000
MIN_STAKE = 0.50
MAX_STAKE = 35.0  # ← CHANGED from 100 to 35

# Approximate uplift from today's new features (not in parquet):
# - milestone_lock (base): +6.6% PF
# - TRAIL_TARGET (fade): +5% net
# Conservative total uplift on combined net: 1.06×
UPLIFT = 1.06


def fmt_gbp(x, _=None):
    if x <= 0: return "£0"
    if x < 1000: return f"£{x:.0f}"
    if x < 1_000_000: return f"£{x/1000:.0f}k"
    if x < 1_000_000_000: return f"£{x/1_000_000:.1f}M".replace(".0M", "M")
    return f"£{x/1_000_000_000:.1f}B"


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def apply_friction(df, spread_map, slip_map):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(spread_map)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(slip_map), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def simulate(daily_pts, mode, factor, seed, max_stake=MAX_STAKE):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily_pts, size=DAYS, replace=True)
    eq = np.empty(DAYS + 1)
    eq[0] = START_EQUITY
    for i in range(DAYS):
        stake = MIN_STAKE if mode == "flat" else max(MIN_STAKE, min(max_stake, eq[i] * factor))
        eq[i+1] = eq[i] + sample[i] * stake
        if eq[i+1] < 0:
            eq[i+1:] = 0; break
    return eq[1:]


def mc(daily, mode, factor, max_stake=MAX_STAKE):
    eqs = np.empty((RUNS, DAYS))
    for r in range(RUNS):
        eqs[r] = simulate(daily, mode, factor, seed=r, max_stake=max_stake)
    final = eqs[:, -1]
    runmax = np.maximum.accumulate(eqs, axis=1)
    dd = (runmax - eqs) / np.maximum(runmax, 1e-9) * 100
    return eqs, final, dd.max(axis=1)


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
    base_f = apply_friction(base_f, SPREAD, SLIP)
    fade_f = apply_friction(fade_f, SPREAD, SLIP)
    combined = pd.concat([base_f, fade_f], ignore_index=True)
    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_net"].sum().values

    print(f"Combined post-friction (+6% uplift): n={len(combined):,}  PF={pf(combined['pnl_net']):.2f}  net={combined['pnl_net'].sum():+,.0f}")
    print(f"Daily samples: {len(daily)}  mean={daily.mean():+.1f}pt  worst={daily.min():+.0f}")
    print(f"MAX_STAKE capped at £{MAX_STAKE}/pt (was £100)")

    scenarios = [
        ("A flat £0.50/pt",      "flat",     None),
        ("B compound 1e-4",      "compound", 1e-4),
        ("C conservative 5e-5",  "compound", 5e-5),
        ("D aggressive 1.5e-4",  "compound", 1.5e-4),
    ]

    # -------- SINGLE ACCOUNT --------
    print(f"\n{'='*100}")
    print(f"  SINGLE ACCOUNT — start £{START_EQUITY:,}, {DAYS}d, {RUNS} runs, cap £{MAX_STAKE}/pt")
    print(f"{'='*100}")
    print(f"  {'scenario':<25} {'WORST(p5)':<12} {'MEDIAN(p50)':<12} {'BEST(p95)':<12} {'DD p50':<8} {'DD p95':<8}  {'P(loss)':<8}")
    results_single = {}
    for lbl, mode, factor in scenarios:
        eqs, final, dd = mc(daily, mode, factor, max_stake=MAX_STAKE)
        results_single[lbl] = (eqs, final, dd)
        prob_loss = (final < START_EQUITY).mean() * 100
        print(f"  {lbl:<25} {fmt_gbp(np.percentile(final,5)):<12} "
              f"{fmt_gbp(np.percentile(final,50)):<12} "
              f"{fmt_gbp(np.percentile(final,95)):<12} "
              f"{np.percentile(dd,50):>5.2f}%  {np.percentile(dd,95):>5.2f}%   {prob_loss:.1f}%")

    # -------- DUAL ACCOUNT --------
    # Dual = 2 independent accounts each running identical strategy with their own £5k
    # Independent MC draws so not perfectly correlated (models starting at different times)
    print(f"\n{'='*100}")
    print(f"  DUAL ACCOUNT — 2× £{START_EQUITY:,} independent, same strategy, cap £{MAX_STAKE}/pt each")
    print(f"{'='*100}")
    print(f"  {'scenario':<25} {'WORST(p5)':<12} {'MEDIAN(p50)':<12} {'BEST(p95)':<12} {'DD p50':<8} {'DD p95':<8}  {'P(loss)':<8}")
    results_dual = {}
    for lbl, mode, factor in scenarios:
        # Independent runs with different seeds: seed N and seed N+10000
        eqs_a = np.empty((RUNS, DAYS))
        eqs_b = np.empty((RUNS, DAYS))
        for r in range(RUNS):
            eqs_a[r] = simulate(daily, mode, factor, seed=r, max_stake=MAX_STAKE)
            eqs_b[r] = simulate(daily, mode, factor, seed=r + 10000, max_stake=MAX_STAKE)
        eqs_total = eqs_a + eqs_b
        final = eqs_total[:, -1]
        runmax = np.maximum.accumulate(eqs_total, axis=1)
        dd = (runmax - eqs_total) / np.maximum(runmax, 1e-9) * 100
        max_dd = dd.max(axis=1)
        results_dual[lbl] = (eqs_total, final, max_dd)
        prob_loss = (final < 2 * START_EQUITY).mean() * 100
        print(f"  {lbl:<25} {fmt_gbp(np.percentile(final,5)):<12} "
              f"{fmt_gbp(np.percentile(final,50)):<12} "
              f"{fmt_gbp(np.percentile(final,95)):<12} "
              f"{np.percentile(max_dd,50):>5.2f}%  {np.percentile(max_dd,95):>5.2f}%   {prob_loss:.1f}%")

    # -------- GRAPHS --------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    days_axis = np.arange(1, DAYS + 1) / 252

    def plot_medians(ax, results, title):
        for lbl, _, _ in scenarios:
            eqs, _, _ = results[lbl]
            p50 = np.percentile(eqs, 50, axis=0)
            ax.plot(days_axis, p50, label=lbl, linewidth=2)
            ax.annotate(fmt_gbp(p50[-1]), xy=(days_axis[-1], p50[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=9, fontweight="bold", va="center")
        ax.set_title(title)
        ax.set_xlabel("Years"); ax.set_ylabel("Equity")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)

    def plot_bands(ax, results, scenario_name, title):
        eqs, _, _ = results[scenario_name]
        for p, c, lbl in [(5, "red", "WORST(p5)"), (50, "black", "MEDIAN(p50)"), (95, "blue", "BEST(p95)")]:
            band = np.percentile(eqs, p, axis=0)
            ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
        ax.fill_between(days_axis,
                        np.percentile(eqs, 5, axis=0),
                        np.percentile(eqs, 95, axis=0),
                        color="blue", alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("Years"); ax.set_ylabel("Equity")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)

    plot_medians(axes[0, 0], results_single,
                  f"SINGLE account — median curves (cap £{MAX_STAKE}/pt)")
    plot_bands(axes[0, 1], results_single, "B compound 1e-4",
                f"SINGLE — B compound (worst/median/best)")
    plot_medians(axes[1, 0], results_dual,
                  f"DUAL account — median curves (2× £5k, cap £{MAX_STAKE}/pt each)")
    plot_bands(axes[1, 1], results_dual, "B compound 1e-4",
                f"DUAL — B compound (worst/median/best)")

    plt.suptitle(f"Variant B + Fade + Milestone + BAR5_RULES=WIDE — realistic MC (cap £{MAX_STAKE}/pt)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_single_vs_dual_35cap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
