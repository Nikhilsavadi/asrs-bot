"""
MC with scaling (flat / compound / aggressive / conservative) on POST-SPREAD
combined daily pnl. Shows best/median/worst paths over 4 years.
"""
import os, sys, time, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

START_EQUITY = 5000
DAYS = 1008
RUNS = 3000
MIN_STAKE = 0.50
MAX_STAKE = 100.0


def fmt_gbp(x, _=None):
    if x <= 0: return "£0"
    if x < 1000: return f"£{x:.0f}"
    if x < 1_000_000: return f"£{x/1000:.0f}k"
    if x < 1_000_000_000: return f"£{x/1_000_000:.1f}M".replace(".0M", "M")
    return f"£{x/1_000_000_000:.1f}B"


def apply_spread(df):
    df = df.copy()
    df["pnl_pts_net"] = df["pnl_pts"] - df["instrument"].map(SPREAD)
    return df


def simulate(daily_pts, mode, factor, seed):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily_pts, size=DAYS, replace=True)
    eq = np.empty(DAYS + 1)
    eq[0] = START_EQUITY
    for i in range(DAYS):
        if mode == "flat":
            stake = MIN_STAKE
        else:
            stake = max(MIN_STAKE, min(MAX_STAKE, eq[i] * factor))
        eq[i+1] = eq[i] + sample[i] * stake
        if eq[i+1] < 0:
            eq[i+1:] = 0
            break
    return eq[1:]


def mc(daily, mode, factor):
    eqs = np.empty((RUNS, DAYS))
    for r in range(RUNS):
        eqs[r] = simulate(daily, mode, factor, seed=r)
    final = eqs[:, -1]
    runmax = np.maximum.accumulate(eqs, axis=1)
    dd = (runmax - eqs) / np.maximum(runmax, 1e-9) * 100
    return eqs, final, dd.max(axis=1)


def main():
    t0 = time.time()
    df = pd.read_parquet(TRADES_CACHE)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False)
    base = df[~df["is_fade"]].copy()
    fade = df[df["is_fade"]].copy()
    base_f = filter_combined(base)
    combined = pd.concat([base_f, fade], ignore_index=True)
    combined = apply_spread(combined)

    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_pts_net"].sum().values
    print(f"Daily pnl samples: {len(daily)}, mean {daily.mean():+.1f}pt, worst {daily.min():+.0f}pt")

    scenarios = [
        ("A flat £0.50/pt",      "flat",     None),
        ("B compound 1e-4",      "compound", 1e-4),
        ("C conservative 5e-5",  "compound", 5e-5),
        ("D aggressive 1.5e-4",  "compound", 1.5e-4),
    ]

    print(f"\n{'='*100}")
    print(f"  MC WITH SCALING — POST-SPREAD combined daily pnl")
    print(f"  Start £{START_EQUITY:,}, {DAYS}d (4yr), {RUNS} runs")
    print(f"{'='*100}")
    print(f"  {'scenario':<25} {'WORST p5':<14} {'MEDIAN p50':<14} {'BEST p95':<14} "
          f"{'DD p50':<8} {'DD p95':<8}")

    results = {}
    for lbl, mode, factor in scenarios:
        eqs, final, dd = mc(daily, mode, factor)
        results[lbl] = (eqs, final, dd)
        print(f"  {lbl:<25} "
              f"{fmt_gbp(np.percentile(final,5)):<13} "
              f"{fmt_gbp(np.percentile(final,50)):<13} "
              f"{fmt_gbp(np.percentile(final,95)):<13} "
              f"{np.percentile(dd,50):>5.2f}%  {np.percentile(dd,95):>5.2f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    days_axis = np.arange(1, DAYS + 1) / 252

    # Left: median curves all scenarios
    ax = axes[0]
    for lbl, _, _ in scenarios:
        eqs, _, _ = results[lbl]
        p50 = np.percentile(eqs, 50, axis=0)
        ax.plot(days_axis, p50, label=lbl, linewidth=2)
        ax.annotate(fmt_gbp(p50[-1]), xy=(days_axis[-1], p50[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, fontweight="bold", va="center")
    ax.set_title("Median equity curves (p50) — POST-SPREAD")
    ax.set_xlabel("Years"); ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=10)

    # Right: B compound with p5/50/95 bands (worst/median/best)
    ax = axes[1]
    eqs, _, _ = results["B compound 1e-4"]
    for p, c, lbl in [(5, "red", "WORST (p5)"), (50, "black", "MEDIAN (p50)"), (95, "blue", "BEST (p95)")]:
        band = np.percentile(eqs, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
    ax.fill_between(days_axis,
                    np.percentile(eqs, 5, axis=0),
                    np.percentile(eqs, 95, axis=0),
                    color="blue", alpha=0.1)
    ax.set_title("B compound 1e-4 — Worst / Median / Best (POST-SPREAD)")
    ax.set_xlabel("Years"); ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=10)

    plt.suptitle(f"Variant B + FADE — MC with scaling, post-spread (start £{START_EQUITY:,}, 4yr)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_scaling_post_spread.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved → {out}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
