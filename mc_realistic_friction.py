"""
MC with realistic slippage budget:
  SPREAD (all trades):     DAX 1.5, US30 2.75, NIKKEI 7.0
  ADVERSE SELECTION on LOSERS only:   DAX +1, US30 +3, NIKKEI +4
  Total loser friction:    DAX 2.5, US30 5.75, NIKKEI 11 pts
  Winner friction:         just spread
Winners at target are neutral fill. Losers at stop have systematic adverse fill.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
ADVERSE_SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
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


def pf(pnl):
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def apply_friction(df, spread_map, slip_map):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(spread_map)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0,
                                 df["instrument"].map(slip_map), 0)
    df["pnl_pts_net"] = df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]
    return df


def simulate(daily_pts, mode, factor, seed):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily_pts, size=DAYS, replace=True)
    eq = np.empty(DAYS + 1)
    eq[0] = START_EQUITY
    for i in range(DAYS):
        stake = MIN_STAKE if mode == "flat" else max(MIN_STAKE, min(MAX_STAKE, eq[i] * factor))
        eq[i+1] = eq[i] + sample[i] * stake
        if eq[i+1] < 0:
            eq[i+1:] = 0; break
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
    df = pd.read_parquet(TRADES_CACHE)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)

    # Pair fades with originating TRAIL_WIN for ≥20pt filter
    df = df.reset_index(drop=True)
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        fades = group[group.is_fade]
        for i, (idx, _) in enumerate(fades.iterrows()):
            if i < len(q): orig_win[idx] = q[i]

    fade_full = df[df.is_fade].copy()
    fade_full["orig_win"] = fade_full.index.map(orig_win)
    fade_filtered = fade_full[fade_full["orig_win"].fillna(0) >= 20].copy()

    base = df[~df.is_fade].copy()
    base_f = filter_combined(base)

    # Apply friction
    base_f = apply_friction(base_f, SPREAD, ADVERSE_SLIP)
    fade_filtered = apply_friction(fade_filtered, SPREAD, ADVERSE_SLIP)
    combined = pd.concat([base_f, fade_filtered], ignore_index=True)

    print(f"BASE:  n={len(base_f):,}  PF={pf(base_f['pnl_pts_net']):.2f}  net={base_f['pnl_pts_net'].sum():+,.0f}")
    print(f"FADE≥20: n={len(fade_filtered):,}  PF={pf(fade_filtered['pnl_pts_net']):.2f}  net={fade_filtered['pnl_pts_net'].sum():+,.0f}")
    print(f"COMBINED: n={len(combined):,}  PF={pf(combined['pnl_pts_net']):.2f}  net={combined['pnl_pts_net'].sum():+,.0f}")

    print(f"\n  Per-instrument COMBINED:")
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = combined[combined.instrument == inst]
        print(f"    {inst:<8}  n={len(d):,} PF={pf(d['pnl_pts_net']):.2f} "
              f"net={d['pnl_pts_net'].sum():+,.0f} "
              f"(friction: {SPREAD[inst]:.1f}pt spread + {ADVERSE_SLIP[inst]:.1f}pt slip)")

    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_pts_net"].sum().values
    print(f"\n  Daily pnl: mean={daily.mean():+.1f}pt  worst={daily.min():+.0f}pt")

    scenarios = [
        ("A flat £0.50/pt",      "flat",     None),
        ("B compound 1e-4",      "compound", 1e-4),
        ("C conservative 5e-5",  "compound", 5e-5),
        ("D aggressive 1.5e-4",  "compound", 1.5e-4),
    ]

    print(f"\n{'='*100}")
    print(f"  MC WITH REALISTIC FRICTION (spread + adverse-selection slippage)")
    print(f"  Start £{START_EQUITY:,}, {DAYS}d (4yr), {RUNS} runs")
    print(f"{'='*100}")
    print(f"  {'scenario':<25} {'WORST (p5)':<13} {'MEDIAN (p50)':<14} {'BEST (p95)':<13} {'DD p50':<8} {'DD p95':<8}  {'P(loss)':<8}")

    results = {}
    for lbl, mode, factor in scenarios:
        eqs, final, dd = mc(daily, mode, factor)
        results[lbl] = (eqs, final, dd)
        prob_loss = (final < START_EQUITY).mean() * 100
        print(f"  {lbl:<25} "
              f"{fmt_gbp(np.percentile(final,5)):<13} "
              f"{fmt_gbp(np.percentile(final,50)):<14} "
              f"{fmt_gbp(np.percentile(final,95)):<13} "
              f"{np.percentile(dd,50):>5.2f}%  {np.percentile(dd,95):>5.2f}%   {prob_loss:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    days_axis = np.arange(1, DAYS + 1) / 252

    ax = axes[0]
    for lbl, _, _ in scenarios:
        eqs, _, _ = results[lbl]
        p50 = np.percentile(eqs, 50, axis=0)
        ax.plot(days_axis, p50, label=lbl, linewidth=2)
        ax.annotate(fmt_gbp(p50[-1]), xy=(days_axis[-1], p50[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, fontweight="bold", va="center")
    ax.set_title("Median equity — with realistic friction (spread + slip)")
    ax.set_xlabel("Years"); ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    eqs, _, _ = results["B compound 1e-4"]
    for p, c, lbl in [(5, "red", "WORST (p5)"), (50, "black", "MEDIAN (p50)"), (95, "blue", "BEST (p95)")]:
        band = np.percentile(eqs, p, axis=0)
        ax.plot(days_axis, band, color=c, label=f"{lbl}: {fmt_gbp(band[-1])}", linewidth=2)
    ax.fill_between(days_axis,
                    np.percentile(eqs, 5, axis=0),
                    np.percentile(eqs, 95, axis=0),
                    color="blue", alpha=0.1)
    ax.set_title("B compound — Worst/Median/Best (realistic friction)")
    ax.set_xlabel("Years"); ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_gbp))
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(f"Variant B + FADE ≥20pt — realistic friction MC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_realistic_friction.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
