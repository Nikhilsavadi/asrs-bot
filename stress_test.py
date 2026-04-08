"""
stress_test.py — Adversarial stress tests on the validated backtest.

Runs five "what if reality is much worse than the data" scenarios on top
of the firstrate backtest, then plots equity curves at £0.50/pt over 1 year.

Scenarios:
  1. BASELINE                  — pure backtest, no modification
  2. FLIP TOP 5% WINNERS       — convert your 5% biggest winners to losers
  3. FLIP TOP 10% WINNERS      — convert your 10% biggest winners to losers
  4. SKIP ALL TRADES > +50pts  — cap winners at +50pts (fat tails removed)
  5. DOUBLE ALL LOSSES         — every loss is 2x bigger
  6. TRIPLE ALL LOSSES + HALVE WINS — apocalyptic scenario
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def apply_scenario(trades: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df = trades.copy()
    pnl = df["pnl_pts"].values

    if scenario == "baseline":
        pass
    elif scenario == "flip_top5":
        cutoff = np.percentile(pnl[pnl > 0], 95)
        flip_mask = pnl >= cutoff
        df.loc[flip_mask, "pnl_pts"] = -pnl[flip_mask]
    elif scenario == "flip_top10":
        cutoff = np.percentile(pnl[pnl > 0], 90)
        flip_mask = pnl >= cutoff
        df.loc[flip_mask, "pnl_pts"] = -pnl[flip_mask]
    elif scenario == "cap_winners_50":
        wins = pnl > 50
        df.loc[wins, "pnl_pts"] = 50
    elif scenario == "double_losses":
        losses = pnl < 0
        df.loc[losses, "pnl_pts"] = pnl[losses] * 2
    elif scenario == "apocalypse":
        losses = pnl < 0
        wins = pnl > 0
        df.loc[losses, "pnl_pts"] = pnl[losses] * 3
        df.loc[wins, "pnl_pts"] = pnl[wins] * 0.5
    else:
        raise ValueError(scenario)

    return df


def stats(df: pd.DataFrame) -> dict:
    pnl = df["pnl_pts"].values
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    pf = w / l if l else float("inf")
    win_rate = (pnl > 0).mean() * 100
    return {
        "trades": len(df),
        "pf": pf,
        "win_rate": win_rate,
        "net_pts": pnl.sum(),
        "avg_win": pnl[pnl > 0].mean() if (pnl > 0).any() else 0,
        "avg_loss": pnl[pnl < 0].mean() if (pnl < 0).any() else 0,
    }


def daily_pnl(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("date")["pnl_pts"].sum().values


def simulate_year(daily_pnls: np.ndarray, account: float, stake: float,
                   n_days: int, runs: int) -> dict:
    rng = np.random.default_rng(42)
    finals = np.zeros(runs)
    max_dds = np.zeros(runs)
    hard_stop_hits = 0

    for i in range(runs):
        idx = rng.integers(0, len(daily_pnls), size=n_days)
        sample = daily_pnls[idx] * stake
        eq = account + np.cumsum(sample)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak).min()
        max_dds[i] = dd
        finals[i] = eq[-1] - account
        if eq.min() <= 4000:
            hard_stop_hits += 1

    return {
        "p5":  np.percentile(finals, 5),
        "p25": np.percentile(finals, 25),
        "p50": np.percentile(finals, 50),
        "p75": np.percentile(finals, 75),
        "p95": np.percentile(finals, 95),
        "worst_dd": max_dds.min(),
        "ruin_pct": hard_stop_hits / runs * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/backtest_firstrate_results.csv")
    ap.add_argument("--account", type=float, default=5000)
    ap.add_argument("--stake", type=float, default=0.5)
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--out", default="data/stress_test.png")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    print(f"Loaded {len(raw):,} trades from {args.csv}")
    print(f"Stake: £{args.stake}/pt, Account: £{args.account:,.0f}, Horizon: {args.days}d, Runs: {args.runs}\n")

    scenarios = [
        ("baseline",         "Baseline (PF 4.22)",     "#1f7a1f"),
        ("flip_top5",        "Flip top 5% winners",    "#ff7f0e"),
        ("flip_top10",       "Flip top 10% winners",   "#d62728"),
        ("cap_winners_50",   "Cap winners @ 50pts",    "#9467bd"),
        ("double_losses",    "Double all losses",      "#8c564b"),
        ("apocalypse",       "Triple losses + halve wins", "#e377c2"),
    ]

    results = {}
    print(f"{'Scenario':<32}{'PF':>7}{'Win%':>7}{'Net pts':>14}  Trade count")
    print("-" * 75)

    for key, label, _color in scenarios:
        df = apply_scenario(raw, key)
        s = stats(df)
        results[key] = (df, s)
        print(f"{label:<32}{s['pf']:>7.2f}{s['win_rate']:>7.1f}{s['net_pts']:>14,.0f}  {s['trades']:,}")

    # Run MC on each scenario
    print(f"\n\n{'='*80}")
    print(f"  1-YEAR MC PROJECTION  (£{args.account:,.0f} start, £{args.stake}/pt, fixed stake, {args.runs:,} sims)")
    print(f"{'='*80}")
    print(f"  {'Scenario':<32}{'p5':>11}{'p50':>11}{'p95':>11}{'WorstDD':>11}{'Ruin%':>8}")
    print("-" * 84)

    mc_results = {}
    for key, label, _ in scenarios:
        df, _ = results[key]
        daily = daily_pnl(df)
        mc = simulate_year(daily, args.account, args.stake, args.days, args.runs)
        mc_results[key] = mc
        print(f"  {label:<32}{mc['p5']:>+11,.0f}{mc['p50']:>+11,.0f}{mc['p95']:>+11,.0f}"
              f"{mc['worst_dd']:>+11,.0f}{mc['ruin_pct']:>7.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    # Top: median equity curves
    ax1 = axes[0]
    rng = np.random.default_rng(42)
    for key, label, color in scenarios:
        df, _ = results[key]
        daily = daily_pnl(df)
        # Median path: simulate runs sims, take median at each step
        n_sims = 500
        all_eq = np.zeros((n_sims, args.days + 1))
        for i in range(n_sims):
            idx = rng.integers(0, len(daily), size=args.days)
            sample = daily[idx] * args.stake
            eq = args.account + np.concatenate([[0], np.cumsum(sample)])
            all_eq[i] = eq
        median_eq = np.median(all_eq, axis=0)
        p25 = np.percentile(all_eq, 25, axis=0)
        p75 = np.percentile(all_eq, 75, axis=0)
        days = np.arange(args.days + 1)
        ax1.fill_between(days, p25, p75, alpha=0.10, color=color)
        ax1.plot(days, median_eq, color=color, linewidth=2.0, label=label)

    ax1.axhline(y=args.account, color="#888", linestyle=":", linewidth=1)
    ax1.axhline(y=4000, color="#d62728", linestyle="--", linewidth=1, label="Hard stop £4k")
    ax1.set_xlabel("Trading day")
    ax1.set_ylabel(f"Account £ (start £{args.account:,.0f}, stake £{args.stake}/pt)")
    ax1.set_title(f"Stress Test — Equity Curves (median + p25/p75 band, fixed stake)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: bar chart of year-end medians
    ax2 = axes[1]
    labels = [s[1] for s in scenarios]
    p5s = [mc_results[s[0]]["p5"] for s in scenarios]
    p50s = [mc_results[s[0]]["p50"] for s in scenarios]
    p95s = [mc_results[s[0]]["p95"] for s in scenarios]
    colors = [s[2] for s in scenarios]
    x = np.arange(len(labels))
    w = 0.25
    ax2.bar(x - w, p5s, w, label="Worst (p5)", color=[c for c in colors], alpha=0.4, edgecolor="black")
    ax2.bar(x,     p50s, w, label="Median", color=colors, edgecolor="black")
    ax2.bar(x + w, p95s, w, label="Best (p95)", color=[c for c in colors], alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax2.set_ylabel("1-Year Net P&L (£)")
    ax2.set_title("1-Year Outcome Distribution by Stress Scenario")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\n  Saved chart → {args.out}")


if __name__ == "__main__":
    main()
