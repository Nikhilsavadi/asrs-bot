"""
monte_carlo.py — Monte Carlo simulation on validated firstrate backtest.

Resamples the 100k validated trade outcomes to project:
  - Distribution of max drawdowns
  - Probability of hitting £4k hard stop
  - 1-year, 6-month, 3-month return distributions
  - Worst-case streaks
"""
import argparse
import numpy as np
import pandas as pd

DEFAULT_RUNS = 5000


def run_mc(csv_path, instruments, stake_per_pt, account_start, hard_stop, runs, horizon_trades):
    df = pd.read_csv(csv_path)
    if instruments:
        df = df[df["instrument"].isin(instruments)]
    pnl_pts = df["pnl_pts"].values
    n_total = len(pnl_pts)
    avg_per_trade = pnl_pts.mean()
    pf = pnl_pts[pnl_pts > 0].sum() / abs(pnl_pts[pnl_pts < 0].sum())

    print(f"\n{'='*64}")
    print(f"  MONTE CARLO — {','.join(instruments) if instruments else 'all instruments'}")
    print(f"{'='*64}")
    print(f"  Source trades: {n_total:,}  |  PF {pf:.2f}  |  Avg {avg_per_trade:+.2f} pts/trade")
    print(f"  Stake: £{stake_per_pt}/pt  |  Account: £{account_start:,}  |  Hard stop: £{hard_stop:,}")
    print(f"  Horizon: {horizon_trades:,} trades per simulation  |  Runs: {runs:,}")

    rng = np.random.default_rng(42)
    final_pnls = np.zeros(runs)
    max_dds = np.zeros(runs)
    hit_hard_stop = 0
    worst_streak = np.zeros(runs)

    for i in range(runs):
        # Bootstrap with replacement (each simulation is an alternative reality)
        sample = rng.choice(pnl_pts, size=horizon_trades, replace=True)
        sample_gbp = sample * stake_per_pt
        equity = account_start + np.cumsum(sample_gbp)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dds[i] = dd.min()
        final_pnls[i] = equity[-1] - account_start

        # Did we hit the hard stop at any point?
        if equity.min() <= hard_stop:
            hit_hard_stop += 1

        # Worst losing streak (in pts)
        cum_neg = 0
        worst_neg = 0
        for s in sample:
            if s < 0:
                cum_neg += s
                worst_neg = min(worst_neg, cum_neg)
            else:
                cum_neg = 0
        worst_streak[i] = worst_neg * stake_per_pt

    # Stats
    pcts = [5, 25, 50, 75, 95]
    print(f"\n  Final P&L distribution (£):")
    for p in pcts:
        print(f"    p{p:>2}: {np.percentile(final_pnls, p):>+12,.0f}")
    print(f"    mean: {final_pnls.mean():>+12,.0f}")
    print(f"    std:  {final_pnls.std():>+12,.0f}")

    print(f"\n  Max drawdown distribution (£):")
    for p in pcts:
        print(f"    p{p:>2}: {np.percentile(max_dds, p):>+12,.0f}")
    print(f"    worst:{max_dds.min():>+12,.0f}")

    print(f"\n  Worst losing streak (£):")
    for p in pcts:
        print(f"    p{p:>2}: {np.percentile(worst_streak, p):>+12,.0f}")
    print(f"    worst:{worst_streak.min():>+12,.0f}")

    profitable = (final_pnls > 0).sum()
    print(f"\n  Profitability: {profitable/runs*100:.1f}% of simulations net positive")
    print(f"  Hard stop hit ({hard_stop}): {hit_hard_stop/runs*100:.2f}% of simulations")

    return {
        "final_pnl_mean": final_pnls.mean(),
        "final_pnl_p5": np.percentile(final_pnls, 5),
        "final_pnl_p95": np.percentile(final_pnls, 95),
        "max_dd_p5": np.percentile(max_dds, 5),
        "max_dd_worst": max_dds.min(),
        "hard_stop_pct": hit_hard_stop/runs*100,
        "profitable_pct": profitable/runs*100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/backtest_firstrate_results.csv")
    ap.add_argument("--stake", type=float, default=1.0, help="£ per point")
    ap.add_argument("--account", type=float, default=5000, help="Starting account £")
    ap.add_argument("--hard_stop", type=float, default=4000, help="Hard stop level £")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--instruments", nargs="*", default=None,
                    help="Filter to specific instruments e.g. US30 NIKKEI")
    args = ap.parse_args()

    # Trade-rate based horizons (~5,500 trades/year on 8 signals)
    horizons = [
        ("1 month",   458),
        ("3 months", 1375),
        ("6 months", 2750),
        ("12 months", 5500),
    ]

    print(f"\n{'#'*64}")
    print(f"  MONTE CARLO SIMULATION")
    print(f"  CSV: {args.csv}")
    print(f"  Stake: £{args.stake}/pt  Account: £{args.account:,}  Hard stop: £{args.hard_stop:,}")
    print(f"{'#'*64}")

    all_results = {}
    for name, n_trades in horizons:
        print(f"\n\n{'#'*30} {name.upper()} {'#'*30}")
        r = run_mc(args.csv, args.instruments, args.stake,
                   args.account, args.hard_stop, args.runs, n_trades)
        all_results[name] = r

    # Summary table
    print(f"\n\n{'='*64}")
    print(f"  SUMMARY")
    print(f"{'='*64}")
    print(f"  {'horizon':<12}{'mean P&L':>14}{'p5 P&L':>14}{'worst DD':>14}{'risk %':>10}")
    for h, r in all_results.items():
        print(f"  {h:<12}{r['final_pnl_mean']:>+14,.0f}{r['final_pnl_p5']:>+14,.0f}"
              f"{r['max_dd_worst']:>+14,.0f}{r['hard_stop_pct']:>9.2f}%")


if __name__ == "__main__":
    main()
