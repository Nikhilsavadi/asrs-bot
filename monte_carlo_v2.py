"""
monte_carlo_v2.py — Realistic Monte Carlo with serial correlation + stress.

Uses BLOCK bootstrap (samples whole trading days, not individual trades) so
clusters of bad days are preserved. Also runs stress scenarios with execution
degradation factors that simulate slippage, missed entries, regime divergence.
"""
import argparse
import numpy as np
import pandas as pd

DEFAULT_RUNS = 5000


def block_bootstrap(daily_pnls, n_days, rng):
    """Sample n_days whole trading days with replacement (preserves intra-day cluster)."""
    n_avail = len(daily_pnls)
    idx = rng.integers(0, n_avail, size=n_days)
    return daily_pnls[idx]


def run_block_mc(csv_path, instruments, stake_per_pt, account_start, hard_stop,
                  runs, days_horizon, degradation):
    df = pd.read_csv(csv_path)
    if instruments:
        df = df[df["instrument"].isin(instruments)]

    # Aggregate to (date, instrument-session-day) → daily P&L per instrument
    df["day_inst"] = df["date"] + "_" + df["instrument"]
    daily = df.groupby("day_inst")["pnl_pts"].sum()
    daily_pnls = daily.values
    n_days_avail = len(daily_pnls)

    # Apply degradation: shrink each daily P&L by factor on losing days,
    # and trim winners proportionally. This simulates "live PF = backtest PF × (1-deg)"
    if degradation > 0:
        # Realistic model: losers get bigger by factor, winners get smaller
        adj = daily_pnls.copy().astype(float)
        adj[adj > 0] *= (1 - degradation)
        adj[adj < 0] *= (1 + degradation)
        daily_pnls = adj

    pf = daily_pnls[daily_pnls > 0].sum() / abs(daily_pnls[daily_pnls < 0].sum())

    rng = np.random.default_rng(42)
    final_pnls = np.zeros(runs)
    max_dds = np.zeros(runs)
    hit_hard_stop = 0

    for i in range(runs):
        sample = block_bootstrap(daily_pnls, days_horizon, rng)
        sample_gbp = sample * stake_per_pt
        equity = account_start + np.cumsum(sample_gbp)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak).min()
        max_dds[i] = dd
        final_pnls[i] = equity[-1] - account_start
        if equity.min() <= hard_stop:
            hit_hard_stop += 1

    return {
        "pf": pf,
        "n_source_days": n_days_avail,
        "p5_pnl": np.percentile(final_pnls, 5),
        "p25_pnl": np.percentile(final_pnls, 25),
        "p50_pnl": np.percentile(final_pnls, 50),
        "p75_pnl": np.percentile(final_pnls, 75),
        "p95_pnl": np.percentile(final_pnls, 95),
        "mean_pnl": final_pnls.mean(),
        "p5_dd": np.percentile(max_dds, 5),
        "worst_dd": max_dds.min(),
        "p_loss": (final_pnls < 0).sum() / runs * 100,
        "p_hard_stop": hit_hard_stop / runs * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/backtest_firstrate_results.csv")
    ap.add_argument("--account", type=float, default=5000)
    ap.add_argument("--hard_stop", type=float, default=4000)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--instruments", nargs="*", default=None)
    args = ap.parse_args()

    # Trading days per horizon
    horizons = [
        ("1 month",   21),
        ("3 months",  63),
        ("6 months", 126),
        ("12 months",252),
    ]

    # Stake / degradation scenarios
    scenarios = [
        ("£0.5/pt   pure backtest",      0.5,  0.00),
        ("£0.5/pt   10% degradation",    0.5,  0.10),
        ("£0.5/pt   30% degradation",    0.5,  0.30),
        ("£0.5/pt   50% degradation",    0.5,  0.50),
        ("£1/pt     pure backtest",      1.0,  0.00),
        ("£1/pt     30% degradation",    1.0,  0.30),
    ]

    print(f"\n{'='*84}")
    print(f"  REALISTIC MONTE CARLO (block bootstrap by day, with degradation stress)")
    print(f"  Source: {args.csv}")
    print(f"  Account: £{args.account:,}  Hard stop: £{args.hard_stop:,}  Runs: {args.runs:,}")
    if args.instruments:
        print(f"  Instruments: {','.join(args.instruments)}")
    print(f"{'='*84}")

    for label, stake, deg in scenarios:
        print(f"\n  {label}")
        print(f"  {'horizon':<11}{'p5':>10}{'p25':>10}{'p50':>10}{'p75':>10}{'p95':>10}"
              f"{'worstDD':>10}{'lose%':>8}{'ruin%':>8}")
        for hname, hdays in horizons:
            r = run_block_mc(args.csv, args.instruments, stake, args.account,
                             args.hard_stop, args.runs, hdays, deg)
            print(f"  {hname:<11}{r['p5_pnl']:>+10,.0f}{r['p25_pnl']:>+10,.0f}"
                  f"{r['p50_pnl']:>+10,.0f}{r['p75_pnl']:>+10,.0f}{r['p95_pnl']:>+10,.0f}"
                  f"{r['worst_dd']:>+10,.0f}{r['p_loss']:>7.1f}%{r['p_hard_stop']:>7.2f}%")


if __name__ == "__main__":
    main()
