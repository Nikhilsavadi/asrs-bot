"""
mc_mfe_compare.py — Monte Carlo drawdown comparison between candle baseline
and MFE hybrid_50_0.5 exit variant.

5000 runs × 4 years × £0.50/pt starting £5k, block-bootstrap by trading day.
Reports equity percentiles + max DD percentiles + % profitable runs.
"""
import numpy as np
import pandas as pd

STARTING = 5000
DAYS = 252 * 4
RUNS = 5000
SEED = 42


def to_daily(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.groupby("date")["pnl_pts"].sum()


def sim(daily_pts, rng):
    vals = daily_pts.values
    sampled = rng.choice(vals, size=DAYS, replace=True)
    return STARTING + np.cumsum(sampled * 0.5)


def report(eq, label):
    final = eq[:, -1]
    runmax = np.maximum.accumulate(eq, axis=1)
    dd = (runmax - eq) / runmax
    maxdd = dd.max(axis=1) * 100
    cagr = ((final / STARTING) ** (1/4) - 1) * 100
    print(f"\n  {label}")
    print(f"    Final £:  p5={np.percentile(final,5):>9,.0f}  p50={np.percentile(final,50):>9,.0f}  p95={np.percentile(final,95):>9,.0f}")
    print(f"    CAGR %:   p5={np.percentile(cagr,5):>6.1f}   p50={np.percentile(cagr,50):>6.1f}   p95={np.percentile(cagr,95):>6.1f}")
    print(f"    Max DD%:  p5={np.percentile(maxdd,5):>6.1f}   p50={np.percentile(maxdd,50):>6.1f}   p95={np.percentile(maxdd,95):>6.1f}")
    print(f"    % profitable runs: {(final > STARTING).mean()*100:.1f}%")
    return final, maxdd


def main():
    d_bl = to_daily("/root/asrs-bot/data/mfe_mc_baseline.csv")
    d_mfe = to_daily("/root/asrs-bot/data/mfe_mc_hybrid_50_0.5.csv")
    print(f"Baseline days: {len(d_bl):,}   MFE days: {len(d_mfe):,}")
    print(f"Sim: {RUNS:,} runs × {DAYS:,} days @ £0.50/pt, starting £{STARTING:,}")
    print("=" * 90)

    rng = np.random.default_rng(SEED)
    eq_bl = np.array([sim(d_bl, rng) for _ in range(RUNS)])
    eq_mfe = np.array([sim(d_mfe, rng) for _ in range(RUNS)])

    f_bl, dd_bl = report(eq_bl, "CANDLE BASELINE")
    f_mfe, dd_mfe = report(eq_mfe, "MFE HYBRID 50_0.5")

    print(f"\n  HEAD-TO-HEAD uplift (MFE − Candle):")
    for q in [5, 25, 50, 75, 95]:
        f = np.percentile(f_mfe, q); b = np.percentile(f_bl, q)
        d = np.percentile(dd_mfe, q); db = np.percentile(dd_bl, q)
        print(f"    p{q:>2}: £{f:>9,.0f} vs £{b:>9,.0f} (Δ £{f-b:>+8,.0f} = {(f/b-1)*100:+5.1f}%)  |  DD {d:>5.1f}% vs {db:>5.1f}% (Δ {d-db:+5.1f}pp)")


if __name__ == "__main__":
    main()
