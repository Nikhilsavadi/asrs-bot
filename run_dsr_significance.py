"""
run_dsr_significance.py — Deflated Sharpe Ratio significance test
═══════════════════════════════════════════════════════════════

One-off script. Run BEFORE going to live with real money.

Computes the Deflated Sharpe Ratio (Bailey & López de Prado 2014) on the
out-of-sample test set (2020-2026), declaring how many strategy variants
were tried during research. DSR deflates the observed Sharpe by a
correction factor that accounts for multiple-testing bias — the more
configs you tried in research, the more your "good" backtest is
expected to be noise.

Decision rule:
  DSR > 0.95 → statistically significant. Safe to scale.
  DSR ∈ [0.80, 0.95] → borderline. Scale slowly, monitor closely.
  DSR < 0.80 → likely overfit. Do NOT scale.

Reference:
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality".
  Journal of Portfolio Management 40(5).
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

Usage:
  python3 run_dsr_significance.py
"""
import math
from collections import defaultdict
import pandas as pd

# How many strategy variants were tried during research?
# Be honest. Underestimating inflates DSR; overestimating deflates it.
# Conservative count: bar4 vs bar5 hybrid, BE buffer values (3 tested),
# trail rules (3 tested), max_entries (3 tested), risk caps (4 tested),
# ~4 instruments tried, ~5 sessions tested. Multiplicative ≈ 100-300.
# Use the wide end to be conservative.
NUM_TRIALS = 300

CSV = "/root/asrs-bot/data/backtest_firstrate_results.csv"

# OOS window: 2020-2026 (the period kept aside for final validation)
OOS_START = "2020-01-01"
OOS_END = "2026-12-31"


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_inv_cdf(p):
    """Inverse CDF (quantile function) using Beasley-Springer-Moro."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    # Acklam's algorithm constants
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
           ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def deflated_sharpe_ratio(returns, num_trials, periods_per_year=252):
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Returns probability that the true Sharpe > 0 after deflating the
    observed Sharpe by the variance of the maximum across N trials.
    """
    n = len(returns)
    if n < 30:
        raise ValueError(f"Need ≥30 returns, got {n}")

    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.5

    # Daily Sharpe and annualised Sharpe
    sr_daily = mean / std
    sr_annual = sr_daily * math.sqrt(periods_per_year)

    # Higher moments
    diffs = [r - mean for r in returns]
    m3 = sum(d ** 3 for d in diffs) / n
    m4 = sum(d ** 4 for d in diffs) / n
    skew = m3 / (std ** 3)
    kurt_excess = m4 / (std ** 4) - 3.0

    # E[SR_max] across N trials, assuming SR_n ~ N(0, 1) — Bailey 2014 eq 5
    # E[max] ≈ (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e))
    # where γ ≈ 0.5772 is Euler-Mascheroni
    euler = 0.5772156649
    e_const = math.e
    inv1 = normal_inv_cdf(1 - 1.0 / num_trials)
    inv2 = normal_inv_cdf(1 - 1.0 / (num_trials * e_const))
    expected_max_sr = (1 - euler) * inv1 + euler * inv2

    # Deflated benchmark Sharpe — daily units (sample SR is in daily units too)
    # The threshold the observed must beat is the *expected* max under null.
    # Convert expected_max_sr (in 'standardized' units) to a daily SR using
    # the sample's null SR distribution.
    # Variance of estimated SR (Mertens 2002):
    var_sr = (1 - skew * sr_daily + ((kurt_excess) / 4.0) * (sr_daily ** 2)) / (n - 1)
    if var_sr <= 0:
        return 0.5
    sd_sr = math.sqrt(var_sr)

    # PSR with deflated benchmark
    sr_threshold = expected_max_sr * sd_sr  # in same units as sr_daily
    z = (sr_daily - sr_threshold) / sd_sr
    dsr = normal_cdf(z)

    return {
        "n_trades": n,
        "mean_pnl": mean,
        "std_pnl": std,
        "sr_daily": sr_daily,
        "sr_annual": sr_annual,
        "skew": skew,
        "excess_kurt": kurt_excess,
        "expected_max_sr_under_null": expected_max_sr,
        "sr_threshold": sr_threshold,
        "dsr": dsr,
    }


def main():
    print("=" * 70)
    print("  DEFLATED SHARPE RATIO SIGNIFICANCE TEST")
    print(f"  OOS window: {OOS_START} → {OOS_END}")
    print(f"  NUM_TRIALS: {NUM_TRIALS} (research configs tried)")
    print("=" * 70)

    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    oos = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].copy()
    print(f"\nLoaded {len(df):,} total trades, {len(oos):,} in OOS window.")

    # Aggregate to daily P&L (one observation per trading day)
    daily = oos.groupby(oos["date"].dt.date)["pnl_pts"].sum()
    print(f"OOS trading days: {len(daily)}")
    print(f"Mean day: {daily.mean():+.1f} pts | Std: {daily.std():.1f} pts")
    print(f"Win rate (days): {(daily > 0).mean() * 100:.1f}%")

    if len(daily) < 30:
        print("ERROR: Not enough OOS days for DSR.")
        return

    result = deflated_sharpe_ratio(daily.tolist(), num_trials=NUM_TRIALS)

    print("\n" + "─" * 70)
    print("  RESULTS")
    print("─" * 70)
    print(f"  Observed daily Sharpe:        {result['sr_daily']:.4f}")
    print(f"  Observed annualised Sharpe:   {result['sr_annual']:.2f}")
    print(f"  Sample skew:                  {result['skew']:+.3f}")
    print(f"  Sample excess kurtosis:       {result['excess_kurt']:+.3f}")
    print(f"  Expected max SR under null:   {result['expected_max_sr_under_null']:.3f}")
    print(f"  Deflated SR threshold (daily): {result['sr_threshold']:.4f}")
    print(f"\n  DEFLATED SHARPE RATIO:        {result['dsr']:.4f}")

    print("\n" + "─" * 70)
    if result["dsr"] > 0.95:
        print("  ✅ DSR > 0.95 — STATISTICALLY SIGNIFICANT")
        print("     Edge is real. Safe to scale to full size.")
    elif result["dsr"] > 0.80:
        print("  ⚠️  DSR ∈ [0.80, 0.95] — BORDERLINE")
        print("     Edge probable but not bulletproof. Scale gradually.")
    else:
        print("  ❌ DSR < 0.80 — LIKELY OVERFIT")
        print("     Do NOT scale. Reduce config search, re-validate.")
    print("─" * 70)


if __name__ == "__main__":
    main()
