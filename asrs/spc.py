"""
spc.py — Statistical process control for live strategy decay
═══════════════════════════════════════════════════════════════

Implements two academic-standard online drift detectors:

  1. Probabilistic Sharpe Ratio (PSR) — Bailey & López de Prado (2012).
     Probability that the *true* Sharpe ratio exceeds a benchmark Sharpe,
     given the observed sample, accounting for skew and excess kurtosis.
     PSR < 0.90 for 5 consecutive days = "your edge may have died."

  2. CUSUM on (live − backtest) PnL residuals.
     Cumulative sum of daily residuals. Breach of ±3σ = "live is
     systematically diverging from backtest."

Both fire Telegram alerts via the caller. Designed to piggyback on the
existing 21:00 UK replay_check cron in asrs/main.py.

References:
  - Bailey & López de Prado (2012) "The Sharpe Ratio Efficient Frontier"
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("ASRS.SPC")

TZ_UK = ZoneInfo("Europe/London")

# Benchmark from the 18yr backtest (firstrate, PF 4.22, 5-min stops)
# Annualised Sharpe — convert to daily for the PSR test.
# 18yr backtest: 100,046 trades, 19/19 years green, mean day +355pts.
# Estimated annualised Sharpe ~2.5 (high but plausible for PF 4.22).
BACKTEST_SHARPE_ANNUAL = 2.5
TRADING_DAYS_PER_YEAR = 252

# PSR alarm thresholds
PSR_ALARM_THRESHOLD = 0.90
PSR_CONSECUTIVE_DAYS_FOR_ALARM = 5

# CUSUM control limits (in daily-residual sigma units)
CUSUM_K = 0.5      # slack/reference value (typical)
CUSUM_H = 3.0      # control limit (±3σ)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _moments(returns: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, skew, kurtosis_excess) of a sample."""
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(returns) / n
    diffs = [r - mean for r in returns]
    var = sum(d * d for d in diffs) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return mean, 0.0, 0.0, 0.0
    m3 = sum(d ** 3 for d in diffs) / n
    m4 = sum(d ** 4 for d in diffs) / n
    skew = m3 / (std ** 3)
    kurt_excess = m4 / (std ** 4) - 3.0
    return mean, std, skew, kurt_excess


def probabilistic_sharpe_ratio(
    returns: list[float],
    benchmark_sharpe_annual: float = BACKTEST_SHARPE_ANNUAL,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    PSR — probability that the true Sharpe (annualised) exceeds the
    benchmark, given a finite sample with possible skew/kurt.

    Returns a value in [0, 1]. Higher = more confident the strategy
    is at least as good as the benchmark.
    """
    n = len(returns)
    if n < 30:
        return 0.5  # not enough data, agnostic
    mean, std, skew, kurt_excess = _moments(returns)
    if std == 0:
        return 0.5
    # Observed daily Sharpe (no rf assumed)
    sr_daily = mean / std
    sr_obs_annual = sr_daily * math.sqrt(periods_per_year)
    sr_bench_daily = benchmark_sharpe_annual / math.sqrt(periods_per_year)
    # PSR formula (Bailey & López de Prado 2012, eq 2):
    #   PSR = Φ( (SR̂ - SR*) · √(n-1) / √(1 - γ3·SR̂ + ((γ4-1)/4)·SR̂²) )
    denom_sq = 1.0 - skew * sr_daily + ((kurt_excess) / 4.0) * (sr_daily ** 2)
    if denom_sq <= 0:
        return 0.5
    z = (sr_daily - sr_bench_daily) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return _norm_cdf(z)


def cusum_check(
    residuals: list[float],
    k: float = CUSUM_K,
    h: float = CUSUM_H,
) -> tuple[float, float, bool]:
    """
    Two-sided CUSUM on standardised residuals.

    Returns (cusum_pos, cusum_neg, alarm).
    """
    if len(residuals) < 5:
        return 0.0, 0.0, False
    # Standardise
    mean, std, _, _ = _moments(residuals)
    if std == 0:
        return 0.0, 0.0, False
    z = [(r - mean) / std for r in residuals]
    cum_pos = 0.0
    cum_neg = 0.0
    alarm = False
    for zi in z:
        cum_pos = max(0.0, cum_pos + zi - k)
        cum_neg = min(0.0, cum_neg + zi + k)
        if cum_pos > h or cum_neg < -h:
            alarm = True
    return cum_pos, cum_neg, alarm


def get_recent_daily_pnl(days: int = 30) -> list[float]:
    """Pull last N trading days of GBP P&L from the journal, ordered oldest→newest.

    Honours RISK_GATE_START_DATE so pre-fix bug-period losses don't
    poison the SPC indicators forever.
    """
    try:
        import os
        from shared.journal_db import _get_conn
        conn = _get_conn()
        start_date = os.getenv("RISK_GATE_START_DATE", "")
        if start_date:
            rows = conn.execute(
                "SELECT date, COALESCE(SUM(pnl_gbp),0) AS pnl "
                "FROM trades WHERE date >= ? GROUP BY date ORDER BY date DESC LIMIT ?",
                (start_date, days),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT date, COALESCE(SUM(pnl_gbp),0) AS pnl "
                "FROM trades GROUP BY date ORDER BY date DESC LIMIT ?",
                (days,),
            ).fetchall()
        return [float(r["pnl"] or 0) for r in reversed(rows)]
    except Exception as e:
        logger.error(f"get_recent_daily_pnl failed: {e}")
        return []


def get_consecutive_low_psr_count() -> int:
    """Count of consecutive trailing days where PSR < threshold."""
    try:
        # Walk backward day-by-day computing PSR on the trailing window.
        # Cheap because we're called once a day.
        all_pnl = get_recent_daily_pnl(days=90)
        if len(all_pnl) < 35:
            return 0
        count = 0
        for end in range(len(all_pnl), 30, -1):
            window = all_pnl[max(0, end - 30):end]
            psr = probabilistic_sharpe_ratio(window)
            if psr < PSR_ALARM_THRESHOLD:
                count += 1
            else:
                break
        return count
    except Exception as e:
        logger.error(f"get_consecutive_low_psr_count failed: {e}")
        return 0


def daily_drift_report() -> dict:
    """
    Run the full SPC suite. Returns a dict with all indicators + an
    'alarm' bool that the caller can use to escalate the Telegram message.

    Designed to be called from asrs/main.py:_replay_check after the replay
    diff is computed.
    """
    pnl_30 = get_recent_daily_pnl(days=30)
    n_days = len(pnl_30)
    # Need ≥30 days of post-fix data before any indicator fires.
    # Until then, all metrics are reported as informational, no alarm.
    enough_data = n_days >= 30

    psr = probabilistic_sharpe_ratio(pnl_30) if enough_data else None
    consecutive_low = get_consecutive_low_psr_count() if enough_data else 0

    # CUSUM on (live - backtest_expected_daily_mean)
    # Backtest mean day = +£177 at £0.50/pt (355 pts × £0.50)
    expected_daily_gbp = 177.0
    residuals = [p - expected_daily_gbp for p in pnl_30]
    cum_pos, cum_neg, cusum_alarm_raw = cusum_check(residuals) if enough_data else (0.0, 0.0, False)

    cusum_alarm = enough_data and cusum_alarm_raw
    psr_alarm = enough_data and psr is not None and psr < PSR_ALARM_THRESHOLD and \
                consecutive_low >= PSR_CONSECUTIVE_DAYS_FOR_ALARM

    return {
        "n_days": len(pnl_30),
        "psr": psr,
        "psr_alarm": psr_alarm,
        "consecutive_low_psr_days": consecutive_low,
        "cusum_pos": cum_pos,
        "cusum_neg": cum_neg,
        "cusum_alarm": cusum_alarm,
        "alarm": psr_alarm or cusum_alarm,
    }


def format_drift_report(report: dict) -> str:
    """Format for Telegram (HTML). Includes inline playbook on alarm."""
    psr = report.get("psr")
    psr_str = f"{psr:.2f}" if psr is not None else "n/a"
    cum_pos = report.get("cusum_pos", 0)
    cum_neg = report.get("cusum_neg", 0)
    n = report.get("n_days", 0)
    psr_alarm = report.get("psr_alarm", False)
    cusum_alarm = report.get("cusum_alarm", False)
    both = psr_alarm and cusum_alarm
    tag = "ALARM" if report.get("alarm") else "OK"

    lines = [
        f"<b>STRATEGY HEALTH</b> [{tag}]",
        f"Window: last {n} days",
        f"PSR: {psr_str} (threshold {PSR_ALARM_THRESHOLD:.2f})",
        f"  Consecutive low days: {report.get('consecutive_low_psr_days', 0)} / {PSR_CONSECUTIVE_DAYS_FOR_ALARM}",
        f"CUSUM: pos={cum_pos:+.2f} neg={cum_neg:+.2f} (limit ±{CUSUM_H:.1f})",
    ]

    # Healthy — no playbook needed, brief message
    if not report.get("alarm"):
        return "\n".join(lines)

    # Alarm fired — include investigation playbook inline
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("<b>WHAT THIS MEANS</b>")

    if both:
        lines.append(
            "Both PSR and CUSUM fired — multi-method consensus that "
            "live performance is statistically distinct from backtest. "
            "Most serious tier of alert."
        )
    elif psr_alarm:
        lines.append(
            "Rolling 30-day Sharpe has been below the backtest benchmark "
            f"for {report.get('consecutive_low_psr_days', 0)} consecutive days. "
            "Underperformance pattern is building. Not yet definitive — "
            "could be a vol regime headwind."
        )
    elif cusum_alarm:
        if cum_neg < -CUSUM_H:
            lines.append(
                "Daily P&L has been systematically below backtest "
                "expectation for several weeks. Drift, not noise. "
                "Most likely culprits: one signal decaying, vol regime "
                "change, or microstructure shift in one instrument."
            )
        else:
            lines.append(
                "Daily P&L has been systematically ABOVE backtest "
                "expectation. Sounds great but is a flag — usually means "
                "the comparison baseline is stale or live is taking "
                "DIFFERENT trades than the backtest. Cross-check the "
                "DAILY PARITY message above for delta drift."
            )

    lines.append("")
    lines.append("<b>NEXT MORNING CHECKLIST</b>")
    lines.append("Do NOT act tonight. Tomorrow before market open:")
    lines.append("1. <code>/pnl 30</code> — break down by signal")
    lines.append("2. Identify worst signal (likely one carrying the loss)")
    lines.append("3. Check vol regime: VIX, recent CPI/FOMC, holidays")
    lines.append("4. Compare absolute loss vs 18yr historical worst window")

    lines.append("")
    lines.append("<b>DECISION RULES</b>")

    if both:
        lines.append(
            "• <b>Pause to Monday</b> if absolute loss is in worst 5% "
            "of historical weeks AND no regime explanation."
        )
        lines.append(
            "• <b>Halve risk</b> (<code>RISK_PCT_PER_TRADE=0.25</code>, restart) "
            "if loss is across multiple signals but within historical worst."
        )
        lines.append(
            "• <b>Continue full size</b> only if one signal is the obvious "
            "culprit AND you disable that signal."
        )
    else:
        lines.append(
            "• <b>Continue full size</b> if one signal is the obvious "
            "culprit (disable just that signal) OR a regime event "
            "explains it (CPI week, FOMC, holiday)."
        )
        lines.append(
            "• <b>Halve risk</b> (<code>RISK_PCT_PER_TRADE=0.25</code>, restart) "
            "if underperformance is across multiple signals with no "
            "clear explanation, but loss is within historical worst."
        )
        lines.append(
            "• <b>Pause to Monday</b> only if BOTH metrics fire on the "
            "same day or absolute loss exceeds historical worst."
        )

    lines.append("")
    lines.append(
        "<i>Investigate calmly. The strategy recovers from drawdowns. "
        "It does not recover from panic-pausing a normal drawdown OR "
        "from ignoring a real decay.</i>"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # CLI smoke test
    r = daily_drift_report()
    print(format_drift_report(r))
    print()
    print(r)
