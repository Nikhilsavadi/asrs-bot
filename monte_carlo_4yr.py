"""
monte_carlo_4yr.py — 4-year MC with scaling ladder, plotted daily/weekly/monthly.

Simulates 4 years of trading with stake auto-scaling per the ladder:
  £0.50/pt → £1/pt at £8k → £2/pt at £15k → £4/pt at £30k → £8/pt at £55k →
  £15/pt at £110k → £30/pt at £200k → £60/pt at £400k (cap)

Block bootstrap by trading day, 30% degradation default.
Outputs PNG with 3 panels: daily equity, weekly bars, monthly bars.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, FixedLocator
from datetime import datetime, timedelta


def _money_fmt(x, _pos=None):
    """Format £ values as 5k, 100k, 1.5M etc."""
    if x >= 1_000_000:
        return f"£{x/1_000_000:.1f}M".replace(".0M", "M")
    if x >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:.0f}"

DEFAULT_RUNS = 2000
TRADING_DAYS_PER_YEAR = 252

# Scaling ladder: (account threshold £, stake £/pt)
# Each step ALSO requires MIN_DAYS_AT_LEVEL trading days at the previous
# stake before upgrading — so the bot can't race up the ladder in weeks.
LADDER = [
    (5000,    0.5),
    (8000,    1.0),
    (15000,   2.0),
    (30000,   4.0),
    (55000,   8.0),
    (110000,  15.0),
    (200000,  30.0),
    (400000,  60.0),     # ceiling — never scales above this
]
MIN_DAYS_AT_LEVEL = 60   # ~3 months — must trade clean at each step before upgrading


def stake_for_account_with_cooldown(account: float, current_stake: float,
                                     days_at_current: int) -> float:
    """Pick the next stake from the ladder, but only if we've spent
    MIN_DAYS_AT_LEVEL at the current one. Always allow downgrades immediately.
    """
    target = LADDER[0][1]
    for thresh, st in LADDER:
        if account >= thresh:
            target = st
        else:
            break
    # Allow downgrade always (drawdown protection)
    if target < current_stake:
        return target
    # Allow upgrade only if cooldown elapsed
    if target > current_stake and days_at_current < MIN_DAYS_AT_LEVEL:
        return current_stake
    return target


def simulate_one(daily_pnls_pts: np.ndarray, account_start: float, n_days: int,
                  degradation: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_days of trading with auto-scaling stake.

    Returns:
        equity[n_days+1]: account value over time (£)
        stakes[n_days]:    stake used each day (£/pt)
    """
    # Apply degradation
    if degradation > 0:
        daily_adj = daily_pnls_pts.astype(float).copy()
        daily_adj[daily_adj > 0] *= (1 - degradation)
        daily_adj[daily_adj < 0] *= (1 + degradation)
    else:
        daily_adj = daily_pnls_pts.astype(float)

    n_avail = len(daily_adj)
    sample_idx = rng.integers(0, n_avail, size=n_days)
    sample_pts = daily_adj[sample_idx]

    equity = np.zeros(n_days + 1)
    equity[0] = account_start
    stakes = np.zeros(n_days)

    current_stake = LADDER[0][1]
    days_at_current = MIN_DAYS_AT_LEVEL  # allow first scale-up immediately if account warrants

    for i in range(n_days):
        new_stake = stake_for_account_with_cooldown(equity[i], current_stake, days_at_current)
        if new_stake != current_stake:
            current_stake = new_stake
            days_at_current = 0
        else:
            days_at_current += 1
        stakes[i] = current_stake
        equity[i + 1] = equity[i] + sample_pts[i] * current_stake

    return equity, stakes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/backtest_firstrate_results.csv")
    ap.add_argument("--account", type=float, default=5000)
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--degradation", type=float, default=0.30)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--out", default="data/monte_carlo_4yr.png")
    ap.add_argument("--instruments", nargs="*", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.instruments:
        df = df[df["instrument"].isin(args.instruments)]

    # Aggregate to daily P&L per (date,instrument), then sum across instruments per date
    daily = df.groupby("date")["pnl_pts"].sum().values

    print(f"Source: {args.csv}")
    print(f"  daily P&L points (cross-instrument): {len(daily):,}")
    print(f"  mean: {daily.mean():+.1f} pts/day  std: {daily.std():.1f}")
    print(f"  Account start: £{args.account:,.0f}  Years: {args.years}  Degradation: {args.degradation*100:.0f}%")
    print(f"  Runs: {args.runs:,}")

    n_days = args.years * TRADING_DAYS_PER_YEAR
    rng = np.random.default_rng(42)

    # Run all simulations, store full equity curves
    all_equity = np.zeros((args.runs, n_days + 1))
    all_stakes = np.zeros((args.runs, n_days))
    for i in range(args.runs):
        eq, st = simulate_one(daily, args.account, n_days, args.degradation, rng)
        all_equity[i] = eq
        all_stakes[i] = st

    # Compute percentile bands for each day
    pcts = [5, 25, 50, 75, 95]
    bands = {p: np.percentile(all_equity, p, axis=0) for p in pcts}

    # Build a date index for plotting
    start_date = datetime.today().date()
    dates = [start_date]
    d = start_date
    while len(dates) < n_days + 1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)
    dates = dates[:n_days + 1]

    # ── Pick a representative SAMPLE PATH for the bar charts ──────────
    # The median across N sims smooths out variance — every day looks
    # positive because positive days slightly outnumber negative ones in
    # any large sample. To show realistic chop, pick ONE simulation
    # whose final P&L is close to the median. That single path has all
    # the variance of real life: losing days, losing weeks, occasional
    # losing months.
    median_eq = bands[50]
    final_pnls = all_equity[:, -1] - args.account
    target = np.percentile(final_pnls, 50)
    sample_idx = int(np.argmin(np.abs(final_pnls - target)))
    sample_eq = all_equity[sample_idx]
    sample_stakes = all_stakes[sample_idx]
    daily_pnl = np.diff(sample_eq)
    print(f"\nSample path #{sample_idx}: final P&L £{final_pnls[sample_idx]:+,.0f} "
          f"(median across all sims: £{target:+,.0f})")
    print(f"  Losing days:    {(daily_pnl < 0).sum()} / {len(daily_pnl)} "
          f"({(daily_pnl < 0).sum()/len(daily_pnl)*100:.1f}%)")
    print(f"  Worst day:      £{daily_pnl.min():+,.0f}")
    print(f"  Worst drawdown: £{(sample_eq - np.maximum.accumulate(sample_eq)).min():+,.0f}")

    weekly_pnl = []
    monthly_pnl = []
    week_buf = []
    month_buf = []
    cur_week = dates[1].isocalendar().week
    cur_month = dates[1].month
    week_dates = []
    month_dates = []
    for i, day in enumerate(dates[1:]):
        if day.isocalendar().week != cur_week:
            weekly_pnl.append(sum(week_buf))
            week_dates.append(dates[i])
            week_buf = []
            cur_week = day.isocalendar().week
        if day.month != cur_month:
            monthly_pnl.append(sum(month_buf))
            month_dates.append(dates[i])
            month_buf = []
            cur_month = day.month
        week_buf.append(daily_pnl[i])
        month_buf.append(daily_pnl[i])
    if week_buf:
        weekly_pnl.append(sum(week_buf))
        week_dates.append(dates[-1])
    if month_buf:
        monthly_pnl.append(sum(month_buf))
        month_dates.append(dates[-1])

    # Stats for the sample path
    losing_weeks = sum(1 for w in weekly_pnl if w < 0)
    losing_months = sum(1 for m in monthly_pnl if m < 0)
    print(f"  Losing weeks:   {losing_weeks} / {len(weekly_pnl)} "
          f"({losing_weeks/max(1,len(weekly_pnl))*100:.1f}%)")
    print(f"  Losing months:  {losing_months} / {len(monthly_pnl)} "
          f"({losing_months/max(1,len(monthly_pnl))*100:.1f}%)")

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1, 1, 0.8], hspace=0.35)

    # 1. Equity curves with confidence bands + sample path overlay
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(dates, bands[5], bands[95], alpha=0.15,
                     color="#2ca02c", label="p5–p95 (90% CI)")
    ax1.fill_between(dates, bands[25], bands[75], alpha=0.30,
                     color="#2ca02c", label="p25–p75 (50% CI)")
    ax1.plot(dates, bands[50], color="#1f7a1f", linewidth=1.5,
             label="Median across all sims", alpha=0.7, linestyle="--")
    ax1.plot(dates, sample_eq, color="#0d4b0d", linewidth=2.0,
             label=f"Sample path #{sample_idx} (real chop)")
    ax1.axhline(y=args.account, color="#888", linestyle=":", linewidth=1, label="Start")

    # Mark stake-up events on the SAMPLE path
    last_stake = LADDER[0][1]
    for i, st in enumerate(sample_stakes):
        if st != last_stake:
            ax1.axvline(x=dates[i], color="#bbbbbb", linewidth=0.5, linestyle=":")
            ax1.annotate(f"£{st:.0f}/pt", xy=(dates[i], sample_eq[i]),
                         xytext=(5, -5), textcoords="offset points",
                         fontsize=7, color="#666")
            last_stake = st

    ax1.set_title(f"4-Year Monte Carlo Equity Curve  "
                  f"(£{args.account:,.0f} start, {args.degradation*100:.0f}% degradation, {args.runs:,} sims)")
    ax1.set_ylabel("Account")
    ax1.set_yscale("log")
    # Friendly £ ticks at 5k, 10k, 50k, 100k, 500k, 1M, 5M, 10M, 50M
    log_ticks = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000,
                 1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000]
    ax1.yaxis.set_major_locator(FixedLocator(log_ticks))
    ax1.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 2. Daily P&L (sample path — shows real losing days)
    ax2 = fig.add_subplot(gs[1])
    colors = ["#2ca02c" if p > 0 else "#d62728" for p in daily_pnl]
    ax2.bar(dates[1:], daily_pnl, color=colors, width=1.5)
    ax2.axhline(y=0, color="#888", linewidth=0.5)
    losing_days_pct = (daily_pnl < 0).sum() / len(daily_pnl) * 100
    ax2.set_title(f"Daily P&L (sample path) — {losing_days_pct:.0f}% losing days")
    ax2.set_ylabel("£/day")
    ax2.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 3. Weekly P&L (sample path)
    ax3 = fig.add_subplot(gs[2])
    wcolors = ["#2ca02c" if p > 0 else "#d62728" for p in weekly_pnl]
    ax3.bar(week_dates, weekly_pnl, color=wcolors, width=5)
    ax3.axhline(y=0, color="#888", linewidth=0.5)
    lw_pct = losing_weeks / max(1, len(weekly_pnl)) * 100
    ax3.set_title(f"Weekly P&L (sample path) — {losing_weeks}/{len(weekly_pnl)} losing weeks ({lw_pct:.0f}%)")
    ax3.set_ylabel("£/week")
    ax3.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 4. Monthly P&L
    ax4 = fig.add_subplot(gs[3])
    mcolors = ["#2ca02c" if p > 0 else "#d62728" for p in monthly_pnl]
    ax4.bar(month_dates, monthly_pnl, color=mcolors, width=20)
    ax4.axhline(y=0, color="#888", linewidth=0.5)
    lm_pct = losing_months / max(1, len(monthly_pnl)) * 100
    ax4.set_title(f"Monthly P&L (sample path) — {losing_months}/{len(monthly_pnl)} losing months ({lm_pct:.0f}%)")
    ax4.set_ylabel("£/month")
    ax4.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved chart → {args.out}")

    # ── Print summary table ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  4-YEAR MONTE CARLO SUMMARY  ({args.degradation*100:.0f}% degradation)")
    print(f"{'='*70}")
    print(f"  {'Year':<8}{'Worst (p5)':>14}{'Conservative':>14}{'Median':>14}{'Optimistic':>14}{'Best (p95)':>14}")
    for yr in range(1, args.years + 1):
        idx = yr * TRADING_DAYS_PER_YEAR
        if idx >= len(bands[50]):
            idx = len(bands[50]) - 1
        row = [yr]
        for p in pcts:
            row.append(f"£{bands[p][idx]:>12,.0f}")
        print(f"  Year {yr}  " + "".join(f"{v:>14}" for v in row[1:]))

    # Sample path stake progression
    print(f"\n  STAKE LADDER PROGRESSION (sample path #{sample_idx}):")
    last_stake = LADDER[0][1]
    for i, st in enumerate(sample_stakes):
        if st != last_stake:
            year_frac = i / TRADING_DAYS_PER_YEAR
            print(f"    Day {i:>4} (Year {year_frac:.1f}): £{last_stake:.1f}/pt → £{st:.1f}/pt  "
                  f"(account ≈ £{sample_eq[i]:,.0f})")
            last_stake = st

    # Median per-year P&L
    print(f"\n  MEDIAN £/PERIOD (rolling):")
    final = bands[50][-1]
    print(f"    Day  1:  £{daily_pnl[0]:+,.0f}")
    print(f"    Week 1:  £{sum(daily_pnl[:5]):+,.0f}")
    print(f"    Month 1: £{sum(daily_pnl[:21]):+,.0f}")
    print(f"    Year 1:  £{bands[50][min(252, len(bands[50])-1)] - args.account:+,.0f}")
    print(f"    Year 2:  £{bands[50][min(504, len(bands[50])-1)] - bands[50][min(252, len(bands[50])-1)]:+,.0f}")
    print(f"    Year 3:  £{bands[50][min(756, len(bands[50])-1)] - bands[50][min(504, len(bands[50])-1)]:+,.0f}")
    print(f"    Year 4:  £{bands[50][min(1008, len(bands[50])-1)] - bands[50][min(756, len(bands[50])-1)]:+,.0f}")
    print(f"    EOY 4:   £{final:,.0f}  ({(final/args.account - 1)*100:.0f}% total return)")


if __name__ == "__main__":
    main()
