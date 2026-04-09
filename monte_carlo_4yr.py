"""
monte_carlo_4yr.py — 4-year MC with vol-targeted contract sizing + NKD gate.

REWRITTEN 2026-04-09 to match the post-IBKR-migration sizing model:
  - Per-instrument integer-contract sizing (DAX FDXS, US30 MYM, NIKKEI NKD)
  - Vol-targeted: contracts = floor(equity × 0.5% / (stop × £/pt))
  - Capped at MAX_CONTRACTS=5 per signal
  - NIKKEI GATED on equity ≥ £30k (NKD margin requirement)
  - NIKKEI uses RISK_PCT=0.66% override (per scaling plan)
  - No more £/pt scaling ladder — sizing scales linearly with equity

Block-bootstrap by trading day. Each sampled day brings ALL three
instruments' P&L points; the simulation applies the per-instrument
contract count current at that equity tier.

Outputs PNG with 4 panels: equity curve + daily/weekly/monthly £.
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
    if abs(x) >= 1_000_000:
        return f"£{x/1_000_000:.1f}M".replace(".0M", "M")
    if abs(x) >= 1_000:
        return f"£{x/1_000:.0f}k"
    return f"£{x:.0f}"


DEFAULT_RUNS = 2000
TRADING_DAYS_PER_YEAR = 252

# Per-instrument config — matches asrs/config.py + asrs/risk_gate.py
INSTRUMENTS = {
    "DAX": {
        "gbp_per_pt": 0.86,           # FDXS €1/pt → ~£0.86/pt
        "typical_stop_pts": 12.0,     # mean bar_range (8.4) + buffer*2 (4)
        "risk_pct": 0.5,
    },
    "US30": {
        "gbp_per_pt": 0.40,           # MYM $0.50/pt → ~£0.40/pt
        "typical_stop_pts": 30.0,     # mean bar_range (19.8) + buffer*2 (10)
        "risk_pct": 0.5,
    },
    "NIKKEI": {
        "gbp_per_pt": 4.00,           # NKD $5/pt → ~£4/pt
        "typical_stop_pts": 14.0,     # mean bar_range (10) + buffer*2 (4)
        "risk_pct": 0.66,             # override per scaling plan
        "min_equity_gbp": 30_000,     # NKD margin requirement gate
    },
}
MAX_CONTRACTS = 10  # bumped from 5 → 10 (2026-04-09); still well below NKD inside-liquidity ~22
STARTING_EQUITY = 5000

# Wife-account model
WIFE_TRIGGER_EQUITY = 60_000   # primary must reach this before gifting
WIFE_GIFT_AMOUNT    = 30_000   # how much to send to wife on trigger
WIFE_COOLDOWN_DAYS  = 60       # primary must be at trigger for N days before gifting


def contracts_for(inst: str, equity: float) -> int:
    """Vol-targeted contract count for one instrument at current equity.

    Returns 0 if instrument is gated out (e.g. NKD below margin threshold).
    Otherwise returns max(1, min(MAX_CONTRACTS, floor(budget / risk_per_lot))).
    """
    cfg = INSTRUMENTS[inst]
    if equity < cfg.get("min_equity_gbp", 0):
        return 0
    budget = equity * cfg["risk_pct"] / 100.0
    risk_per_lot = cfg["typical_stop_pts"] * cfg["gbp_per_pt"]
    raw = budget / risk_per_lot
    return max(1, min(MAX_CONTRACTS, int(raw)))


def _day_pnl_gbp(idx: int, equity: float,
                  daily_inst_pts: dict[str, np.ndarray],
                  degradation: float) -> tuple[float, dict]:
    """Compute one day's £ P&L given current equity (used by both accounts)."""
    day_gbp = 0.0
    qty_per_inst = {}
    for inst in INSTRUMENTS:
        qty = contracts_for(inst, equity)
        qty_per_inst[inst] = qty
        if qty == 0:
            continue
        pts = daily_inst_pts[inst][idx]
        if degradation > 0:
            pts = pts * (1 - degradation) if pts > 0 else pts * (1 + degradation)
        day_gbp += pts * qty * INSTRUMENTS[inst]["gbp_per_pt"]
    return day_gbp, qty_per_inst


def simulate_one(daily_inst_pts: dict[str, np.ndarray], n_days: int,
                 degradation: float, rng,
                 start_equity: float = STARTING_EQUITY,
                 dual_account: bool = False) -> tuple[np.ndarray, dict]:
    """
    Simulate n_days. Single or dual-account mode.

    Single mode: bot trades one account that grows from start_equity.

    Dual mode: starts as single. When primary equity ≥ WIFE_TRIGGER_EQUITY
    for WIFE_COOLDOWN_DAYS, gift WIFE_GIFT_AMOUNT to wife account. From
    that day forward, BOTH accounts run independently against the SAME
    sampled day (correlation = 1, which is realistic — same strategy on
    same instruments will move together).

    Returns: (combined_equity[n_days+1], debug)
    """
    n_avail = len(next(iter(daily_inst_pts.values())))
    sample_idx = rng.integers(0, n_avail, size=n_days)

    primary = np.zeros(n_days + 1)
    wife    = np.zeros(n_days + 1)
    primary[0] = start_equity
    wife[0] = 0.0

    contracts_hist = {inst: np.zeros(n_days, dtype=int) for inst in INSTRUMENTS}
    nikkei_on_day = -1
    wife_open_day = -1
    wife_nikkei_on_day = -1
    days_above_trigger = 0

    for i in range(n_days):
        idx = sample_idx[i]

        # Primary
        p_pnl, p_qty = _day_pnl_gbp(idx, primary[i], daily_inst_pts, degradation)
        for inst, q in p_qty.items():
            contracts_hist[inst][i] = q
        if nikkei_on_day < 0 and p_qty.get("NIKKEI", 0) > 0:
            nikkei_on_day = i
        primary_after = max(0.0, primary[i] + p_pnl)

        # Wife (only if already opened)
        if wife_open_day >= 0 and wife[i] > 0:
            w_pnl, w_qty = _day_pnl_gbp(idx, wife[i], daily_inst_pts, degradation)
            wife_after = max(0.0, wife[i] + w_pnl)
            if wife_nikkei_on_day < 0 and w_qty.get("NIKKEI", 0) > 0:
                wife_nikkei_on_day = i
        else:
            wife_after = wife[i]

        # Wife trigger check (only in dual mode, only if not yet opened)
        if dual_account and wife_open_day < 0:
            if primary_after >= WIFE_TRIGGER_EQUITY:
                days_above_trigger += 1
            else:
                days_above_trigger = 0
            # Also need PSR-equivalent: 30+ days post-NKD-on
            ready = (
                days_above_trigger >= WIFE_COOLDOWN_DAYS
                and nikkei_on_day >= 0
                and (i - nikkei_on_day) >= 60
                and primary_after - WIFE_GIFT_AMOUNT >= 30_000
            )
            if ready:
                primary_after -= WIFE_GIFT_AMOUNT
                wife_after = WIFE_GIFT_AMOUNT
                wife_open_day = i

        primary[i + 1] = primary_after
        wife[i + 1] = wife_after

    combined = primary + wife
    return combined, {
        "contracts": contracts_hist,
        "nikkei_on_day": nikkei_on_day,
        "wife_open_day": wife_open_day,
        "wife_nikkei_on_day": wife_nikkei_on_day,
        "primary": primary,
        "wife": wife,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/backtest_firstrate_results.csv")
    ap.add_argument("--account", type=float, default=STARTING_EQUITY)
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--degradation", type=float, default=0.30)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--out", default="data/monte_carlo_4yr.png")
    ap.add_argument("--no-nikkei", action="store_true",
                    help="Disable NIKKEI entirely (DAX+US30 only forever)")
    ap.add_argument("--dual", action="store_true",
                    help="Enable wife-account model (gifts £30k when primary ≥ £60k for 60d)")
    args = ap.parse_args()

    if args.no_nikkei:
        INSTRUMENTS.pop("NIKKEI", None)
        print("NIKKEI disabled (--no-nikkei flag)")

    df = pd.read_csv(args.csv)
    print(f"Source: {args.csv} ({len(df):,} trades)")

    # Pivot to per-(date, instrument) daily pts. Missing combos → 0.
    daily_per_inst = (
        df.groupby(["date", "instrument"])["pnl_pts"].sum().unstack(fill_value=0)
    )
    daily_inst_pts = {}
    for inst in INSTRUMENTS:
        if inst in daily_per_inst.columns:
            daily_inst_pts[inst] = daily_per_inst[inst].values.astype(float)
        else:
            daily_inst_pts[inst] = np.zeros(len(daily_per_inst))
    n_avail = len(daily_per_inst)
    print(f"  {n_avail:,} historical trading days")
    for inst, pts in daily_inst_pts.items():
        print(f"    {inst:7} mean +{pts.mean():.1f} pts/day, "
              f"std {pts.std():.1f}, win days {(pts > 0).mean()*100:.1f}%")

    print(f"\nAccount start: £{args.account:,.0f}  Years: {args.years}  "
          f"Degradation: {args.degradation*100:.0f}%  Runs: {args.runs:,}")
    print(f"MAX_CONTRACTS: {MAX_CONTRACTS}")
    print(f"NIKKEI gate:   equity ≥ £{INSTRUMENTS.get('NIKKEI', {}).get('min_equity_gbp', 0):,}")
    if args.dual:
        print(f"DUAL ACCOUNT: wife gets £{WIFE_GIFT_AMOUNT:,} when primary ≥ "
              f"£{WIFE_TRIGGER_EQUITY:,} for {WIFE_COOLDOWN_DAYS}d (post-NKD-on)")

    # Sizing preview at key equity tiers
    print(f"\nSIZING PREVIEW (contracts per instrument at each equity tier):")
    print(f"  {'Equity':<12}{'DAX':>6}{'US30':>6}{'NKD':>6}{'Daily £ exp':>15}")
    for tier in [5000, 10000, 15000, 20000, 30000, 50000, 100000, 250000, 500000]:
        sizes = {}
        exp_gbp = 0.0
        for inst in INSTRUMENTS:
            q = contracts_for(inst, tier)
            sizes[inst] = q
            mean_pts = daily_inst_pts[inst].mean() * (1 - args.degradation)
            exp_gbp += mean_pts * q * INSTRUMENTS[inst]["gbp_per_pt"]
        sz_str = f"{sizes.get('DAX',0):>6}{sizes.get('US30',0):>6}{sizes.get('NIKKEI',0):>6}"
        print(f"  £{tier:>10,}{sz_str}  £{exp_gbp:>+13,.0f}")

    n_days = args.years * TRADING_DAYS_PER_YEAR
    rng = np.random.default_rng(42)

    all_equity = np.zeros((args.runs, n_days + 1))
    nikkei_on_days = np.zeros(args.runs)
    wife_open_days = np.zeros(args.runs)
    for i in range(args.runs):
        eq, dbg = simulate_one(daily_inst_pts, n_days, args.degradation, rng,
                                start_equity=args.account, dual_account=args.dual)
        all_equity[i] = eq
        nikkei_on_days[i] = dbg["nikkei_on_day"]
        wife_open_days[i] = dbg["wife_open_day"]

    pcts = [5, 25, 50, 75, 95]
    bands = {p: np.percentile(all_equity, p, axis=0) for p in pcts}

    # Date index
    start_date = datetime.today().date()
    dates = [start_date]
    d = start_date
    while len(dates) < n_days + 1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)
    dates = dates[:n_days + 1]

    # Pick representative sample path
    final_pnls = all_equity[:, -1] - args.account
    target = np.percentile(final_pnls, 50)
    sample_idx = int(np.argmin(np.abs(final_pnls - target)))
    sample_eq = all_equity[sample_idx]
    daily_pnl = np.diff(sample_eq)

    # Re-simulate the sample path to get its contracts trajectory (deterministic w/ seed)
    rng2 = np.random.default_rng(42)
    for i in range(sample_idx + 1):
        _, sample_dbg = simulate_one(daily_inst_pts, n_days, args.degradation, rng2,
                                      start_equity=args.account)
    sample_contracts = sample_dbg["contracts"]
    sample_nikkei_on = sample_dbg["nikkei_on_day"]

    print(f"\nSAMPLE PATH #{sample_idx}: final P&L £{final_pnls[sample_idx]:+,.0f}")
    print(f"  Final equity:   £{sample_eq[-1]:,.0f}")
    print(f"  Worst day:      £{daily_pnl.min():+,.0f}")
    print(f"  Worst drawdown: £{(sample_eq - np.maximum.accumulate(sample_eq)).min():+,.0f}")
    print(f"  Losing days:    {(daily_pnl < 0).sum()}/{len(daily_pnl)} "
          f"({(daily_pnl < 0).sum()/len(daily_pnl)*100:.1f}%)")
    if sample_nikkei_on >= 0:
        yr = sample_nikkei_on / TRADING_DAYS_PER_YEAR
        print(f"  NKD enabled:    day {sample_nikkei_on} (year {yr:.2f}) — equity £{sample_eq[sample_nikkei_on]:,.0f}")
    else:
        print(f"  NKD enabled:    NEVER (sample never hit £30k)")

    # Aggregate
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

    losing_weeks = sum(1 for w in weekly_pnl if w < 0)
    losing_months = sum(1 for m in monthly_pnl if m < 0)
    print(f"  Losing weeks:   {losing_weeks}/{len(weekly_pnl)} "
          f"({losing_weeks/max(1,len(weekly_pnl))*100:.1f}%)")
    print(f"  Losing months:  {losing_months}/{len(monthly_pnl)} "
          f"({losing_months/max(1,len(monthly_pnl))*100:.1f}%)")

    # When does each percentile sim cross £30k (NKD trigger)?
    nkd_trigger_day = {}
    for p in pcts:
        crossings = np.where(bands[p] >= 30_000)[0]
        nkd_trigger_day[p] = int(crossings[0]) if len(crossings) > 0 else -1

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1, 1, 0.8], hspace=0.35)

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
    ax1.axhline(y=30_000, color="#d62728", linestyle="--", linewidth=1,
                label="NKD enable threshold (£30k)", alpha=0.6)

    # Mark NKD enable on sample path
    if sample_nikkei_on >= 0:
        ax1.axvline(x=dates[sample_nikkei_on], color="#d62728", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.annotate("NKD ON", xy=(dates[sample_nikkei_on], sample_eq[sample_nikkei_on]),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=9, color="#d62728", fontweight="bold")

    ax1.set_title(f"4-Year Monte Carlo Equity Curve  "
                  f"(£{args.account:,.0f} start, {args.degradation*100:.0f}% degradation, "
                  f"{args.runs:,} sims, NKD gated at £30k)")
    ax1.set_ylabel("Account")
    ax1.set_yscale("log")
    log_ticks = [5_000, 10_000, 30_000, 50_000, 100_000, 250_000, 500_000,
                 1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000]
    ax1.yaxis.set_major_locator(FixedLocator(log_ticks))
    ax1.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

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

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  4-YEAR MONTE CARLO SUMMARY  ({args.degradation*100:.0f}% degradation, NKD@£30k)")
    print(f"{'='*78}")
    print(f"  {'Year':<8}{'p5 (worst)':>14}{'p25':>14}{'Median':>14}{'p75':>14}{'p95 (best)':>14}")
    for yr in range(1, args.years + 1):
        idx = yr * TRADING_DAYS_PER_YEAR
        if idx >= len(bands[50]):
            idx = len(bands[50]) - 1
        row = "".join(f"£{bands[p][idx]:>12,.0f}" for p in pcts)
        print(f"  Year {yr}  {row}")

    print(f"\n  NKD ENABLE TIMING (when each percentile sim crosses £30k):")
    for p in pcts:
        d = nkd_trigger_day[p]
        if d < 0:
            print(f"    p{p:<3}: NEVER")
        else:
            yr = d / TRADING_DAYS_PER_YEAR
            print(f"    p{p:<3}: day {d} (year {yr:.2f})")

    if args.dual:
        opened = wife_open_days[wife_open_days >= 0]
        if len(opened) > 0:
            print(f"\n  WIFE ACCOUNT OPENED ({len(opened)}/{args.runs} sims = "
                  f"{len(opened)/args.runs*100:.0f}%):")
            for p in pcts:
                d = np.percentile(opened, p)
                yr = d / TRADING_DAYS_PER_YEAR
                print(f"    p{p:<3}: day {int(d)} (year {yr:.2f})")
        else:
            print(f"\n  WIFE ACCOUNT: never opened in any sim")

    median_final = bands[50][-1]
    print(f"\n  YEAR-BY-YEAR MEDIAN £:")
    print(f"    Year 1:  £{bands[50][min(252, len(bands[50])-1)] - args.account:+,.0f}")
    print(f"    Year 2:  £{bands[50][min(504, len(bands[50])-1)] - bands[50][min(252, len(bands[50])-1)]:+,.0f}")
    print(f"    Year 3:  £{bands[50][min(756, len(bands[50])-1)] - bands[50][min(504, len(bands[50])-1)]:+,.0f}")
    print(f"    Year 4:  £{bands[50][min(1008, len(bands[50])-1)] - bands[50][min(756, len(bands[50])-1)]:+,.0f}")
    print(f"    EOY 4:   £{median_final:,.0f}  ({(median_final/args.account - 1)*100:.0f}% total return)")


if __name__ == "__main__":
    main()
