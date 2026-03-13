"""
charts.py — Backtest Visualisation
═══════════════════════════════════════════════════════════════════════════════

Generates publication-quality charts from backtest results:
  1. Equity curve (cumulative P&L over time)
  2. Win/loss ratio breakdown (bar chart + expectancy)
  3. Monthly P&L heatmap
  4. P&L distribution histogram
  5. V58 overnight bias comparison
  6. Drawdown chart

Usage:
    from charts import generate_all_charts
    generate_all_charts(results, output_dir="data/charts")

All charts saved as PNG files + a combined dashboard.
"""

import os
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless — no display needed on VPS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

CHART_DIR = os.path.join(os.path.dirname(__file__), "data", "dax", "charts")


def generate_all_charts(results: list, output_dir: str = None) -> list[str]:
    """
    Generate all charts from backtest results.
    Returns list of saved file paths.
    """
    if output_dir is None:
        output_dir = CHART_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_trades = [t for r in results for t in r.trades]
    if not all_trades:
        logger.warning("No trades to chart")
        return []

    saved = []

    # Individual charts
    saved.append(chart_equity_curve(results, output_dir))
    saved.append(chart_win_loss_ratio(all_trades, output_dir))
    saved.append(chart_monthly_pnl(results, output_dir))
    saved.append(chart_pnl_distribution(all_trades, output_dir))
    saved.append(chart_drawdown(results, output_dir))
    saved.append(chart_v58_analysis(all_trades, output_dir))
    saved.append(chart_day_of_week(all_trades, output_dir))

    # Combined dashboard
    saved.append(chart_dashboard(results, all_trades, output_dir))

    saved = [s for s in saved if s]  # Remove None entries
    logger.info(f"Generated {len(saved)} charts in {output_dir}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
#  1. EQUITY CURVE
# ══════════════════════════════════════════════════════════════════════════════

def chart_equity_curve(results: list, output_dir: str) -> str:
    """Cumulative P&L equity curve with drawdown shading."""
    dates = []
    equity = []
    running = 0

    for r in results:
        if r.triggered:
            dates.append(datetime.strptime(r.date, "%Y-%m-%d"))
            running += r.total_pnl
            equity.append(running)

    if not dates:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    # Equity line
    ax.plot(dates, equity, color="#2196F3", linewidth=1.5, label="Equity (pts)")
    ax.fill_between(dates, 0, equity,
                    where=[e >= 0 for e in equity],
                    alpha=0.15, color="#4CAF50")
    ax.fill_between(dates, 0, equity,
                    where=[e < 0 for e in equity],
                    alpha=0.15, color="#F44336")

    # Drawdown shading
    peaks = np.maximum.accumulate(equity)
    dd = [e - p for e, p in zip(equity, peaks)]

    # Zero line
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Formatting
    ax.set_title("ASRS Equity Curve", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (points)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Stats annotation
    final_pnl = equity[-1]
    max_dd = abs(min(dd)) if dd else 0
    total_days = len(dates)
    stats_text = (
        f"Final P&L: {final_pnl:+.0f} pts\n"
        f"Max Drawdown: {max_dd:.0f} pts\n"
        f"Trading Days: {total_days}"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()

    path = os.path.join(output_dir, "equity_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  2. WIN/LOSS RATIO
# ══════════════════════════════════════════════════════════════════════════════

def chart_win_loss_ratio(trades: list, output_dir: str) -> str:
    """Win/loss breakdown with expectancy calculation."""
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts < 0]
    flat = [t for t in trades if t.pnl_pts == 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl_pts for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pts for t in losses]) if losses else 0

    # Expectancy = (Win% × Avg Win) + (Loss% × Avg Loss)
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

    # Profit factor = Gross wins / Gross losses
    gross_win = sum(t.pnl_pts for t in wins)
    gross_loss = abs(sum(t.pnl_pts for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Risk/reward ratio
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Win/Loss count
    ax1 = axes[0]
    bars = ax1.bar(["Winners", "Losers", "Flat"],
                   [len(wins), len(losses), len(flat)],
                   color=["#4CAF50", "#F44336", "#9E9E9E"],
                   edgecolor="white", linewidth=1.5)
    ax1.set_title("Trade Outcomes", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, [len(wins), len(losses), len(flat)]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Trades")

    # Panel 2: Avg Win vs Avg Loss
    ax2 = axes[1]
    bars2 = ax2.bar(["Avg Win", "Avg Loss"],
                    [avg_win, abs(avg_loss)],
                    color=["#4CAF50", "#F44336"],
                    edgecolor="white", linewidth=1.5)
    ax2.set_title("Win vs Loss Size", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, [avg_win, avg_loss]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:+.1f}", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Points")

    # Panel 3: Key metrics
    ax3 = axes[2]
    ax3.axis("off")
    metrics = [
        ("Win Rate", f"{win_rate:.1f}%"),
        ("Avg Win", f"{avg_win:+.1f} pts"),
        ("Avg Loss", f"{avg_loss:+.1f} pts"),
        ("R:R Ratio", f"{rr_ratio:.2f}"),
        ("Expectancy", f"{expectancy:+.1f} pts/trade"),
        ("Profit Factor", f"{profit_factor:.2f}"),
        ("Total Trades", f"{len(trades)}"),
        ("Gross P&L", f"{gross_win - gross_loss:+.0f} pts"),
    ]

    verdict = "✅ EDGE EXISTS" if expectancy > 0 and profit_factor > 1.2 else "⚠️ MARGINAL" if expectancy > 0 else "❌ NO EDGE"
    verdict_color = "#4CAF50" if "EXISTS" in verdict else "#FF9800" if "MARGINAL" in verdict else "#F44336"

    y_start = 0.95
    ax3.text(0.5, y_start + 0.05, verdict, fontsize=16, fontweight="bold",
             ha="center", va="top", color=verdict_color,
             transform=ax3.transAxes)

    for i, (label, value) in enumerate(metrics):
        y = y_start - (i * 0.11)
        ax3.text(0.1, y, label, fontsize=11, ha="left",
                 transform=ax3.transAxes, color="#555")
        ax3.text(0.9, y, value, fontsize=11, ha="right",
                 fontweight="bold", transform=ax3.transAxes)

    fig.suptitle("ASRS Win/Loss Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "win_loss_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  3. MONTHLY P&L HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def chart_monthly_pnl(results: list, output_dir: str) -> str:
    """Monthly P&L bar chart with cumulative line overlay."""
    monthly = defaultdict(float)
    for r in results:
        m = r.date[:7]
        monthly[m] += r.total_pnl

    months = sorted(monthly.keys())
    pnls = [round(monthly[m], 1) for m in months]
    cum = list(np.cumsum(pnls))

    fig, ax1 = plt.subplots(figsize=(14, 6))

    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
    bars = ax1.bar(range(len(months)), pnls, color=colors, alpha=0.7, edgecolor="white")
    ax1.set_ylabel("Monthly P&L (points)", color="#333")
    ax1.set_title("ASRS Monthly Performance", fontsize=16, fontweight="bold", pad=15)

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, pnls)):
        if val != 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (2 if val >= 0 else -8),
                     f"{val:+.0f}", ha="center", fontsize=8, fontweight="bold")

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(months)), cum, color="#2196F3", linewidth=2,
             marker="o", markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative P&L (points)", color="#2196F3")

    # X-axis
    ax1.set_xticks(range(len(months)))
    ax1.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax1.grid(True, alpha=0.2, axis="y")

    # Win months / loss months count
    win_months = sum(1 for p in pnls if p > 0)
    lose_months = sum(1 for p in pnls if p < 0)
    ax1.text(0.02, 0.95,
             f"Win months: {win_months}/{len(months)} ({round(win_months/len(months)*100)}%)",
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "monthly_pnl.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  4. P&L DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_pnl_distribution(trades: list, output_dir: str) -> str:
    """Histogram of trade P&L outcomes."""
    pnls = [t.pnl_pts for t in trades]

    fig, ax = plt.subplots(figsize=(12, 5))

    n, bins, patches = ax.hist(pnls, bins=40, edgecolor="white", linewidth=0.8)

    # Color bars green/red based on sign
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0:
            patch.set_facecolor("#4CAF50")
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor("#F44336")
            patch.set_alpha(0.7)

    # Mean line
    mean_pnl = np.mean(pnls)
    ax.axvline(mean_pnl, color="#2196F3", linewidth=2, linestyle="--",
               label=f"Mean: {mean_pnl:+.1f} pts")

    # Median line
    median_pnl = np.median(pnls)
    ax.axvline(median_pnl, color="#FF9800", linewidth=2, linestyle=":",
               label=f"Median: {median_pnl:+.1f} pts")

    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title("P&L Distribution per Trade", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("P&L (points)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "pnl_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  5. DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════

def chart_drawdown(results: list, output_dir: str) -> str:
    """Underwater equity chart (drawdown from peak)."""
    dates = []
    equity = []
    running = 0

    for r in results:
        if r.triggered:
            dates.append(datetime.strptime(r.date, "%Y-%m-%d"))
            running += r.total_pnl
            equity.append(running)

    if not dates:
        return None

    peaks = np.maximum.accumulate(equity)
    dd = [e - p for e, p in zip(equity, peaks)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dates, dd, 0, color="#F44336", alpha=0.4)
    ax.plot(dates, dd, color="#F44336", linewidth=1)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    max_dd = min(dd)
    max_dd_idx = dd.index(max_dd)
    ax.annotate(f"Max DD: {max_dd:.0f} pts",
                xy=(dates[max_dd_idx], max_dd),
                xytext=(30, -20), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#D32F2F",
                arrowprops=dict(arrowstyle="->", color="#D32F2F"))

    ax.set_title("Drawdown from Peak", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Drawdown (points)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "drawdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  6. V58 OVERNIGHT BIAS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def chart_v58_analysis(trades: list, output_dir: str) -> str:
    """Compare performance when following V58 bias vs going against it."""
    # Group trades by overnight bias
    groups = defaultdict(list)
    for t in trades:
        bias = t.overnight_bias or "NO_DATA"
        groups[bias].append(t)

    if not groups or all(k == "NO_DATA" for k in groups):
        return None

    # Also check: does fading work?
    aligned = []    # V58 says SHORT and trade is SHORT, or LONG and LONG
    against = []    # V58 says SHORT but trade is LONG, etc.

    for t in trades:
        if t.overnight_bias == "SHORT_ONLY" and t.direction == "SHORT":
            aligned.append(t)
        elif t.overnight_bias == "SHORT_ONLY" and t.direction == "LONG":
            against.append(t)
        elif t.overnight_bias == "LONG_ONLY" and t.direction == "LONG":
            aligned.append(t)
        elif t.overnight_bias == "LONG_ONLY" and t.direction == "SHORT":
            against.append(t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: P&L by bias type
    ax1 = axes[0]
    bias_types = sorted(groups.keys())
    x_pos = range(len(bias_types))
    avg_pnls = [np.mean([t.pnl_pts for t in groups[b]]) for b in bias_types]
    counts = [len(groups[b]) for b in bias_types]
    colors = []
    for p in avg_pnls:
        if p > 2:
            colors.append("#4CAF50")
        elif p < -2:
            colors.append("#F44336")
        else:
            colors.append("#FF9800")

    bars = ax1.bar(x_pos, avg_pnls, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bias_types, fontsize=9)
    ax1.set_title("Avg P&L by Overnight Bias", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Avg P&L (points)")
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    for bar, val, n in zip(bars, avg_pnls, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.5 if val >= 0 else -1.5),
                 f"{val:+.1f}\n(n={n})", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Aligned vs Against
    ax2 = axes[1]
    if aligned or against:
        categories = []
        values = []
        bar_colors = []

        if aligned:
            wr = sum(1 for t in aligned if t.pnl_pts > 0) / len(aligned) * 100
            categories.append(f"With V58\n({len(aligned)} trades)")
            values.append(np.mean([t.pnl_pts for t in aligned]))
            bar_colors.append("#4CAF50")

        if against:
            wr = sum(1 for t in against if t.pnl_pts > 0) / len(against) * 100
            categories.append(f"Against V58\n({len(against)} trades)")
            values.append(np.mean([t.pnl_pts for t in against]))
            bar_colors.append("#F44336")

        bars2 = ax2.bar(range(len(categories)), values, color=bar_colors,
                        edgecolor="white", linewidth=1.5)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, fontsize=10)

        for bar, val in zip(bars2, values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.5 if val >= 0 else -1.5),
                     f"{val:+.1f}", ha="center", fontsize=12, fontweight="bold")

        ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax2.set_title("Following V58 vs Going Against", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Avg P&L (points)")
    else:
        ax2.text(0.5, 0.5, "No overnight data\navailable",
                 ha="center", va="center", fontsize=14, transform=ax2.transAxes)
        ax2.set_title("V58 Alignment", fontsize=13, fontweight="bold")

    fig.suptitle("V58 Overnight Range Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "v58_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  7. DAY OF WEEK
# ══════════════════════════════════════════════════════════════════════════════

def chart_day_of_week(trades: list, output_dir: str) -> str:
    """Performance by day of week."""
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_groups = defaultdict(list)
    for t in trades:
        day_groups[t.day_of_week].append(t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Avg P&L
    ax1 = axes[0]
    avg_pnls = [np.mean([t.pnl_pts for t in day_groups[d]]) if d in day_groups else 0
                for d in days_order]
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in avg_pnls]
    bars = ax1.bar(range(5), avg_pnls, color=colors, edgecolor="white")
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([d[:3] for d in days_order])
    ax1.set_title("Avg P&L by Day", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Avg P&L (points)")
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    for bar, val in zip(bars, avg_pnls):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if val >= 0 else -1),
                 f"{val:+.1f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 2: Win rate
    ax2 = axes[1]
    win_rates = []
    for d in days_order:
        ts = day_groups.get(d, [])
        wr = sum(1 for t in ts if t.pnl_pts > 0) / len(ts) * 100 if ts else 0
        win_rates.append(wr)
    colors2 = ["#4CAF50" if wr >= 50 else "#F44336" for wr in win_rates]
    bars2 = ax2.bar(range(5), win_rates, color=colors2, edgecolor="white")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([d[:3] for d in days_order])
    ax2.set_title("Win Rate by Day", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Win Rate (%)")
    ax2.axhline(y=50, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Day of Week Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "day_of_week.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  8. COMBINED DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def chart_dashboard(results: list, trades: list, output_dir: str) -> str:
    """Single-page dashboard combining key metrics."""
    wins = [t for t in trades if t.pnl_pts > 0]
    losses = [t for t in trades if t.pnl_pts < 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl_pts for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pts for t in losses]) if losses else 0
    expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
    gross_win = sum(t.pnl_pts for t in wins)
    gross_loss = abs(sum(t.pnl_pts for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = sum(t.pnl_pts for t in trades)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("ASRS BACKTEST DASHBOARD", fontsize=20, fontweight="bold", y=0.98)

    # Layout: 3 rows, 3 cols
    # Row 1: Equity curve (spans 2 cols) + Key metrics
    ax_eq = fig.add_subplot(3, 3, (1, 2))
    ax_metrics = fig.add_subplot(3, 3, 3)
    # Row 2: Monthly bars (spans 2 cols) + Win/Loss pie
    ax_monthly = fig.add_subplot(3, 3, (4, 5))
    ax_pie = fig.add_subplot(3, 3, 6)
    # Row 3: Distribution + Drawdown + Day of week
    ax_dist = fig.add_subplot(3, 3, 7)
    ax_dd = fig.add_subplot(3, 3, 8)
    ax_dow = fig.add_subplot(3, 3, 9)

    # ── Equity Curve ───────────────────────────────────────────────
    dates, equity = [], []
    running = 0
    for r in results:
        if r.triggered:
            dates.append(datetime.strptime(r.date, "%Y-%m-%d"))
            running += r.total_pnl
            equity.append(running)

    ax_eq.plot(dates, equity, color="#2196F3", linewidth=1.2)
    ax_eq.fill_between(dates, 0, equity,
                       where=[e >= 0 for e in equity], alpha=0.1, color="#4CAF50")
    ax_eq.fill_between(dates, 0, equity,
                       where=[e < 0 for e in equity], alpha=0.1, color="#F44336")
    ax_eq.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax_eq.set_ylabel("P&L (pts)")
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax_eq.tick_params(axis="x", rotation=45, labelsize=8)
    ax_eq.grid(True, alpha=0.2)

    # ── Key Metrics ────────────────────────────────────────────────
    ax_metrics.axis("off")
    verdict = "✅ EDGE" if expectancy > 0 and profit_factor > 1.2 else "⚠️ MARGINAL" if expectancy > 0 else "❌ NO EDGE"
    metrics_text = (
        f"{verdict}\n\n"
        f"Total P&L:     {total_pnl:+.0f} pts\n"
        f"Trades:        {len(trades)}\n"
        f"Win Rate:      {win_rate:.1f}%\n"
        f"Avg Win:       {avg_win:+.1f}\n"
        f"Avg Loss:      {avg_loss:+.1f}\n"
        f"R:R Ratio:     {abs(avg_win/avg_loss) if avg_loss else 0:.2f}\n"
        f"Expectancy:    {expectancy:+.1f}\n"
        f"Profit Factor: {profit_factor:.2f}\n\n"
        f"€1/pt:  €{total_pnl:+,.0f}\n"
        f"€5/pt:  €{total_pnl*5:+,.0f}\n"
        f"€25/pt: €{total_pnl*25:+,.0f}"
    )
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=10, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5"))

    # ── Monthly P&L ────────────────────────────────────────────────
    monthly = defaultdict(float)
    for r in results:
        monthly[r.date[:7]] += r.total_pnl
    months = sorted(monthly.keys())
    mpnls = [monthly[m] for m in months]
    colors_m = ["#4CAF50" if p >= 0 else "#F44336" for p in mpnls]
    ax_monthly.bar(range(len(months)), mpnls, color=colors_m, alpha=0.7)
    ax_monthly.set_xticks(range(len(months)))
    ax_monthly.set_xticklabels(months, rotation=45, fontsize=7, ha="right")
    ax_monthly.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax_monthly.set_title("Monthly P&L", fontsize=12, fontweight="bold")
    ax_monthly.grid(True, alpha=0.2, axis="y")

    # ── Win/Loss Pie ───────────────────────────────────────────────
    ax_pie.pie([len(wins), len(losses)],
               labels=[f"Win {len(wins)}", f"Loss {len(losses)}"],
               colors=["#4CAF50", "#F44336"],
               autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 10})
    ax_pie.set_title("Win/Loss Split", fontsize=12, fontweight="bold")

    # ── Distribution ───────────────────────────────────────────────
    pnls = [t.pnl_pts for t in trades]
    n, bins, patches = ax_dist.hist(pnls, bins=30, edgecolor="white", linewidth=0.5)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor("#4CAF50" if left >= 0 else "#F44336")
        patch.set_alpha(0.7)
    ax_dist.axvline(np.mean(pnls), color="#2196F3", linewidth=1.5, linestyle="--")
    ax_dist.set_title("P&L Distribution", fontsize=12, fontweight="bold")
    ax_dist.set_xlabel("Points", fontsize=9)

    # ── Drawdown ───────────────────────────────────────────────────
    peaks = np.maximum.accumulate(equity)
    dd = [e - p for e, p in zip(equity, peaks)]
    ax_dd.fill_between(dates, dd, 0, color="#F44336", alpha=0.3)
    ax_dd.plot(dates, dd, color="#F44336", linewidth=0.8)
    ax_dd.set_title(f"Drawdown (max: {min(dd):.0f} pts)", fontsize=12, fontweight="bold")
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax_dd.tick_params(axis="x", rotation=45, labelsize=7)

    # ── Day of Week ────────────────────────────────────────────────
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    days_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dg = defaultdict(list)
    for t in trades:
        dg[t.day_of_week].append(t)
    dow_pnls = [np.mean([t.pnl_pts for t in dg[d]]) if d in dg else 0 for d in days_full]
    colors_d = ["#4CAF50" if p >= 0 else "#F44336" for p in dow_pnls]
    ax_dow.bar(range(5), dow_pnls, color=colors_d, edgecolor="white")
    ax_dow.set_xticks(range(5))
    ax_dow.set_xticklabels(days_order)
    ax_dow.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax_dow.set_title("Avg P&L by Day", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
