"""
reports.py -- Performance report generator (equity curves + stats)
==================================================================
Produces PNG images with matplotlib for Telegram delivery.
Dark theme (#1a1a2e), three equity curves (weekly/monthly/YTD),
governance checks, and per-instrument breakdown.
"""

import io
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import numpy as np

logger = logging.getLogger(__name__)

TZ_UK = ZoneInfo("Europe/London")

# R-value mapping: 1R = max_risk_gbp per instrument
RISK_MAP = {"DAX": 50, "US30": 50, "NIKKEI": 75}

# Governance thresholds
BENCH_PF = 2.66
BENCH_WIN_RATE = 0.52
BENCH_EXPECTANCY = 0.85

# Colors
BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"
GREEN = "#00ff88"
RED = "#ff4444"
AMBER = "#ffaa00"
TEXT_COLOR = "#e0e0e0"
MUTED = "#888888"


def _fetch_trades(since: str | None = None) -> list[dict]:
    """Fetch trades from journal DB, optionally filtered by date."""
    from shared.journal_db import _get_conn
    conn = _get_conn()
    if since:
        rows = conn.execute(
            "SELECT * FROM trades WHERE date >= ? ORDER BY id ASC",
            (since,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _to_r(trade: dict) -> float:
    """Convert a trade's pnl_pts to R-multiple."""
    inst = trade.get("instrument", "")
    risk = RISK_MAP.get(inst, 50)
    pnl = trade.get("pnl_pts", 0) or 0
    return pnl / risk if risk else 0


def _calc_stats(trades: list[dict]) -> dict:
    """Calculate full stats dict from a list of trade dicts."""
    if not trades:
        return {
            "trades": 0, "cum_r": 0, "win_rate": 0, "pf": 0,
            "expectancy": 0, "avg_win_r": 0, "avg_loss_r": 0,
            "best_day_r": 0, "worst_day_r": 0,
            "consec_loss": 0, "worst_streak": 0,
            "largest_loss_r": 0, "monthly_r": 0,
            "r_values": [], "per_instrument": {},
        }

    r_values = [_to_r(t) for t in trades]
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Consecutive loss streak
    max_streak = 0
    current_streak = 0
    for r in r_values:
        if r <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    # Current consecutive losses (from end)
    consec_loss = 0
    for r in reversed(r_values):
        if r <= 0:
            consec_loss += 1
        else:
            break

    # Per-day R totals
    day_r: dict[str, float] = {}
    for t, r in zip(trades, r_values):
        d = t.get("date", "")
        day_r[d] = day_r.get(d, 0) + r

    best_day = max(day_r.values()) if day_r else 0
    worst_day = min(day_r.values()) if day_r else 0

    # Per instrument
    per_inst: dict[str, dict] = {}
    for inst in ["DAX", "US30", "NIKKEI"]:
        inst_trades = [t for t in trades if t.get("instrument") == inst]
        inst_r = [_to_r(t) for t in inst_trades]
        inst_wins = [r for r in inst_r if r > 0]
        inst_losses = [r for r in inst_r if r <= 0]
        gw = sum(inst_wins) if inst_wins else 0
        gl = abs(sum(inst_losses)) if inst_losses else 0
        per_inst[inst] = {
            "trades": len(inst_trades),
            "pf": gw / gl if gl > 0 else float("inf"),
            "cum_r": sum(inst_r),
        }

    # Rolling PF (last N trades)
    def rolling_pf(n):
        recent = r_values[-n:] if len(r_values) >= n else r_values
        w = sum(r for r in recent if r > 0)
        l = abs(sum(r for r in recent if r <= 0))
        return w / l if l > 0 else float("inf")

    return {
        "trades": len(trades),
        "cum_r": sum(r_values),
        "win_rate": len(wins) / len(r_values) if r_values else 0,
        "pf": pf,
        "pf_20": rolling_pf(20),
        "pf_50": rolling_pf(50),
        "pf_100": rolling_pf(100),
        "expectancy": sum(r_values) / len(r_values) if r_values else 0,
        "avg_win_r": sum(wins) / len(wins) if wins else 0,
        "avg_loss_r": sum(losses) / len(losses) if losses else 0,
        "best_day_r": round(best_day, 2),
        "worst_day_r": round(worst_day, 2),
        "consec_loss": consec_loss,
        "worst_streak": max_streak,
        "largest_loss_r": min(r_values) if r_values else 0,
        "monthly_r": sum(r_values),  # for the period queried
        "r_values": r_values,
        "per_instrument": per_inst,
    }


def _get_period_bounds() -> dict[str, str]:
    """Return ISO date strings for week/month/YTD start."""
    now = datetime.now(TZ_UK)
    # Week: Monday of current week
    mon = now.date() - timedelta(days=now.weekday())
    # Month: 1st of current month
    m1 = now.date().replace(day=1)
    # YTD: 1st Jan
    y1 = now.date().replace(month=1, day=1)
    return {
        "weekly": mon.isoformat(),
        "monthly": m1.isoformat(),
        "ytd": y1.isoformat(),
    }


def _draw_equity_curve(ax, r_values: list[float], title: str):
    """Draw a single equity curve on the given axes."""
    if not r_values:
        ax.set_facecolor(BG_COLOR)
        ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, pad=4)
        ax.tick_params(colors=MUTED, labelsize=7)
        return

    cum_r = np.cumsum(r_values)
    x = np.arange(1, len(cum_r) + 1)

    ax.set_facecolor(BG_COLOR)
    ax.plot(x, cum_r, color=GREEN, linewidth=1.5, solid_capstyle="round")
    ax.fill_between(x, cum_r, alpha=0.08, color=GREEN)
    ax.axhline(0, color=MUTED, linewidth=0.5, linestyle="--")

    ax.set_title(title, color=TEXT_COLOR, fontsize=10, pad=4)
    ax.set_xlabel("Trade #", color=MUTED, fontsize=7)
    ax.set_ylabel("Cumulative R", color=MUTED, fontsize=7)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.5)


def _draw_stats_table(ax, weekly: dict, monthly: dict, ytd: dict):
    """Draw the stats table on the given axes."""
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    def _fmt(v, fmt=".2f", is_pct=False):
        if v == float("inf"):
            return "INF"
        if is_pct:
            return f"{v * 100:.1f}%"
        return f"{v:{fmt}}"

    def _color(val, threshold, higher_good=True):
        if val == float("inf"):
            return GREEN
        if higher_good:
            return RED if val < threshold else GREEN
        else:
            return RED if val > threshold else GREEN

    # Build rows: (label, weekly_val, monthly_val, ytd_val, color_func)
    rows = []

    # -- Governance --
    rows.append(("--- GOVERNANCE ---", "", "", "", TEXT_COLOR))
    for label, key, n in [("PF (last 20)", "pf_20", 20), ("PF (last 50)", "pf_50", 50), ("PF (last 100)", "pf_100", 100)]:
        for s, period_label in [(weekly, "W"), (monthly, "M"), (ytd, "Y")]:
            pass
        w_val = _fmt(weekly.get(key, 0))
        m_val = _fmt(monthly.get(key, 0))
        y_val = _fmt(ytd.get(key, 0))
        # Color based on YTD
        c = _color(ytd.get(key, 0), BENCH_PF)
        rows.append((label, w_val, m_val, y_val, c))

    rows.append(("Consec Loss (current)", str(weekly["consec_loss"]), str(monthly["consec_loss"]), str(ytd["consec_loss"]),
                 RED if ytd["consec_loss"] >= 20 else AMBER if ytd["consec_loss"] >= 15 else GREEN))
    rows.append(("Worst Streak", str(weekly["worst_streak"]), str(monthly["worst_streak"]), str(ytd["worst_streak"]),
                 RED if ytd["worst_streak"] >= 40 else AMBER if ytd["worst_streak"] >= 20 else GREEN))
    rows.append(("Largest Loss", _fmt(weekly["largest_loss_r"]), _fmt(monthly["largest_loss_r"]), _fmt(ytd["largest_loss_r"]),
                 RED if ytd["largest_loss_r"] < -1.5 else GREEN))
    rows.append(("Monthly R Total", "", _fmt(monthly["cum_r"]), "",
                 RED if monthly["cum_r"] < -40 else AMBER if monthly["cum_r"] < -20 else GREEN))

    # -- Performance --
    rows.append(("--- PERFORMANCE ---", "", "", "", TEXT_COLOR))
    rows.append(("Cumulative R", _fmt(weekly["cum_r"]), _fmt(monthly["cum_r"]), _fmt(ytd["cum_r"]), TEXT_COLOR))
    rows.append(("Win Rate", _fmt(weekly["win_rate"], is_pct=True), _fmt(monthly["win_rate"], is_pct=True), _fmt(ytd["win_rate"], is_pct=True),
                 _color(ytd["win_rate"], BENCH_WIN_RATE)))
    rows.append(("Expectancy", _fmt(weekly["expectancy"]), _fmt(monthly["expectancy"]), _fmt(ytd["expectancy"]),
                 _color(ytd["expectancy"], BENCH_EXPECTANCY)))
    rows.append(("Avg Win R", _fmt(weekly["avg_win_r"]), _fmt(monthly["avg_win_r"]), _fmt(ytd["avg_win_r"]), TEXT_COLOR))
    rows.append(("Avg Loss R", _fmt(weekly["avg_loss_r"]), _fmt(monthly["avg_loss_r"]), _fmt(ytd["avg_loss_r"]), TEXT_COLOR))
    rows.append(("Best Day R", _fmt(weekly["best_day_r"]), _fmt(monthly["best_day_r"]), _fmt(ytd["best_day_r"]), TEXT_COLOR))
    rows.append(("Worst Day R", _fmt(weekly["worst_day_r"]), _fmt(monthly["worst_day_r"]), _fmt(ytd["worst_day_r"]), TEXT_COLOR))

    # -- Per Instrument --
    rows.append(("--- PER INSTRUMENT ---", "", "", "", TEXT_COLOR))
    for inst in ["DAX", "US30", "NIKKEI"]:
        w_i = weekly["per_instrument"].get(inst, {"trades": 0, "pf": 0, "cum_r": 0})
        m_i = monthly["per_instrument"].get(inst, {"trades": 0, "pf": 0, "cum_r": 0})
        y_i = ytd["per_instrument"].get(inst, {"trades": 0, "pf": 0, "cum_r": 0})
        rows.append((f"{inst} Trades", str(w_i["trades"]), str(m_i["trades"]), str(y_i["trades"]), TEXT_COLOR))
        rows.append((f"{inst} PF", _fmt(w_i["pf"]), _fmt(m_i["pf"]), _fmt(y_i["pf"]),
                     _color(y_i["pf"], BENCH_PF)))
        rows.append((f"{inst} Cum R", _fmt(w_i["cum_r"]), _fmt(m_i["cum_r"]), _fmt(y_i["cum_r"]), TEXT_COLOR))

    # Render table
    n_rows = len(rows)
    col_labels = ["", "Weekly", "Monthly", "YTD"]
    cell_text = [[r[0], r[1], r[2], r[3]] for r in rows]
    cell_colors = []
    for r in rows:
        c = r[4]
        bg = to_rgba(BG_COLOR, 1.0)
        cell_colors.append([bg, bg, bg, bg])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    # Style cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(MUTED)
        cell.set_linewidth(0.3)
        if row == 0:
            # Header
            cell.set_facecolor(PANEL_COLOR)
            cell.set_text_props(color=TEXT_COLOR, fontweight="bold", fontsize=8)
        else:
            cell.set_facecolor(BG_COLOR)
            r_data = rows[row - 1]
            color = r_data[4]
            if r_data[0].startswith("---"):
                cell.set_facecolor(PANEL_COLOR)
                cell.set_text_props(color=AMBER, fontweight="bold", fontsize=7)
            elif col == 0:
                cell.set_text_props(color=TEXT_COLOR, fontsize=7)
            else:
                cell.set_text_props(color=color, fontsize=7)


def generate_full_report() -> bytes:
    """Generate full performance report as PNG bytes."""
    bounds = _get_period_bounds()
    weekly_trades = _fetch_trades(bounds["weekly"])
    monthly_trades = _fetch_trades(bounds["monthly"])
    ytd_trades = _fetch_trades(bounds["ytd"])

    weekly = _calc_stats(weekly_trades)
    monthly = _calc_stats(monthly_trades)
    ytd = _calc_stats(ytd_trades)

    fig = plt.figure(figsize=(12, 16), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 3], hspace=0.3)

    # Equity curves
    ax_w = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    ax_y = fig.add_subplot(gs[2])

    _draw_equity_curve(ax_w, weekly["r_values"], "Weekly Equity (R)")
    _draw_equity_curve(ax_m, monthly["r_values"], "Monthly Equity (R)")
    _draw_equity_curve(ax_y, ytd["r_values"], "YTD Equity (R)")

    # Stats table
    ax_t = fig.add_subplot(gs[3])
    _draw_stats_table(ax_t, weekly, monthly, ytd)

    # Title
    now = datetime.now(TZ_UK).strftime("%d %b %Y %H:%M UK")
    fig.suptitle(f"ASRS Performance Report -- {now}",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_chart_only() -> bytes:
    """Generate just the 3 equity curves as PNG bytes."""
    bounds = _get_period_bounds()
    weekly_trades = _fetch_trades(bounds["weekly"])
    monthly_trades = _fetch_trades(bounds["monthly"])
    ytd_trades = _fetch_trades(bounds["ytd"])

    weekly = _calc_stats(weekly_trades)
    monthly = _calc_stats(monthly_trades)
    ytd = _calc_stats(ytd_trades)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.4)

    _draw_equity_curve(axes[0], weekly["r_values"], "Weekly Equity (R)")
    _draw_equity_curve(axes[1], monthly["r_values"], "Monthly Equity (R)")
    _draw_equity_curve(axes[2], ytd["r_values"], "YTD Equity (R)")

    now = datetime.now(TZ_UK).strftime("%d %b %Y %H:%M UK")
    fig.suptitle(f"ASRS Equity Curves -- {now}",
                 color=TEXT_COLOR, fontsize=12, fontweight="bold", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def check_governance_alerts() -> list[str]:
    """Check governance thresholds and return list of alert messages."""
    alerts = []
    now = datetime.now(TZ_UK)

    # All trades for streak checks
    all_trades = _fetch_trades()
    if not all_trades:
        return alerts

    all_stats = _calc_stats(all_trades)

    # Consecutive losses
    if all_stats["consec_loss"] >= 40:
        alerts.append(
            f"CRITICAL: {all_stats['consec_loss']} consecutive losses -- "
            f"trading should be halted immediately"
        )
    elif all_stats["consec_loss"] >= 20:
        alerts.append(
            f"WARNING: {all_stats['consec_loss']} consecutive losses -- "
            f"review strategy and consider reducing size"
        )

    # Largest single loss
    if all_stats["largest_loss_r"] < -1.5:
        alerts.append(
            f"ALERT: Single loss of {all_stats['largest_loss_r']:.2f}R "
            f"exceeds 1.5R threshold"
        )

    # Monthly drawdown
    m1 = now.date().replace(day=1).isoformat()
    monthly_trades = _fetch_trades(m1)
    monthly_stats = _calc_stats(monthly_trades)

    if monthly_stats["cum_r"] <= -40:
        alerts.append(
            f"CRITICAL: Monthly R at {monthly_stats['cum_r']:.1f}R -- "
            f"exceeds -40R limit, halt trading"
        )
    elif monthly_stats["cum_r"] <= -20:
        alerts.append(
            f"WARNING: Monthly R at {monthly_stats['cum_r']:.1f}R -- "
            f"approaching -40R limit"
        )

    return alerts
