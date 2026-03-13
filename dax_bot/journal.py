"""
journal.py -- Persistent Trade Journal & Performance Analytics
==============================================================================

Delegates to shared SQLite journal (shared/journal_db.py).
Keeps the same public API so callers don't change.
"""

import logging
from shared import journal_db

logger = logging.getLogger(__name__)


def append_trade(trade: dict, state=None):
    """Append a completed trade to the journal."""
    journal_db.insert_trade("DAX", trade, state)


def append_trades(trades: list, state=None):
    """Append multiple completed trades."""
    for t in trades:
        if t.get("exit"):
            append_trade(t, state)


def load_all_trades() -> list[dict]:
    """Load all DAX trades from the journal."""
    return journal_db.get_recent_trades(n=10000, instrument="DAX")


def get_stats() -> dict:
    """Calculate comprehensive performance statistics for dashboard."""
    # Dashboard expects the old format with equity_curve etc.
    # Fetch all trades and compute in the old way for compatibility
    trades = load_all_trades()

    if not trades:
        return {
            "total_trades": 0, "total_pnl": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "expectancy": 0,
            "max_drawdown": 0, "profit_factor": 0,
            "avg_slippage": 0, "total_slippage": 0,
            "equity_curve": [], "drawdown_curve": [],
            "daily_pnl": {}, "weekly_pnl": {}, "monthly_pnl": {},
            "trades": [],
        }

    from datetime import datetime

    pnls = [t.get("pnl_pts", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    slippages = [t.get("slippage_total", 0) for t in trades]

    # Equity curve
    equity = []
    running = 0
    peak = 0
    max_dd = 0
    drawdown_curve = []
    for t in trades:
        running += t.get("pnl_pts", 0)
        equity.append({"date": t.get("date", ""), "equity": round(running, 1)})
        peak = max(peak, running)
        dd = round(peak - running, 1)
        max_dd = max(max_dd, dd)
        drawdown_curve.append({"date": t.get("date", ""), "drawdown": -dd})

    # Daily P&L
    daily = {}
    for t in trades:
        d = t.get("date", "")
        daily[d] = round(daily.get(d, 0) + t.get("pnl_pts", 0), 1)

    # Weekly P&L
    weekly = {}
    for t in trades:
        try:
            dt = datetime.strptime(t.get("date", ""), "%Y-%m-%d")
            week = dt.strftime("%Y-W%W")
            weekly[week] = round(weekly.get(week, 0) + t.get("pnl_pts", 0), 1)
        except ValueError:
            pass

    # Monthly P&L
    monthly = {}
    for t in trades:
        try:
            m = t.get("date", "")[:7]
            monthly[m] = round(monthly.get(m, 0) + t.get("pnl_pts", 0), 1)
        except (IndexError, ValueError):
            pass

    total_pnl = round(sum(pnls), 1)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    return {
        "total_trades": len(trades),
        "total_pnl": total_pnl,
        "total_pnl_eur": round(total_pnl, 1),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win": round(sum(wins) / len(wins), 1) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 1) if losses else 0,
        "expectancy": round(total_pnl / len(trades), 1) if trades else 0,
        "max_drawdown": round(max_dd, 1),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.99,
        "avg_slippage": round(sum(slippages) / len(slippages), 2) if slippages else 0,
        "total_slippage": round(sum(slippages), 1),
        "best_trade": round(max(pnls), 1) if pnls else 0,
        "worst_trade": round(min(pnls), 1) if pnls else 0,
        "consecutive_wins": _max_streak(pnls, positive=True),
        "consecutive_losses": _max_streak(pnls, positive=False),
        "trading_days": len(daily),
        "equity_curve": equity,
        "drawdown_curve": drawdown_curve,
        "daily_pnl": daily,
        "weekly_pnl": weekly,
        "monthly_pnl": monthly,
        "trades": trades,
    }


def _max_streak(pnls: list, positive: bool) -> int:
    max_streak = 0
    current = 0
    for p in pnls:
        if (positive and p > 0) or (not positive and p < 0):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
