"""
risk_gate.py — Portfolio-level risk gate for ASRS bot
═══════════════════════════════════════════════════════════════

Hard gates checked at every signal entry point. Returns (allow, reason).
Gates are designed to be ENABLED, not advisory — strategy code must
refuse the trade when allow=False.

Gates checked (in order):
  1. /pause sentinel → block
  2. Daily loss limit (default 3% of equity)
  3. Weekly loss limit (default 6% of equity)
  4. Max concurrent open positions (default 3 of 8 signals)
  5. Consecutive loss kill switch (default 6 in a row → halt to next morning)

Equity sourced from: STARTING_EQUITY_GBP env (default 5000) + cumulative
journal P&L. Tracking is GBP-denominated because all max_risk_gbp configs
already are.

Env overrides (set via export or systemd Environment=):
  STARTING_EQUITY_GBP        default 5000.0
  DAILY_LOSS_LIMIT_PCT       default 3.0
  WEEKLY_LOSS_LIMIT_PCT      default 6.0
  MAX_CONCURRENT_POSITIONS   default 3
  CONSECUTIVE_LOSS_KILL      default 6
  RISK_PCT_PER_TRADE         default 0.5  (used by sizing, not gate)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("ASRS.RiskGate")

TZ_UK = ZoneInfo("Europe/London")


@dataclass
class RiskGateConfig:
    """Risk gate config read from env vars.

    Uses default_factory so each RiskGateConfig() call re-reads os.environ.
    Without this, the env is frozen at class-definition time (module import)
    and changing STARTING_EQUITY_GBP etc. requires a full python restart —
    which breaks testing AND breaks any runtime env override.
    """
    starting_equity_gbp: float = field(
        default_factory=lambda: float(os.getenv("STARTING_EQUITY_GBP", "5000")))
    daily_loss_limit_pct: float = field(
        default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0")))
    weekly_loss_limit_pct: float = field(
        default_factory=lambda: float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "6.0")))
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "3")))
    consecutive_loss_kill: int = field(
        default_factory=lambda: int(os.getenv("CONSECUTIVE_LOSS_KILL", "6")))
    risk_pct_per_trade: float = field(
        default_factory=lambda: float(os.getenv("RISK_PCT_PER_TRADE", "0.5")))
    # Ignore trades before this YYYY-MM-DD when computing equity / limits.
    # Use this to reset the gate baseline after fixing bugs (so historic
    # bug-period losses don't permanently lock the gate).
    start_date: str = field(
        default_factory=lambda: os.getenv("RISK_GATE_START_DATE", ""))


CFG = RiskGateConfig()


def _journal_sum(since_date: str | None = None) -> float:
    """Sum pnl_gbp from journal, optionally only since a given date.
    Used so we can isolate gate state from pre-fix bug-period losses."""
    try:
        from shared.journal_db import _get_conn
        conn = _get_conn()
        if since_date:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl_gbp),0) FROM trades WHERE date >= ?",
                (since_date,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl_gbp),0) FROM trades"
            ).fetchone()
        return float(row[0] or 0)
    except Exception as e:
        logger.error(f"_journal_sum failed: {e}")
        return 0.0


def current_equity_gbp() -> float:
    """STARTING_EQUITY + cumulative realised P&L from journal (since start_date if set)."""
    return CFG.starting_equity_gbp + _journal_sum(CFG.start_date or None)


def today_pnl_gbp() -> float:
    try:
        from shared.journal_db import get_trades_for_date
        today = datetime.now(TZ_UK).date().isoformat()
        # Honour RISK_GATE_START_DATE: if today is before it, return 0
        if CFG.start_date and today < CFG.start_date:
            return 0.0
        return float(sum(t.get("pnl_gbp", 0) or 0 for t in get_trades_for_date(today)))
    except Exception as e:
        logger.error(f"today_pnl_gbp failed: {e}")
        return 0.0


def week_pnl_gbp() -> float:
    """Week P&L, but never older than RISK_GATE_START_DATE."""
    try:
        from shared.journal_db import _get_conn
        from datetime import timedelta
        conn = _get_conn()
        today = datetime.now(TZ_UK).date()
        monday = today - timedelta(days=today.weekday())
        floor_str = monday.isoformat()
        if CFG.start_date and CFG.start_date > floor_str:
            floor_str = CFG.start_date
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl_gbp),0) FROM trades WHERE date >= ? AND date <= ?",
            (floor_str, today.isoformat()),
        ).fetchone()
        return float(row[0] or 0)
    except Exception as e:
        logger.error(f"week_pnl_gbp failed: {e}")
        return 0.0


def consecutive_losses() -> int:
    """Count of consecutive losing trades from the most recent backwards."""
    try:
        from shared.journal_db import get_recent_trades
        trades = get_recent_trades(n=CFG.consecutive_loss_kill + 5)
        # get_recent_trades returns oldest→newest after reverse, walk backwards
        count = 0
        for t in reversed(trades):
            if (t.get("pnl_gbp") or 0) < 0:
                count += 1
            else:
                break
        return count
    except Exception as e:
        logger.error(f"consecutive_losses failed: {e}")
        return 0


def concurrent_open_positions(all_signals) -> int:
    """Count signals currently in LONG or SHORT phase."""
    try:
        from asrs.strategy import Phase
        return sum(
            1 for s in all_signals
            if getattr(s.state, "phase", None) in (Phase.LONG, Phase.SHORT)
        )
    except Exception as e:
        logger.error(f"concurrent_open_positions failed: {e}")
        return 0


def check_entry_allowed(signal_name: str, all_signals=None) -> tuple[bool, str]:
    """
    Master gate. Call before placing any new entry.

    Returns (allow, reason). reason is human-readable for Telegram alerts.
    """
    # Lazy ALL_SIGNALS lookup if not passed
    if all_signals is None:
        try:
            from asrs.main import ALL_SIGNALS
            all_signals = ALL_SIGNALS
        except Exception:
            all_signals = []

    # 1. /pause sentinel
    try:
        from telegram_cmd import is_paused
        if is_paused():
            return False, "PAUSED (via /pause or /kill)"
    except Exception:
        pass

    equity = current_equity_gbp()
    if equity <= 0:
        return False, f"Equity ≤ 0 (£{equity:.0f}) — refusing all entries"

    # 2. Daily loss limit
    today = today_pnl_gbp()
    daily_limit_gbp = -(equity * CFG.daily_loss_limit_pct / 100.0)
    if today <= daily_limit_gbp:
        return False, (
            f"Daily loss limit hit: today £{today:.0f} ≤ £{daily_limit_gbp:.0f} "
            f"({CFG.daily_loss_limit_pct}% of £{equity:.0f})"
        )

    # 3. Weekly loss limit
    week = week_pnl_gbp()
    weekly_limit_gbp = -(equity * CFG.weekly_loss_limit_pct / 100.0)
    if week <= weekly_limit_gbp:
        return False, (
            f"Weekly loss limit hit: week £{week:.0f} ≤ £{weekly_limit_gbp:.0f} "
            f"({CFG.weekly_loss_limit_pct}% of £{equity:.0f})"
        )

    # 4. Max concurrent positions
    open_count = concurrent_open_positions(all_signals)
    if open_count >= CFG.max_concurrent_positions:
        return False, (
            f"Max concurrent positions: {open_count} ≥ {CFG.max_concurrent_positions}"
        )

    # 5. Consecutive loss kill
    losses = consecutive_losses()
    if losses >= CFG.consecutive_loss_kill:
        return False, (
            f"Consecutive loss kill: {losses} ≥ {CFG.consecutive_loss_kill} losses in a row"
        )

    return True, "OK"


def position_size_contracts(
    signal_name: str,
    instrument: str,
    stop_distance_pts: float,
    gbp_per_pt: float,
    max_contracts: int,
) -> int:
    """
    Vol-targeted position sizing. Returns the number of contracts that
    risks exactly RISK_PCT_PER_TRADE of current equity for the given
    stop distance, capped at max_contracts and floored at 1.

    qty = floor(risk_budget_gbp / (stop_distance_pts * gbp_per_pt))
    """
    if stop_distance_pts <= 0 or gbp_per_pt <= 0:
        return 1
    equity = current_equity_gbp()
    risk_budget_gbp = equity * CFG.risk_pct_per_trade / 100.0
    raw = risk_budget_gbp / (stop_distance_pts * gbp_per_pt)
    qty = max(1, min(max_contracts, int(raw)))
    logger.info(
        f"[{signal_name}] sizing: equity=£{equity:.0f} budget=£{risk_budget_gbp:.0f} "
        f"stop={stop_distance_pts:.0f}pts £/pt={gbp_per_pt:.2f} → "
        f"raw={raw:.2f} → qty={qty} (cap={max_contracts})"
    )
    return qty


def status_report() -> str:
    """Snapshot of all gate values, for /status or startup log."""
    equity = current_equity_gbp()
    return (
        f"Risk gate ({CFG.daily_loss_limit_pct}%/{CFG.weekly_loss_limit_pct}% "
        f"/ {CFG.max_concurrent_positions} pos / {CFG.consecutive_loss_kill}L)\n"
        f"  Equity: £{equity:.0f}  (start £{CFG.starting_equity_gbp:.0f})\n"
        f"  Today:  £{today_pnl_gbp():.0f} / -£{equity * CFG.daily_loss_limit_pct / 100:.0f}\n"
        f"  Week:   £{week_pnl_gbp():.0f} / -£{equity * CFG.weekly_loss_limit_pct / 100:.0f}\n"
        f"  Cons L: {consecutive_losses()} / {CFG.consecutive_loss_kill}"
    )
