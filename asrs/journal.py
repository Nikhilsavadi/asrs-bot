"""
journal.py -- Trade journal (SQLite)
====================================
Thin wrapper around shared/journal_db.py. Formats trade dicts from
SignalState into the schema expected by insert_trade.
"""

import logging
from shared.journal_db import insert_trade, init_db, get_monthly_report, get_stats
from shared.journal_db import get_cumulative_pnl, get_cumulative_pnl_by_instrument
from shared.journal_db import get_recent_trades, get_trades_for_date
from shared.journal_db import get_weekly_pnl, seed_scaling_ladder, migrate_csv

logger = logging.getLogger(__name__)


def log_trade(instrument: str, trade: dict, state=None) -> int:
    """
    Log a completed trade to the SQLite journal.
    Maps field names from SignalState.trades entries to journal schema.
    Returns the new trade row ID.
    """
    if not trade:
        logger.warning(f"[{instrument}] log_trade called with empty trade dict")
        return 0

    # Build trade dict compatible with journal_db.insert_trade
    record = {
        "num": trade.get("num", 0),
        "direction": trade.get("direction", ""),
        "entry": trade.get("entry", 0),
        "exit": trade.get("exit", 0),
        "entry_intended": trade.get("entry_intended", 0),
        "exit_intended": trade.get("exit_intended", 0),
        "time": trade.get("time", ""),
        "exit_time": trade.get("exit_time", ""),
        "pnl_pts": trade.get("pnl_pts", 0),
        "contracts": trade.get("contracts_stopped", 1),
        "contracts_stopped": trade.get("contracts_stopped", 0),
        "adds_used": trade.get("adds_used", 0),
        "add_pnl_pts": trade.get("pnl_adds", 0),
        "entry_slippage": trade.get("entry_slippage", 0),
        "exit_slippage": trade.get("exit_slippage", 0),
        "slippage_total": trade.get("slippage_total", 0),
        "tp1_filled": trade.get("tp1_filled", False),
        "tp2_filled": trade.get("tp2_filled", False),
        "mfe": trade.get("mfe", 0),
        "exit_reason": trade.get("exit_reason", ""),
        "signal_bar": trade.get("signal_bar", 4),
        "bar5_rule": trade.get("bar5_rule", ""),
        "gap_dir": trade.get("gap_dir", ""),
        "session": trade.get("session", "S1"),
        "signal_type": trade.get("signal_type", "BRACKET"),
    }

    try:
        trade_id = insert_trade(instrument, record, state)
        logger.info(f"[{instrument}] Trade logged to DB: #{record['num']} "
                     f"{record['direction']} {record['pnl_pts']:+.1f}pts")
        return trade_id
    except Exception as e:
        logger.error(f"[{instrument}] Journal insert failed: {e}", exc_info=True)
        return 0
