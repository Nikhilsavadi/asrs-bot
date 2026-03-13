"""
journal.py -- FTSE Trade Journal (delegates to shared SQLite)
"""

import logging
from shared import journal_db

logger = logging.getLogger(__name__)


def append_trade(trade: dict):
    """Append a completed trade to the journal."""
    journal_db.insert_trade("FTSE", trade)


def load_all_trades() -> list[dict]:
    """Load all FTSE trades from the journal."""
    return journal_db.get_recent_trades(n=10000, instrument="FTSE")


def get_weekly_pnl() -> float:
    """Calculate P&L for the current week."""
    result = journal_db.get_weekly_pnl(instrument="FTSE")
    return round(result["pnl_gbp"], 1)
