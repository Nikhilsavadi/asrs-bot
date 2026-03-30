"""
sync_to_railway.py — Push ASRS trade data to Railway dashboard
===============================================================

Reads from unified SQLite journal and POSTs to Railway backend.
Called periodically from asrs/main.py scheduler.

Environment:
    RAILWAY_API_URL     Railway backend URL
    BOT_SYNC_SECRET     Shared secret for auth
"""

import os
import httpx
import logging

logger = logging.getLogger(__name__)

RAILWAY_API_URL = os.environ.get("RAILWAY_API_URL", "https://nikhiltrades.up.railway.app")
BOT_SYNC_SECRET = os.environ.get("BOT_SYNC_SECRET", "changeme")


def read_trades(instrument: str) -> list[dict]:
    """Read trades from unified SQLite journal."""
    try:
        from shared.journal_db import get_recent_trades
        rows = get_recent_trades(instrument=instrument, limit=5000)
        trades = []
        for r in rows:
            trades.append({
                "date": r.get("date", ""),
                "trade_num": r.get("num", 1),
                "direction": r.get("direction", ""),
                "entry": r.get("entry", 0),
                "exit": r.get("exit", 0),
                "pnl_pts": r.get("pnl_pts", 0),
                "mfe": r.get("mfe", 0),
                "bar_type": r.get("range_flag", ""),
                "bar_width": r.get("bar_range", 0),
                "stake": r.get("stake", 1),
                "stop_phase": "",
                "exit_reason": r.get("exit_reason", ""),
                "adds_used": r.get("adds_used", 0),
                "signal": r.get("signal", ""),
            })
        return trades
    except Exception as e:
        logger.error(f"read_trades({instrument}) failed: {e}")
        return []


def sync(bot: str, trades: list[dict]):
    """POST trades to Railway backend."""
    url = f"{RAILWAY_API_URL}/api/bot/sync"
    payload = {
        "secret": BOT_SYNC_SECRET,
        "bot": bot,
        "trades": trades,
    }
    try:
        resp = httpx.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"[{bot}] Synced {result.get('synced', 0)} trades to dashboard")
    except Exception as e:
        logger.error(f"[{bot}] Dashboard sync failed: {e}")


def sync_all():
    """Sync all instruments to Railway dashboard."""
    for inst in ["DAX", "US30", "NIKKEI"]:
        trades = read_trades(inst)
        if trades:
            sync(inst, trades)


if __name__ == "__main__":
    sync_all()
