"""
sync_to_railway.py — Push DAX + FTSE trade data to Railway dashboard
=====================================================================

Reads trade journals from CSV and POSTs to Railway backend.
Run via cron every 5 minutes or after each trade.

Usage:
    python3 sync_to_railway.py              # Sync both bots
    python3 sync_to_railway.py --dax        # DAX only
    python3 sync_to_railway.py --ftse       # FTSE only

Environment:
    RAILWAY_API_URL     Railway backend URL (e.g. https://nikhiltrades.up.railway.app)
    BOT_SYNC_SECRET     Shared secret for auth
"""

import csv
import json
import os
import sys
import httpx

RAILWAY_API_URL = os.environ.get("RAILWAY_API_URL", "https://nikhiltrades.up.railway.app")
BOT_SYNC_SECRET = os.environ.get("BOT_SYNC_SECRET", "changeme")

DAX_JOURNAL = os.path.join(os.path.dirname(__file__), "data", "dax", "trade_journal.csv")
FTSE_JOURNAL = os.path.join(os.path.dirname(__file__), "data", "ftse", "ftse_trades.csv")


def read_dax_trades() -> list[dict]:
    """Read DAX trade journal CSV."""
    if not os.path.exists(DAX_JOURNAL):
        return []
    trades = []
    with open(DAX_JOURNAL, "r") as f:
        for row in csv.DictReader(f):
            trades.append({
                "date": row.get("date", ""),
                "trade_num": int(row.get("trade_num", 1) or 1),
                "direction": row.get("direction", ""),
                "entry": float(row.get("entry", 0) or 0),
                "exit": float(row.get("exit", 0) or 0),
                "pnl_pts": float(row.get("pnl_pts", 0) or 0),
                "mfe": float(row.get("mfe", 0) or 0),
                "stop_phase": "",
                "exit_reason": "",
                "entry_slippage": row.get("entry_slippage", ""),
                "exit_slippage": row.get("exit_slippage", ""),
                "tp1_filled": row.get("tp1_filled", ""),
                "tp2_filled": row.get("tp2_filled", ""),
                "bar_range": row.get("bar_range", ""),
                "overnight_bias": row.get("overnight_bias", ""),
            })
    return trades


def read_ftse_trades() -> list[dict]:
    """Read FTSE trade journal CSV."""
    if not os.path.exists(FTSE_JOURNAL):
        return []
    trades = []
    with open(FTSE_JOURNAL, "r") as f:
        for row in csv.DictReader(f):
            trades.append({
                "date": row.get("date", ""),
                "trade_num": 1,
                "direction": row.get("direction", ""),
                "entry": float(row.get("entry", 0) or 0),
                "exit": float(row.get("exit", 0) or 0),
                "pnl_pts": float(row.get("pnl_pts", 0) or 0),
                "mfe": float(row.get("mfe", 0) or 0),
                "bar_type": row.get("bar_type", ""),
                "bar_width": float(row.get("bar_width", 0) or 0),
                "stake": float(row.get("stake", 1) or 1),
                "stop_phase": row.get("stop_phase", ""),
                "exit_reason": row.get("exit_reason", ""),
            })
    return trades


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
        print(f"[{bot}] Synced {result.get('synced', 0)} trades")
    except Exception as e:
        print(f"[{bot}] Sync failed: {e}")


def main():
    args = sys.argv[1:]

    do_dax = "--dax" in args or not args
    do_ftse = "--ftse" in args or not args

    if do_dax:
        dax = read_dax_trades()
        print(f"[DAX] Read {len(dax)} trades from journal")
        sync("DAX", dax)

    if do_ftse:
        ftse = read_ftse_trades()
        print(f"[FTSE] Read {len(ftse)} trades from journal")
        sync("FTSE", ftse)


if __name__ == "__main__":
    main()
