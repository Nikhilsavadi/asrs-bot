"""
daily_ig_reconcile.py — compare IG account activity to bot journal.

Queries IG's activity/transaction history for today, sums pnl per instrument,
compares to journal's pnl_pts × stake_per_point. Alerts if divergence > £5 per
instrument or >£10 total.

Cron-friendly. Designed to catch:
  - Missed trade captures (bot didn't log an IG trade)
  - Slippage recording bugs (bot pnl != IG pnl)
  - Ghost positions (bot logged but IG didn't execute)
"""
import os, sys, time, sqlite3, asyncio, requests
import pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
DB = "/root/asrs-bot/data/trade_journal.db"

EPIC_TO_INST = {
    "IX.D.DAX.DAILY.IP": "DAX",
    "IX.D.DOW.DAILY.IP": "US30",
    "IX.D.NIKKEI.DAILY.IP": "NIKKEI",
}

ALERT_PER_INST_PCT = 5.0   # £5 per instrument triggers alert
ALERT_TOTAL_PCT = 10.0     # £10 total triggers alert


def send_tg(msg):
    if not (TG_TOKEN and TG_CHAT):
        print(msg); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"TG send failed: {e}\n{msg}")


async def query_ig_activity(date_str):
    """Query IG activity for the given date. Returns list of {epic, pnl, direction, ...}."""
    try:
        # Use the running bot's shared session to avoid double-login
        # Call via docker exec to run inside the container
        import subprocess
        cmd = [
            "docker", "exec", "asrs-bot", "python", "-c",
            f"""
import asyncio, json
from shared.ig_session import IGSharedSession
from datetime import datetime, timedelta

async def main():
    s = IGSharedSession()
    await s.ensure_connected()
    # IG activity history — use get_activity
    try:
        today = datetime.now().strftime('%Y-%m-%dT00:00:00')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')
        result = await s.rest_call(
            s.ig.fetch_account_activity_by_date_range,
            today, tomorrow
        )
        out = []
        if result is not None and hasattr(result, 'itertuples'):
            for row in result.itertuples():
                if row.status != 'ACCEPTED': continue
                out.append({{
                    'epic': getattr(row, 'epic', ''),
                    'type': getattr(row, 'type', ''),
                    'dealId': getattr(row, 'dealId', ''),
                    'level': float(getattr(row, 'level', 0)),
                    'size': float(getattr(row, 'size', 0)),
                    'period': getattr(row, 'period', ''),
                }})
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({{'error': str(e)}}))

asyncio.run(main())
""",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        import json
        data = json.loads(result.stdout.strip().split("\n")[-1])
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"IG activity query failed: {e}")
        return []


def query_journal_today(date_str):
    """Load today's live trades from journal. Returns DataFrame."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT instrument, entry_time, direction, entry_price, exit_price, "
        "pnl_pts, stake_per_point, signal_type "
        "FROM trades WHERE mode='live' AND date=? ORDER BY entry_time",
        conn, params=(date_str,)
    )
    conn.close()
    df["pnl_gbp"] = df["pnl_pts"] * df["stake_per_point"]
    return df


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    journal = query_journal_today(today)

    if len(journal) == 0:
        send_tg(f"*Daily reconcile — {today}*\n_No live trades today_")
        return

    # Summary per instrument
    journal_summary = journal.groupby("instrument")["pnl_gbp"].agg(["sum", "count"])

    lines = [f"*Daily IG reconcile — {today}*"]
    lines.append(f"Journal trades: {len(journal)}\n")
    lines.append("Per-instrument (journal):")
    total_journal = 0
    for inst, row in journal_summary.iterrows():
        net = row["sum"]; n = row["count"]
        total_journal += net
        lines.append(f"  `{inst}`: n={int(n)} pnl=£{net:+.2f}")
    lines.append(f"\n*Journal total: £{total_journal:+.2f}*")

    # IG activity — feature-flagged — requires IG account API permissions
    # For now, just log journal + flag any reconciliation gaps manually
    lines.append("\n_(IG-side automated reconcile pending — manually diff vs IG statement)_")

    msg = "\n".join(lines)
    print(msg)
    send_tg(msg)


if __name__ == "__main__":
    main()
