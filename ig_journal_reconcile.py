"""
ig_journal_reconcile.py — authoritative IG-vs-journal reconciliation.

Pulls IG's OWN transaction history (the broker's record of realised P&L) and
diffs it against our SQLite journal per-instrument over a rolling window.
This finally implements the IG side that daily_ig_reconcile.py left stubbed.

Designed to run AFTER market close (default cron 22:30 UK) so the fresh IG
login cannot disrupt the live bot's session. Read-only against IG.

Catches:
  - Journal mis-recording vs broker truth (e.g. the get_recent_close_price
    fallback bug logging local trail levels instead of real IG fills)
  - Missed/ghost trades (count mismatch)
  - Cumulative £ drift between what we think we made and what IG paid

Usage:
  python3 ig_journal_reconcile.py [--days N | --from YYYY-MM-DD --to YYYY-MM-DD]
"""
import os, sys, re, argparse, sqlite3, asyncio, requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from shared.ig_session import IGSharedSession

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
DB = "/root/asrs-bot/data/trade_journal.db"

# IG instrumentName fragments → our instrument key
NAME_MATCHERS = [
    ("DAX", ("germany 40", "germany40", "dax")),
    ("US30", ("wall street", "us 30", "us tech", "dow")),  # us tech excluded below
    ("XAUUSD", ("gold",)),
    ("NIKKEI", ("japan 225", "nikkei")),
]
ALERT_PER_INST = 5.0   # £ per-instrument delta that triggers a flag
ALERT_TOTAL = 10.0     # £ total delta that triggers a flag


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


def map_instrument(name: str) -> str | None:
    n = (name or "").lower()
    if "us tech" in n or "nasdaq" in n:
        return None  # not traded
    for key, frags in NAME_MATCHERS:
        if any(f in n for f in frags):
            return key
    return None


def parse_pnl(val) -> float:
    """IG profitAndLoss is a string like '£-5.30' / 'E1,234.50'. → float."""
    if val is None:
        return 0.0
    s = str(val)
    s = re.sub(r"[^0-9.\-]", "", s.replace(",", ""))
    try:
        return float(s) if s not in ("", "-", ".") else 0.0
    except ValueError:
        return 0.0


async def fetch_ig(from_dt: datetime, to_dt: datetime) -> dict:
    """Return {instrument: {'pnl': £, 'n': count}} from IG transaction history,
    plus '_funding' and '_unmapped' buckets. Read-only."""
    s = IGSharedSession()
    if not await s.connect():
        raise RuntimeError("IG connect failed")
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None,
        lambda: s.ig.fetch_transaction_history(
            # page_size must be a large truthy value: trading_ig does
            # `if page_size:` so 0 is dropped and IG truncates to its small
            # default page. 5000 >> any plausible 12-day transaction count.
            trans_type="ALL", from_date=from_dt, to_date=to_dt, page_size=5000
        ),
    )
    await s.disconnect()

    out: dict = {}
    funding = 0.0
    unmapped = []
    if df is None or not hasattr(df, "itertuples") or len(df) == 0:
        return {"_funding": 0.0, "_unmapped": []}
    cols = {c.lower(): c for c in df.columns}

    def col(row, *names):
        for nm in names:
            if nm in cols:
                return getattr(row, cols[nm], None)
        return None

    for row in df.itertuples():
        name = col(row, "instrumentname", "instrument_name")
        pnl = parse_pnl(col(row, "profitandloss", "profit_and_loss"))
        ttype = str(col(row, "transactiontype", "transaction_type") or "").upper()
        cash = col(row, "cashtransaction", "cash_transaction")
        # Funding / interest / cash adjustments — track separately, not trade P&L.
        # cashTransaction may be a real bool or the string "true".
        is_cash = cash is True or (isinstance(cash, str) and cash.lower() == "true")
        if is_cash or "WITH" in ttype or "DEPO" in ttype:
            funding += pnl
            continue
        inst = map_instrument(name)
        if inst is None:
            if pnl:
                unmapped.append((name, pnl))
            continue
        d = out.setdefault(inst, {"pnl": 0.0, "n": 0})
        d["pnl"] += pnl
        d["n"] += 1
    out["_funding"] = funding
    out["_unmapped"] = unmapped
    return out


def fetch_journal(from_s: str, to_s: str) -> dict:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT instrument, COUNT(*), ROUND(SUM(pnl_gbp),2) "
        "FROM trades WHERE mode='live' AND date>=? AND date<=? GROUP BY instrument",
        (from_s, to_s),
    ).fetchall()
    conn.close()
    return {r[0]: {"n": r[1], "pnl": r[2] or 0.0} for r in rows}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", dest="to", default=None)
    args = ap.parse_args()

    if args.frm:
        from_dt = datetime.strptime(args.frm, "%Y-%m-%d")
        to_dt = datetime.strptime(args.to, "%Y-%m-%d") + timedelta(days=1) if args.to else datetime.now()
    else:
        to_dt = datetime.now()
        from_dt = (to_dt - timedelta(days=args.days)).replace(hour=0, minute=0, second=0)
    from_s, to_s = from_dt.strftime("%Y-%m-%d"), (to_dt - timedelta(seconds=1)).strftime("%Y-%m-%d")

    try:
        ig = await fetch_ig(from_dt, to_dt)
    except Exception as e:
        send_tg(f"*IG reconcile FAILED* {from_s}→{to_s}\nIG pull error: `{e}`")
        raise

    jrnl = fetch_journal(from_s, to_s)
    funding = ig.pop("_funding", 0.0)
    unmapped = ig.pop("_unmapped", [])

    insts = sorted(set(ig) | set(jrnl))
    lines = [f"*IG⇄Journal reconcile  {from_s}→{to_s}*"]
    tot_ig = tot_j = 0.0
    flagged = False
    for inst in insts:
        ig_pnl = ig.get(inst, {}).get("pnl", 0.0); ig_n = ig.get(inst, {}).get("n", 0)
        j_pnl = jrnl.get(inst, {}).get("pnl", 0.0); j_n = jrnl.get(inst, {}).get("n", 0)
        delta = j_pnl - ig_pnl
        tot_ig += ig_pnl; tot_j += j_pnl
        mark = "⚠️" if abs(delta) > ALERT_PER_INST else "✅"
        if abs(delta) > ALERT_PER_INST:
            flagged = True
        lines.append(
            f"{mark} `{inst}` IG £{ig_pnl:+.2f}(n{ig_n}) | Jrnl £{j_pnl:+.2f}(n{j_n}) "
            f"| Δ(J−IG) £{delta:+.2f}"
        )
    tot_delta = tot_j - tot_ig
    lines.append(f"\n*TOTAL* IG £{tot_ig:+.2f} | Jrnl £{tot_j:+.2f} | Δ £{tot_delta:+.2f}")
    if funding:
        lines.append(f"_IG funding/cash (excl): £{funding:+.2f}_")
    if unmapped:
        lines.append(f"_Unmapped IG rows: {unmapped[:5]}_")
    if abs(tot_delta) > ALERT_TOTAL:
        flagged = True
    lines.append("\n*⚠️ DIVERGENCE — investigate*" if flagged else "\n_in sync_")

    msg = "\n".join(lines)
    print(msg)
    send_tg(msg)


if __name__ == "__main__":
    asyncio.run(main())
