"""
dax_trade_diff.py — trade-level IG-vs-journal diff for one instrument.

The aggregate reconcile (ig_journal_reconcile.py) flagged DAX: IG −£9.55 vs
journal +£30.45 over May20–Jun1, with IG showing 33 deals to the journal's 29.
This lists IG deals next to journal rows PER DAY so the mis-records and the
4-trade gap are pinpointed (root cause: the get_recent_close_price fallback bug).
Read-only; run after close.

Usage: python3 dax_trade_diff.py [--inst DAX] [--days 13]
"""
import os, sys, re, argparse, sqlite3, asyncio
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass
from shared.ig_session import IGSharedSession
from ig_journal_reconcile import parse_pnl, map_instrument

DB = "/root/asrs-bot/data/trade_journal.db"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="DAX")
    ap.add_argument("--days", type=int, default=13)
    args = ap.parse_args()
    inst = args.inst

    # NAIVE UTC, no microseconds — IG rejects tz-aware isoformat (the "+00:00"
    # becomes a URL space and fails parsing). Host runs UTC; daily aggregation
    # makes any few-hour boundary fuzz irrelevant.
    now = datetime.utcnow().replace(microsecond=0)
    from_dt = now - timedelta(days=args.days)

    s = IGSharedSession()
    if not await s.connect():
        print("IG connect failed"); return
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None,
        lambda: s.ig.fetch_transaction_history(
            trans_type="ALL_DEAL", from_date=from_dt, to_date=now, page_size=5000),
    )
    await s.disconnect()

    ig_by_day = defaultdict(list)
    if df is not None and hasattr(df, "itertuples") and len(df):
        cols = {c.lower(): c for c in df.columns}
        g = lambda r, n: getattr(r, cols[n], None) if n in cols else None
        for r in df.itertuples():
            if map_instrument(g(r, "instrumentname")) != inst:
                continue
            d = str(g(r, "dateutc") or g(r, "date") or "")[:10]
            ig_by_day[d].append({
                "t": str(g(r, "dateutc") or "")[11:19],
                "pnl": parse_pnl(g(r, "profitandloss")),
                "open": g(r, "openlevel"), "close": g(r, "closelevel"),
                "size": g(r, "size"),
            })

    conn = sqlite3.connect(DB); cur = conn.cursor()
    jr = cur.execute(
        "SELECT id,date,entry_time,exit_time,entry_price,exit_price,pnl_gbp,"
        "exit_reason,signal_type FROM trades WHERE mode='live' AND instrument=? "
        "AND date>=? ORDER BY date,entry_time",
        (inst, from_dt.strftime("%Y-%m-%d")),
    ).fetchall()
    conn.close()
    jr_by_day = defaultdict(list)
    for row in jr:
        jr_by_day[row[1]].append(row)

    days = sorted(set(ig_by_day) | set(jr_by_day))
    print(f"=== {inst} trade-level IG vs Journal  (last {args.days}d) ===")
    tot_ig = tot_jr = 0.0
    for d in days:
        ig = ig_by_day.get(d, []); js = jr_by_day.get(d, [])
        ig_sum = sum(x["pnl"] for x in ig); jr_sum = sum((r[6] or 0) for r in js)
        tot_ig += ig_sum; tot_jr += jr_sum
        flag = "  ⚠️" if (len(ig) != len(js) or abs(ig_sum - jr_sum) > 1.0) else ""
        print(f"\n{d}  IG n={len(ig)} £{ig_sum:+.2f}  |  Jrnl n={len(js)} £{jr_sum:+.2f}{flag}")
        if not flag:
            continue
        print("   IG  :", ", ".join(f"{x['t']} {x['open']}→{x['close']} £{x['pnl']:+.2f}" for x in ig) or "(none)")
        print("   Jrnl:", ", ".join(f"#{r[0]} {r[2]}→{r[3]} {r[4]}→{r[5]} £{(r[6] or 0):+.2f} {r[7]}" for r in js) or "(none)")
    print(f"\nTOTAL  IG £{tot_ig:+.2f}  |  Jrnl £{tot_jr:+.2f}  |  Δ(J−IG) £{tot_jr - tot_ig:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())
