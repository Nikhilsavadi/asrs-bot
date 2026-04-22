"""
nightly_live_replay.py — daily LIVE vs A-replay vs B-replay on IG ticks.

For TODAY's live trades only, replay each trade's exit under:
  A trail (prev_low with tight_threshold)
  B trail (prev_close always)
using the exact IG tick data the bot saw.

Posts per-instrument delta summary to Telegram. Intended for cron ~22:30 UTC.
"""
import os, sys, sqlite3, time, pandas as pd, requests
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from replay_exit_a_vs_b import simulate_exit, load_ticks, build_5min_bars, CFG

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
DB = "/root/asrs-bot/data/trade_journal.db"


def send_tg(msg: str):
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


def main():
    t0 = time.time()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        "SELECT instrument, date, direction, entry_price, entry_intended, "
        "exit_price, pnl_pts, entry_time, bar_range "
        "FROM trades WHERE mode='live' AND date=? ORDER BY entry_time",
        conn, params=(today,)
    )
    conn.close()

    if len(live) == 0:
        send_tg(f"*Nightly live replay* — {today}\n_no live trades today_")
        return

    cache = {}
    def get(epic, ds, tz):
        key = (epic, ds)
        if key not in cache:
            ticks = load_ticks(epic, ds)
            cache[key] = (ticks, build_5min_bars(ticks, tz)) if ticks is not None else None
        return cache[key]

    totals = {"LIVE": {}, "A": {}, "B": {}}
    for inst in ["DAX", "US30", "NIKKEI"]:
        for k in totals: totals[k][inst] = {"n": 0, "net": 0.0}
    skipped = 0

    for _, tr in live.iterrows():
        inst = tr["instrument"]; cfg = CFG[inst]
        cached = get(cfg["epic"], tr["date"], cfg["tz"])
        if cached is None:
            skipped += 1; continue
        ticks, bars = cached

        payload = dict(tr)
        res_a = simulate_exit(payload, ticks, bars, cfg, tr["date"], variant="A")
        res_b = simulate_exit(payload, ticks, bars, cfg, tr["date"], variant="B")
        if tr["direction"] == "LONG":
            pnl_a = res_a["exit_price"] - float(tr["entry_price"])
            pnl_b = res_b["exit_price"] - float(tr["entry_price"])
        else:
            pnl_a = float(tr["entry_price"]) - res_a["exit_price"]
            pnl_b = float(tr["entry_price"]) - res_b["exit_price"]
        live_pnl = float(tr["pnl_pts"])

        totals["LIVE"][inst]["n"] += 1; totals["LIVE"][inst]["net"] += live_pnl
        totals["A"][inst]["n"] += 1; totals["A"][inst]["net"] += pnl_a
        totals["B"][inst]["n"] += 1; totals["B"][inst]["net"] += pnl_b

    lines = [
        f"*Nightly live replay — {today}*",
        f"Live trades: {len(live)} ({skipped} skipped)",
        "",
    ]
    gl = ga = gb = gn = 0
    for inst in ["DAX", "US30", "NIKKEI"]:
        n = totals["LIVE"][inst]["n"]
        if n == 0: continue
        l = totals["LIVE"][inst]["net"]; a = totals["A"][inst]["net"]; b = totals["B"][inst]["net"]
        lines.append(f"*{inst}* ({n} trades)")
        lines.append(f"  LIVE:  {l:+.1f}pt")
        lines.append(f"  A sim: {a:+.1f}pt (live vs A: {l-a:+.1f})")
        lines.append(f"  B sim: {b:+.1f}pt (live vs B: {l-b:+.1f})")
        gl += l; ga += a; gb += b; gn += n

    lines += [
        "",
        f"*TOTAL* ({gn} trades)",
        f"  LIVE:  {gl:+.1f}pt",
        f"  A sim: {ga:+.1f}pt  ({gl-ga:+.1f} vs LIVE)",
        f"  B sim: {gb:+.1f}pt  ({gl-gb:+.1f} vs LIVE)",
        f"  B−A:   {gb-ga:+.1f}pt",
        "",
        f"_elapsed: {time.time()-t0:.0f}s_",
    ]
    msg = "\n".join(lines)
    print(msg)
    send_tg(msg)


if __name__ == "__main__":
    main()
