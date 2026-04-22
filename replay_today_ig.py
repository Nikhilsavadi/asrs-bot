"""
replay_today_ig.py — Pull today's IG bars via REST and run the backtest
engine against them, then compare to live trades.

IG-equivalent of replay_today.py (which uses IBKR).

Usage:
    python3 replay_today_ig.py                  # today
    python3 replay_today_ig.py 2026-04-09       # specific date
    REPLAY_INSTRUMENTS=DAX python3 replay_today_ig.py
"""
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

from shared.ig_session import IGSharedSession
from asrs.config import INSTRUMENTS as ASRS_INSTRUMENTS
import backtest as bt

# Match live config (including BE buffer for fairness)
for c in bt.INSTRUMENTS.values():
    c["max_entries"] = 3
    c["add_max"] = 0
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["max_risk_gbp"] = 50.0


async def fetch_ig_bars(session, epic: str, tz_str: str) -> pd.DataFrame:
    """Fetch 2 days of 5-min bars from IG REST for an epic."""
    try:
        tz = ZoneInfo(tz_str)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=2)

        result = await session.rest_call(
            session.ig.fetch_historical_prices_by_epic_and_date_range,
            epic=epic, resolution="MINUTE_5",
            start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if result is None:
            return pd.DataFrame()

        prices = result.get("prices", result) if isinstance(result, dict) else result
        if prices is None:
            return pd.DataFrame()

        # trading_ig returns a pandas MultiIndex DataFrame with cols like
        # ('bid','Open'), ('bid','High'), ('ask','Open'), etc.
        if hasattr(prices, "columns") and isinstance(prices.columns, pd.MultiIndex):
            bid = prices["bid"]
            ask = prices["ask"]
            df = pd.DataFrame({
                "Open":  ((bid["Open"]  + ask["Open"])  / 2).round(1),
                "High":  ((bid["High"]  + ask["High"])  / 2).round(1),
                "Low":   ((bid["Low"]   + ask["Low"])   / 2).round(1),
                "Close": ((bid["Close"] + ask["Close"]) / 2).round(1),
            })
            # Index is UTC datetimes already from IG
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()

        # Fallback: list-of-records (older API shape)
        if hasattr(prices, "to_dict"):
            prices = prices.to_dict("records")
        if not prices:
            return pd.DataFrame()
        rows = []
        for p in prices:
            ts = p.get("snapshotTime") or p.get("snapshotTimeUTC")
            if not ts: continue
            try:
                o = round((float(p.get("openPrice", {}).get("bid", 0))
                           + float(p.get("openPrice", {}).get("ask", 0))) / 2, 1)
                h = round((float(p.get("highPrice", {}).get("bid", 0))
                           + float(p.get("highPrice", {}).get("ask", 0))) / 2, 1)
                l = round((float(p.get("lowPrice", {}).get("bid", 0))
                           + float(p.get("lowPrice", {}).get("ask", 0))) / 2, 1)
                c = round((float(p.get("closePrice", {}).get("bid", 0))
                           + float(p.get("closePrice", {}).get("ask", 0))) / 2, 1)
            except (ValueError, TypeError):
                continue
            rows.append({"time": ts, "Open": o, "High": h, "Low": l, "Close": c})
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["time"], utc=True)
        return df[["Open","High","Low","Close"]].sort_index()

    except Exception as e:
        print(f"  fetch_ig_bars failed: {e}")
        return pd.DataFrame()


def run_for_inst(df, inst_name, target_date):
    cfg = bt.INSTRUMENTS[inst_name]
    df_local = df.copy()
    df_local.index = df_local.index.tz_convert(cfg["timezone"])
    df_local = df_local[df_local.index.date == target_date]
    if df_local.empty:
        return []
    df_local["_hour"] = df_local.index.hour
    df_local["_minute"] = df_local.index.minute

    ohlc = df_local[["Open", "High", "Low", "Close"]].values
    hours = df_local["_hour"].values
    minutes = df_local["_minute"].values

    sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
    out = []
    for session in sessions:
        oh = cfg[f"s{session}_open_hour"]; om = cfg[f"s{session}_open_minute"]
        eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
        trades = bt.simulate_session(ohlc, hours, minutes, oh, om, eh, em, cfg)
        for t in trades:
            t["session"] = f"S{session}"
            out.append(t)
    return out


async def main():
    if len(sys.argv) > 1:
        target_date_str = sys.argv[1]
    else:
        target_date_str = os.getenv("REPLAY_DATE", datetime.utcnow().strftime("%Y-%m-%d"))
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    inst_filter = os.getenv("REPLAY_INSTRUMENTS", "").strip()
    if inst_filter:
        wanted = {x.strip().upper() for x in inst_filter.split(",")}
    else:
        wanted = set(ASRS_INSTRUMENTS.keys())

    s = IGSharedSession.get_instance()
    await s.connect()
    print(f"Pulling bars for {target_date_str} from IG REST...\n")

    bt_results = {}
    for inst_name in sorted(wanted):
        if inst_name not in ASRS_INSTRUMENTS:
            continue
        epic = ASRS_INSTRUMENTS[inst_name]["epic"]
        tz_str = ASRS_INSTRUMENTS[inst_name]["timezone"]
        df = await fetch_ig_bars(s, epic, tz_str)
        if df.empty:
            print(f"  {inst_name}: no bars")
            continue
        trades = run_for_inst(df, inst_name, target_date)
        bt_results[inst_name] = trades

    # Compare to live journal
    db_path = os.getenv("JOURNAL_DB", "/app/data/trade_journal.db")
    if not os.path.exists(db_path):
        db_path = "/root/asrs-bot/data/trade_journal.db"
    c = sqlite3.connect(db_path)
    placeholders = ",".join(["?"] * len(wanted))
    live_rows = c.execute(
        f"""SELECT instrument, trade_num, direction, entry_price, exit_price,
                   pnl_pts, exit_reason, entry_time
            FROM trades
            WHERE date=? AND instrument IN ({placeholders})
            ORDER BY rowid""",
        (target_date_str, *sorted(wanted)),
    ).fetchall()

    print(f"\n{'='*78}")
    print(f"  COMPARISON  {target_date_str}")
    print(f"{'='*78}")

    for inst in [i for i in ["DAX", "US30", "NIKKEI"] if i in wanted]:
        print(f"\n---- {inst} ----")
        bt_t = bt_results.get(inst, [])
        live_t = [r for r in live_rows if r[0] == inst]

        bt_net = sum(t["pnl_pts"] for t in bt_t)
        live_net = sum(float(r[5] or 0) for r in live_t)

        print(f"  BACKTEST: {len(bt_t)} trades  net {bt_net:+.0f} pts")
        for t in bt_t:
            print(f"    {t['session']} {t['direction']:<6} {t['entry']:>9.1f} -> {t['exit']:>9.1f}  {t['pnl_pts']:>+7.0f}  {t['reason']}")
        print(f"  LIVE:     {len(live_t)} trades  net {live_net:+.0f} pts")
        for r in live_t:
            print(f"    #{r[1]} {r[2]:<6} {float(r[3] or 0):>9.1f} -> {float(r[4] or 0):>9.1f}  {float(r[5] or 0):>+7.0f}  {r[6]}")
        print(f"  DELTA: {live_net - bt_net:+.0f} pts")

    bt_total = sum(sum(t["pnl_pts"] for t in trades) for trades in bt_results.values())
    live_total = sum(float(r[5] or 0) for r in live_rows)
    print(f"\n{'='*78}")
    print(f"  TOTAL  Backtest: {bt_total:+.0f} pts  Live: {live_total:+.0f} pts  Delta: {live_total - bt_total:+.0f}")
    print(f"{'='*78}")


if __name__ == "__main__":
    asyncio.run(main())
