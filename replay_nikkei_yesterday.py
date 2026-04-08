"""
replay_nikkei_yesterday.py — Pull fresh IBKR NKD bars for the overnight session
and run the backtest engine against them, then compare side-by-side with the
live paper trades from the journal.
"""
import asyncio
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ib_async import Future
from shared.ib_session import IBSharedSession
import backtest as bt

# Match live config
for c in bt.INSTRUMENTS.values():
    c["max_entries"] = 3
    c["add_max"] = 0  # adds disabled
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
# Use 50pt risk cap (matches current live)
bt.INSTRUMENTS["NIKKEI"]["max_risk_gbp"] = 50.0


async def fetch_nkd_bars(session, days=2):
    """Pull last 2 days of 5-min NKD (front-month) RTH bars."""
    c = Future(symbol="NKD", exchange="CME", currency="USD")
    details = await session.ib.reqContractDetailsAsync(c)
    today = datetime.now().date()
    contract = None
    for d in sorted(details, key=lambda x: x.contract.lastTradeDateOrContractMonth):
        try:
            exp = datetime.strptime(d.contract.lastTradeDateOrContractMonth, "%Y%m%d").date()
        except Exception:
            continue
        if exp > today and exp.month in (3, 6, 9, 12):
            contract = d.contract
            break
    if contract is None:
        return None, None

    bars = await session.ib.reqHistoricalDataAsync(
        contract, endDateTime="", durationStr=f"{days} D",
        barSizeSetting="5 mins", whatToShow="TRADES",
        useRTH=False,         # NKD trades nearly 24h, RTH would filter out the Tokyo session
        formatDate=2,
    )
    if not bars:
        return contract, pd.DataFrame()

    df = pd.DataFrame([{
        "Open": float(b.open), "High": float(b.high),
        "Low": float(b.low), "Close": float(b.close),
    } for b in bars])
    idx = pd.DatetimeIndex([
        b.date if isinstance(b.date, datetime) else pd.to_datetime(b.date, utc=True)
        for b in bars
    ])
    df.index = idx
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return contract, df


def run_backtest_for_date(df, target_date_jst):
    """Run the backtest's simulate_session for NIKKEI on a specific JST date."""
    cfg = bt.INSTRUMENTS["NIKKEI"]
    # Convert df index to JST (NIKKEI's timezone)
    df_jst = df.copy()
    df_jst.index = df_jst.index.tz_convert("Asia/Tokyo")
    df_jst = df_jst[df_jst.index.date == target_date_jst]
    df_jst["_hour"] = df_jst.index.hour
    df_jst["_minute"] = df_jst.index.minute

    ohlc = df_jst[["Open", "High", "Low", "Close"]].values
    hours = df_jst["_hour"].values
    minutes = df_jst["_minute"].values

    out = []
    for session in (1, 2, 3):
        oh = cfg[f"s{session}_open_hour"]
        om = cfg[f"s{session}_open_minute"]
        eh = cfg["session_end_hour"]
        em = cfg["session_end_minute"]
        trades = bt.simulate_session(ohlc, hours, minutes, oh, om, eh, em, cfg)
        for t in trades:
            t["session"] = f"S{session}"
            out.append(t)
    return out, df_jst


async def main():
    s = IBSharedSession.get_instance()
    await s.connect()
    print("Pulling NKD bars from IBKR (last 2 days, 24h not RTH)...")
    contract, df = await fetch_nkd_bars(s, days=2)
    if contract is None or df is None or df.empty:
        print("Failed to fetch bars")
        return
    print(f"Got {len(df)} bars  conId={contract.conId}")
    print(f"Range: {df.index.min()} → {df.index.max()}")

    # Yesterday's date in JST: 2026-04-08 (the bot logged trades dated 2026-04-08)
    target_date = datetime(2026, 4, 8).date()
    print(f"\nReplaying NIKKEI strategy on {target_date} (JST)...\n")

    bt_trades, df_jst = run_backtest_for_date(df, target_date)
    print(f"Bars in JST window for {target_date}: {len(df_jst)}")
    if len(df_jst):
        print(f"First bar: {df_jst.index[0]}  Last bar: {df_jst.index[-1]}")

    # Compare
    print(f"\n{'='*78}")
    print(f"  BACKTEST trades on NKD for {target_date}")
    print(f"{'='*78}")
    if not bt_trades:
        print("  No backtest trades")
    else:
        print(f"  {'session':<6}{'dir':<6}{'entry':>10}{'exit':>10}{'pts':>10}  reason")
        for t in bt_trades:
            print(f"  {t['session']:<6}{t['direction']:<6}{t['entry']:>10.0f}{t['exit']:>10.0f}"
                  f"{t['pnl_pts']:>+10.1f}  {t['reason']}")
        net = sum(t["pnl_pts"] for t in bt_trades)
        wins = sum(1 for t in bt_trades if t["pnl_pts"] > 0)
        losses = sum(1 for t in bt_trades if t["pnl_pts"] < 0)
        print(f"\n  BACKTEST NET: {net:+.1f} pts  ({wins}W/{losses}L)")

    # Live
    print(f"\n{'='*78}")
    print(f"  LIVE paper trades from journal for NIKKEI 2026-04-08")
    print(f"{'='*78}")
    c = sqlite3.connect("/root/asrs-bot/data/trade_journal.db")
    rows = c.execute(
        "SELECT trade_num, direction, entry_price, exit_price, pnl_pts, exit_reason, entry_time, range_flag "
        "FROM trades WHERE mode='paper' AND instrument='NIKKEI' AND date='2026-04-08' "
        "ORDER BY entry_time"
    ).fetchall()
    if not rows:
        print("  No live trades")
    else:
        # SELECT order: trade_num, direction, entry_price, exit_price,
        # pnl_pts, exit_reason, entry_time, range_flag
        print(f"  {'time':<6}{'#':<3}{'dir':<6}{'entry':>10}{'exit':>10}{'pts':>10}  reason")
        for r in rows:
            num, dir_, entry, exit_, pnl, reason, time_, rf = r
            print(f"  {time_:<6}{str(num):<3}{dir_:<6}{float(entry or 0):>10.0f}"
                  f"{float(exit_ or 0):>10.0f}{float(pnl or 0):>+10.1f}  {reason}")
        # pnl_pts is index 4 in the SELECT
        live_net = sum(float(r[4] or 0) for r in rows)
        live_w = sum(1 for r in rows if float(r[4] or 0) > 0)
        live_l = sum(1 for r in rows if float(r[4] or 0) < 0)
        print(f"\n  LIVE NET: {live_net:+.1f} pts  ({live_w}W/{live_l}L)")

    # Verdict
    print(f"\n{'='*78}")
    print(f"  COMPARISON")
    print(f"{'='*78}")
    bt_net = sum(t["pnl_pts"] for t in bt_trades) if bt_trades else 0
    live_net = sum(float(r[4] or 0) for r in rows) if rows else 0
    print(f"  Backtest:  {bt_net:+.1f} pts  ({len(bt_trades)} trades)")
    print(f"  Live:      {live_net:+.1f} pts  ({len(rows)} trades)")
    print(f"  Delta:     {live_net - bt_net:+.1f} pts")

    await s.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
