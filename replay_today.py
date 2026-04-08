"""
replay_today.py — Pull fresh IBKR bars for today and run the backtest
engine against them, then compare to live trades.

Confirms whether live execution matches what the backtest would do.
"""
import asyncio
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from ib_async import Future, Contract
from shared.ib_session import IBSharedSession
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


# Map instrument to IBKR symbol/exchange (NOTE: live uses FDXS/MYM/NIY for size,
# but for the backtest comparison we want the LIQUID contract: FDAX/YM/NKD)
INSTRUMENTS = {
    "DAX": {"symbol": "DAX", "exchange": "EUREX", "currency": "EUR", "tc": "FDAX"},
    "US30": {"symbol": "YM", "exchange": "CBOT", "currency": "USD", "tc": ""},
    "NIKKEI": {"symbol": "NKD", "exchange": "CME", "currency": "USD", "tc": ""},
}


async def fetch_bars(session, spec):
    kwargs = dict(symbol=spec["symbol"], exchange=spec["exchange"], currency=spec["currency"])
    if spec["tc"]:
        kwargs["tradingClass"] = spec["tc"]
    c = Future(**kwargs)
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
        contract, endDateTime="", durationStr="2 D",
        barSizeSetting="5 mins", whatToShow="TRADES", useRTH=False, formatDate=2,
    )
    if not bars:
        return contract, pd.DataFrame()
    df = pd.DataFrame([{"Open": float(b.open), "High": float(b.high),
                        "Low": float(b.low), "Close": float(b.close)} for b in bars])
    idx = pd.DatetimeIndex([b.date if isinstance(b.date, datetime)
                            else pd.to_datetime(b.date, utc=True) for b in bars])
    df.index = idx
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return contract, df


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
    s = IBSharedSession.get_instance()
    await s.connect()
    print(f"Pulling today's bars from IBKR (LIQUID contracts)...\n")

    bt_results = {}
    for inst, spec in INSTRUMENTS.items():
        contract, df = await fetch_bars(s, spec)
        if contract is None or df is None or df.empty:
            print(f"  {inst}: failed to fetch")
            continue
        target = datetime.now().date() if datetime.now().hour > 0 else datetime.now().date()
        # NIKKEI Tokyo session ran on 2026-04-08 JST which started at 01:00 UTC,
        # but the date label is 2026-04-08. DAX/US30 trade today.
        target_date = datetime(2026, 4, 8).date()
        trades = run_for_inst(df, inst, target_date)
        bt_results[inst] = trades

    # Compare to live
    c = sqlite3.connect("/root/asrs-bot/data/trade_journal.db")
    live_rows = c.execute("""SELECT instrument, trade_num, direction, entry_price, exit_price,
                                    pnl_pts, exit_reason, entry_time
                             FROM trades WHERE mode='paper' AND date='2026-04-08'
                             ORDER BY rowid""").fetchall()

    print(f"\n{'='*78}")
    print(f"  COMPARISON  2026-04-08")
    print(f"{'='*78}")

    for inst in ["DAX", "US30", "NIKKEI"]:
        print(f"\n──── {inst} ────")
        bt_t = bt_results.get(inst, [])
        live_t = [r for r in live_rows if r[0] == inst]

        bt_net = sum(t["pnl_pts"] for t in bt_t)
        live_net = sum(float(r[5] or 0) for r in live_t)

        print(f"  BACKTEST: {len(bt_t)} trades  net {bt_net:+.0f} pts")
        for t in bt_t:
            print(f"    {t['session']} {t['direction']:<6} {t['entry']:>9.0f} → {t['exit']:>9.0f}  {t['pnl_pts']:>+7.0f}  {t['reason']}")
        print(f"  LIVE:     {len(live_t)} trades  net {live_net:+.0f} pts")
        for r in live_t:
            print(f"    #{r[1]} {r[2]:<6} {float(r[3] or 0):>9.0f} → {float(r[4] or 0):>9.0f}  {float(r[5] or 0):>+7.0f}  {r[6]}")
        print(f"  DELTA: {live_net - bt_net:+.0f} pts")

    bt_total = sum(sum(t["pnl_pts"] for t in trades) for trades in bt_results.values())
    live_total = sum(float(r[5] or 0) for r in live_rows)
    print(f"\n{'='*78}")
    print(f"  TOTAL  Backtest: {bt_total:+.0f} pts  Live: {live_total:+.0f} pts  Delta: {live_total - bt_total:+.0f}")
    print(f"{'='*78}")

    await s.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
