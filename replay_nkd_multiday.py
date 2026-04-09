"""
replay_nkd_multiday.py — Replay last N trading days on NKD (Nikkei Dollar)
to compare against the bot's NIY (Nikkei Yen) live performance.

NIY is what the bot trades on the £5k paper account because it's smaller
(¥500/pt vs $5/pt). NKD is what the 18yr backtest was built on AND what
the live account will trade at scale. This script answers: "if the bot
were on NKD all along, what would Nikkei P&L look like?"

Usage:
    python3 replay_nkd_multiday.py            # last 6 trading days
    python3 replay_nkd_multiday.py 10         # last N trading days
"""
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

from ib_async import IB, Future
from shared.ib_session import IBSharedSession
import backtest as bt

# Match live config
for c in bt.INSTRUMENTS.values():
    c["max_entries"] = 3
    c["add_max"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["max_risk_gbp"] = 50.0

# £/pt for the two contracts (rough)
NKD_GBP_PER_PT = 4.0   # $5/pt × ~0.80 GBP/USD
NIY_GBP_PER_PT = 2.65  # ¥500/pt × ~0.0053 GBP/JPY


async def resolve_contract(session, symbol_spec):
    sym, exch, ccy, tc = symbol_spec
    kwargs = dict(symbol=sym, exchange=exch, currency=ccy)
    if tc:
        kwargs["tradingClass"] = tc
    c = Future(**kwargs)
    details = await session.ib.reqContractDetailsAsync(c)
    today = datetime.now().date()
    for d in sorted(details, key=lambda x: x.contract.lastTradeDateOrContractMonth):
        try:
            exp = datetime.strptime(d.contract.lastTradeDateOrContractMonth, "%Y%m%d").date()
        except Exception:
            continue
        if exp > today and exp.month in (3, 6, 9, 12):
            return d.contract
    return None


async def fetch_one_day(session, contract, target_date: date) -> pd.DataFrame:
    """Pull 1 day of 5-min bars ending at the given date 23:59 UTC.
    IBKR caps 5-min bar requests at 86400s = 1 day per call."""
    end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)
    end_str = end.strftime("%Y%m%d-%H:%M:%S")
    bars = await session.ib.reqHistoricalDataAsync(
        contract, endDateTime=end_str,
        durationStr="86400 S",
        barSizeSetting="5 mins", whatToShow="TRADES",
        useRTH=False, formatDate=2,
    )
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "Open": float(b.open), "High": float(b.high),
        "Low": float(b.low), "Close": float(b.close),
        "Volume": float(getattr(b, "volume", 0) or 0),
    } for b in bars])
    idx = pd.DatetimeIndex([
        b.date if isinstance(b.date, datetime)
        else pd.to_datetime(b.date, utc=True) for b in bars
    ])
    df.index = idx
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def run_for_day(df, target_date):
    """Run the backtest engine on one day's worth of NIKKEI bars."""
    cfg = bt.INSTRUMENTS["NIKKEI"]
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


def get_live_niy_pnl(target_date_str: str) -> tuple[int, float]:
    """Pull NIY live P&L from the journal for a given date (in pts)."""
    try:
        c = sqlite3.connect("/root/asrs-bot/data/trade_journal.db")
        rows = c.execute(
            "SELECT pnl_pts FROM trades WHERE instrument='NIKKEI' AND date=?",
            (target_date_str,),
        ).fetchall()
        return len(rows), sum(float(r[0] or 0) for r in rows)
    except Exception as e:
        return 0, 0.0


async def main():
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    # Build trading-day list (skip weekends)
    target_dates = []
    d = date(2026, 4, 9)  # today
    while len(target_dates) < n_days:
        if d.weekday() < 5:
            target_dates.append(d)
        d -= timedelta(days=1)
    target_dates = sorted(target_dates)

    s = IBSharedSession.get_instance()
    await s.connect()

    print("=" * 88)
    print(f"  NKD vs NIY MULTI-DAY REPLAY  ({n_days} trading days)")
    print("=" * 88)

    async def fetch_for(label, spec):
        c = await resolve_contract(s, spec)
        if c is None:
            print(f"FAILED to resolve {label}")
            return None
        print(f"\nResolved {label}: {c.localSymbol} (expiry {c.lastTradeDateOrContractMonth})")
        day_dfs = {}
        for td in target_dates:
            try:
                df_day = await fetch_one_day(s, c, td)
                if not df_day.empty:
                    day_dfs[td] = df_day
                    print(f"  {label} {td.isoformat()}: {len(df_day)} bars")
            except Exception as e:
                print(f"  {label} {td.isoformat()}: error {e}")
            await asyncio.sleep(0.5)
        if not day_dfs:
            return None
        out = pd.concat(day_dfs.values()).sort_index()
        return out[~out.index.duplicated(keep="last")]

    df_nkd = await fetch_for("NKD", ("NKD", "CME", "USD", ""))
    df_niy = await fetch_for("NIY", ("NIY", "CME", "JPY", "NIY"))
    if df_nkd is None or df_niy is None:
        print("\nFAILED to fetch contracts")
        return
    print(f"\nNKD: {len(df_nkd)} bars  NIY: {len(df_niy)} bars")

    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    print()
    print(f"{'Date':<12} {'NKD-bt':>10} {'NIY-bt':>10} {'NIY-live':>10}  {'£@NKD':>10} {'£@NIY-bt':>10} {'£@NIY-live':>12}")
    print("-" * 88)

    bt_total_pts = 0.0
    bt_total_gbp = 0.0
    live_total_pts = 0.0
    live_total_gbp = 0.0
    bt_total_trades = 0
    live_total_trades = 0

    detail_rows = []
    nkd_total_pts = niy_bt_total_pts = live_total_pts = 0.0
    for td in target_dates:
        td_str = td.isoformat()
        nkd_trades = run_for_day(df_nkd, td)
        niy_trades = run_for_day(df_niy, td)
        nkd_pts = sum(t["pnl_pts"] for t in nkd_trades)
        niy_pts = sum(t["pnl_pts"] for t in niy_trades)
        live_n, live_pts = get_live_niy_pnl(td_str)
        nkd_total_pts += nkd_pts
        niy_bt_total_pts += niy_pts
        live_total_pts += live_pts
        detail_rows.append((td_str, nkd_trades, niy_trades))
        print(f"{td_str:<12} {nkd_pts:>+10.0f} {niy_pts:>+10.0f} {live_pts:>+10.0f}  "
              f"{nkd_pts*NKD_GBP_PER_PT:>+10.0f} {niy_pts*NIY_GBP_PER_PT:>+10.0f} "
              f"{live_pts*NIY_GBP_PER_PT:>+12.0f}")

    print("-" * 88)
    print(f"{'TOTAL':<12} {nkd_total_pts:>+10.0f} {niy_bt_total_pts:>+10.0f} {live_total_pts:>+10.0f}  "
          f"{nkd_total_pts*NKD_GBP_PER_PT:>+10.0f} {niy_bt_total_pts*NIY_GBP_PER_PT:>+10.0f} "
          f"{live_total_pts*NIY_GBP_PER_PT:>+12.0f}")
    print()
    print(f"  NKD backtest theoretical:    {nkd_total_pts:>+8.0f} pts  =  £{nkd_total_pts*NKD_GBP_PER_PT:>+7.0f}")
    print(f"  NIY backtest theoretical:    {niy_bt_total_pts:>+8.0f} pts  =  £{niy_bt_total_pts*NIY_GBP_PER_PT:>+7.0f}")
    print(f"  NIY live actual:             {live_total_pts:>+8.0f} pts  =  £{live_total_pts*NIY_GBP_PER_PT:>+7.0f}")
    print()
    print(f"  Contract delta (NKD vs NIY backtest): {nkd_total_pts - niy_bt_total_pts:+.0f} pts (microstructure)")
    print(f"  Live execution delta (NIY bt vs live):{niy_bt_total_pts - live_total_pts:+.0f} pts (slippage + bugs)")
    print()
    if verbose:
        print()
        print("=" * 88)
        print("  PER-TRADE DETAIL (NKD)")
        print("=" * 88)
        for td_str, nkd_trades, _ in detail_rows:
            print(f"\n── {td_str} ──")
            for t in nkd_trades:
                print(f"  {t['session']:<3} {t.get('direction', '?'):<6} "
                      f"entry={t.get('entry', 0):>9.0f} → exit={t.get('exit', 0):>9.0f}  "
                      f"{t.get('pnl_pts', 0):>+8.0f}pts  {t.get('reason', '')}")
        print()

    print("Notes:")
    print("  • Backtest uses NKD (Nikkei Dollar, $5/pt) — same contract as the 18yr firstrate backtest.")
    print("  • Live uses NIY (Nikkei Yen, ¥500/pt) — smaller contract for £5k account.")
    print("  • £ values: NKD £4/pt, NIY £2.65/pt (approx FX).")
    print("  • Backtest assumes next-bar-open fills, no slippage. Live includes")
    print("    stop slippage, intra-bar tick wicks, and any divergences from today.")

    await s.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
