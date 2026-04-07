"""
replay_live.py — Compare live trades against firstrate backtest on same dates.

For each (instrument, date) in the live trade journal, run the firstrate
backtest for that single day and compare:
- # of trades taken (live vs backtest)
- direction(s) taken
- net pnl_pts (live vs backtest)

Answers: "On the days the bot lost money live, what would the validated
backtest have done on the same dates?"
"""
import sqlite3
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

import backtest as bt
from backtest_firstrate import FIRSTRATE_FILES, load_firstrate

# Match live config: max_entries=3, S3 sessions for US30 + NIKKEI
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"]   = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"]   = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0


def fetch_live_trades(db_path):
    c = sqlite3.connect(db_path)
    rows = c.execute(
        "SELECT instrument, date, direction, pnl_pts, exit_reason "
        "FROM trades WHERE mode='live' ORDER BY date, instrument, entry_time"
    ).fetchall()
    c.close()
    return pd.DataFrame(rows, columns=["instrument", "date", "direction", "pnl_pts", "exit_reason"])


def run_backtest_for_dates(inst, dates):
    """Run firstrate backtest for one instrument, return dict {date_str: [trades]}."""
    cfg = bt.INSTRUMENTS[inst]
    fr = FIRSTRATE_FILES[inst]
    df = load_firstrate(fr["file"], fr["src_tz"], cfg["timezone"])

    target_dates = set(pd.to_datetime(d).date() for d in dates)
    df = df[df["_date"].isin(target_dates)]

    if df.empty:
        return {}

    ohlc = df[["Open", "High", "Low", "Close"]].values
    hours = df["_hour"].values
    minutes = df["_minute"].values
    dts = df["_date"].values

    out = {}
    sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
    for session in sessions:
        oh = cfg[f"s{session}_open_hour"]; om = cfg[f"s{session}_open_minute"]
        eh = cfg["session_end_hour"];      em = cfg["session_end_minute"]
        for day in sorted(set(dts)):
            mask = dts == day
            trs = bt.simulate_session(
                ohlc[mask], hours[mask], minutes[mask], oh, om, eh, em, cfg
            )
            for t in trs:
                t["session"] = session
            out.setdefault(str(day), []).extend(trs)
    return out


def main():
    live = fetch_live_trades("/root/asrs-bot/data/trade_journal.db")
    if live.empty:
        print("No live trades found.")
        return

    print(f"Loaded {len(live)} live trades  ({live['date'].min()} → {live['date'].max()})")
    print(f"Live total P&L: {live['pnl_pts'].sum():+.1f} pts\n")

    # Run backtest per instrument across all live dates
    bt_by_inst = {}
    for inst in live["instrument"].unique():
        dates = sorted(live[live["instrument"] == inst]["date"].unique())
        bt_by_inst[inst] = run_backtest_for_dates(inst, dates)

    # Compare per (instrument, date)
    print(f"{'date':<12}{'inst':<8}{'live#':>6}{'bt#':>5}{'live_pnl':>11}{'bt_pnl':>10}{'gap':>10}")
    print("-" * 62)

    rows = []
    for (inst, date), grp in live.groupby(["instrument", "date"]):
        live_n = len(grp)
        live_pnl = grp["pnl_pts"].sum()
        bt_trades = bt_by_inst.get(inst, {}).get(date, [])
        bt_n = len(bt_trades)
        bt_pnl = sum(t["pnl_pts"] for t in bt_trades)
        gap = live_pnl - bt_pnl
        rows.append({"date": date, "inst": inst, "live_n": live_n, "bt_n": bt_n,
                     "live_pnl": live_pnl, "bt_pnl": bt_pnl, "gap": gap})
        print(f"{date:<12}{inst:<8}{live_n:>6}{bt_n:>5}{live_pnl:>+11.1f}{bt_pnl:>+10.1f}{gap:>+10.1f}")

    df = pd.DataFrame(rows)
    print("-" * 62)
    print(f"{'TOTAL':<20}{df['live_n'].sum():>6}{df['bt_n'].sum():>5}"
          f"{df['live_pnl'].sum():>+11.1f}{df['bt_pnl'].sum():>+10.1f}"
          f"{df['gap'].sum():>+10.1f}")

    # Per instrument summary
    print("\nPer instrument:")
    for inst, grp in df.groupby("inst"):
        print(f"  {inst:6} live {grp['live_pnl'].sum():+8.1f}  "
              f"bt {grp['bt_pnl'].sum():+8.1f}  gap {grp['gap'].sum():+8.1f}")

    # Days where backtest disagrees significantly
    print("\nBiggest gaps (live underperformance vs backtest):")
    for _, r in df.sort_values("gap").head(8).iterrows():
        print(f"  {r['date']} {r['inst']:6}  live {r['live_pnl']:+7.1f}  "
              f"bt {r['bt_pnl']:+7.1f}  gap {r['gap']:+7.1f}  "
              f"(live {int(r['live_n'])} trades, bt {int(r['bt_n'])} trades)")


if __name__ == "__main__":
    main()
