"""
backtest_ibkr_fdxm.py — Run ASRS backtest on REAL IBKR FDXM historical data.

This is the validation that determines whether DAX comes back online
in the live bot. Uses the same backtest engine that produced the
firstrate-validated PF 4.22, but on the actual instrument we'd trade.

Compares against firstrate FDAX backtest for the same date range.
"""
from __future__ import annotations

import time
import pandas as pd
from zoneinfo import ZoneInfo

import backtest as bt

# Match LIVE bot config (3 sessions for US30/NIKKEI, 2 for DAX)
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3


def load_ibkr_fdxm(filepath="data/ibkr/FDAX_5min.csv", min_session_volume: int = 500):
    """
    Load IBKR-pulled FDAX bars and convert to backtest format.

    Filters out days where the contract was not the liquid front-month.
    Heuristic: keep only days where total volume during 08:00-10:00 UTC
    (DAX cash session opening hour) exceeds `min_session_volume`. This
    removes back-month thin-trading days that pollute the dataset.
    """
    df = pd.read_csv(filepath)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("dt")
    df.columns = [c.capitalize() for c in df.columns]
    # IBKR returns UTC; convert to CET (DAX timezone)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ZoneInfo("Europe/Berlin"))
    df = df[df.index.dayofweek < 5]
    df["_hour"] = df.index.hour
    df["_minute"] = df.index.minute
    df["_date"] = df.index.date

    # Liquidity filter: only keep days with real session volume
    if "Volume" in df.columns:
        morning = df[(df["_hour"] >= 9) & (df["_hour"] < 11)]
        daily_vol = morning.groupby("_date")["Volume"].sum()
        liquid_days = set(daily_vol[daily_vol >= min_session_volume].index)
        before = len(set(df["_date"]))
        df = df[df["_date"].isin(liquid_days)]
        after = len(liquid_days)
        print(f"  Liquidity filter: {after}/{before} trading days kept "
              f"(min session volume {min_session_volume})")
        df = df.drop(columns=["Volume"])
    return df


def run_backtest_on_ibkr(df, cfg):
    """Run the standard simulate_session loop for both DAX sessions."""
    import numpy as np

    ohlc = df[["Open", "High", "Low", "Close"]].values
    hours = df["_hour"].values
    minutes = df["_minute"].values
    dates = df["_date"].values
    unique_dates = sorted(set(dates))

    all_trades = []
    for session in (1, 2):
        signal_name = f"DAX_S{session}"
        open_h = cfg[f"s{session}_open_hour"]
        open_m = cfg[f"s{session}_open_minute"]
        eod_h = cfg["session_end_hour"]
        eod_m = cfg["session_end_minute"]

        sig_trades = []
        for day in unique_dates:
            mask = dates == day
            day_trades = bt.simulate_session(
                ohlc[mask], hours[mask], minutes[mask],
                open_h, open_m, eod_h, eod_m, cfg
            )
            for t in day_trades:
                t["date"] = str(day)
                t["signal"] = signal_name
                t["instrument"] = "DAX"
                sig_trades.append(t)

        if sig_trades:
            pnl = sum(t["pnl_pts"] for t in sig_trades)
            w = sum(1 for t in sig_trades if t["pnl_pts"] > 0)
            l = sum(1 for t in sig_trades if t["pnl_pts"] < 0)
            gw = sum(t["pnl_pts"] for t in sig_trades if t["pnl_pts"] > 0)
            gl = abs(sum(t["pnl_pts"] for t in sig_trades if t["pnl_pts"] < 0))
            pf = gw / gl if gl > 0 else float("inf")
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"  {signal_name}: {len(sig_trades):>5} trades | PF {pf:.2f} | "
                  f"W/L {w}/{l} ({wr:.0f}%) | Net {pnl:+,.0f}pts")
            all_trades.extend(sig_trades)

    return all_trades


def main():
    t0 = time.time()
    print("=" * 64)
    print("  DAX BACKTEST ON IBKR FDXM (REAL INSTRUMENT)")
    print("=" * 64)

    df = load_ibkr_fdxm()
    print(f"\n  Data: {df.index.min().date()} → {df.index.max().date()}  ({len(df):,} bars)")

    cfg = bt.INSTRUMENTS["DAX"]
    trades = run_backtest_on_ibkr(df, cfg)

    if not trades:
        print("\n  No trades generated.")
        return

    # Aggregate stats
    tp = sum(t["pnl_pts"] for t in trades)
    tw = sum(1 for t in trades if t["pnl_pts"] > 0)
    tl = sum(1 for t in trades if t["pnl_pts"] < 0)
    tf = sum(1 for t in trades if t["pnl_pts"] == 0)
    gw = sum(t["pnl_pts"] for t in trades if t["pnl_pts"] > 0)
    gl = abs(sum(t["pnl_pts"] for t in trades if t["pnl_pts"] < 0))
    pf = gw / gl if gl > 0 else float("inf")

    print(f"\n  COMBINED ({time.time()-t0:.0f}s)")
    print(f"  Total trades: {len(trades):,}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Win/Loss/Flat: {tw}/{tl}/{tf}  ({tw/(tw+tl)*100:.0f}%)")
    print(f"  Net P&L: {tp:+,.0f} pts")
    print(f"  Avg win: {gw/tw:.1f}  |  Avg loss: {gl/tl:.1f}")

    df_t = pd.DataFrame(trades)
    df_t["year"] = pd.to_datetime(df_t["date"]).dt.year
    print(f"\n  Per year:")
    for year, grp in df_t.groupby("year"):
        yw = grp[grp["pnl_pts"] > 0]["pnl_pts"].sum()
        yl = abs(grp[grp["pnl_pts"] < 0]["pnl_pts"].sum())
        ypf = yw / yl if yl > 0 else float("inf")
        print(f"    {year}: {len(grp):>5} trades | PF {ypf:.2f} | Net {grp['pnl_pts'].sum():+,.0f}pts")

    eq = df_t["pnl_pts"].cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"\n  Max drawdown: {dd:,.0f} pts")

    df_t.to_csv("data/backtest_ibkr_fdxm_results.csv", index=False)

    # Compare against firstrate FDAX for same date range
    print("\n" + "=" * 64)
    print("  COMPARISON: firstrate FDAX vs IBKR FDXM (same window)")
    print("=" * 64)
    try:
        fr = pd.read_csv("data/backtest_firstrate_results.csv")
        fr["dt"] = pd.to_datetime(fr["date"])
        fr_dax = fr[fr["instrument"] == "DAX"]
        ib_start = df_t["date"].min()
        ib_end = df_t["date"].max()
        fr_window = fr_dax[(fr_dax["date"] >= ib_start) & (fr_dax["date"] <= ib_end)]
        if len(fr_window):
            fr_w = fr_window[fr_window["pnl_pts"] > 0]["pnl_pts"].sum()
            fr_l = abs(fr_window[fr_window["pnl_pts"] < 0]["pnl_pts"].sum())
            fr_pf = fr_w / fr_l if fr_l else float("inf")
            print(f"  firstrate FDAX: {len(fr_window):,} trades | PF {fr_pf:.2f} | "
                  f"Net {fr_window['pnl_pts'].sum():+,.0f}pts")
            print(f"  IBKR FDXM:      {len(trades):,} trades | PF {pf:.2f} | "
                  f"Net {tp:+,.0f}pts")
            print(f"\n  PF delta: {pf - fr_pf:+.2f}  ({(pf/fr_pf - 1) * 100:+.1f}%)")
    except Exception as e:
        print(f"  Comparison failed: {e}")


if __name__ == "__main__":
    main()
