"""
backtest_firstrate.py — Run ASRS backtest on firstratedata.com futures (FDAX/YM/NKD).

Usage:
    python3 backtest_firstrate.py            # all instruments, full history
    python3 backtest_firstrate.py --years 3  # last 3 years only
    python3 backtest_firstrate.py --instrument DAX
"""
import argparse
import pandas as pd
from zoneinfo import ZoneInfo

import backtest as bt

# Match LIVE bot config: 3 entries per session, 3 sessions for US30 + NIKKEI
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"]   = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"]   = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0

DATA_DIR = "data/firstrate"

# Map ASRS instrument → firstrate file + native timezone of the file
FIRSTRATE_FILES = {
    "DAX":    {"file": f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt", "src_tz": "Europe/Berlin"},
    "US30":   {"file": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",   "src_tz": "America/New_York"},
    "NIKKEI": {"file": f"{DATA_DIR}/NKD_full_5min_continuous_ratio_adjusted.txt",  "src_tz": "America/New_York"},
}


def load_firstrate(filepath: str, src_tz: str, target_tz: str) -> pd.DataFrame:
    """Load firstrate CSV (no header) and convert to ASRS backtest format."""
    df = pd.read_csv(
        filepath,
        header=None,
        names=["dt", "Open", "High", "Low", "Close", "Volume"],
    )
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index = df.index.tz_localize(ZoneInfo(src_tz)).tz_convert(ZoneInfo(target_tz))
    df = df[df.index.dayofweek < 5]
    df["_hour"] = df.index.hour
    df["_minute"] = df.index.minute
    df["_date"] = df.index.date
    return df


def run(years: int | None, only: str | None):
    import time
    import numpy as np

    t0 = time.time()
    all_trades = []

    for inst_name, cfg in bt.INSTRUMENTS.items():
        if only and inst_name != only:
            continue
        if inst_name not in FIRSTRATE_FILES:
            continue

        fr = FIRSTRATE_FILES[inst_name]
        print(f"\n{'=' * 60}\n  {inst_name}  ({fr['file'].split('/')[-1]})\n{'=' * 60}")

        df = load_firstrate(fr["file"], fr["src_tz"], cfg["timezone"])
        if years:
            cutoff = df.index.max() - pd.Timedelta(days=365 * years)
            df = df[df.index >= cutoff]
        print(f"  Data: {df.index.min().date()} → {df.index.max().date()}  ({len(df):,} bars)")

        ohlc = df[["Open", "High", "Low", "Close"]].values
        hours = df["_hour"].values
        minutes = df["_minute"].values
        dates = df["_date"].values
        unique_dates = sorted(set(dates))

        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        for session in sessions:
            signal_name = f"{inst_name}_S{session}"
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
                    t["instrument"] = inst_name
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

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}\n  COMBINED  ({elapsed:.1f}s)\n{'=' * 60}")
    if not all_trades:
        print("  No trades.")
        return

    tp = sum(t["pnl_pts"] for t in all_trades)
    tw = sum(1 for t in all_trades if t["pnl_pts"] > 0)
    tl = sum(1 for t in all_trades if t["pnl_pts"] < 0)
    tf = sum(1 for t in all_trades if t["pnl_pts"] == 0)
    gw = sum(t["pnl_pts"] for t in all_trades if t["pnl_pts"] > 0)
    gl = abs(sum(t["pnl_pts"] for t in all_trades if t["pnl_pts"] < 0))
    pf = gw / gl if gl > 0 else float("inf")
    print(f"  Trades: {len(all_trades):,}  |  PF {pf:.2f}  |  W/L/F {tw}/{tl}/{tf}  |  Net {tp:+,.0f}pts")
    if tw and tl:
        print(f"  Avg win {gw/tw:.1f}  |  Avg loss {gl/tl:.1f}")

    df_t = pd.DataFrame(all_trades)
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

    out = "data/backtest_firstrate_results.csv"
    df_t.to_csv(out, index=False)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=None, help="Restrict to last N years")
    ap.add_argument("--instrument", choices=["DAX", "US30", "NIKKEI"], default=None)
    args = ap.parse_args()
    run(args.years, args.instrument)
