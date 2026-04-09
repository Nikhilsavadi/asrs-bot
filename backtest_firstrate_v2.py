"""
backtest_firstrate_v2.py — Run ASRS backtest with HONEST re-entry semantics.

Loads both 5-min AND 1-min firstrate data per instrument. The 5-min file
is the primary engine; the 1-min file is used to verify re-entry levels
were actually touched (closing the fictional-fill bug from v1).

Usage:
    python3 backtest_firstrate_v2.py                # full history, all instruments
    python3 backtest_firstrate_v2.py --years 3      # last 3 years
    python3 backtest_firstrate_v2.py --instrument DAX
    python3 backtest_firstrate_v2.py --compare      # also run v1 side-by-side
"""
import argparse
import time
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

import backtest as bt
import backtest_v2 as bt2

# Match LIVE bot config
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"]   = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"]   = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0

DATA_DIR = "data/firstrate"

FIRSTRATE_FILES = {
    "DAX": {
        "file_5m": f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt",
        "file_1m": f"{DATA_DIR}/FDAX_full_1min_continuous_ratio_adjusted.txt",
        "src_tz": "Europe/Berlin",
    },
    "US30": {
        "file_5m": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",
        "file_1m": f"{DATA_DIR}/YM_full_1min_continuous_ratio_adjusted.txt",
        "src_tz": "America/New_York",
    },
    "NIKKEI": {
        "file_5m": f"{DATA_DIR}/NKD_full_5min_continuous_ratio_adjusted.txt",
        "file_1m": f"{DATA_DIR}/NKD_full_1min_continuous_ratio_adjusted.txt",
        "src_tz": "America/New_York",
    },
}


def load_firstrate(filepath: str, src_tz: str, target_tz: str) -> pd.DataFrame:
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


def run(years: int | None, only: str | None, compare: bool = False):
    t0 = time.time()
    all_trades_v2 = []
    all_trades_v1 = []  # only populated if --compare

    for inst_name, cfg in bt.INSTRUMENTS.items():
        if only and inst_name != only:
            continue
        if inst_name not in FIRSTRATE_FILES:
            continue

        fr = FIRSTRATE_FILES[inst_name]
        print(f"\n{'=' * 60}\n  {inst_name}  ({fr['file_5m'].split('/')[-1]})\n{'=' * 60}")

        # Load 5-min (primary)
        df5 = load_firstrate(fr["file_5m"], fr["src_tz"], cfg["timezone"])
        # Load 1-min (drill-down)
        print(f"  Loading 1-min file... ", end="", flush=True)
        df1 = load_firstrate(fr["file_1m"], fr["src_tz"], cfg["timezone"])
        print(f"{len(df1):,} bars")

        if years:
            cutoff = df5.index.max() - pd.Timedelta(days=365 * years)
            df5 = df5[df5.index >= cutoff]
            df1 = df1[df1.index >= cutoff]

        print(f"  5-min: {df5.index.min().date()} → {df5.index.max().date()}  ({len(df5):,} bars)")
        print(f"  1-min: {df1.index.min().date()} → {df1.index.max().date()}  ({len(df1):,} bars)")

        ohlc5 = df5[["Open", "High", "Low", "Close"]].values
        hours5 = df5["_hour"].values
        minutes5 = df5["_minute"].values
        dates5 = df5["_date"].values

        ohlc1 = df1[["Open", "High", "Low", "Close"]].values
        hours1 = df1["_hour"].values
        minutes1 = df1["_minute"].values
        dates1 = df1["_date"].values

        unique_dates = sorted(set(dates5))

        # Pre-build a date → 1-min index map for fast lookup
        date_to_1min_idx: dict = {}
        cur_date = None
        cur_start = 0
        for i, d in enumerate(dates1):
            if d != cur_date:
                if cur_date is not None:
                    date_to_1min_idx[cur_date] = (cur_start, i)
                cur_date = d
                cur_start = i
        if cur_date is not None:
            date_to_1min_idx[cur_date] = (cur_start, len(dates1))

        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        for session in sessions:
            signal_name = f"{inst_name}_S{session}"
            open_h = cfg[f"s{session}_open_hour"]
            open_m = cfg[f"s{session}_open_minute"]
            eod_h = cfg["session_end_hour"]
            eod_m = cfg["session_end_minute"]

            sig_trades_v2 = []
            sig_trades_v1 = []

            for day in unique_dates:
                mask5 = dates5 == day
                day_ohlc5 = ohlc5[mask5]
                day_hours5 = hours5[mask5]
                day_minutes5 = minutes5[mask5]

                # 1-min slice for this day
                day_1m_range = date_to_1min_idx.get(day)
                if day_1m_range is not None:
                    s, e = day_1m_range
                    day_ohlc1 = ohlc1[s:e]
                    day_hours1 = hours1[s:e]
                    day_minutes1 = minutes1[s:e]
                    day_min_of_day_1m = day_hours1 * 60 + day_minutes1
                else:
                    day_ohlc1 = None
                    day_min_of_day_1m = None

                # v2 (honest)
                day_trades_v2 = bt2.simulate_session_v2(
                    day_ohlc5, day_hours5, day_minutes5,
                    open_h, open_m, eod_h, eod_m, cfg,
                    day_1min=day_ohlc1,
                    one_min_hours=day_hours1 if day_1m_range else None,
                    one_min_minutes=day_min_of_day_1m,
                )
                for t in day_trades_v2:
                    t["date"] = str(day)
                    t["signal"] = signal_name
                    t["instrument"] = inst_name
                    sig_trades_v2.append(t)

                # v1 (original, for --compare)
                if compare:
                    day_trades_v1 = bt.simulate_session(
                        day_ohlc5, day_hours5, day_minutes5,
                        open_h, open_m, eod_h, eod_m, cfg,
                    )
                    for t in day_trades_v1:
                        t["date"] = str(day)
                        t["signal"] = signal_name
                        t["instrument"] = inst_name
                        sig_trades_v1.append(t)

            if sig_trades_v2:
                _print_signal_summary(signal_name, sig_trades_v2, "v2")
                all_trades_v2.extend(sig_trades_v2)
            if compare and sig_trades_v1:
                _print_signal_summary(signal_name, sig_trades_v1, "v1")
                all_trades_v1.extend(sig_trades_v1)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}\n  COMBINED v2 (HONEST re-entry)  ({elapsed:.1f}s)\n{'=' * 60}")
    _print_combined(all_trades_v2)

    if compare and all_trades_v1:
        print(f"\n{'=' * 60}\n  COMBINED v1 (ORIGINAL fictional re-entry)\n{'=' * 60}")
        _print_combined(all_trades_v1)

        print(f"\n{'=' * 60}\n  V1 vs V2 DELTA\n{'=' * 60}")
        v1_pf = _pf(all_trades_v1)
        v2_pf = _pf(all_trades_v2)
        v1_net = sum(t["pnl_pts"] for t in all_trades_v1)
        v2_net = sum(t["pnl_pts"] for t in all_trades_v2)
        print(f"  Trades: {len(all_trades_v1):,} (v1) vs {len(all_trades_v2):,} (v2)  "
              f"= {len(all_trades_v1) - len(all_trades_v2):+,} fictional re-entries removed")
        print(f"  PF:     {v1_pf:.2f}    (v1) vs {v2_pf:.2f}    (v2)")
        print(f"  Net pts: {v1_net:+,.0f} (v1) vs {v2_net:+,.0f} (v2)  "
              f"= {v2_net - v1_net:+,.0f} delta")
        print()
        print("  → v2 numbers are the HONEST baseline. Use these for live")
        print("    expectations / MC scenarios / scaling plan anchors.")

    out = "data/backtest_firstrate_v2_results.csv"
    pd.DataFrame(all_trades_v2).to_csv(out, index=False)
    print(f"\n  v2 trades saved → {out}")
    if compare:
        out_v1 = "data/backtest_firstrate_v1_results.csv"
        pd.DataFrame(all_trades_v1).to_csv(out_v1, index=False)
        print(f"  v1 trades saved → {out_v1}")


def _pf(trades):
    gw = sum(t["pnl_pts"] for t in trades if t["pnl_pts"] > 0)
    gl = abs(sum(t["pnl_pts"] for t in trades if t["pnl_pts"] < 0))
    return gw / gl if gl > 0 else float("inf")


def _print_signal_summary(name, trades, tag):
    pnl = sum(t["pnl_pts"] for t in trades)
    w = sum(1 for t in trades if t["pnl_pts"] > 0)
    l = sum(1 for t in trades if t["pnl_pts"] < 0)
    pf = _pf(trades)
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f"  [{tag}] {name}: {len(trades):>5} trades | PF {pf:.2f} | "
          f"W/L {w}/{l} ({wr:.0f}%) | Net {pnl:+,.0f}pts")


def _print_combined(trades):
    if not trades:
        print("  No trades.")
        return
    tp = sum(t["pnl_pts"] for t in trades)
    tw = sum(1 for t in trades if t["pnl_pts"] > 0)
    tl = sum(1 for t in trades if t["pnl_pts"] < 0)
    pf = _pf(trades)
    print(f"  Trades: {len(trades):,}  |  PF {pf:.2f}  |  "
          f"W/L {tw}/{tl} ({tw / max(1, tw + tl) * 100:.0f}%)  |  "
          f"Net {tp:+,.0f}pts")

    df_t = pd.DataFrame(trades)
    df_t["year"] = pd.to_datetime(df_t["date"]).dt.year
    print(f"\n  Per year:")
    for year, grp in df_t.groupby("year"):
        ypf = _pf(grp.to_dict("records"))
        print(f"    {year}: {len(grp):>5} trades | PF {ypf:.2f} | "
              f"Net {grp['pnl_pts'].sum():+,.0f}pts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--instrument", choices=["DAX", "US30", "NIKKEI"], default=None)
    ap.add_argument("--compare", action="store_true",
                    help="Also run v1 (original) side-by-side and print delta")
    args = ap.parse_args()
    run(args.years, args.instrument, compare=args.compare)
