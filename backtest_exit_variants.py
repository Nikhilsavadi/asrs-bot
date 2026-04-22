"""
backtest_exit_variants.py — hunt for exit-logic improvements.

Runs the full v2 engine with different cfg variants:
  A) Baseline        — current live params
  B) No BE           — breakeven_pts = 99999 (never fires)
  C) Earlier BE      — breakeven_pts × 0.5
  D) Wider tight     — tight_threshold × 2 (winners run longer)
  E) No tight switch — tight_threshold = 99999 (always candle trail)
  F) No BE + Wider tight
  G) No BE + No tight switch
  H) BE at 1R (adaptive) — breakeven_pts = typical bar range

Also applies COMBINED reverse-reentry filter post-hoc for fair comparison
against current live config.

Train/test split 2008-2017 / 2018-2026.
Outputs PF, net, WR per variant per instrument, and combined.
"""
import time
from copy import deepcopy
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

import backtest as bt
import backtest_v2 as bt2

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

# Match the reference v2 backtest (backtest_firstrate_v2.py uses max_entries=3)
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3


def load_firstrate(filepath: str, src_tz: str, target_tz: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None,
                     names=["dt", "Open", "High", "Low", "Close", "Volume"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index = df.index.tz_localize(ZoneInfo(src_tz)).tz_convert(ZoneInfo(target_tz))
    df = df[df.index.dayofweek < 5]
    df["_hour"] = df.index.hour
    df["_minute"] = df.index.minute
    df["_date"] = df.index.date
    return df


VARIANTS = {
    "A_baseline":      lambda c: c,
    "B_no_BE":         lambda c: {**c, "breakeven_pts": 99999.0},
    "C_earlier_BE":    lambda c: {**c, "breakeven_pts": c["breakeven_pts"] * 0.5},
    "D_wider_tight":   lambda c: {**c, "tight_threshold": c["tight_threshold"] * 2.0},
    "E_no_tight":      lambda c: {**c, "tight_threshold": 99999.0},
    "F_no_BE_wider":   lambda c: {**c, "breakeven_pts": 99999.0,
                                  "tight_threshold": c["tight_threshold"] * 2.0},
    "G_no_BE_no_tight": lambda c: {**c, "breakeven_pts": 99999.0,
                                   "tight_threshold": 99999.0},
}


def run_variant(variant_fn, data_cache: dict) -> list:
    """Run v2 backtest for all instruments/sessions with a cfg variant."""
    all_trades = []
    for inst_name, cached in data_cache.items():
        cfg_base = bt.INSTRUMENTS[inst_name]
        cfg = variant_fn(deepcopy(cfg_base))
        for session in cached["sessions"]:
            open_h = cfg[f"s{session}_open_hour"]
            open_m = cfg[f"s{session}_open_minute"]
            eod_h = cfg["session_end_hour"]
            eod_m = cfg["session_end_minute"]
            signal_name = f"{inst_name}_S{session}"

            for day in cached["unique_dates"]:
                day_5m = cached["by_date_5m"].get(day)
                if day_5m is None:
                    continue
                day_ohlc5, day_hours5, day_minutes5 = day_5m

                day_1m = cached["by_date_1m"].get(day)
                if day_1m is not None:
                    day_ohlc1, day_hours1, day_min_of_day_1m = day_1m
                else:
                    day_ohlc1 = None
                    day_hours1 = None
                    day_min_of_day_1m = None

                trades = bt2.simulate_session_v2(
                    day_ohlc5, day_hours5, day_minutes5,
                    open_h, open_m, eod_h, eod_m, cfg,
                    day_1min=day_ohlc1,
                    one_min_hours=day_hours1,
                    one_min_minutes=day_min_of_day_1m,
                )
                for t in trades:
                    t["date"] = str(day)
                    t["signal"] = signal_name
                    t["instrument"] = inst_name
                    all_trades.append(t)
    return all_trades


def build_cache() -> dict:
    cache = {}
    for inst_name, fr in FIRSTRATE_FILES.items():
        cfg = bt.INSTRUMENTS[inst_name]
        print(f"Loading {inst_name} ...")
        df5 = load_firstrate(fr["file_5m"], fr["src_tz"], cfg["timezone"])
        df1 = load_firstrate(fr["file_1m"], fr["src_tz"], cfg["timezone"])
        print(f"  {inst_name}: 5m={len(df5):,}  1m={len(df1):,}")

        ohlc5 = df5[["Open", "High", "Low", "Close"]].values
        hours5 = df5["_hour"].values
        minutes5 = df5["_minute"].values
        dates5 = df5["_date"].values

        ohlc1 = df1[["Open", "High", "Low", "Close"]].values
        hours1 = df1["_hour"].values
        minutes1 = df1["_minute"].values
        dates1 = df1["_date"].values

        by_date_5m, by_date_1m = {}, {}
        for day in sorted(set(dates5)):
            m5 = dates5 == day
            by_date_5m[day] = (ohlc5[m5], hours5[m5], minutes5[m5])
        cur_date, cur_start = None, 0
        for i, d in enumerate(dates1):
            if d != cur_date:
                if cur_date is not None:
                    m1 = slice(cur_start, i)
                    by_date_1m[cur_date] = (
                        ohlc1[m1],
                        hours1[m1],
                        (hours1[m1] * 60 + minutes1[m1]),
                    )
                cur_date, cur_start = d, i
        if cur_date is not None:
            m1 = slice(cur_start, len(dates1))
            by_date_1m[cur_date] = (ohlc1[m1], hours1[m1],
                                     hours1[m1] * 60 + minutes1[m1])

        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        cache[inst_name] = {
            "by_date_5m": by_date_5m,
            "by_date_1m": by_date_1m,
            "unique_dates": sorted(set(dates5)),
            "sessions": sessions,
        }
    return cache


def apply_combined_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"}
    )
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    drop = df["is_re"] & ~df["same"] & (
        df["instrument"].isin(["DAX", "US30"])
        | ((df["instrument"] == "NIKKEI") & df["first_won"])
    )
    return df[~drop].copy()


def stats(d: pd.DataFrame):
    if len(d) == 0: return None
    w = d[d["pnl_pts"] > 0]; l = d[d["pnl_pts"] < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    net = d.pnl_pts.sum()
    wr = len(w) / len(d) * 100
    return len(d), pf, net, wr


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache built in {time.time()-t0:.0f}s")

    results = {}
    for vname, vfn in VARIANTS.items():
        t1 = time.time()
        print(f"\n--- running {vname}")
        trades = run_variant(vfn, cache)
        df = pd.DataFrame(trades)
        df["date_d"] = pd.to_datetime(df["date"])
        df["year"] = df["date_d"].dt.year
        filt = apply_combined_filter(df)
        results[vname] = filt
        print(f"  {vname}: {len(trades):,} raw → {len(filt):,} filtered "
              f"({time.time()-t1:.0f}s)")

    print("\n" + "=" * 95)
    print(f"  EXIT-LOGIC VARIANTS — COMBINED filter applied, train/test")
    print("=" * 95)
    print(f"\n  {'variant':<22} {'period':<10} {'n':>6} {'PF':>6} {'net':>10} {'WR':>5}")
    for vname in VARIANTS:
        d = results[vname]
        for pname, mask in [
            ("FULL",  d["year"] >= 2000),
            ("TRAIN", d["year"] <= 2017),
            ("TEST",  d["year"] >= 2018),
        ]:
            sub = d[mask]
            s = stats(sub)
            if s:
                n, pf, net, wr = s
                print(f"  {vname:<22} {pname:<10} {n:>6}  PF={pf:>5.2f}  "
                      f"net={net:>+9,.0f}  WR={wr:>4.1f}%")

    print("\n  Per-instrument FULL period:")
    for vname in VARIANTS:
        d = results[vname]
        parts = []
        for inst in ["DAX", "US30", "NIKKEI"]:
            sub = d[d["instrument"] == inst]
            s = stats(sub)
            if s:
                _, pf, net, _ = s
                parts.append(f"{inst}={pf:.2f}")
        print(f"  {vname:<22} {'  '.join(parts)}")

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
