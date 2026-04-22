"""
backtest_atr_trail.py — test ATR-based trail vs candle-trail baseline.

Runs the full 18yr v2 engine swapping candle-trail for ATR trail.
Tests multiple atr_mult values: 1.5, 2.0, 2.5, 3.0

COMBINED filter applied post-hoc for fair PF comparison.
Train/test split 2008-2017 / 2018-2026.
"""
import time
from copy import deepcopy
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

import backtest as bt
from backtest_v2_atr import simulate_session_atr
import backtest_v2 as bt2

DATA_DIR = "data/firstrate"
FIRSTRATE_FILES = {
    "DAX": {"file_5m": f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt",
            "file_1m": f"{DATA_DIR}/FDAX_full_1min_continuous_ratio_adjusted.txt",
            "src_tz": "Europe/Berlin"},
    "US30": {"file_5m": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",
             "file_1m": f"{DATA_DIR}/YM_full_1min_continuous_ratio_adjusted.txt",
             "src_tz": "America/New_York"},
    "NIKKEI": {"file_5m": f"{DATA_DIR}/NKD_full_5min_continuous_ratio_adjusted.txt",
               "file_1m": f"{DATA_DIR}/NKD_full_1min_continuous_ratio_adjusted.txt",
               "src_tz": "America/New_York"},
}

for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3

ATR_GRID = [
    # (period, mult)
    (14, 0.25), (14, 0.35), (14, 0.40), (14, 0.50), (14, 0.60),
    (7,  0.50),
    (21, 0.50),
    (7,  0.35),
    (21, 0.35),
]


def load_firstrate(filepath, src_tz, target_tz):
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


def build_cache():
    cache = {}
    for inst_name, fr in FIRSTRATE_FILES.items():
        cfg = bt.INSTRUMENTS[inst_name]
        print(f"Loading {inst_name} ...")
        df5 = load_firstrate(fr["file_5m"], fr["src_tz"], cfg["timezone"])
        df1 = load_firstrate(fr["file_1m"], fr["src_tz"], cfg["timezone"])
        ohlc5 = df5[["Open", "High", "Low", "Close"]].values
        dates5 = df5["_date"].values
        by_date_5m = {}
        for day in sorted(set(dates5)):
            m5 = dates5 == day
            by_date_5m[day] = (ohlc5[m5], df5["_hour"].values[m5], df5["_minute"].values[m5])
        ohlc1 = df1[["Open", "High", "Low", "Close"]].values
        dates1 = df1["_date"].values
        hours1 = df1["_hour"].values
        minutes1 = df1["_minute"].values
        by_date_1m = {}
        cur_date, cur_start = None, 0
        for i, d in enumerate(dates1):
            if d != cur_date:
                if cur_date is not None:
                    s = slice(cur_start, i)
                    by_date_1m[cur_date] = (ohlc1[s], hours1[s], hours1[s]*60+minutes1[s])
                cur_date, cur_start = d, i
        if cur_date is not None:
            s = slice(cur_start, len(dates1))
            by_date_1m[cur_date] = (ohlc1[s], hours1[s], hours1[s]*60+minutes1[s])
        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        cache[inst_name] = {
            "by_date_5m": by_date_5m, "by_date_1m": by_date_1m,
            "dates": sorted(set(dates5)), "sessions": sessions,
        }
    return cache


def run_variant(cache, atr_mult, simulator_fn, atr_period=14):
    all_trades = []
    for inst_name, cached in cache.items():
        cfg = deepcopy(bt.INSTRUMENTS[inst_name])
        cfg["atr_period"] = atr_period
        cfg["atr_mult"] = atr_mult
        for session in cached["sessions"]:
            open_h = cfg[f"s{session}_open_hour"]
            open_m = cfg[f"s{session}_open_minute"]
            eod_h = cfg["session_end_hour"]
            eod_m = cfg["session_end_minute"]
            signal_name = f"{inst_name}_S{session}"
            for day in cached["dates"]:
                d5 = cached["by_date_5m"].get(day)
                if d5 is None: continue
                day_ohlc5, day_hours5, day_minutes5 = d5
                d1 = cached["by_date_1m"].get(day)
                if d1:
                    day_ohlc1, day_hours1, day_min1 = d1
                else:
                    day_ohlc1 = day_hours1 = day_min1 = None
                trades = simulator_fn(
                    day_ohlc5, day_hours5, day_minutes5,
                    open_h, open_m, eod_h, eod_m, cfg,
                    day_1min=day_ohlc1,
                    one_min_hours=day_hours1,
                    one_min_minutes=day_min1,
                )
                for t in trades:
                    t["date"] = str(day); t["signal"] = signal_name
                    t["instrument"] = inst_name
                    all_trades.append(t)
    return all_trades


def apply_combined_filter(df):
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"})
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    drop = df["is_re"] & ~df["same"] & (
        df["instrument"].isin(["DAX", "US30"])
        | ((df["instrument"] == "NIKKEI") & df["first_won"]))
    return df[~drop].copy()


def stats(d):
    if len(d) == 0: return None
    w = d[d["pnl_pts"] > 0]; l = d[d["pnl_pts"] < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    return len(d), pf, d.pnl_pts.sum(), len(w)/len(d)*100


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s")

    # Baseline v2 with current candle trail
    print(f"\n--- baseline CANDLE trail ---")
    bl = run_variant(cache, 0, bt2.simulate_session_v2)
    df_bl = pd.DataFrame(bl)
    df_bl["year"] = pd.to_datetime(df_bl["date"]).dt.year
    filt_bl = apply_combined_filter(df_bl)

    # ATR grid variants
    atr_results = {}
    for period, mult in ATR_GRID:
        t1 = time.time()
        key = f"P{period}_M{mult}"
        print(f"\n--- ATR period={period}, mult={mult}× ---")
        tr = run_variant(cache, mult, simulate_session_atr, atr_period=period)
        df = pd.DataFrame(tr)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        filt = apply_combined_filter(df)
        atr_results[key] = filt
        print(f"  {key}: {len(tr):,} raw → {len(filt):,} filtered ({time.time()-t1:.0f}s)")

    print(f"\n{'='*95}\n  ATR vs CANDLE TRAIL — 18yr v2, COMBINED filter\n{'='*95}")
    for label, d in [("CANDLE (baseline)", filt_bl)] + [(f"ATR {k}", atr_results[k]) for k in [f"P{p}_M{m}" for p,m in ATR_GRID]]:
        print(f"\n  {label}")
        for pname, years in [
            ("FULL ", lambda y: True),
            ("TRAIN", lambda y: y <= 2017),
            ("TEST ", lambda y: y >= 2018),
        ]:
            sub = d[d["year"].apply(years)]
            s = stats(sub)
            if s:
                n, pf, net, wr = s
                print(f"    {pname:<6} n={n:>6}  PF={pf:>5.2f}  net={net:>+9,.0f}  WR={wr:>4.1f}%")
        parts = []
        for inst in ["DAX", "US30", "NIKKEI"]:
            s = stats(d[d["instrument"] == inst])
            if s:
                parts.append(f"{inst}={s[1]:.2f}")
        print(f"    per-inst: {'  '.join(parts)}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
