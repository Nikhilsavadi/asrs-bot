"""
backtest_adds_test.py — test adds (pyramiding) re-enabled with proper PnL.

Uses backtest_v2_adds (candle trail preserved, adds now compute real PnL).
Tests multiple add_max values: 0 (baseline), 1, 2.
Tests multiple add_trigger scalings: 0.5×default, 1×, 1.5×.

COMBINED filter applied, train/test split.
"""
import time, sys
from copy import deepcopy
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

import backtest as bt
import backtest_v2 as bt2
from backtest_v2_adds import simulate_session_adds

DATA_DIR = "data/firstrate"
FR = {
    "DAX": {"5m": f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt",
            "1m": f"{DATA_DIR}/FDAX_full_1min_continuous_ratio_adjusted.txt", "tz": "Europe/Berlin"},
    "US30": {"5m": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",
             "1m": f"{DATA_DIR}/YM_full_1min_continuous_ratio_adjusted.txt", "tz": "America/New_York"},
    "NIKKEI": {"5m": f"{DATA_DIR}/NKD_full_5min_continuous_ratio_adjusted.txt",
               "1m": f"{DATA_DIR}/NKD_full_1min_continuous_ratio_adjusted.txt", "tz": "America/New_York"},
}

for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3

VARIANTS = [
    # (label, add_max, trigger_scale)
    ("add0_baseline",    0, 1.0),
    ("add1_default",     1, 1.0),
    ("add2_default",     2, 1.0),
    ("add1_tight",       1, 0.5),
    ("add1_loose",       1, 1.5),
    ("add2_tight",       2, 0.5),
    ("add2_loose",       2, 1.5),
]


def load(f, tz_src, tz_tgt):
    df = pd.read_csv(f, header=None, names=["dt","Open","High","Low","Close","V"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index = df.index.tz_localize(ZoneInfo(tz_src)).tz_convert(ZoneInfo(tz_tgt))
    df = df[df.index.dayofweek < 5]
    df["_h"] = df.index.hour; df["_m"] = df.index.minute; df["_d"] = df.index.date
    return df


def build_cache():
    cache = {}
    for inst, meta in FR.items():
        cfg = bt.INSTRUMENTS[inst]
        print(f"Loading {inst} ...")
        d5 = load(meta["5m"], meta["tz"], cfg["timezone"])
        d1 = load(meta["1m"], meta["tz"], cfg["timezone"])
        ohlc5 = d5[["Open","High","Low","Close"]].values
        dates5 = d5["_d"].values
        by5 = {}
        for dd in sorted(set(dates5)):
            m = dates5 == dd
            by5[dd] = (ohlc5[m], d5["_h"].values[m], d5["_m"].values[m])
        ohlc1 = d1[["Open","High","Low","Close"]].values
        dates1 = d1["_d"].values; h1 = d1["_h"].values; m1 = d1["_m"].values
        by1 = {}
        cd, cs = None, 0
        for i, dd in enumerate(dates1):
            if dd != cd:
                if cd is not None:
                    s = slice(cs, i)
                    by1[cd] = (ohlc1[s], h1[s], h1[s]*60+m1[s])
                cd, cs = dd, i
        if cd is not None:
            s = slice(cs, len(dates1))
            by1[cd] = (ohlc1[s], h1[s], h1[s]*60+m1[s])
        sessions = [s for s in (1,2,3) if f"s{s}_open_hour" in cfg]
        cache[inst] = {"by5": by5, "by1": by1, "dates": sorted(set(dates5)), "sessions": sessions}
    return cache


def run(cache, add_max, trigger_scale, simulator):
    trades = []
    for inst, cached in cache.items():
        cfg = deepcopy(bt.INSTRUMENTS[inst])
        cfg["add_max"] = add_max
        cfg["add_trigger"] = cfg["add_trigger"] * trigger_scale
        for s in cached["sessions"]:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            name = f"{inst}_S{s}"
            for d in cached["dates"]:
                d5 = cached["by5"].get(d)
                if d5 is None: continue
                o5, h5, m5 = d5
                d1 = cached["by1"].get(d)
                if d1: o1, h1, min1 = d1
                else: o1 = h1 = min1 = None
                ts = simulator(o5, h5, m5, oh, om, eh, em, cfg,
                              day_1min=o1, one_min_hours=h1, one_min_minutes=min1)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def filter_combined(df):
    df = df.sort_values(["date","signal","entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date","signal"]).cumcount()+1
    first = df[df["tn"]==1][["date","signal","direction","pnl_pts"]].rename(
        columns={"direction":"fd","pnl_pts":"fp"})
    df = df.merge(first, on=["date","signal"], how="left")
    df["first_won"] = df["fp"]>0
    df["is_re"] = df["tn"]>1
    df["same"] = df["direction"]==df["fd"]
    drop = df["is_re"] & ~df["same"] & (
        df["instrument"].isin(["DAX","US30"])
        | ((df["instrument"]=="NIKKEI") & df["first_won"]))
    return df[~drop].copy()


def stats(d):
    if len(d)==0: return None
    w=d[d["pnl_pts"]>0]; l=d[d["pnl_pts"]<0]
    pf = w.pnl_pts.sum()/max(abs(l.pnl_pts.sum()),0.001)
    return len(d), pf, d.pnl_pts.sum(), len(w)/len(d)*100


def main():
    t0 = time.time()
    cache = build_cache()
    print(f"Cache in {time.time()-t0:.0f}s\n")

    results = {}
    for lbl, am, ts in VARIANTS:
        t1 = time.time()
        print(f"--- {lbl} (add_max={am}, trigger_scale={ts}) ---")
        trades = run(cache, am, ts, simulate_session_adds)
        df = pd.DataFrame(trades)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        filt = filter_combined(df)
        results[lbl] = filt
        print(f"  {lbl}: {len(trades):,} → {len(filt):,} filtered  ({time.time()-t1:.0f}s)")

    print(f"\n{'='*95}\n  ADDS VARIANTS — 18yr v2, COMBINED filter\n{'='*95}")
    for lbl, _, _ in VARIANTS:
        d = results[lbl]
        print(f"\n  {lbl}")
        for pn, yf in [("FULL", lambda y: True),
                       ("TRAIN", lambda y: y<=2017),
                       ("TEST", lambda y: y>=2018)]:
            s = stats(d[d["year"].apply(yf)])
            if s:
                n, pf, net, wr = s
                print(f"    {pn:<6} n={n:>5}  PF={pf:>5.2f}  net={net:>+9,.0f}  WR={wr:>4.1f}%")
        parts = [f"{inst}={stats(d[d['instrument']==inst])[1]:.2f}" for inst in ["DAX","US30","NIKKEI"]]
        print(f"    per-inst: {'  '.join(parts)}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
