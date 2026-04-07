"""
backtest_cross.py — Cross-market validation of ASRS strategy.

For each new instrument (ES, NQ, FESX), derive params by scaling from a
reference instrument (DAX or US30) based on the ratio of median 5-min bar
range. Then run the same backtest engine.

Goal: confirm the strategy edge generalizes beyond the 3 live markets.
"""
import argparse
import time
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import backtest as bt

# Match live config
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3

DATA_DIR = "data/firstrate"

# new instrument → (file, src_tz, target_tz, reference_instrument, sessions)
NEW = {
    "ES":   {"file": f"{DATA_DIR}/ES_full_5min_continuous_ratio_adjusted.txt",
             "src_tz": "America/New_York", "target_tz": "America/New_York",
             "ref": "US30",
             "s1_open": (9, 30), "s2_open": (11, 0), "eod": (16, 0)},
    "NQ":   {"file": f"{DATA_DIR}/NQ_full_5min_continuous_ratio_adjusted.txt",
             "src_tz": "America/New_York", "target_tz": "America/New_York",
             "ref": "US30",
             "s1_open": (9, 30), "s2_open": (11, 0), "eod": (16, 0)},
    "FESX": {"file": f"{DATA_DIR}/FESX_full_5min_continuous_ratio_adjusted.txt",
             "src_tz": "Europe/Berlin",   "target_tz": "Europe/Berlin",
             "ref": "DAX",
             "s1_open": (9, 0),  "s2_open": (14, 0), "eod": (17, 30)},
}

# Reference median bar range (5-min) for scaling. Computed from firstrate files.
# Filled in at runtime from reference instrument's data.
REF_FILES = {
    "DAX":  f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt",
    "US30": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",
}
REF_TZ = {"DAX": "Europe/Berlin", "US30": "America/New_York"}


def load_csv(filepath, src_tz, target_tz):
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


def median_bar_range(df, hour_lo, hour_hi):
    """Median 5-min bar range during the active session window."""
    sub = df[(df["_hour"] >= hour_lo) & (df["_hour"] < hour_hi)]
    return float((sub["High"] - sub["Low"]).median())


def scale_params(ref_cfg, ref_med, new_med):
    """Scale point-based params by the bar-range ratio."""
    r = new_med / ref_med
    out = dict(ref_cfg)
    for k in ("buffer", "narrow_range", "wide_range", "max_bar_range",
              "breakeven_pts", "tight_threshold", "add_trigger", "max_risk_gbp"):
        out[k] = round(ref_cfg[k] * r, 1)
    return out, r


def run_one(name, info, ref_med):
    print(f"\n{'=' * 60}\n  {name}  (ref={info['ref']})\n{'=' * 60}")
    df = load_csv(info["file"], info["src_tz"], info["target_tz"])
    print(f"  Data: {df.index.min().date()} → {df.index.max().date()}  ({len(df):,} bars)")

    # Use 1 hour after open to compute "active session" median for scaling
    s1h, _ = info["s1_open"]
    new_med = median_bar_range(df, s1h, s1h + 1)
    ref_cfg = bt.INSTRUMENTS[info["ref"]]
    cfg, ratio = scale_params(ref_cfg, ref_med, new_med)
    cfg["s1_open_hour"], cfg["s1_open_minute"] = info["s1_open"]
    cfg["s2_open_hour"], cfg["s2_open_minute"] = info["s2_open"]
    cfg["session_end_hour"], cfg["session_end_minute"] = info["eod"]
    cfg["max_entries"] = 3
    cfg["add_max"] = 2
    print(f"  Bar-range ratio vs {info['ref']}: {ratio:.2f}x  (new med {new_med:.1f} / ref med {ref_med:.1f})")
    print(f"  Scaled params: buffer={cfg['buffer']} narrow={cfg['narrow_range']} "
          f"wide={cfg['wide_range']} risk={cfg['max_risk_gbp']} BE={cfg['breakeven_pts']} "
          f"add={cfg['add_trigger']} trail={cfg['tight_threshold']}")

    ohlc = df[["Open", "High", "Low", "Close"]].values
    hours = df["_hour"].values
    minutes = df["_minute"].values
    dates = df["_date"].values
    unique_dates = sorted(set(dates))

    all_sig_trades = []
    for session in (1, 2):
        signal_name = f"{name}_S{session}"
        oh, om = (cfg["s1_open_hour"], cfg["s1_open_minute"]) if session == 1 \
                 else (cfg["s2_open_hour"], cfg["s2_open_minute"])
        eh, em = cfg["session_end_hour"], cfg["session_end_minute"]

        sig_trades = []
        for day in unique_dates:
            mask = dates == day
            day_trades = bt.simulate_session(
                ohlc[mask], hours[mask], minutes[mask], oh, om, eh, em, cfg
            )
            for t in day_trades:
                t["date"] = str(day); t["signal"] = signal_name; t["instrument"] = name
                sig_trades.append(t)

        if sig_trades:
            pnl = sum(t["pnl_pts"] for t in sig_trades)
            w = sum(1 for t in sig_trades if t["pnl_pts"] > 0)
            l = sum(1 for t in sig_trades if t["pnl_pts"] < 0)
            gw = sum(t["pnl_pts"] for t in sig_trades if t["pnl_pts"] > 0)
            gl = abs(sum(t["pnl_pts"] for t in sig_trades if t["pnl_pts"] < 0))
            pf = gw / gl if gl > 0 else float("inf")
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            avg_w = gw / w if w else 0
            avg_l = gl / l if l else 0
            print(f"  {signal_name}: {len(sig_trades):>5} trades | PF {pf:.2f} | "
                  f"W/L {w}/{l} ({wr:.0f}%) | Net {pnl:+,.0f}pts | "
                  f"avg+{avg_w:.1f}/-{avg_l:.1f}")
            all_sig_trades.extend(sig_trades)
    return all_sig_trades


def compute_ref_median(ref_name, hour_lo):
    df = load_csv(REF_FILES[ref_name], REF_TZ[ref_name], REF_TZ[ref_name])
    return median_bar_range(df, hour_lo, hour_lo + 1)


def main():
    t0 = time.time()
    print("Computing reference bar-range medians...")
    # DAX session start 9:00 CET; US30 session start 9:30 ET
    ref_med = {
        "DAX":  compute_ref_median("DAX", 9),
        "US30": compute_ref_median("US30", 9),  # 9:30 → use hour 9
    }
    print(f"  DAX median 5m range (9-10 CET):  {ref_med['DAX']:.1f}")
    print(f"  US30 median 5m range (9-10 ET):  {ref_med['US30']:.1f}")

    all_trades = []
    for name, info in NEW.items():
        rmed = ref_med[info["ref"]]
        all_trades.extend(run_one(name, info, rmed))

    print(f"\n{'=' * 60}\n  COMBINED CROSS-MARKET  ({time.time()-t0:.0f}s)\n{'=' * 60}")
    if not all_trades:
        return
    df_t = pd.DataFrame(all_trades)
    for inst, grp in df_t.groupby("instrument"):
        gw = grp[grp["pnl_pts"] > 0]["pnl_pts"].sum()
        gl = abs(grp[grp["pnl_pts"] < 0]["pnl_pts"].sum())
        pf = gw / gl if gl > 0 else float("inf")
        print(f"  {inst:6} {len(grp):>5} trades | PF {pf:.2f} | Net {grp['pnl_pts'].sum():+,.0f}pts")

    df_t["year"] = pd.to_datetime(df_t["date"]).dt.year
    print("\n  Per year (combined):")
    for year, grp in df_t.groupby("year"):
        gw = grp[grp["pnl_pts"] > 0]["pnl_pts"].sum()
        gl = abs(grp[grp["pnl_pts"] < 0]["pnl_pts"].sum())
        pf = gw / gl if gl > 0 else float("inf")
        print(f"    {year}: {len(grp):>5} | PF {pf:.2f} | Net {grp['pnl_pts'].sum():+,.0f}")

    df_t.to_csv("data/backtest_cross_results.csv", index=False)
    print("\n  Saved → data/backtest_cross_results.csv")


if __name__ == "__main__":
    main()
