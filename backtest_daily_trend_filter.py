"""
backtest_daily_trend_filter.py — Prior-day regime filter for ORB entries.

Hypothesis: on "up" days (prior-day bullish), the profitable direction is
usually LONG — skip SHORT entries. Vice versa for "down" days.

Variants tested:
  1. prior_close > prior_open (simple candle direction)
  2. prior_close > prior 5-day SMA (medium-term trend)
  3. prior_close > prior_high (strong bullish)
  4. 2-of-3: at least 2 of (C>O, C>SMA5, C>prior H-L midpoint)

For each variant:
  A. FILTER mode — drop counter-trend entries entirely
  B. SIZING mode — keep all entries but would size up aligned (simulated by
     counting aligned trades at 2x weight when computing PF)
"""
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

V2 = "/root/asrs-bot/data/backtest_firstrate_v2_results.csv"
FIRSTRATE = {
    "DAX":    ("/root/asrs-bot/data/firstrate/FDAX_full_5min_continuous_ratio_adjusted.txt", "Europe/Berlin"),
    "US30":   ("/root/asrs-bot/data/firstrate/YM_full_5min_continuous_ratio_adjusted.txt",   "America/New_York"),
    "NIKKEI": ("/root/asrs-bot/data/firstrate/NKD_full_5min_continuous_ratio_adjusted.txt",  "America/New_York"),
}


def daily_from_5min(path, tz):
    df = pd.read_csv(path, header=None, names=["dt","Open","High","Low","Close","V"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index = df.index.tz_localize(ZoneInfo(tz))
    df = df[df.index.dayofweek < 5]
    df["_d"] = df.index.date
    g = df.groupby("_d").agg(
        Open=("Open","first"),
        High=("High","max"),
        Low=("Low","min"),
        Close=("Close","last"),
    )
    return g


def load_and_filter():
    df = pd.read_csv(V2)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
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


def stats(d, label):
    if len(d) == 0: print(f"  {label:<50} empty"); return
    w = d[d.pnl_pts > 0]; l = d[d.pnl_pts < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    net = d.pnl_pts.sum()
    wr = len(w) / len(d) * 100
    print(f"  {label:<50} n={len(d):>6}  PF={pf:>5.2f}  net={net:>+9,.0f}  WR={wr:>4.1f}%")


def main():
    print("Loading trades + daily bars...")
    trades = load_and_filter()
    trades["date_d"] = trades["date"].dt.date
    print(f"  {len(trades):,} trades")

    # Build daily bar classification for each instrument
    daily_bias = {}
    for inst, (path, tz) in FIRSTRATE.items():
        d = daily_from_5min(path, tz)
        d["prev_O"] = d["Open"].shift(1)
        d["prev_C"] = d["Close"].shift(1)
        d["prev_H"] = d["High"].shift(1)
        d["prev_L"] = d["Low"].shift(1)
        d["prev_sma5"] = d["Close"].shift(1).rolling(5).mean()
        d["prev_mid_hl"] = (d["prev_H"] + d["prev_L"]) / 2
        # Variants
        d["bias_co"] = np.where(d["prev_C"] > d["prev_O"], "LONG", "SHORT")
        d["bias_sma"] = np.where(d["prev_C"] > d["prev_sma5"], "LONG", "SHORT")
        d["bias_midhl"] = np.where(d["prev_C"] > d["prev_mid_hl"], "LONG", "SHORT")
        # 2-of-3 consensus
        ok_co = (d["prev_C"] > d["prev_O"]).astype(int)
        ok_sma = (d["prev_C"] > d["prev_sma5"]).astype(int)
        ok_mid = (d["prev_C"] > d["prev_mid_hl"]).astype(int)
        d["bias_2of3"] = np.where(ok_co + ok_sma + ok_mid >= 2, "LONG", "SHORT")
        daily_bias[inst] = d

    # For each variant, attach bias to each trade and compute PF with filter
    print(f"\n\nBaseline (no filter):")
    stats(trades, "FULL")
    for period, mask in [("TRAIN", trades["year"] <= 2017),
                          ("TEST", trades["year"] >= 2018)]:
        stats(trades[mask], f"  {period}")

    for variant in ["bias_co", "bias_sma", "bias_midhl", "bias_2of3"]:
        print(f"\n=== Variant: {variant} — drop counter-trend entries ===")
        # Attach bias per trade
        trades_annot = trades.copy()
        biases = []
        for idx, row in trades_annot.iterrows():
            d_bias = daily_bias.get(row["instrument"])
            if d_bias is None:
                biases.append(None); continue
            b = d_bias[variant].get(row["date_d"])
            biases.append(b if pd.notna(b) else None)
        trades_annot["bias"] = biases
        # Drop if direction != bias
        aligned = trades_annot[trades_annot["direction"] == trades_annot["bias"]]
        stats(aligned, "FULL")
        for period, mask in [("TRAIN", aligned["year"] <= 2017),
                              ("TEST", aligned["year"] >= 2018)]:
            stats(aligned[mask], f"  {period}")
        # Per-instrument
        for inst in ["DAX", "US30", "NIKKEI"]:
            stats(aligned[aligned["instrument"] == inst], f"    {inst}")


if __name__ == "__main__":
    main()
