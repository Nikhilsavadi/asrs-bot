"""
analyse_post_trend.py — Test: do we do worse on the day AFTER a trending day?

A "trending day" is defined by directional close:
  trend_score = (Close - Open) / (High - Low)
  trend_score >= THRESH → bull trend day
  trend_score <= -THRESH → bear trend day
  |trend_score| < THRESH → choppy/range day

We then look at NEXT day's trades and compare PF.
"""
import pandas as pd
from zoneinfo import ZoneInfo
from pathlib import Path

V2_TRADES = "/app/data/backtest_firstrate_v2_results.csv"
DATA = "/app/data/firstrate"
FILES = {
    "DAX":    ("FDAX_full_5min_continuous_ratio_adjusted.txt", "Europe/Berlin"),
    "US30":   ("YM_full_5min_continuous_ratio_adjusted.txt",   "America/New_York"),
    "NIKKEI": ("NKD_full_5min_continuous_ratio_adjusted.txt",  "America/New_York"),
}

# Session hours (match live config)
SESSIONS = {
    "DAX":    (9, 0, 17, 30),
    "US30":   (9, 30, 16, 0),
    "NIKKEI": (10, 0, 15, 0),
}


def daily_trend(path: str, tz: str, sess: tuple) -> pd.DataFrame:
    """Build daily OHLC from the session hours of the 5-min file."""
    df = pd.read_csv(path, header=None, names=["dt","Open","High","Low","Close","Volume"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index = df.index.tz_localize(ZoneInfo(tz))
    df = df[df.index.dayofweek < 5]
    oh, om, ch, cm = sess
    start_m = oh*60 + om
    end_m   = ch*60 + cm
    df["_m"] = df.index.hour*60 + df.index.minute
    df = df[(df["_m"] >= start_m) & (df["_m"] < end_m)]
    df["_date"] = df.index.date
    g = df.groupby("_date").agg(
        Open=("Open","first"),
        High=("High","max"),
        Low=("Low","min"),
        Close=("Close","last"),
    )
    g["range"]   = g["High"] - g["Low"]
    g["body"]    = g["Close"] - g["Open"]
    # Normalised trend: +1 pure bull, -1 pure bear, 0 doji
    g["trend"]   = g["body"] / g["range"].replace(0, 1)
    return g


def classify(trend: float, thresh: float) -> str:
    if trend >= thresh:  return "BULL"
    if trend <= -thresh: return "BEAR"
    return "CHOP"


def run(thresh: float):
    trades = pd.read_csv(V2_TRADES)
    trades["date"] = pd.to_datetime(trades["date"]).dt.date

    per_inst_daily = {}
    for inst, (fname, tz) in FILES.items():
        p = f"{DATA}/{fname}"
        if not Path(p).exists():
            print(f"missing {p}")
            continue
        per_inst_daily[inst] = daily_trend(p, tz, SESSIONS[inst])

    # Attach prior-day classification per trade (same instrument)
    results = []
    for inst, dd in per_inst_daily.items():
        sub = trades[trades["instrument"] == inst].copy()
        dd = dd.reset_index()
        dd["_date_next"] = dd["_date"].shift(-1)  # the day this trend predicts
        dd["prev_cls"]   = dd["trend"].apply(lambda t: classify(t, thresh))
        merged = sub.merge(
            dd[["_date_next","prev_cls","trend"]].rename(columns={"_date_next":"date","trend":"prev_trend"}),
            on="date", how="left"
        )
        results.append(merged)
    m = pd.concat(results)

    def stats(df: pd.DataFrame, label: str):
        if len(df) == 0:
            return
        wins   = df[df["pnl_pts"] > 0]
        losses = df[df["pnl_pts"] < 0]
        w  = wins["pnl_pts"].sum()
        l  = abs(losses["pnl_pts"].sum()) or 0.001
        pf = w / l
        net = df["pnl_pts"].sum()
        wr  = len(wins) / len(df) * 100
        print(f"  {label:<30} n={len(df):>6}  PF={pf:>5.2f}  net={net:>+9,.0f}  WR={wr:.0f}%")

    print(f"\n=== threshold |trend| >= {thresh} ===")
    print(f"Total trades with prior-day class: {len(m.dropna(subset=['prev_cls']))}")
    print()
    print("OVERALL (all instruments):")
    stats(m, "all")
    for c in ["BULL","BEAR","CHOP"]:
        stats(m[m["prev_cls"] == c], f"prior={c}")

    print()
    print("PER INSTRUMENT:")
    for inst in ["DAX","US30","NIKKEI"]:
        sub = m[m["instrument"] == inst]
        if len(sub) == 0:
            continue
        print(f"\n  {inst}:")
        stats(sub, "all")
        for c in ["BULL","BEAR","CHOP"]:
            stats(sub[sub["prev_cls"] == c], f"prior={c}")

    print()
    print("BULL vs BEAR vs CHOP — aggregated (TREND = BULL+BEAR):")
    trend_rows = m[m["prev_cls"].isin(["BULL","BEAR"])]
    chop_rows  = m[m["prev_cls"] == "CHOP"]
    stats(trend_rows, "prior=TREND (BULL or BEAR)")
    stats(chop_rows,  "prior=CHOP")


if __name__ == "__main__":
    for t in [0.5, 0.6, 0.7, 0.8]:
        run(t)
