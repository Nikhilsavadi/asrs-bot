"""
backtest_expresso.py — Tom Hougaard's Expresso / Karaoke Expresso opening-range
breakout applied to DAX, US30, NIKKEI on 18yr firstrate data.

Rules:
  - Identify the opening-range bar (5 or 8 min from session open)
  - Buy stop: range_high + 2pt
  - Sell stop: range_low  - 2pt
  - Stop: either bar opposite OR fixed 40pt (two variants tested)
  - Exit: EOD (no discretion — conservative assumption)
  - Single entry per session

Tested:
  A. Expresso (5-min range)
  B. Karaoke (8-min range)
  C. Expresso with 40pt fixed stop
  D. Karaoke with 40pt fixed stop

Train (2008-17) / test (2018-26).
"""
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

DATA = "/root/asrs-bot/data/firstrate"
FILES = {
    "DAX":    (f"{DATA}/FDAX_full_5min_continuous_ratio_adjusted.txt", "Europe/Berlin",    9, 0,  17, 30),
    "US30":   (f"{DATA}/YM_full_5min_continuous_ratio_adjusted.txt",   "America/New_York", 9, 30, 16, 0),
    "NIKKEI": (f"{DATA}/NKD_full_5min_continuous_ratio_adjusted.txt",  "America/New_York", 10, 0, 15, 0),
}
FIVE_MIN = 5
EIGHT_MIN = 10  # 8-min rounded UP to 10 (2 × 5-min bars) since data is 5-min granularity
BUFFER = 2.0
FIXED_STOP = 40.0


def load(path, tz):
    df = pd.read_csv(path, header=None, names=["dt","O","H","L","C","V"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").tz_localize(ZoneInfo(tz))
    df = df[df.index.dayofweek < 5]
    df["_hm"] = df.index.hour * 60 + df.index.minute
    df["_d"] = df.index.date
    return df


def simulate(df, open_hm, close_hm, range_len_min, buffer, fixed_stop=None):
    """
    range_len_min: how many minutes of opening range (5 or 10 for our 5-min data)
    fixed_stop: None = use bar-opposite; else = stop distance in pts
    """
    trades = []
    for day, day_df in df.groupby("_d"):
        day_df = day_df[(day_df["_hm"] >= open_hm) & (day_df["_hm"] < close_hm)].sort_index()
        if len(day_df) < 3:
            continue
        # Opening range = first N minutes worth of 5-min bars
        n_bars = max(1, range_len_min // 5)
        range_bars = day_df.head(n_bars)
        range_h = range_bars["H"].max()
        range_l = range_bars["L"].min()
        buy_lvl  = range_h + buffer
        sell_lvl = range_l - buffer

        trading = day_df.iloc[n_bars:]
        if len(trading) == 0: continue

        position = None
        entry = 0.0
        stop = 0.0
        for i, row in trading.iterrows():
            h, l, c = row["H"], row["L"], row["C"]
            # Stop check first
            if position == "LONG" and l <= stop:
                trades.append({"date": day, "direction": "LONG",
                               "entry": entry, "exit": stop,
                               "pnl_pts": stop - entry, "reason": "STOP"})
                position = None
                break  # one trade per day
            if position == "SHORT" and h >= stop:
                trades.append({"date": day, "direction": "SHORT",
                               "entry": entry, "exit": stop,
                               "pnl_pts": entry - stop, "reason": "STOP"})
                position = None
                break
            # Entry check
            if position is None:
                if h >= buy_lvl:
                    position = "LONG"; entry = buy_lvl
                    stop = entry - fixed_stop if fixed_stop else range_l
                elif l <= sell_lvl:
                    position = "SHORT"; entry = sell_lvl
                    stop = entry + fixed_stop if fixed_stop else range_h
        # EOD close if still open
        if position is not None:
            last_c = trading.iloc[-1]["C"]
            pnl = (last_c - entry) if position == "LONG" else (entry - last_c)
            trades.append({"date": day, "direction": position,
                           "entry": entry, "exit": last_c,
                           "pnl_pts": pnl, "reason": "EOD"})
    return trades


def stats(trades, label):
    if not trades:
        print(f"  {label:<40}  no trades"); return
    pnls = [t["pnl_pts"] for t in trades]
    net = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = sum(wins) / max(abs(sum(losses)), 0.001)
    wr = len(wins) / len(trades) * 100
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    print(f"  {label:<40}  n={len(trades):>5}  PF={pf:>5.2f}  net={net:>+9,.0f}  "
          f"WR={wr:>4.1f}%  avgW={avg_w:>5.1f}  avgL={avg_l:>5.1f}")


def main():
    variants = [
        ("Expresso 5m, bar-stop",    FIVE_MIN,  None),
        ("Karaoke 8m, bar-stop",     EIGHT_MIN, None),
        ("Expresso 5m, 40pt-stop",   FIVE_MIN,  FIXED_STOP),
        ("Karaoke 8m, 40pt-stop",    EIGHT_MIN, FIXED_STOP),
    ]

    for inst, (path, tz, oh, om, eh, em) in FILES.items():
        print(f"\n{'='*95}\n  {inst}\n{'='*95}")
        df = load(path, tz)
        print(f"  Data: {df.index.min().date()} → {df.index.max().date()}  ({len(df):,} bars)")
        open_hm = oh * 60 + om
        close_hm = eh * 60 + em

        for label, rng, fstop in variants:
            trades = simulate(df, open_hm, close_hm, rng, BUFFER, fstop)
            if not trades: continue
            tdf = pd.DataFrame(trades)
            tdf["year"] = pd.to_datetime(tdf["date"]).dt.year
            print(f"\n  {label}")
            stats(trades, "    ALL")
            stats(tdf[tdf["year"] <= 2017].to_dict("records"), "    TRAIN 2008-17")
            stats(tdf[tdf["year"] >= 2018].to_dict("records"), "    TEST  2018-26")


if __name__ == "__main__":
    main()
