"""
backtest_dowflip_5min.py — Apply TrendFollowerV3's Dow-Theory HH/HL flip
logic to 5-min bars of DAX, US30, NIKKEI.

Rules (from the BN Future spreadsheet):
  - Each bar classified "R" or "D" based on HH/HL pattern (Dow Theory)
  - Trade fires on mode FLIP: D→R = BUY, R→D = SELL
  - Entry at the flip bar's close
  - Stop at ~3% away (approx — we use bar-derived ATR or fixed multiplier)
  - Hold until next flip (new trade closes previous)

We test:
  - Intraday only (exit at session close instead of holding overnight)
  - No filter (raw Dow flip)
  - Compare to current ORB strategy on same data
"""
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

DATA = "/root/asrs-bot/data/firstrate"
FILES = {
    "DAX":    (f"{DATA}/FDAX_full_5min_continuous_ratio_adjusted.txt", "Europe/Berlin",    (9,0),  (17,30)),
    "US30":   (f"{DATA}/YM_full_5min_continuous_ratio_adjusted.txt",   "America/New_York", (9,30), (16,0)),
    "NIKKEI": (f"{DATA}/NKD_full_5min_continuous_ratio_adjusted.txt",  "America/New_York", (10,0), (15,0)),
}
STOP_PCT = 0.003  # 0.3% — tighter than daily's 3% because 5-min timeframe


def load(path, tz):
    df = pd.read_csv(path, header=None, names=["dt","O","H","L","C","V"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").tz_localize(ZoneInfo(tz))
    df = df[df.index.dayofweek < 5]
    df["_hm"] = df.index.hour * 60 + df.index.minute
    df["_d"] = df.index.date
    return df


def classify_dow_mode(df):
    """Classify each 5-min bar as R or D via Dow Theory:
    - HH and not LL → R
    - LL and not HH → D
    - Both (outside bar) → R if close > prev_close else D
    - Neither (inside bar) → persist previous
    """
    prev_H = df["H"].shift(1)
    prev_L = df["L"].shift(1)
    prev_C = df["C"].shift(1)
    higher_h = df["H"] > prev_H
    lower_l  = df["L"] < prev_L

    mode = np.empty(len(df), dtype=object)
    prev = "D"
    for i in range(len(df)):
        if pd.isna(prev_H.iat[i]):
            mode[i] = prev; continue
        hh = higher_h.iat[i]; ll = lower_l.iat[i]
        if hh and not ll:    new = "R"
        elif ll and not hh:  new = "D"
        elif hh and ll:      new = "R" if df["C"].iat[i] > prev_C.iat[i] else "D"
        else:                new = prev
        mode[i] = new
        prev = new
    return mode


def simulate_dowflip(df, open_hm, close_hm, stop_pct):
    """Take trades on mode flips. Exit at session close each day.
    Returns list of trade dicts.
    """
    df = df[(df["_hm"] >= open_hm) & (df["_hm"] < close_hm)].copy()
    df["mode"] = classify_dow_mode(df)
    df["prev_mode"] = df["mode"].shift(1)
    df["flip"] = (df["mode"] != df["prev_mode"]) & df["prev_mode"].notna()

    trades = []
    for day, day_df in df.groupby("_d"):
        day_df = day_df.sort_index().reset_index(drop=True)
        pos = None  # (direction, entry_price, stop, entry_idx)
        for i, row in day_df.iterrows():
            if pos is not None:
                direction, entry, stop, ei = pos
                # Stop check (intra-bar)
                if direction == "LONG" and row["L"] <= stop:
                    trades.append({"date": day, "direction": "LONG",
                                   "entry": entry, "exit": stop,
                                   "pnl_pts": stop - entry, "reason": "STOP"})
                    pos = None
                elif direction == "SHORT" and row["H"] >= stop:
                    trades.append({"date": day, "direction": "SHORT",
                                   "entry": entry, "exit": stop,
                                   "pnl_pts": entry - stop, "reason": "STOP"})
                    pos = None
                # Flip-based exit + reverse
                elif row["flip"]:
                    exit_price = row["C"]
                    if direction == "LONG":
                        pnl = exit_price - entry
                    else:
                        pnl = entry - exit_price
                    trades.append({"date": day, "direction": direction,
                                   "entry": entry, "exit": exit_price,
                                   "pnl_pts": pnl, "reason": "FLIP"})
                    # Open opposite
                    if row["mode"] == "R":
                        pos = ("LONG", row["C"], row["C"] * (1 - stop_pct), i)
                    else:
                        pos = ("SHORT", row["C"], row["C"] * (1 + stop_pct), i)
            else:
                # No position — open on flip
                if row["flip"]:
                    if row["mode"] == "R":
                        pos = ("LONG", row["C"], row["C"] * (1 - stop_pct), i)
                    else:
                        pos = ("SHORT", row["C"], row["C"] * (1 + stop_pct), i)
        # EOD close
        if pos is not None:
            direction, entry, stop, ei = pos
            last = day_df.iloc[-1]
            exit_price = last["C"]
            pnl = exit_price - entry if direction == "LONG" else entry - exit_price
            trades.append({"date": day, "direction": direction,
                           "entry": entry, "exit": exit_price,
                           "pnl_pts": pnl, "reason": "EOD"})
    return trades


def stats(trades, label):
    if not trades: print(f"  {label:<25}  no trades"); return
    pnls = [t["pnl_pts"] for t in trades]
    net = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = sum(wins) / max(abs(sum(losses)), 0.001)
    wr = len(wins) / len(trades) * 100
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    print(f"  {label:<25}  n={len(trades):>6}  PF={pf:>5.2f}  net={net:>+10,.0f}  "
          f"WR={wr:>4.1f}%  avgW={avg_w:>5.1f}  avgL={avg_l:>5.1f}")


def main():
    for inst, (path, tz, open_t, close_t) in FILES.items():
        print(f"\n{'='*90}\n  {inst}\n{'='*90}")
        df = load(path, tz)
        print(f"  Data: {df.index.min().date()} → {df.index.max().date()}  ({len(df):,} bars)")
        open_hm = open_t[0]*60 + open_t[1]
        close_hm = close_t[0]*60 + close_t[1]
        trades = simulate_dowflip(df, open_hm, close_hm, STOP_PCT)
        if not trades: print("  no trades"); continue
        # Annotate year
        tdf = pd.DataFrame(trades)
        tdf["year"] = pd.to_datetime(tdf["date"]).dt.year
        stats(trades, "ALL")
        train = tdf[tdf["year"] <= 2017].to_dict("records")
        test = tdf[tdf["year"] >= 2018].to_dict("records")
        stats(train, "  TRAIN 2008-17")
        stats(test,  "  TEST 2018-26")


if __name__ == "__main__":
    main()
