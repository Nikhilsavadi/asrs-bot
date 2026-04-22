"""
backtest_bar1_momentum.py — First-bar wide-and-close-extreme continuation.

Signal: Session-1 first 5-min bar has range > wide_range AND closes in the
bottom 20% (→ SHORT) or top 20% (→ LONG) of its own range.

Entry: MARKET at bar 1 close.
Exit:  bar 4 close (15 min hold) OR 50pt loss stop, whichever first.

Tested on 18yr FirstRate data, DAX/US30/NIKKEI.
"""
import os, sys, time, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR

STOPS = {"DAX": 50, "US30": 50, "NIKKEI": 50}        # per user spec
STOPS_SCALED = {"DAX": 40, "US30": 80, "NIKKEI": 150} # scaled to instrument noise

HOLD_BARS = 3  # enter at bar 1 close, exit 3 bars later = 15 min hold


def sim_day(day_5m, hours, mins, open_h, open_m, eod_h, eod_m, cfg, stop_pts):
    if len(day_5m) == 0: return []
    highs = day_5m[:, 1]; lows = day_5m[:, 2]; closes = day_5m[:, 3]; opens = day_5m[:, 0]
    open_mins = open_h*60 + open_m; eod_mins = eod_h*60 + eod_m
    mod = hours*60 + mins
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return []
    idxs = np.where(sess)[0]
    if len(idxs) < HOLD_BARS + 1: return []

    bar1 = idxs[0]
    b_h, b_l, b_c, b_o = highs[bar1], lows[bar1], closes[bar1], opens[bar1]
    rng = b_h - b_l
    if rng <= cfg["wide_range"] or rng <= 0:
        return []  # not wide

    # Close-position filter
    pos = (b_c - b_l) / rng  # 0 = at low, 1 = at high
    if pos >= 0.8:
        direction = "LONG"
    elif pos <= 0.2:
        direction = "SHORT"
    else:
        return []

    entry = b_c
    stop = entry - stop_pts if direction == "LONG" else entry + stop_pts
    exit_bar = bar1 + HOLD_BARS

    # Walk bars 2..4 looking for stop hit
    for j in range(bar1 + 1, min(exit_bar + 1, len(highs))):
        bh, bl, bc = highs[j], lows[j], closes[j]
        if mod[j] >= eod_mins:
            exit_price = bc
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            return [{"direction": direction, "entry": entry, "exit": exit_price,
                     "pnl_pts": round(pnl, 1), "reason": "EOD",
                     "bar_range": round(rng, 1), "close_pos": round(pos, 3)}]
        # Stop hit check
        if (direction == "LONG" and bl <= stop) or (direction == "SHORT" and bh >= stop):
            pnl = -stop_pts
            return [{"direction": direction, "entry": entry, "exit": stop,
                     "pnl_pts": round(pnl, 1), "reason": "STOP",
                     "bar_range": round(rng, 1), "close_pos": round(pos, 3)}]
        # Exit at bar 4 close (HOLD_BARS after bar1)
        if j == exit_bar:
            exit_price = bc
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            return [{"direction": direction, "entry": entry, "exit": exit_price,
                     "pnl_pts": round(pnl, 1), "reason": "TIME",
                     "bar_range": round(rng, 1), "close_pos": round(pos, 3)}]
    return []


def run_inst(inst, stop_pts):
    cfg = bt.INSTRUMENTS[inst]
    meta = FR[inst]
    d5 = load(meta["5m"], meta["tz"], cfg["timezone"])
    ohlc = d5[["Open","High","Low","Close"]].values
    dates = d5["_d"].values
    hours = d5["_h"].values
    mins = d5["_m"].values

    oh = cfg["s1_open_hour"]; om = cfg["s1_open_minute"]
    eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]

    trades = []
    for dd in sorted(set(dates)):
        mask = dates == dd
        day_ohlc = ohlc[mask]; day_h = hours[mask]; day_m = mins[mask]
        ts = sim_day(day_ohlc, day_h, day_m, oh, om, eh, em, cfg, stop_pts)
        for t in ts:
            t["date"] = str(dd); t["instrument"] = inst
            trades.append(t)
    return trades


def stats(trades, label):
    if not trades:
        print(f"  {label:<40} n=0")
        return
    d = pd.DataFrame(trades)
    w = d[d["pnl_pts"] > 0]; l = d[d["pnl_pts"] < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    wr = len(w) / len(d) * 100
    avgw = w.pnl_pts.mean() if len(w) else 0
    avgl = l.pnl_pts.mean() if len(l) else 0
    net = d.pnl_pts.sum()
    print(f"  {label:<40} n={len(d):>5}  PF={pf:>5.2f}  net={net:>+9,.0f}  "
          f"WR={wr:>5.1f}%  avgW={avgw:>+6.1f}  avgL={avgl:>+6.1f}")


def main():
    t0 = time.time()
    for stop_label, stops in [("50pt (user spec)", STOPS), ("scaled", STOPS_SCALED)]:
        print(f"\n{'='*90}\n  STOP = {stop_label}\n{'='*90}")
        all_trades = []
        for inst in ["DAX", "US30", "NIKKEI"]:
            print(f"\n--- {inst} (stop={stops[inst]}pt, wide_range={bt.INSTRUMENTS[inst]['wide_range']}) ---")
            tr = run_inst(inst, stops[inst])
            all_trades.extend(tr)
            stats(tr, f"{inst} ALL")
            d = pd.DataFrame(tr) if tr else pd.DataFrame()
            if len(d):
                d["year"] = pd.to_datetime(d["date"]).dt.year
                stats(d[d.year<=2017].to_dict("records"), f"{inst} TRAIN 2008-17")
                stats(d[d.year>=2018].to_dict("records"), f"{inst} TEST 2018-26")
                for dr in ("LONG", "SHORT"):
                    stats(d[d.direction==dr].to_dict("records"), f"{inst}   {dr}")
        print(f"\n--- COMBINED ---")
        stats(all_trades, "ALL 3 instruments")
    print(f"\nelapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
