"""
backtest_bar1_variants.py — two variants of first-bar filter.

V1 FLIP: wide bar, close in bottom 20% → LONG (mean-reversion)
                  close in top 20%    → SHORT
V2 HALF: wide threshold cut 50% (more signals), same momentum direction as V0.

Entry: MARKET at bar 1 close.
Exit:  bar 4 close (15 min) OR 50pt loss stop.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import load, FR

HOLD_BARS = 3
STOPS = {"DAX": 50, "US30": 50, "NIKKEI": 50}


def sim_day(day_5m, hours, mins, open_h, open_m, eod_h, eod_m, wide_pts, stop_pts, flip: bool,
            stop_frac_of_range: float = 0.0):
    """If stop_frac_of_range > 0, stop = frac * bar_range (volatility-adjusted)."""
    if len(day_5m) == 0: return []
    highs = day_5m[:, 1]; lows = day_5m[:, 2]; closes = day_5m[:, 3]
    open_mins = open_h*60 + open_m; eod_mins = eod_h*60 + eod_m
    mod = hours*60 + mins
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return []
    idxs = np.where(sess)[0]
    if len(idxs) < HOLD_BARS + 1: return []

    bar1 = idxs[0]
    b_h, b_l, b_c = highs[bar1], lows[bar1], closes[bar1]
    rng = b_h - b_l
    if rng <= wide_pts or rng <= 0:
        return []

    pos = (b_c - b_l) / rng
    if pos >= 0.8:
        direction = "SHORT" if flip else "LONG"
    elif pos <= 0.2:
        direction = "LONG" if flip else "SHORT"
    else:
        return []

    entry = b_c
    effective_stop = stop_frac_of_range * rng if stop_frac_of_range > 0 else stop_pts
    stop = entry - effective_stop if direction == "LONG" else entry + effective_stop
    exit_bar = bar1 + HOLD_BARS

    for j in range(bar1 + 1, min(exit_bar + 1, len(highs))):
        bh, bl, bc = highs[j], lows[j], closes[j]
        if mod[j] >= eod_mins:
            exit_price = bc
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            return [{"direction": direction, "entry": entry, "exit": exit_price,
                     "pnl_pts": round(pnl, 1), "reason": "EOD",
                     "bar_range": round(rng, 1), "close_pos": round(pos, 3)}]
        if (direction == "LONG" and bl <= stop) or (direction == "SHORT" and bh >= stop):
            return [{"direction": direction, "entry": entry, "exit": stop,
                     "pnl_pts": round(-effective_stop, 1), "reason": "STOP",
                     "bar_range": round(rng, 1), "close_pos": round(pos, 3)}]
        if j == exit_bar:
            exit_price = bc
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            return [{"direction": direction, "entry": entry, "exit": exit_price,
                     "pnl_pts": round(pnl, 1), "reason": "TIME",
                     "bar_range": round(rng, 1), "close_pos": round(pos, 3)}]
    return []


def run_inst(inst, wide_pts, stop_pts, flip, stop_frac=0.0):
    cfg = bt.INSTRUMENTS[inst]; meta = FR[inst]
    d5 = load(meta["5m"], meta["tz"], cfg["timezone"])
    ohlc = d5[["Open","High","Low","Close"]].values
    dates = d5["_d"].values; hours = d5["_h"].values; mins = d5["_m"].values
    oh, om = cfg["s1_open_hour"], cfg["s1_open_minute"]
    eh, em = cfg["session_end_hour"], cfg["session_end_minute"]
    trades = []
    for dd in sorted(set(dates)):
        mask = dates == dd
        ts = sim_day(ohlc[mask], hours[mask], mins[mask], oh, om, eh, em,
                     wide_pts, stop_pts, flip, stop_frac_of_range=stop_frac)
        for t in ts:
            t["date"] = str(dd); t["instrument"] = inst
            trades.append(t)
    return trades


def stats(trades, label):
    if not trades:
        print(f"  {label:<40} n=0"); return
    d = pd.DataFrame(trades)
    w = d[d["pnl_pts"] > 0]; l = d[d["pnl_pts"] < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    print(f"  {label:<40} n={len(d):>5}  PF={pf:>5.2f}  net={d.pnl_pts.sum():>+9,.0f}  "
          f"WR={len(w)/len(d)*100:>5.1f}%")


def run_variant(name, wide_multiplier, flip, stop_frac=0.0):
    print(f"\n{'='*90}\n  {name}\n{'='*90}")
    all_t = []
    for inst in ["DAX", "US30", "NIKKEI"]:
        wide = bt.INSTRUMENTS[inst]["wide_range"] * wide_multiplier
        tr = run_inst(inst, wide, STOPS[inst], flip, stop_frac=stop_frac)
        all_t.extend(tr)
        stop_desc = f"stop={stop_frac:.0%} of bar" if stop_frac else f"stop={STOPS[inst]}pt"
        print(f"\n--- {inst} (wide>{wide}, {stop_desc}, flip={flip}) ---")
        stats(tr, f"{inst} ALL")
        if tr:
            d = pd.DataFrame(tr); d["year"] = pd.to_datetime(d["date"]).dt.year
            stats(d[d.year<=2017].to_dict("records"), f"{inst} TRAIN 2008-17")
            stats(d[d.year>=2018].to_dict("records"), f"{inst} TEST 2018-26")
            for dr in ("LONG", "SHORT"):
                stats(d[d.direction==dr].to_dict("records"), f"{inst}   {dr}")
    print(f"\n--- COMBINED ---")
    stats(all_t, "ALL 3")


def main():
    t0 = time.time()
    run_variant("V1 FLIP: mean-reversion (50pt stop)",
                wide_multiplier=1.0, flip=True)
    run_variant("V2 VOL-STOP: stop = 50% of bar range (momentum)",
                wide_multiplier=1.0, flip=False, stop_frac=0.5)
    run_variant("V3 VOL-STOP + FLIP: stop = 50% of bar range (mean-reversion)",
                wide_multiplier=1.0, flip=True, stop_frac=0.5)
    print(f"\nelapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
