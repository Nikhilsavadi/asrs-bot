"""
backtest_signal_bar_position.py — does signal bar position matter?

For each trade, classify signal bar's position within session-range-so-far:
  TOP    = signal bar high == max high across bars 1-4 (or 1-5 for bar5 signals)
  BOTTOM = signal bar low == min low
  MID    = neither

Then split by trade direction (LONG/SHORT) and report PF per bucket.

Hypothesis: LONGs on a TOP signal bar may be exhaustion (avoid); SHORTs on
TOP may be strong (take). Vice versa for BOTTOM.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_fade import build_cache
from backtest_v2 import _find_real_reentry_in_1min

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["session_end_hour"] = 15
bt.INSTRUMENTS["NIKKEI"]["session_end_minute"] = 0

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}


def sim_with_position(day_5min, hours_5m, minutes_5m, open_h, open_m, eod_h, eod_m,
                       cfg, day_1min=None, one_min_hours=None, one_min_minutes=None):
    """Simulate like sim_with_fade but capture signal_bar_pos."""
    if len(day_5min) == 0: return []
    highs = day_5min[:, 1]; lows = day_5min[:, 2]; closes = day_5min[:, 3]; opens = day_5min[:, 0]
    open_mins = open_h*60+open_m; eod_mins = eod_h*60+eod_m
    mod = hours_5m*60 + minutes_5m
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any(): return []
    idxs = np.where(sess)[0]
    if len(idxs) < 5: return []

    bar4_idx = idxs[3]; bar4_h = highs[bar4_idx]; bar4_l = lows[bar4_idx]
    bar_range_4 = bar4_h - bar4_l
    if bar_range_4 < cfg["narrow_range"]:
        sig_h, sig_l = bar4_h, bar4_l; bar_num = 4
    elif bar_range_4 > cfg["wide_range"]:
        if len(idxs) < 5: return []
        bar5_idx = idxs[4]; sig_h = highs[bar5_idx]; sig_l = lows[bar5_idx]
        bar_num = 5
        if sig_h - sig_l > cfg["max_bar_range"] or sig_h - sig_l <= 0: return []
    else:
        sig_h, sig_l = bar4_h, bar4_l; bar_num = 4

    # Session-so-far range INCLUDING the signal bar (bars 1..bar_num)
    n_bars_to_include = bar_num
    session_bars_so_far = idxs[:n_bars_to_include]
    session_h = highs[session_bars_so_far].max()
    session_l = lows[session_bars_so_far].min()

    # Position flag
    if abs(sig_h - session_h) < 0.01:
        pos_flag = "TOP"
    elif abs(sig_l - session_l) < 0.01:
        pos_flag = "BOTTOM"
    else:
        pos_flag = "MID"

    bar_range = sig_h - sig_l
    buy_level = round(sig_h + cfg["buffer"], 1)
    sell_level = round(sig_l - cfg["buffer"], 1)
    first_scan = bar4_idx+1 if bar_num == 4 else bar4_idx+2
    sess_end = idxs[-1]

    fl = fs = -1
    for j in range(first_scan, sess_end + 1):
        if mod[j] >= eod_mins: break
        if highs[j] >= buy_level and fl == -1: fl = j
        if lows[j] <= sell_level and fs == -1: fs = j
        if fl != -1 and fs != -1: break
    if fl == -1 and fs == -1: return []
    if fl != -1 and fs != -1:
        if fl < fs: direction, entry, stop, start = "LONG", buy_level, sell_level, fl
        elif fs < fl: direction, entry, stop, start = "SHORT", sell_level, buy_level, fs
        else:
            if opens[fl] >= buy_level: direction, entry, stop, start = "LONG", buy_level, sell_level, fl
            else: direction, entry, stop, start = "SHORT", sell_level, buy_level, fs
    elif fl >= 0: direction, entry, stop, start = "LONG", buy_level, sell_level, fl
    else: direction, entry, stop, start = "SHORT", sell_level, buy_level, fs

    trades = []; entries_used = 0; max_e = cfg["max_entries"]
    active = True; be = False; waiting = False; last_stop = 0
    first_direction = None; first_pnl = None

    j = start
    while j <= sess_end:
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                pnl = (opens[j]-entry) if direction == "LONG" else (entry-opens[j])
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(opens[j], 1), "pnl_pts": round(pnl, 1),
                               "reason": "EOD", "bar_num": bar_num,
                               "pos_flag": pos_flag, "tn": entries_used + 1})
            break
        bh, bl, bc = highs[j], lows[j], closes[j]
        if active:
            if (direction == "LONG" and bl <= stop) or (direction == "SHORT" and bh >= stop):
                pnl = (stop - entry) if direction == "LONG" else (entry - stop)
                t = {"direction": direction, "entry": round(entry, 1),
                     "exit": round(stop, 1), "pnl_pts": round(pnl, 1),
                     "reason": "STOP", "bar_num": bar_num,
                     "pos_flag": pos_flag, "tn": entries_used + 1}
                trades.append(t)
                if first_direction is None:
                    first_direction = direction; first_pnl = pnl
                entries_used += 1; active = False; waiting = entries_used < max_e
                last_stop = bm + 5
                j += 1; continue
            unr = (bc - entry) if direction == "LONG" else (entry - bc)
            if not be and unr >= cfg.get("breakeven_pts", 15):
                be = True
                if direction == "LONG" and stop < entry: stop = entry
                elif direction == "SHORT" and stop > entry: stop = entry
            if j > start:
                prev_c = closes[j - 1]
                if direction == "LONG" and prev_c > stop: stop = round(prev_c, 1)
                elif direction == "SHORT" and prev_c < stop: stop = round(prev_c, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop, eod_mins,
                                                  buy_level, sell_level)
                if res is None: waiting = False
                else:
                    re_min, re_dir, re_fill = res
                    direction = re_dir; entry = re_fill
                    stop = sell_level if re_dir == "LONG" else buy_level
                    be = False; active = True
                    if first_direction is None:
                        first_direction = direction
                    while j+1 < len(mod) and mod[j+1] < re_min: j += 1
            else: waiting = False
        j += 1
    return trades


def run():
    cache = build_cache()
    trades = []
    for inst, cached in cache.items():
        cfg = bt.INSTRUMENTS[inst]
        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        for s in sessions:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            name = f"{inst}_S{s}"
            for d in cached["dates"]:
                d5 = cached["by5"].get(d)
                if d5 is None: continue
                o5, h5, m5 = d5
                d1 = cached["by1"].get(d)
                o1, h1, min1 = d1 if d1 else (None, None, None)
                ts = sim_with_position(o5, h5, m5, oh, om, eh, em, cfg,
                                        day_1min=o1, one_min_hours=h1, one_min_minutes=min1)
                for t in ts:
                    t["date"] = str(d); t["signal"] = name; t["instrument"] = inst
                    trades.append(t)
    return trades


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * 1.06
    return df


def filter_rr(df):
    df = df.sort_values(["date", "signal", "tn"]).reset_index(drop=True)
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"})
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    opp = df["is_re"] & ~df["same"]
    drop = opp & (df["instrument"].isin(["DAX", "US30"])
                  | ((df["instrument"] == "NIKKEI") & df["first_won"]))
    return df[~drop].copy()


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def main():
    t0 = time.time()
    print("Running backtest with signal bar position tracking ...", flush=True)
    trades = run()
    df = pd.DataFrame(trades)
    print(f"  total trades: {len(df):,} ({time.time()-t0:.0f}s)", flush=True)

    df = filter_rr(df)
    df = apply_friction(df)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    print(f"\n{'='*100}")
    print(f"  SIGNAL-BAR POSITION × DIRECTION (18yr, Variant B, reverse-reentry filter, post-friction)")
    print(f"{'='*100}")

    # Overall distribution
    for inst in ["ALL", "DAX", "US30", "NIKKEI"]:
        sub = df if inst == "ALL" else df[df.instrument == inst]
        print(f"\n  --- {inst} ---")
        print(f"  {'pos':<8} {'dir':<8} {'n':>7} {'PF':>6} {'net':>12} {'avg':>8} {'WR':>6}")
        for pos in ["TOP", "MID", "BOTTOM"]:
            for direction in ["LONG", "SHORT"]:
                s = sub[(sub.pos_flag == pos) & (sub.direction == direction)]
                if len(s) == 0: continue
                w = s[s.pnl_net > 0].pnl_net.sum()
                l = abs(s[s.pnl_net < 0].pnl_net.sum())
                p = w / max(l, 0.001)
                avg = s.pnl_net.mean()
                wr = (s.pnl_net > 0).mean() * 100
                flag = "⚠️ " if p < 1.0 else ("  " if p < 1.5 else "✓ ")
                print(f"  {pos:<8} {direction:<8} {len(s):>7,} {p:>6.2f} {s.pnl_net.sum():>+12,.0f} {avg:>+8.2f} {wr:>5.1f}% {flag}")

    # Total by pos only
    print(f"\n  --- by position only (all instruments + all directions) ---")
    print(f"  {'pos':<8} {'n':>7} {'PF':>6} {'net':>12} {'avg':>8} {'WR':>6}")
    for pos in ["TOP", "MID", "BOTTOM"]:
        s = df[df.pos_flag == pos]
        if len(s) == 0: continue
        p = pf(s["pnl_net"])
        wr = (s.pnl_net > 0).mean() * 100
        print(f"  {pos:<8} {len(s):>7,} {p:>6.2f} {s.pnl_net.sum():>+12,.0f} {s.pnl_net.mean():>+8.2f} {wr:>5.1f}%")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
