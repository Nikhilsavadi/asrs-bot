"""
mfe_retro_replay.py — retrospective MFE-hybrid vs current (candle) exit.

The MFE-hybrid forward-walk was never operationalised (see memory
project_mfe_exit_candidate_2026_04_15). This runs it properly, after the fact,
over the logged IG tick CSVs for every live trade day we have data for.

Method (clean attribution):
  • Same entry/re-entry logic for both arms (both call the shared bar4/5
    breakout + 1-min re-entry engine), so they only differ in the EXIT.
  • CANDLE arm  = sim_with_fade base exit (Variant B trail-to-prev-close + BE) —
    i.e. exactly the live engine, fade off.
  • MFE arm     = identical Variant-B trail PLUS the MFE-fraction overlay
    (stop = max(variant_b_stop, peak − retrace×mfe), activated after
    activate_mult × initial_stop). So the ONLY policy delta is the overlay.
  • Both arms get the same reverse-reentry filter + spread/slippage friction.
  • Judged on RELATIVE lift on the same days — NOT the stale absolute-PF rule.

Usage: python3 mfe_retro_replay.py            # all dates with tick CSVs
       python3 mfe_retro_replay.py --csv out.csv
"""
import os, sys, glob, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_fade import sim_with_fade
from backtest_v2 import _find_real_reentry_in_1min
from nightly_backtest_vs_live import (
    INST_META, SPREAD, SLIP, fetch_bars_local, bars_to_numpy,
    apply_friction, filter_reverse_reentry,
)

TICK_DIR = "/root/asrs-bot/data/ticks"
MFE_RETRACE = 0.5
MFE_ACTIVATE_MULT = 0.5
ARMS_INSTRUMENTS = ["DAX", "US30"]  # where MFE was specced + we have tick data


def simulate_session_mfe_vb(day_5min, hours_5m, minutes_5m, open_h, open_m,
                            eod_h, eod_m, cfg,
                            day_1min=None, one_min_hours=None, one_min_minutes=None):
    """Variant-B candle trail (mirrors sim_with_fade exactly) + MFE overlay.
    Entry/re-entry logic is identical to sim_with_fade. No fade layer."""
    if len(day_5min) == 0:
        return []
    retrace = cfg.get("mfe_retrace", MFE_RETRACE)
    activate_mult = cfg.get("mfe_activate_mult", MFE_ACTIVATE_MULT)
    highs = day_5min[:, 1]; lows = day_5min[:, 2]; closes = day_5min[:, 3]; opens = day_5min[:, 0]
    open_mins = open_h * 60 + open_m; eod_mins = eod_h * 60 + eod_m
    mod = hours_5m * 60 + minutes_5m
    sess = (mod >= open_mins) & (mod < eod_mins)
    if not sess.any():
        return []
    idxs = np.where(sess)[0]
    if len(idxs) < 5:
        return []

    # --- entry selection: identical to sim_with_fade ---
    bar4_idx = idxs[3]; bar4_h = highs[bar4_idx]; bar4_l = lows[bar4_idx]
    bar_range = bar4_h - bar4_l; range_flag = "NARROW"; bar_num = 4
    if bar_range < cfg["narrow_range"]:
        sig_h, sig_l = bar4_h, bar4_l
    elif bar_range > cfg["wide_range"]:
        if len(idxs) < 5:
            return []
        bar5_idx = idxs[4]; sig_h = highs[bar5_idx]; sig_l = lows[bar5_idx]
        bar_range = sig_h - sig_l; bar_num = 5
        if bar_range > cfg["max_bar_range"] or bar_range <= 0:
            return []
        range_flag = "WIDE" if bar_range > cfg["wide_range"] else "NORMAL"
    else:
        sig_h, sig_l = bar4_h, bar4_l; range_flag = "NORMAL"
    buy_level = round(sig_h + cfg["buffer"], 1); sell_level = round(sig_l - cfg["buffer"], 1)
    first_scan = bar4_idx + 1 if bar_num == 4 else bar4_idx + 2
    sess_end = idxs[-1]
    fl = fs = -1
    for j in range(first_scan, sess_end + 1):
        if mod[j] >= eod_mins:
            break
        if highs[j] >= buy_level and fl == -1:
            fl = j
        if lows[j] <= sell_level and fs == -1:
            fs = j
        if fl != -1 and fs != -1:
            break
    if fl == -1 and fs == -1:
        return []
    if fl != -1 and fs != -1:
        if fl < fs:
            direction, entry, stop, start = "LONG", buy_level, sell_level, fl
        elif fs < fl:
            direction, entry, stop, start = "SHORT", sell_level, buy_level, fs
        else:
            if opens[fl] >= buy_level:
                direction, entry, stop, start = "LONG", buy_level, sell_level, fl
            else:
                direction, entry, stop, start = "SHORT", sell_level, buy_level, fs
    elif fl >= 0:
        direction, entry, stop, start = "LONG", buy_level, sell_level, fl
    else:
        direction, entry, stop, start = "SHORT", sell_level, buy_level, fs

    trades = []; entries_used = 0; max_e = cfg["max_entries"]
    active = True; be = False; mfe = 0.0; waiting = False; last_stop = 0
    peak = entry
    init_stop_dist = abs(entry - stop)
    activate_pts = activate_mult * init_stop_dist

    for j in range(start, sess_end + 1):
        bm = mod[j]
        if bm >= eod_mins:
            if active:
                pnl = (opens[j] - entry) if direction == "LONG" else (entry - opens[j])
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(opens[j], 1), "pnl_pts": round(pnl, 1),
                    "mfe": round(mfe, 1), "adds": 0, "reason": "EOD", "bar_num": bar_num,
                    "range_flag": range_flag, "bar_range": round(bar_range, 1), "exit_idx": j})
            break
        bh, bl, bc = highs[j], lows[j], closes[j]
        if active:
            if (direction == "LONG" and bl <= stop) or (direction == "SHORT" and bh >= stop):
                pnl = (stop - entry) if direction == "LONG" else (entry - stop)
                trades.append({"direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(pnl, 1), "mfe": round(mfe, 1),
                    "adds": 0, "reason": "TRAIL_WIN" if pnl > 0 else "STOP", "bar_num": bar_num,
                    "range_flag": range_flag, "bar_range": round(bar_range, 1), "exit_idx": j})
                entries_used += 1; active = False; waiting = entries_used < max_e
                last_stop = bm + 5
                continue
            # MFE + peak
            if direction == "LONG":
                if bh > peak: peak = bh
                m = bh - entry
                if m > mfe: mfe = m
                unr = bc - entry
            else:
                if bl < peak: peak = bl
                m = entry - bl
                if m > mfe: mfe = m
                unr = entry - bc
            # Breakeven (identical to sim_with_fade)
            if not be and unr >= cfg["breakeven_pts"]:
                be = True
                if direction == "LONG" and stop < entry: stop = entry
                elif direction == "SHORT" and stop > entry: stop = entry
            # Variant-B candle trail (identical to sim_with_fade)
            if j > start:
                prev_c = closes[j - 1]
                if direction == "LONG":
                    if prev_c > stop: stop = round(prev_c, 1)
                else:
                    if prev_c < stop: stop = round(prev_c, 1)
            # MFE overlay — the ONLY addition vs the candle arm
            if mfe >= activate_pts:
                if direction == "LONG":
                    ms = peak - retrace * mfe
                    if ms > stop: stop = round(ms, 1)
                else:
                    ms = peak + retrace * mfe
                    if ms < stop: stop = round(ms, 1)
        elif waiting:
            if day_1min is not None and one_min_minutes is not None:
                res = _find_real_reentry_in_1min(day_1min, one_min_minutes, last_stop,
                                                 eod_mins, buy_level, sell_level)
                if res is None:
                    waiting = False
                else:
                    re_min, re_dir, re_fill = res
                    direction = re_dir; entry = re_fill
                    stop = sell_level if re_dir == "LONG" else buy_level
                    be = False; mfe = 0.0; active = True
                    peak = entry; init_stop_dist = abs(entry - stop)
                    activate_pts = activate_mult * init_stop_dist
                    while j + 1 < len(mod) and mod[j + 1] < re_min:
                        j += 1
            else:
                waiting = False
    return trades


def run_arm(date, inst, bars5, bars1, exit_engine):
    """Run one exit engine for one instrument/date across its sessions.
    Returns post-friction base trades (reverse-reentry filtered)."""
    cfg = dict(bt.INSTRUMENTS[inst])
    cfg["mfe_retrace"] = MFE_RETRACE; cfg["mfe_activate_mult"] = MFE_ACTIVATE_MULT
    d5 = bars5.get(inst); d1 = bars1.get(inst)
    if d5 is None or d5.empty:
        return []
    o5, h5, m5 = bars_to_numpy(d5)
    o1 = h1 = m1abs = None
    if d1 is not None and not d1.empty:
        o1, h1, m1 = bars_to_numpy(d1)
        m1abs = h1 * 60 + m1
    sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
    trades_all = []
    for s in sessions:
        oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
        eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
        if exit_engine == "candle":
            ts = sim_with_fade(o5, h5, m5, oh, om, eh, em, cfg,
                               day_1min=o1, one_min_hours=h1, one_min_minutes=m1abs,
                               fade_stop=0)  # base only, no fade
        else:
            ts = simulate_session_mfe_vb(o5, h5, m5, oh, om, eh, em, cfg,
                                         day_1min=o1, one_min_hours=h1, one_min_minutes=m1abs)
        for t in ts:
            t["session"] = s
        trades_all.extend(ts)
    if not trades_all:
        return []
    df = pd.DataFrame(trades_all)
    df["instrument"] = inst
    df["signal"] = df["session"].apply(lambda s: f"{inst}_S{s}")
    df["date"] = date
    df["entry"] = df["entry"].astype(float)
    df = filter_reverse_reentry(df)
    return apply_friction(df.to_dict("records"), inst)


def pf_stats(trades):
    nets = [t.get("pnl_net", t["pnl_pts"]) for t in trades]
    if not nets:
        return dict(n=0, net=0.0, pf=0.0, win=0.0)
    wins = [x for x in nets if x > 0]; gl = -sum(x for x in nets if x <= 0)
    pf = (sum(wins) / gl) if gl > 0 else float("inf")
    return dict(n=len(nets), net=sum(nets), pf=pf, win=100 * len(wins) / len(nets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    # All dates with a DAX or US30 tick CSV
    dates = set()
    for inst in ARMS_INSTRUMENTS:
        epic = INST_META[inst]["epic"]
        for p in glob.glob(f"{TICK_DIR}/{epic}_*.csv"):
            dates.add(os.path.basename(p).replace(epic + "_", "").replace(".csv", ""))
    dates = sorted(dates)
    if not dates:
        print("No tick CSVs found."); return
    print(f"MFE retrospective replay — {len(dates)} dates {dates[0]}→{dates[-1]}")
    print(f"  overlay: retrace={MFE_RETRACE} activate_mult={MFE_ACTIVATE_MULT} (as specced)")
    print("=" * 78)

    agg = {inst: {"candle": [], "mfe": []} for inst in ARMS_INSTRUMENTS}
    rows = []
    for date in dates:
        bars5, bars1 = {}, {}
        for inst in ARMS_INSTRUMENTS:
            meta = INST_META[inst]
            bars5[inst] = fetch_bars_local(meta["epic"], date, meta["tz"], "MINUTE_5")
            bars1[inst] = fetch_bars_local(meta["epic"], date, meta["tz"], "MINUTE")
        for inst in ARMS_INSTRUMENTS:
            c = run_arm(date, inst, bars5, bars1, "candle")
            m = run_arm(date, inst, bars5, bars1, "mfe")
            agg[inst]["candle"].extend(c); agg[inst]["mfe"].extend(m)
            cn = sum(t.get("pnl_net", t["pnl_pts"]) for t in c)
            mn = sum(t.get("pnl_net", t["pnl_pts"]) for t in m)
            if c or m:
                rows.append(dict(date=date, inst=inst, candle_net=round(cn, 1),
                                 mfe_net=round(mn, 1), delta=round(mn - cn, 1),
                                 n_candle=len(c), n_mfe=len(m)))

    grand = {"candle": [], "mfe": []}
    for inst in ARMS_INSTRUMENTS:
        cs = pf_stats(agg[inst]["candle"]); ms = pf_stats(agg[inst]["mfe"])
        grand["candle"].extend(agg[inst]["candle"]); grand["mfe"].extend(agg[inst]["mfe"])
        print(f"\n---- {inst} ----")
        print(f"  CANDLE : n={cs['n']:3d}  net={cs['net']:+8.1f}pt  PF={cs['pf']:.2f}  win={cs['win']:.0f}%")
        print(f"  MFE    : n={ms['n']:3d}  net={ms['net']:+8.1f}pt  PF={ms['pf']:.2f}  win={ms['win']:.0f}%")
        lift = ms["net"] - cs["net"]
        pct = (lift / abs(cs["net"]) * 100) if cs["net"] else float("nan")
        print(f"  → MFE lift: {lift:+.1f}pt ({pct:+.0f}% vs candle)  PF {cs['pf']:.2f}→{ms['pf']:.2f}")

    gc = pf_stats(grand["candle"]); gm = pf_stats(grand["mfe"])
    lift = gm["net"] - gc["net"]
    print("\n" + "=" * 78)
    print(f"  TOTAL  CANDLE net {gc['net']:+.1f}pt PF {gc['pf']:.2f}  |  "
          f"MFE net {gm['net']:+.1f}pt PF {gm['pf']:.2f}  |  lift {lift:+.1f}pt")
    print(f"  £ at £0.50/pt:  candle £{gc['net']*0.5:+.2f}  mfe £{gm['net']*0.5:+.2f}  "
          f"lift £{lift*0.5:+.2f}")
    print("=" * 78)
    # Honest read
    n = gc["n"]
    if n < 30:
        verdict = f"INSUFFICIENT: only {n} base trades replayed — not enough to decide."
    elif lift > 0 and gm["pf"] > gc["pf"]:
        verdict = "MFE BEATS candle on this live sample — candidate to forward-test live (canary)."
    elif lift <= 0:
        verdict = "MFE does NOT beat candle on live data — do NOT ship; backtest edge not reproducing."
    else:
        verdict = "MIXED — MFE net up but PF not clearly better; needs more data."
    print(f"  VERDICT: {verdict}")

    if args.csv and rows:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"  per-day detail → {args.csv}")


if __name__ == "__main__":
    main()
