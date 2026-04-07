"""
backtest_1m.py — ASRS backtest with 1-minute stop/entry resolution.

Mirrors backtest.simulate_session but:
- Stops are checked at 1-minute granularity (not 5-min lows/highs)
- Entry triggers (buy_level/sell_level) checked at 1-minute granularity
- Trail / breakeven / adds still update on 5-minute boundaries (matches live bot)

This gives the "honest" answer: did the stop actually fire before the bar's
favorable extreme, or was the 5-min backtest cheating by seeing the bar's
high before its low?
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
FIRSTRATE = {
    "DAX":    {"file5": f"{DATA_DIR}/FDAX_full_5min_continuous_ratio_adjusted.txt",
               "file1": f"{DATA_DIR}/FDAX_full_1min_continuous_ratio_adjusted.txt",
               "src_tz": "Europe/Berlin"},
    "US30":   {"file5": f"{DATA_DIR}/YM_full_5min_continuous_ratio_adjusted.txt",
               "file1": f"{DATA_DIR}/YM_full_1min_continuous_ratio_adjusted.txt",
               "src_tz": "America/New_York"},
    "NIKKEI": {"file5": f"{DATA_DIR}/NKD_full_5min_continuous_ratio_adjusted.txt",
               "file1": f"{DATA_DIR}/NKD_full_1min_continuous_ratio_adjusted.txt",
               "src_tz": "America/New_York"},
}

BAR5_RULES = ["NORMAL", "WIDE"]


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
    df["_mins"] = df["_hour"] * 60 + df["_minute"]
    return df


def simulate_session_1m(d5, d1, open_h, open_m, eod_h, eod_m, cfg):
    """
    d5: dict with 'ohlc' (n5,4), 'mins' (n5,)
    d1: dict with 'ohlc' (n1,4), 'mins' (n1,)
    """
    ohlc5 = d5["ohlc"]; mins5 = d5["mins"]
    ohlc1 = d1["ohlc"]; mins1 = d1["mins"]
    n5 = len(ohlc5)
    if n5 == 0:
        return []

    open_mins = open_h * 60 + open_m
    eod_mins = eod_h * 60 + eod_m
    buffer = cfg["buffer"]

    # --- Bar 4/5 detection on 5-min (same logic as bt.simulate_session) ---
    mfo = mins5 - open_mins
    bnums = mfo // 5 + 1
    b4 = np.where((mfo >= 0) & (bnums == 4))[0]
    if len(b4) == 0:
        return []
    b4i = b4[0]
    bar4_h, bar4_l = ohlc5[b4i, 1], ohlc5[b4i, 2]
    bar4_range = bar4_h - bar4_l

    if bar4_range < cfg["narrow_range"]: rf = "NARROW"
    elif bar4_range > cfg["wide_range"]: rf = "WIDE"
    else: rf = "NORMAL"

    sig_h, sig_l, bar_num = bar4_h, bar4_l, 4
    if rf in BAR5_RULES:
        b5 = np.where((mfo >= 0) & (bnums == 5))[0]
        if len(b5) > 0:
            b5i = b5[0]
            sig_h, sig_l = ohlc5[b5i, 1], ohlc5[b5i, 2]
            bar_num = 5

    bar_range = sig_h - sig_l
    if bar_range < cfg["narrow_range"]: rf = "NARROW"
    elif bar_range > cfg["wide_range"]: rf = "WIDE"
    else: rf = "NORMAL"
    if bar_range > cfg["max_bar_range"]:
        return []

    bar_high = round(sig_h, 1)
    bar_low = round(sig_l, 1)
    buy_level = round(bar_high + buffer, 1)
    sell_level = round(bar_low - buffer, 1)

    # Risk cap
    if (bar_range + buffer * 2) > cfg["max_risk_gbp"]:
        max_stop = cfg["max_risk_gbp"]
        bar_high = round(sell_level + max_stop, 1)
        bar_low = round(buy_level - max_stop, 1)
        bar_range = max_stop

    # Identify the 5-min signal bar end-minute → start scanning 1-min after it
    sig_idx = np.where((mfo >= 0) & (bnums == bar_num))[0]
    if len(sig_idx) == 0:
        return []
    sig5i = sig_idx[0]
    sig_end_min = mins5[sig5i] + 5  # 1-min bars at this minute and after

    # Subset 1-min bars after signal bar
    rem1_mask = mins1 >= sig_end_min
    r1 = ohlc1[rem1_mask]
    rm1 = mins1[rem1_mask]
    rn1 = len(r1)
    if rn1 == 0:
        return []

    # Build per-1m index → which 5-min bar it belongs to (post-signal index)
    # 5-min bar index in rem5: (mins1[k] - sig_end_min) // 5
    rem5_start_idx = sig5i + 1
    rem_ohlc5 = ohlc5[rem5_start_idx:]
    rem_mins5 = mins5[rem5_start_idx:]
    rn5 = len(rem_ohlc5)

    # State
    trades = []
    entries_used = 0
    max_entries = cfg["max_entries"]
    active = False
    waiting = False
    breakeven_hit = False
    adds_used = 0
    last_add_price = 0.0
    direction = entry = stop = mfe = 0.0
    add_pnl = 0.0

    # Determine first entry from 1-min bars
    first_long = -1
    first_short = -1
    for k in range(rn1):
        if rm1[k] >= eod_mins:
            break
        if first_long < 0 and r1[k, 1] >= buy_level:
            first_long = k
        if first_short < 0 and r1[k, 2] <= sell_level:
            first_short = k
        if first_long >= 0 and first_short >= 0:
            break
    if first_long < 0 and first_short < 0:
        return []

    if first_long >= 0 and first_short >= 0:
        if first_long < first_short:
            direction, entry, stop = "LONG", buy_level, sell_level; k_start = first_long
        elif first_short < first_long:
            direction, entry, stop = "SHORT", sell_level, buy_level; k_start = first_short
        else:
            # same 1-min bar — use open as tiebreaker
            if r1[first_long, 0] >= buy_level:
                direction, entry, stop = "LONG", buy_level, sell_level; k_start = first_long
            else:
                direction, entry, stop = "SHORT", sell_level, buy_level; k_start = first_short
    elif first_long >= 0:
        direction, entry, stop = "LONG", buy_level, sell_level; k_start = first_long
    else:
        direction, entry, stop = "SHORT", sell_level, buy_level; k_start = first_short

    active = True
    last_add_price = entry

    # Walk 1-min bars; at 5-min boundaries do trail/breakeven/adds
    last_5m_idx = -1
    for k in range(k_start, rn1):
        m = rm1[k]
        if m >= eod_mins:
            if active:
                ep = r1[k, 0]
                pnl = (ep - entry) if direction == "LONG" else (entry - ep)
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(ep, 1), "pnl_pts": round(pnl + add_pnl, 1),
                               "mfe": round(mfe, 1), "adds": adds_used, "reason": "EOD",
                               "bar_num": bar_num, "range_flag": rf,
                               "bar_range": round(bar_range, 1)})
            break

        bo, bh, bl, bc = r1[k, 0], r1[k, 1], r1[k, 2], r1[k, 3]

        if active:
            # Stop check at 1-min granularity
            if direction == "LONG" and bl <= stop:
                pnl = stop - entry
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(stop, 1), "pnl_pts": round(pnl + add_pnl, 1),
                               "mfe": round(mfe, 1), "adds": adds_used, "reason": "STOP",
                               "bar_num": bar_num, "range_flag": rf,
                               "bar_range": round(bar_range, 1)})
                entries_used += 1
                active = False
                waiting = entries_used < max_entries
                continue
            if direction == "SHORT" and bh >= stop:
                pnl = entry - stop
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(stop, 1), "pnl_pts": round(pnl + add_pnl, 1),
                               "mfe": round(mfe, 1), "adds": adds_used, "reason": "STOP",
                               "bar_num": bar_num, "range_flag": rf,
                               "bar_range": round(bar_range, 1)})
                entries_used += 1
                active = False
                waiting = entries_used < max_entries
                continue

            # MFE
            if direction == "LONG":
                mm = bh - entry
                if mm > mfe: mfe = mm
            else:
                mm = entry - bl
                if mm > mfe: mfe = mm

        elif waiting:
            # Re-entry trigger at 1-min
            tl = bh >= buy_level
            ts = bl <= sell_level
            if tl and ts:
                if bo >= buy_level: ts = False
                elif bo <= sell_level: tl = False
                else:
                    tl = bc >= bo
                    ts = not tl
            if tl:
                direction, entry, stop = "LONG", buy_level, sell_level
                active, waiting = True, False
                breakeven_hit, adds_used, add_pnl = False, 0, 0.0
                last_add_price = entry
                mfe = max(0.0, bh - entry)
            elif ts:
                direction, entry, stop = "SHORT", sell_level, buy_level
                active, waiting = True, False
                breakeven_hit, adds_used, add_pnl = False, 0, 0.0
                last_add_price = entry
                mfe = max(0.0, entry - bl)

        # 5-min boundary check: when we cross into a new 5-min bar, run
        # trail / breakeven / adds based on the COMPLETED previous 5-min bar.
        cur_5m_idx = (m - sig_end_min) // 5  # which post-signal 5-min bar this 1-min belongs to
        if cur_5m_idx != last_5m_idx and last_5m_idx >= 0 and active:
            # Use 5-min bar at index last_5m_idx (just completed)
            if last_5m_idx < rn5:
                p_o, p_h, p_l, p_c = rem_ohlc5[last_5m_idx]
                # Breakeven
                if not breakeven_hit:
                    unreal = (p_c - entry) if direction == "LONG" else (entry - p_c)
                    if unreal >= cfg["breakeven_pts"]:
                        breakeven_hit = True
                        if direction == "LONG" and stop < entry: stop = entry
                        elif direction == "SHORT" and stop > entry: stop = entry
                # Trail (uses bar that completed before previous — j-1 in 5min logic)
                if last_5m_idx >= 1:
                    pp_o, pp_h, pp_l, pp_c = rem_ohlc5[last_5m_idx - 1]
                    if direction == "LONG":
                        profit = pp_c - entry
                        ns = pp_c if profit >= cfg["tight_threshold"] else pp_l
                        if ns > stop: stop = round(ns, 1)
                    else:
                        profit = entry - pp_c
                        ns = pp_c if profit >= cfg["tight_threshold"] else pp_h
                        if ns < stop: stop = round(ns, 1)
                # Adds
                if adds_used < cfg["add_max"]:
                    pfr = (p_c - last_add_price) if direction == "LONG" else (last_add_price - p_c)
                    if pfr >= cfg["add_trigger"]:
                        adds_used += 1
                        last_add_price = p_c
                        breakeven_hit = True
        last_5m_idx = cur_5m_idx

    return trades


def run(years, only):
    t0 = time.time()
    all_trades = []

    for inst_name, cfg in bt.INSTRUMENTS.items():
        if only and inst_name != only: continue
        if inst_name not in FIRSTRATE: continue

        fr = FIRSTRATE[inst_name]
        print(f"\n{'=' * 60}\n  {inst_name}  (1-min stop resolution)\n{'=' * 60}")
        t_load = time.time()
        df5 = load_csv(fr["file5"], fr["src_tz"], cfg["timezone"])
        df1 = load_csv(fr["file1"], fr["src_tz"], cfg["timezone"])
        if years:
            cutoff5 = df5.index.max() - pd.Timedelta(days=365 * years)
            df5 = df5[df5.index >= cutoff5]
            cutoff1 = df1.index.max() - pd.Timedelta(days=365 * years)
            df1 = df1[df1.index >= cutoff1]
        print(f"  5m bars: {len(df5):,}  |  1m bars: {len(df1):,}  |  load {time.time()-t_load:.0f}s")

        ohlc5 = df5[["Open", "High", "Low", "Close"]].values
        mins5 = df5["_mins"].values
        dates5 = df5["_date"].values

        ohlc1 = df1[["Open", "High", "Low", "Close"]].values
        mins1 = df1["_mins"].values
        dates1 = df1["_date"].values

        unique_dates = sorted(set(dates5))

        for session in (1, 2):
            signal_name = f"{inst_name}_S{session}"
            open_h = cfg[f"s{session}_open_hour"]
            open_m = cfg[f"s{session}_open_minute"]
            eod_h = cfg["session_end_hour"]
            eod_m = cfg["session_end_minute"]

            sig_trades = []
            for day in unique_dates:
                m5 = dates5 == day
                m1 = dates1 == day
                d5 = {"ohlc": ohlc5[m5], "mins": mins5[m5]}
                d1 = {"ohlc": ohlc1[m1], "mins": mins1[m1]}
                day_trades = simulate_session_1m(d5, d1, open_h, open_m, eod_h, eod_m, cfg)
                for t in day_trades:
                    t["date"] = str(day)
                    t["signal"] = signal_name
                    t["instrument"] = inst_name
                    sig_trades.append(t)

            if sig_trades:
                pnl = sum(t["pnl_pts"] for t in sig_trades)
                w = sum(1 for t in sig_trades if t["pnl_pts"] > 0)
                l = sum(1 for t in sig_trades if t["pnl_pts"] < 0)
                gw = sum(t["pnl_pts"] for t in sig_trades if t["pnl_pts"] > 0)
                gl = abs(sum(t["pnl_pts"] for t in sig_trades if t["pnl_pts"] < 0))
                pf = gw / gl if gl > 0 else float("inf")
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                print(f"  {signal_name}: {len(sig_trades):>5} trades | PF {pf:.2f} | "
                      f"W/L {w}/{l} ({wr:.0f}%) | Net {pnl:+,.0f}pts")
                all_trades.extend(sig_trades)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}\n  COMBINED  ({elapsed:.0f}s)\n{'=' * 60}")
    if not all_trades:
        print("  No trades."); return

    tp = sum(t["pnl_pts"] for t in all_trades)
    tw = sum(1 for t in all_trades if t["pnl_pts"] > 0)
    tl = sum(1 for t in all_trades if t["pnl_pts"] < 0)
    tf = sum(1 for t in all_trades if t["pnl_pts"] == 0)
    gw = sum(t["pnl_pts"] for t in all_trades if t["pnl_pts"] > 0)
    gl = abs(sum(t["pnl_pts"] for t in all_trades if t["pnl_pts"] < 0))
    pf = gw / gl if gl > 0 else float("inf")
    print(f"  Trades: {len(all_trades):,}  |  PF {pf:.2f}  |  W/L/F {tw}/{tl}/{tf}  |  Net {tp:+,.0f}pts")
    if tw and tl:
        print(f"  Avg win {gw/tw:.1f}  |  Avg loss {gl/tl:.1f}")

    df_t = pd.DataFrame(all_trades)
    df_t["year"] = pd.to_datetime(df_t["date"]).dt.year
    print(f"\n  Per year:")
    for year, grp in df_t.groupby("year"):
        yw = grp[grp["pnl_pts"] > 0]["pnl_pts"].sum()
        yl = abs(grp[grp["pnl_pts"] < 0]["pnl_pts"].sum())
        ypf = yw / yl if yl > 0 else float("inf")
        print(f"    {year}: {len(grp):>5} trades | PF {ypf:.2f} | Net {grp['pnl_pts'].sum():+,.0f}pts")

    eq = df_t["pnl_pts"].cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"\n  Max drawdown: {dd:,.0f} pts")

    out = "data/backtest_1m_results.csv"
    df_t.to_csv(out, index=False)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--instrument", choices=["DAX", "US30", "NIKKEI"], default=None)
    args = ap.parse_args()
    run(args.years, args.instrument)
