"""
backtest.py — ASRS strategy backtest (optimized, vectorized)
=============================================================
Matches asrs/strategy.py rules exactly. Uses numpy for speed.
"""

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import time

BAR5_RULES = ["NORMAL", "WIDE"]

INSTRUMENTS = {
    "DAX": {
        "buffer": 2.0, "narrow_range": 15, "wide_range": 40,
        "max_risk_gbp": 50.0, "max_entries": 2, "max_bar_range": 120,
        "breakeven_pts": 15.0, "tight_threshold": 100.0,
        "add_trigger": 25.0, "add_max": 2,
        "s1_open_hour": 9, "s1_open_minute": 0,
        "s2_open_hour": 14, "s2_open_minute": 0,
        "session_end_hour": 17, "session_end_minute": 30,
        "timezone": "Europe/Berlin",
        "data_file": "gold_bot/ger40_m5.csv",
    },
    "US30": {
        "buffer": 5.0, "narrow_range": 30, "wide_range": 100,
        "max_risk_gbp": 50.0, "max_entries": 2, "max_bar_range": 300,
        "breakeven_pts": 20.0, "tight_threshold": 80.0,
        "add_trigger": 30.0, "add_max": 2,
        "s1_open_hour": 9, "s1_open_minute": 30,
        "s2_open_hour": 11, "s2_open_minute": 0,
        "session_end_hour": 16, "session_end_minute": 0,
        "timezone": "America/New_York",
        "data_file": "gold_bot/us30_m5.csv",
    },
    "NIKKEI": {
        "buffer": 2.0, "narrow_range": 50, "wide_range": 150,
        "max_risk_gbp": 75.0, "max_entries": 2, "max_bar_range": 250,
        "breakeven_pts": 50.0, "tight_threshold": 300.0,
        "add_trigger": 80.0, "add_max": 2,
        "s1_open_hour": 10, "s1_open_minute": 0,
        "s2_open_hour": 12, "s2_open_minute": 0,
        "session_end_hour": 15, "session_end_minute": 0,
        "timezone": "Asia/Tokyo",
        "data_file": "gold_bot/jpnidxjpy-m5-bid-2020-01-01-2026-03-19.csv",
    },
}


def load_data(filepath: str, tz_str: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("dt")
    else:
        df.index = pd.to_datetime(df.index)
    df.columns = [c.capitalize() for c in df.columns]
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    tz = ZoneInfo(tz_str)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)
    df = df[df.index.dayofweek < 5]
    # Pre-compute columns
    df["_hour"] = df.index.hour
    df["_minute"] = df.index.minute
    df["_date"] = df.index.date
    return df


def simulate_session(day_bars: np.ndarray, hours: np.ndarray, minutes: np.ndarray,
                     open_h: int, open_m: int, eod_h: int, eod_m: int,
                     cfg: dict) -> list[dict]:
    """
    Simulate one session for one day. Arrays: [Open, High, Low, Close] per bar.
    Returns list of trade dicts.
    """
    n = len(day_bars)
    if n == 0:
        return []

    buffer = cfg["buffer"]
    open_mins = open_h * 60 + open_m
    eod_mins = eod_h * 60 + eod_m

    # Compute bar numbers
    bar_mins = hours * 60 + minutes
    mins_from_open = bar_mins - open_mins
    bar_nums = mins_from_open // 5 + 1

    # Find bar 4
    bar4_mask = (mins_from_open >= 0) & (bar_nums == 4)
    bar4_idx = np.where(bar4_mask)[0]
    if len(bar4_idx) == 0:
        return []
    b4i = bar4_idx[0]
    bar4_h = day_bars[b4i, 1]  # High
    bar4_l = day_bars[b4i, 2]  # Low
    bar4_range = bar4_h - bar4_l

    # Range classification
    if bar4_range < cfg["narrow_range"]:
        range_flag = "NARROW"
    elif bar4_range > cfg["wide_range"]:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    # Bar 5 hybrid
    sig_h, sig_l = bar4_h, bar4_l
    bar_num = 4
    if range_flag in BAR5_RULES:
        bar5_mask = (mins_from_open >= 0) & (bar_nums == 5)
        bar5_idx = np.where(bar5_mask)[0]
        if len(bar5_idx) > 0:
            b5i = bar5_idx[0]
            sig_h = day_bars[b5i, 1]
            sig_l = day_bars[b5i, 2]
            bar_num = 5

    bar_range = sig_h - sig_l
    # Reclassify
    if bar_range < cfg["narrow_range"]:
        range_flag = "NARROW"
    elif bar_range > cfg["wide_range"]:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    if bar_range > cfg["max_bar_range"]:
        return []

    bar_high = round(sig_h, 1)
    bar_low = round(sig_l, 1)
    buy_level = round(bar_high + buffer, 1)
    sell_level = round(bar_low - buffer, 1)

    # Risk cap
    risk_pts = bar_range + buffer * 2
    if risk_pts > cfg["max_risk_gbp"]:
        max_stop = cfg["max_risk_gbp"]
        bar_high = round(sell_level + max_stop, 1)
        bar_low = round(buy_level - max_stop, 1)
        bar_range = max_stop

    # Get bars after signal bar
    sig_bar_mask = (mins_from_open >= 0) & (bar_nums == bar_num)
    sig_idx = np.where(sig_bar_mask)[0]
    if len(sig_idx) == 0:
        return []
    start_after = sig_idx[0] + 1
    if start_after >= n:
        return []

    # Remaining bars (as arrays for speed)
    rem_open = day_bars[start_after:, 0]
    rem_high = day_bars[start_after:, 1]
    rem_low = day_bars[start_after:, 2]
    rem_close = day_bars[start_after:, 3]
    rem_mins = bar_mins[start_after:]
    rem_n = len(rem_open)

    # Find first trigger
    first_long = -1
    first_short = -1
    for j in range(rem_n):
        if rem_mins[j] >= eod_mins:
            break
        if first_long < 0 and rem_high[j] >= buy_level:
            first_long = j
        if first_short < 0 and rem_low[j] <= sell_level:
            first_short = j
        if first_long >= 0 and first_short >= 0:
            break

    if first_long < 0 and first_short < 0:
        return []

    if first_long >= 0 and first_short >= 0:
        if first_long < first_short:
            direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
        elif first_short < first_long:
            direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
        else:
            if rem_open[first_long] >= buy_level:
                direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
            else:
                direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
    elif first_long >= 0:
        direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
    else:
        direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short

    # Simulate trades
    trades = []
    entries_used = 0
    max_entries = cfg["max_entries"]
    active = True
    breakeven_hit = False
    adds_used = 0
    add_pnl = 0.0
    last_add_price = entry
    mfe = 0.0
    waiting = False

    for j in range(start, rem_n):
        bm = rem_mins[j]
        if bm >= eod_mins:
            if active:
                ep = rem_open[j]
                pnl = (ep - entry) if direction == "LONG" else (entry - ep)
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(ep, 1), "pnl_pts": round(pnl + add_pnl, 1),
                               "mfe": round(mfe, 1), "adds": adds_used, "reason": "EOD",
                               "bar_num": bar_num, "range_flag": range_flag,
                               "bar_range": round(bar_range, 1)})
            break

        bh = rem_high[j]
        bl = rem_low[j]
        bc = rem_close[j]
        bo = rem_open[j]

        if active:
            # Stop check
            if direction == "LONG" and bl <= stop:
                pnl = stop - entry
                trades.append({"direction": direction, "entry": round(entry, 1),
                               "exit": round(stop, 1), "pnl_pts": round(pnl + add_pnl, 1),
                               "mfe": round(mfe, 1), "adds": adds_used, "reason": "STOP",
                               "bar_num": bar_num, "range_flag": range_flag,
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
                               "bar_num": bar_num, "range_flag": range_flag,
                               "bar_range": round(bar_range, 1)})
                entries_used += 1
                active = False
                waiting = entries_used < max_entries
                continue

            # MFE
            if direction == "LONG":
                m = bh - entry
                if m > mfe: mfe = m
                unrealized = bc - entry
            else:
                m = entry - bl
                if m > mfe: mfe = m
                unrealized = entry - bc

            # Breakeven
            if not breakeven_hit and unrealized >= cfg["breakeven_pts"]:
                breakeven_hit = True
                if direction == "LONG" and stop < entry:
                    stop = entry
                elif direction == "SHORT" and stop > entry:
                    stop = entry

            # Candle trail (use previous bar j-1 if exists and j > start)
            if j > start:
                prev_h = rem_high[j - 1]
                prev_l = rem_low[j - 1]
                prev_c = rem_close[j - 1]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop:
                        stop = round(ns, 1)
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop:
                        stop = round(ns, 1)

            # Adds
            if adds_used < cfg["add_max"]:
                ref = last_add_price
                pfr = (bc - ref) if direction == "LONG" else (ref - bc)
                if pfr >= cfg["add_trigger"]:
                    adds_used += 1
                    last_add_price = bc
                    breakeven_hit = True

        elif waiting:
            tl = bh >= buy_level
            ts = bl <= sell_level
            if tl and ts:
                if bo >= buy_level:
                    ts = False
                elif bo <= sell_level:
                    tl = False
                else:
                    tl = bc >= bo
                    ts = not tl

            if tl:
                direction, entry, stop = "LONG", buy_level, sell_level
                active, waiting = True, False
                breakeven_hit, adds_used, add_pnl = False, 0, 0.0
                last_add_price = entry
                mfe = max(0, bh - entry)
            elif ts:
                direction, entry, stop = "SHORT", sell_level, buy_level
                active, waiting = True, False
                breakeven_hit, adds_used, add_pnl = False, 0, 0.0
                last_add_price = entry
                mfe = max(0, entry - bl)

    return trades


def run_backtest():
    t0 = time.time()
    all_trades = []

    for inst_name, cfg in INSTRUMENTS.items():
        print(f"\n{'=' * 60}")
        print(f"  {inst_name}")
        print(f"{'=' * 60}")

        df = load_data(cfg["data_file"], cfg["timezone"])
        print(f"  Data: {df.index.min().date()} to {df.index.max().date()} ({len(df)} bars)")

        # Pre-extract numpy arrays grouped by date
        ohlc = df[["Open", "High", "Low", "Close"]].values
        hours = df["_hour"].values
        minutes = df["_minute"].values
        dates = df["_date"].values

        unique_dates = sorted(set(dates))

        for session in (1, 2):
            signal_name = f"{inst_name}_S{session}"
            open_h = cfg[f"s{session}_open_hour"]
            open_m = cfg[f"s{session}_open_minute"]
            eod_h = cfg["session_end_hour"]
            eod_m = cfg["session_end_minute"]

            signal_trades = []

            for day in unique_dates:
                mask = dates == day
                day_ohlc = ohlc[mask]
                day_hours = hours[mask]
                day_minutes = minutes[mask]

                day_trades = simulate_session(
                    day_ohlc, day_hours, day_minutes,
                    open_h, open_m, eod_h, eod_m, cfg
                )
                for t in day_trades:
                    t["date"] = str(day)
                    t["signal"] = signal_name
                    t["instrument"] = inst_name
                    signal_trades.append(t)

            if signal_trades:
                pnl = sum(t["pnl_pts"] for t in signal_trades)
                wins = sum(1 for t in signal_trades if t["pnl_pts"] > 0)
                losses = sum(1 for t in signal_trades if t["pnl_pts"] < 0)
                gw = sum(t["pnl_pts"] for t in signal_trades if t["pnl_pts"] > 0)
                gl = abs(sum(t["pnl_pts"] for t in signal_trades if t["pnl_pts"] < 0))
                pf = gw / gl if gl > 0 else float("inf")
                wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
                print(f"  {signal_name}: {len(signal_trades)} trades | PF {pf:.2f} | "
                      f"W/L {wins}/{losses} ({wr:.0f}%) | Net {pnl:+,.0f}pts")
                all_trades.extend(signal_trades)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  COMBINED RESULTS  ({elapsed:.0f}s)")
    print(f"{'=' * 60}")

    tp = sum(t["pnl_pts"] for t in all_trades)
    tw = sum(1 for t in all_trades if t["pnl_pts"] > 0)
    tl = sum(1 for t in all_trades if t["pnl_pts"] < 0)
    tf = sum(1 for t in all_trades if t["pnl_pts"] == 0)
    gw = sum(t["pnl_pts"] for t in all_trades if t["pnl_pts"] > 0)
    gl = abs(sum(t["pnl_pts"] for t in all_trades if t["pnl_pts"] < 0))
    pf = gw / gl if gl > 0 else float("inf")

    print(f"  Total trades: {len(all_trades)}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Win/Loss/Flat: {tw}/{tl}/{tf} ({tw / (tw + tl) * 100:.0f}%)")
    print(f"  Net P&L: {tp:+,.0f} pts")
    print(f"  Gross wins: {gw:,.0f} | Gross losses: {gl:,.0f}")
    if tw > 0 and tl > 0:
        print(f"  Avg win: {gw / tw:.1f} | Avg loss: {gl / tl:.1f}")

    df_t = pd.DataFrame(all_trades)
    df_t["year"] = pd.to_datetime(df_t["date"]).dt.year
    print(f"\n  Per year:")
    for year, grp in df_t.groupby("year"):
        yw = grp[grp["pnl_pts"] > 0]["pnl_pts"].sum()
        yl = abs(grp[grp["pnl_pts"] < 0]["pnl_pts"].sum())
        ypf = yw / yl if yl > 0 else float("inf")
        print(f"    {year}: {len(grp)} trades | PF {ypf:.2f} | Net {grp['pnl_pts'].sum():+,.0f}pts")

    eq = df_t["pnl_pts"].cumsum()
    dd = eq - eq.cummax()
    print(f"\n  Max drawdown: {dd.min():,.0f} pts")

    df_t.to_csv("data/backtest_results.csv", index=False)
    print(f"  Saved to data/backtest_results.csv")


if __name__ == "__main__":
    run_backtest()
