"""
backtest_wick_grid.py — 2D grid: stop-confirm time × wick depth tolerance.

For each combo, simulates variant B trail (prev_close always) + tunable
wick-forgiveness on exits. Tests on 5 days of tick data.

Exit logic per tick:
  breach = bid <= (stop - wick_depth)   # LONG; stop - 0 = any touch
  If breach and timer not running → start timer
  If breach and timer >= confirm_secs → EXIT at current bid
  If NOT breach (price recovered above stop) → reset timer
"""
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo

TICK_DIR = Path("/app/data/ticks") if Path("/app/data/ticks").exists() else Path("/root/asrs-bot/data/ticks")

INSTRUMENTS = {
    "DAX": {
        "epic": "IX.D.DAX.DAILY.IP",
        "tz": "Europe/Berlin",
        "buffer": 2.0, "narrow": 15, "wide": 40, "max_range": 120,
        "be_pts": 15.0, "be_buf": 5.0, "tight": 100.0, "max_entries": 3,
        "sessions": [(9, 0), (14, 0)], "eod": (17, 30),
        "reverse_reentry": "never",
    },
    "US30": {
        "epic": "IX.D.DOW.DAILY.IP",
        "tz": "America/New_York",
        "buffer": 5.0, "narrow": 30, "wide": 100, "max_range": 300,
        "be_pts": 20.0, "be_buf": 5.0, "tight": 80.0, "max_entries": 3,
        "sessions": [(9, 30), (11, 0), (13, 0)], "eod": (16, 0),
        "reverse_reentry": "never",
    },
    "NIKKEI": {
        "epic": "IX.D.NIKKEI.DAILY.IP",
        "tz": "Asia/Tokyo",
        "buffer": 2.0, "narrow": 50, "wide": 150, "max_range": 250,
        "be_pts": 50.0, "be_buf": 10.0, "tight": 300.0, "max_entries": 3,
        "sessions": [(10, 0), (12, 0), (13, 0)], "eod": (15, 0),
        "reverse_reentry": "after_loss",
    },
}


def load_ticks(epic, date_str, tz):
    path = TICK_DIR / f"{epic}_{date_str}.csv"
    if not path.exists(): return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["utm", "bid", "ofr"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["bid", "ofr", "mid"])
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["utm"])
    df["dt"] = pd.to_datetime(df["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def build_bars(ticks):
    ticks = ticks.copy()
    ticks["bar_start"] = ticks["dt"].dt.floor("5min")
    bars = ticks.groupby("bar_start").agg(
        Open=("mid", "first"), High=("mid", "max"),
        Low=("mid", "min"), Close=("mid", "last")
    ).reset_index()
    return bars


def simulate_session(cfg, ticks, bars, session_open, session_num,
                     confirm_secs, wick_depth, trail_close):
    """Simulate one session with specified confirm timer + wick depth."""
    trades = []
    oh, om = session_open
    open_m = oh*60+om
    eod_m = cfg["eod"][0]*60 + cfg["eod"][1]
    bars["hm"] = bars["bar_start"].dt.hour*60 + bars["bar_start"].dt.minute
    session_bars = bars[(bars["hm"] >= open_m) & (bars["hm"] < eod_m)].reset_index(drop=True)
    if len(session_bars) < 5: return trades

    bar4 = session_bars.iloc[3]
    bar4_range = bar4["High"] - bar4["Low"]
    if bar4_range < cfg["narrow"] and bar4_range > 0:
        sig_h, sig_l = bar4["High"], bar4["Low"]; sig_bar_start = bar4["bar_start"]
    else:
        if len(session_bars) < 5: return trades
        bar5 = session_bars.iloc[4]
        sig_h, sig_l = bar5["High"], bar5["Low"]; sig_bar_start = bar5["bar_start"]
        if sig_h - sig_l > cfg["max_range"] or sig_h - sig_l <= 0: return trades
    if sig_h - sig_l > cfg["max_range"] or sig_h - sig_l <= 0: return trades

    buy_lv = round(float(sig_h) + cfg["buffer"], 1)
    sell_lv = round(float(sig_l) - cfg["buffer"], 1)
    sig_h_f, sig_l_f = float(sig_h), float(sig_l)
    tick_start = sig_bar_start + pd.Timedelta(minutes=5)
    session_tz = sig_bar_start.tz
    session_end = pd.Timestamp(sig_bar_start.date(), tz=session_tz) + pd.Timedelta(
        hours=cfg["eod"][0], minutes=cfg["eod"][1])

    trading = ticks[(ticks["dt"] >= tick_start) & (ticks["dt"] < session_end)].reset_index(drop=True)
    if len(trading) == 0: return trades
    bars_idx = bars.set_index("bar_start")

    entries_used, max_entries = 0, cfg["max_entries"]
    position = None; entry_price = 0.0; current_stop = 0.0
    be_hit = False; waiting_reentry = False; re_gate_cleared = True
    last_bar_processed = None; stop_breach_since = 0.0
    first_direction, first_pnl = None, None
    rr_policy = cfg.get("reverse_reentry", "always")

    bids = trading["bid"].values; ofrs = trading["ofr"].values; dts = trading["dt"].values

    for i in range(len(trading)):
        bid, ofr, dt = bids[i], ofrs[i], dts[i]
        bar_start = pd.Timestamp(dt).floor("5min")
        if pd.Timestamp(dt).tz is None:
            bar_start = bar_start.tz_localize(bars_idx.index.tz)

        # Trail on new bar
        if position is not None and last_bar_processed is not None and bar_start != last_bar_processed:
            if last_bar_processed in bars_idx.index:
                pb = bars_idx.loc[last_bar_processed]
                prev_h, prev_l, prev_c = float(pb["High"]), float(pb["Low"]), float(pb["Close"])
                if position == "LONG":
                    unr = prev_c - entry_price
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry_price - cfg["be_buf"]
                        if new_be > current_stop: current_stop = new_be; stop_breach_since = 0
                    prof = prev_c - entry_price
                    if trail_close:
                        ns = prev_c
                    else:
                        ns = prev_c if prof >= cfg["tight"] else prev_l
                    if ns > current_stop: current_stop = round(ns, 1); stop_breach_since = 0
                else:
                    unr = entry_price - prev_c
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry_price + cfg["be_buf"]
                        if new_be < current_stop: current_stop = new_be; stop_breach_since = 0
                    prof = entry_price - prev_c
                    if trail_close:
                        ns = prev_c
                    else:
                        ns = prev_c if prof >= cfg["tight"] else prev_h
                    if ns < current_stop: current_stop = round(ns, 1); stop_breach_since = 0
        last_bar_processed = bar_start

        # Stop check with 60s timer + PANIC-DEPTH bypass
        # wick_depth param = panic depth: if price goes this far past stop, bypass timer
        if position is not None:
            tick_ts = pd.Timestamp(dt).timestamp()
            # Any breach of stop starts timer
            breached = False
            panic = False
            if position == "LONG":
                if bid <= current_stop:
                    breached = True
                    if wick_depth > 0 and bid <= (current_stop - wick_depth):
                        panic = True
            elif position == "SHORT":
                if ofr >= current_stop:
                    breached = True
                    if wick_depth > 0 and ofr >= (current_stop + wick_depth):
                        panic = True

            if breached:
                if stop_breach_since == 0:
                    stop_breach_since = tick_ts
                elif panic or (tick_ts - stop_breach_since) >= confirm_secs:
                    # Confirmed — exit at current bid/ofr
                    exit_price = float(bid) if position == "LONG" else float(ofr)
                    if position == "LONG":
                        trades.append({"pnl_pts": round(exit_price-entry_price, 1),
                                       "direction": "LONG", "entry": entry_price, "exit": exit_price,
                                       "reason": "STOP", "session": f"S{session_num}"})
                    else:
                        trades.append({"pnl_pts": round(entry_price-exit_price, 1),
                                       "direction": "SHORT", "entry": entry_price, "exit": exit_price,
                                       "reason": "STOP", "session": f"S{session_num}"})
                    if first_pnl is None: first_pnl = trades[-1]["pnl_pts"]
                    position = None; stop_breach_since = 0
                    if entries_used < max_entries:
                        waiting_reentry = True; re_gate_cleared = False
                    continue
            else:
                # Price recovered above stop (not just above stop-depth): reset timer
                recovered = (position == "LONG" and bid > current_stop) or \
                            (position == "SHORT" and ofr < current_stop)
                if recovered:
                    stop_breach_since = 0

        # Re-entry gate
        if waiting_reentry and not re_gate_cleared:
            mid = (bid + ofr) / 2
            if (sell_lv - 2) <= mid <= (buy_lv + 2):
                re_gate_cleared = True; waiting_reentry = False

        # Entry
        if position is None and entries_used < max_entries and re_gate_cleared:
            if entries_used == 0:
                long_ok = short_ok = True
            else:
                first_won = (first_pnl is not None and first_pnl > 0)
                block = rr_policy == "never" or (rr_policy == "after_loss" and first_won)
                if block:
                    long_ok = first_direction == "LONG"
                    short_ok = first_direction == "SHORT"
                else: long_ok = short_ok = True
            if ofr >= buy_lv and long_ok:
                position = "LONG"; entry_price = buy_lv; current_stop = sig_l_f
                be_hit = False; entries_used += 1
                if first_direction is None: first_direction = "LONG"
                stop_breach_since = 0
            elif bid <= sell_lv and short_ok:
                position = "SHORT"; entry_price = sell_lv; current_stop = sig_h_f
                be_hit = False; entries_used += 1
                if first_direction is None: first_direction = "SHORT"
                stop_breach_since = 0

    if position is not None:
        exit_price = float(bids[-1]) if position == "LONG" else float(ofrs[-1])
        pnl = (exit_price-entry_price) if position == "LONG" else (entry_price-exit_price)
        trades.append({"pnl_pts": round(pnl, 1), "direction": position,
                       "entry": entry_price, "exit": exit_price,
                       "reason": "EOD", "session": f"S{session_num}"})
    return trades


def run_all(confirm_secs, wick_depth, trail_close):
    """Run all instruments, all dates, all sessions — return flat trade list."""
    all_trades = []
    for inst, cfg in INSTRUMENTS.items():
        pattern = f"{cfg['epic']}_*.csv"
        files = sorted(TICK_DIR.glob(pattern))
        for f in files:
            date_str = f.stem.split("_")[-1]
            ticks = load_ticks(cfg['epic'], date_str, cfg["tz"])
            if ticks is None or len(ticks) < 100: continue
            bars = build_bars(ticks)
            if len(bars) < 10: continue
            for sn, session in enumerate(cfg["sessions"], start=1):
                ts = simulate_session(cfg, ticks, bars, session, sn,
                                      confirm_secs, wick_depth, trail_close)
                all_trades.extend(ts)
    return all_trades


def summary(trades):
    if not trades: return 0, 0.0, 0.0
    net = sum(t["pnl_pts"] for t in trades)
    wins = sum(t["pnl_pts"] for t in trades if t["pnl_pts"] > 0)
    losses = sum(t["pnl_pts"] for t in trades if t["pnl_pts"] < 0)
    pf = wins / abs(losses) if losses < 0 else float("inf")
    return len(trades), pf, net


def main():
    import time as _t
    # Smaller grid — focus on key points
    times = [60]  # standard timer, test panic-depth instead
    depths = [0, 5, 10, 15, 20, 30]  # panic-bypass thresholds
    results = {}
    for trail_label, trail_close in [("A prev_low", False), ("B prev_close", True)]:
        print(f"\n=== Trail: {trail_label} ===", flush=True)
        for t_s in times:
            for d in depths:
                t0 = _t.time()
                trades = run_all(confirm_secs=t_s, wick_depth=d, trail_close=trail_close)
                n, pf, net = summary(trades)
                results[(trail_label, t_s, d)] = (n, pf, net)
                pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
                print(f"  [{trail_label}] t={t_s}s d={d}pt  n={n:>3}  PF={pf_str}  net={net:>+5.0f}  ({_t.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*90}\n  GRID SUMMARY\n{'='*90}", flush=True)
    for trail_label, _ in [("A prev_low", False), ("B prev_close", True)]:
        print(f"\n  Trail: {trail_label}", flush=True)
        hdr = f"    {'time\\depth':<14}" + "".join(f"{d:>14}pt" for d in depths)
        print(hdr, flush=True)
        for t_s in times:
            row = f"    {t_s}s{'':<11}"
            for d in depths:
                _, pf, net = results[(trail_label, t_s, d)]
                pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
                row += f"  PF={pf_s:<5} ({net:+5.0f})"
            print(row, flush=True)


if __name__ == "__main__":
    main()
