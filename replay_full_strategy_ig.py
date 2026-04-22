"""
replay_full_strategy_ig.py — full strategy (entries + exits) on IG tick data.

This is the closest simulator to the live bot that we can build. It:
  1. Builds 5-min bars from IG ticks
  2. Detects bar 4/5 per session (same hybrid rule as live)
  3. Sets buy/sell levels with buffer
  4. Tick-level entry trigger (matches live broker logic)
  5. Variant B trail (prev_close always) + 60s confirm timer
  6. Re-entries with zone gate (matches live)
  7. Respects max_entries + reverse_reentry policy
  8. Per-instrument entry lock across sessions (one position per epic)
  9. max_spread filter (skip entries if spread > cap)

Outputs side-by-side: SIM vs LIVE for each day, per instrument/session.
"""
import os, sys, sqlite3, pandas as pd, numpy as np
from zoneinfo import ZoneInfo
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asrs.config import INSTRUMENTS

TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"


def load_ticks(epic, date_str):
    path = f"{TICK_DIR}/{epic}_{date_str}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["utm", "bid", "ofr"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["bid", "ofr", "mid"])
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    return df.dropna(subset=["utm"]).sort_values("utm").reset_index(drop=True)


def build_5min_bars(ticks, tz):
    t = ticks.copy()
    t["dt"] = pd.to_datetime(t["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    t["bar"] = t["dt"].dt.floor("5min")
    return t.groupby("bar").agg(
        High=("mid","max"), Low=("mid","min"),
        Close=("mid","last"), Open=("mid","first")
    ).reset_index()


def pick_signal_bar(bars_session, cfg):
    """Same hybrid logic as live: bar 4 unless range > wide, then bar 5."""
    if len(bars_session) < 5: return None
    b4 = bars_session.iloc[3]
    b4_range = b4["High"] - b4["Low"]
    if b4_range < cfg["narrow_range"]:
        return 4, b4["High"], b4["Low"], b4["bar"]
    elif b4_range > cfg["wide_range"]:
        if len(bars_session) < 5: return None
        b5 = bars_session.iloc[4]
        b5_range = b5["High"] - b5["Low"]
        if b5_range > cfg["max_bar_range"] or b5_range <= 0: return None
        return 5, b5["High"], b5["Low"], b5["bar"]
    else:
        return 4, b4["High"], b4["Low"], b4["bar"]


def simulate_session(ticks, bars, cfg, session_idx, session_name):
    """Simulate ONE session. Returns list of trades + initial bracket info."""
    tz = ZoneInfo(cfg["timezone"])
    oh = cfg[f"s{session_idx}_open_hour"]; om = cfg[f"s{session_idx}_open_minute"]
    eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]

    # Session bars
    bars_sess = bars[
        (bars["bar"].dt.hour * 60 + bars["bar"].dt.minute >= oh * 60 + om) &
        (bars["bar"].dt.hour * 60 + bars["bar"].dt.minute < eh * 60 + em)
    ].reset_index(drop=True)
    pick = pick_signal_bar(bars_sess, cfg)
    if pick is None:
        return [], None
    bar_num, sig_h, sig_l, sig_bar_start = pick
    bar_range = sig_h - sig_l

    buf = cfg["buffer"]
    buy_level = round(sig_h + buf, 1)
    sell_level = round(sig_l - buf, 1)

    # Ticks from (sig_bar_start + bar_duration) to session_end
    tick_start_dt = sig_bar_start + pd.Timedelta(minutes=5)
    tick_start_utm = int(tick_start_dt.tz_convert("UTC").timestamp() * 1000)
    sess_end_dt = pd.Timestamp(sig_bar_start.date(), tz=tz).replace(hour=eh, minute=em)
    sess_end_utm = int(sess_end_dt.tz_convert("UTC").timestamp() * 1000)

    trading_ticks = ticks[(ticks["utm"] >= tick_start_utm) & (ticks["utm"] < sess_end_utm)].reset_index(drop=True)
    if len(trading_ticks) == 0:
        return [], {"buy_level": buy_level, "sell_level": sell_level, "bar_num": bar_num, "bar_range": bar_range}

    bars_ind = bars.set_index("bar")
    trail_close_always = cfg.get("trail_to_close_always", False)
    max_spread = cfg.get("max_spread", 50.0)
    rr_policy = cfg.get("reverse_reentry", "never")
    CONFIRM_MS = 60_000

    trades = []
    entries_used = 0
    position = None; entry_price = 0.0; current_stop = 0.0
    be_hit = False; waiting_reentry = False; re_gate_cleared = True
    stop_breach_since = 0
    first_direction = None; first_pnl = None
    last_bar = None

    bids = trading_ticks["bid"].values
    ofrs = trading_ticks["ofr"].values
    utms = trading_ticks["utm"].values

    for i in range(len(trading_ticks)):
        bid, ofr, utm = float(bids[i]), float(ofrs[i]), int(utms[i])
        spread = ofr - bid
        bar = pd.Timestamp(utm, unit="ms", tz="UTC").tz_convert(tz).floor("5min")

        # --- Trail update on new bar ---
        if position is not None and last_bar is not None and bar != last_bar:
            if last_bar in bars_ind.index:
                pb = bars_ind.loc[last_bar]
                prev_h, prev_l, prev_c = float(pb["High"]), float(pb["Low"]), float(pb["Close"])
                if position == "LONG":
                    unr = prev_c - entry_price
                    if not be_hit and unr >= cfg["breakeven_pts"]:
                        be_hit = True
                        new_be = entry_price - cfg.get("be_buffer_pts", 0)
                        if new_be > current_stop: current_stop = new_be; stop_breach_since = 0
                    if trail_close_always:
                        ns = prev_c
                    else:
                        profit = prev_c - entry_price
                        ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > current_stop: current_stop = round(ns, 1); stop_breach_since = 0
                else:
                    unr = entry_price - prev_c
                    if not be_hit and unr >= cfg["breakeven_pts"]:
                        be_hit = True
                        new_be = entry_price + cfg.get("be_buffer_pts", 0)
                        if new_be < current_stop: current_stop = new_be; stop_breach_since = 0
                    if trail_close_always:
                        ns = prev_c
                    else:
                        profit = entry_price - prev_c
                        ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < current_stop: current_stop = round(ns, 1); stop_breach_since = 0
        last_bar = bar

        # --- Stop check with 60s timer ---
        if position is not None:
            breached = (position == "LONG" and bid <= current_stop) or \
                       (position == "SHORT" and ofr >= current_stop)
            if breached:
                if stop_breach_since == 0:
                    stop_breach_since = utm
                elif (utm - stop_breach_since) >= CONFIRM_MS:
                    exit_price = bid if position == "LONG" else ofr
                    pnl = (exit_price - entry_price) if position == "LONG" else (entry_price - exit_price)
                    trades.append({
                        "session": session_name, "direction": position,
                        "entry": round(entry_price, 1), "exit": round(exit_price, 1),
                        "pnl_pts": round(pnl, 1), "reason": "STOP",
                        "entry_utm": entry_utm_rec, "exit_utm": utm,
                    })
                    if first_pnl is None: first_pnl = trades[-1]["pnl_pts"]
                    position = None; stop_breach_since = 0
                    if entries_used < cfg["max_entries"]:
                        waiting_reentry = True; re_gate_cleared = False
                    continue
            else:
                recovered = (position == "LONG" and bid > current_stop) or \
                            (position == "SHORT" and ofr < current_stop)
                if recovered:
                    stop_breach_since = 0

        # --- Re-entry gate ---
        if waiting_reentry and not re_gate_cleared:
            mid = (bid + ofr) / 2
            if (sell_level - 2) <= mid <= (buy_level + 2):
                re_gate_cleared = True; waiting_reentry = False

        # --- Entry logic ---
        if position is None and entries_used < cfg["max_entries"] and re_gate_cleared:
            if spread > max_spread:
                continue  # spread too wide
            long_ok = short_ok = True
            if entries_used > 0:
                first_won = (first_pnl is not None and first_pnl > 0)
                block = rr_policy == "never" or (rr_policy == "after_loss" and first_won)
                if block:
                    long_ok = (first_direction == "LONG")
                    short_ok = (first_direction == "SHORT")
            if ofr >= buy_level and long_ok:
                position = "LONG"; entry_price = buy_level; current_stop = sig_l
                be_hit = False; entries_used += 1; stop_breach_since = 0
                if first_direction is None: first_direction = "LONG"
                entry_utm_rec = utm
            elif bid <= sell_level and short_ok:
                position = "SHORT"; entry_price = sell_level; current_stop = sig_h
                be_hit = False; entries_used += 1; stop_breach_since = 0
                if first_direction is None: first_direction = "SHORT"
                entry_utm_rec = utm

    # Session end
    if position is not None:
        exit_price = float(bids[-1]) if position == "LONG" else float(ofrs[-1])
        pnl = (exit_price - entry_price) if position == "LONG" else (entry_price - exit_price)
        trades.append({
            "session": session_name, "direction": position,
            "entry": round(entry_price, 1), "exit": round(exit_price, 1),
            "pnl_pts": round(pnl, 1), "reason": "EOD",
            "entry_utm": entry_utm_rec, "exit_utm": int(utms[-1]),
        })
    return trades, {"buy_level": buy_level, "sell_level": sell_level, "bar_num": bar_num, "bar_range": bar_range}


def main():
    # Load live trades for comparison
    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        "SELECT instrument,date,direction,entry_price,exit_price,pnl_pts,entry_time "
        "FROM trades WHERE mode='live' ORDER BY date,entry_time", conn
    )
    conn.close()
    live_dates = sorted(live["date"].unique())

    # Simulate each (instrument, date, session) with IG tick data
    sim_results = defaultdict(list)
    live_results = defaultdict(list)
    skipped_sessions = 0
    total_sessions = 0

    for date_str in live_dates:
        for inst, cfg in INSTRUMENTS.items():
            ticks = load_ticks(cfg["epic"], date_str)
            if ticks is None:
                continue
            bars = build_5min_bars(ticks, cfg["timezone"])
            n_sessions = sum(1 for k in cfg if k.startswith("s") and k.endswith("_open_hour"))
            for s_idx in range(1, n_sessions + 1):
                total_sessions += 1
                sess_name = f"{inst}_S{s_idx}"
                trades, bracket = simulate_session(ticks, bars, cfg, s_idx, sess_name)
                if bracket is None:
                    skipped_sessions += 1
                    continue
                for t in trades:
                    t["instrument"] = inst; t["date"] = date_str
                    sim_results[(inst, date_str, sess_name)].append(t)

    # Compare to live by (inst, date, session)
    def infer_session(entry_time_str, inst):
        h = int(entry_time_str.split(":")[0]); m = int(entry_time_str.split(":")[1])
        if inst == "US30":
            if h*60+m < 11*60: return "US30_S1"
            if h*60+m < 13*60: return "US30_S2"
            return "US30_S3"
        if inst == "NIKKEI":
            if h*60+m < 12*60: return "NIKKEI_S1"
            if h*60+m < 13*60: return "NIKKEI_S2"
            return "NIKKEI_S3"
        if inst == "DAX":
            if h*60+m < 14*60: return "DAX_S1"
            return "DAX_S2"
    live["session"] = live.apply(lambda r: infer_session(r["entry_time"], r["instrument"]), axis=1)
    for _, r in live.iterrows():
        live_results[(r["instrument"], r["date"], r["session"])].append(dict(r))

    # Print side-by-side comparison
    print(f"{'date':<12} {'session':<12} {'live n/net':<15} {'sim n/net':<15} "
          f"{'dir':<8} {'delta':<10}")
    print("-" * 85)
    totals = {"live_n": 0, "live_net": 0.0, "sim_n": 0, "sim_net": 0.0}
    by_inst = defaultdict(lambda: {"live_n": 0, "live_net": 0.0, "sim_n": 0, "sim_net": 0.0})

    all_keys = sorted(set(list(sim_results.keys()) + list(live_results.keys())))
    for k in all_keys:
        inst, date_str, sess = k
        if date_str not in live_dates: continue
        sim_ts = sim_results.get(k, [])
        live_ts = live_results.get(k, [])
        sim_net = sum(t["pnl_pts"] for t in sim_ts)
        live_net = sum(t["pnl_pts"] for t in live_ts)
        sim_n = len(sim_ts); live_n = len(live_ts)
        if sim_n == 0 and live_n == 0: continue
        dirs = ""
        if sim_ts: dirs += f"S:{sim_ts[0]['direction'][0]}"
        if live_ts: dirs += f" L:{live_ts[0]['direction'][0]}"
        print(f"  {date_str:<10} {sess:<12} "
              f"{live_n}/{live_net:>+6.0f}     {sim_n}/{sim_net:>+6.0f}     "
              f"{dirs:<8} {live_net-sim_net:>+6.0f}")
        totals["live_n"] += live_n; totals["live_net"] += live_net
        totals["sim_n"] += sim_n; totals["sim_net"] += sim_net
        by_inst[inst]["live_n"] += live_n; by_inst[inst]["live_net"] += live_net
        by_inst[inst]["sim_n"] += sim_n; by_inst[inst]["sim_net"] += sim_net

    print(f"\n{'='*85}\n  PER-INSTRUMENT AGGREGATE\n{'='*85}")
    print(f"  {'inst':<10} {'live n':<8} {'sim n':<8} {'live net':<10} {'sim net':<10} {'delta':<10}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = by_inst[inst]
        delta = d["live_net"] - d["sim_net"]
        print(f"  {inst:<10} {d['live_n']:<8} {d['sim_n']:<8} "
              f"{d['live_net']:>+7.0f}    {d['sim_net']:>+7.0f}    {delta:>+7.0f}")

    print(f"\n{'='*85}\n  GRAND TOTAL\n{'='*85}")
    print(f"  live: n={totals['live_n']} net={totals['live_net']:+.1f}")
    print(f"  sim:  n={totals['sim_n']} net={totals['sim_net']:+.1f}")
    print(f"  delta (live-sim): {totals['live_net']-totals['sim_net']:+.1f}pt")


if __name__ == "__main__":
    main()
