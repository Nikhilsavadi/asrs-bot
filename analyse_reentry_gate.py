"""
analyse_reentry_gate.py — compare TICK vs BAR-CLOSE entry gates on 12 days of live data.

Isolates the effect of the re-entry gate semantics by:
  1. Reconstructing bracket levels per (instrument, date, session) from first live trade.
  2. Replaying all entries under two gates:
       TICK: current live (fires on any tick crossing, 2pt return-to-zone re-arm)
       BAR1: bar-close gate (1-min bar must close past level to trigger)
       BAR5: bar-close gate (5-min bar must close past level to trigger)
  3. Same Variant B exit logic (prev_close trail + 60s timer) for all gates.
  4. Respecting reverse_reentry policy per instrument.

Output: per-instrument comparison of trade count, net pnl, PF between gates.
"""
import os, sys, sqlite3, pandas as pd, numpy as np
from zoneinfo import ZoneInfo
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"

CFG = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",
               "buffer": 2.0, "be_pts": 15.0, "be_buf": 5.0,
               "max_entries": 3, "reverse_reentry": "never",
               "sessions": [("S1", 9, 0), ("S2", 14, 0)], "eod": (17, 30)},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York",
               "buffer": 5.0, "be_pts": 20.0, "be_buf": 5.0,
               "max_entries": 3, "reverse_reentry": "never",
               "sessions": [("S1", 9, 30), ("S2", 11, 0), ("S3", 13, 0)], "eod": (16, 0)},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",
               "buffer": 2.0, "be_pts": 50.0, "be_buf": 10.0,
               "max_entries": 3, "reverse_reentry": "after_loss",
               "sessions": [("S1", 10, 0), ("S2", 12, 0), ("S3", 13, 0)], "eod": (15, 0)},
}


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


def build_bars(ticks, tz, freq):
    t = ticks.copy()
    t["dt"] = pd.to_datetime(t["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    t["bar"] = t["dt"].dt.floor(freq)
    return t.groupby("bar").agg(
        High=("mid","max"), Low=("mid","min"),
        Close=("mid","last"), Open=("mid","first")
    ).reset_index()


def infer_session(entry_time_str, cfg):
    h, m = int(entry_time_str.split(":")[0]), int(entry_time_str.split(":")[1])
    sess_name = None
    for name, oh, om in cfg["sessions"]:
        if h*60 + m >= oh*60 + om:
            sess_name = name
    return sess_name


def get_session_levels(live_trades, inst, date_str, cfg):
    """
    For each session with a trade, reconstruct the bracket levels from first trade.
    Returns dict: {session_name: (buy_level, sell_level, sig_h, sig_l, open_time)}
    """
    by_session = {}
    by_session_ordered = sorted(live_trades, key=lambda t: t["entry_time"])
    for tr in by_session_ordered:
        if tr["instrument"] != inst or tr["date"] != date_str:
            continue
        sess = infer_session(tr["entry_time"], cfg)
        if sess in by_session:
            continue  # already have first trade
        buf = cfg["buffer"]
        bar_range = float(tr.get("bar_range") or 0)
        entry_intended = float(tr.get("entry_intended") or tr["entry_price"])
        if tr["direction"] == "LONG":
            sig_h = entry_intended - buf
            sig_l = sig_h - bar_range
        else:
            sig_l = entry_intended + buf
            sig_h = sig_l + bar_range
        buy_level = round(sig_h + buf, 1)
        sell_level = round(sig_l - buf, 1)
        # Session open time
        for name, oh, om in cfg["sessions"]:
            if name == sess:
                open_time = (oh, om)
                break
        by_session[sess] = {
            "buy_level": buy_level, "sell_level": sell_level,
            "sig_h": sig_h, "sig_l": sig_l,
            "open_time": open_time,
        }
    return by_session


def simulate_session(ticks, bars1, bars5, bar_cls_5, cfg, levels_info, sess_info, gate):
    """
    Simulate one session with specified gate.
    gate: "TICK" | "BAR1" | "BAR5"
    Returns list of trades.
    """
    buy_level = levels_info["buy_level"]
    sell_level = levels_info["sell_level"]
    sig_h = levels_info["sig_h"]
    sig_l = levels_info["sig_l"]
    buf = cfg["buffer"]

    trades = []
    entries_used = 0
    position = None
    entry_price = 0.0
    current_stop = 0.0
    be_hit = False
    waiting_reentry = False
    re_gate_cleared = True  # first entry: gate is clear
    stop_breach_since = 0
    first_direction = None
    first_pnl = None
    last_bar1 = None
    last_bar5 = None

    # Pending-gate tracking (for BAR1/BAR5)
    pending_long = False
    pending_short = False
    pending_bar = None

    # Prev-1min bars for trail (Variant B uses 5-min bars but we still need to
    # build them from ticks). Here we use 5-min for trail.
    bars5_ind = bars5.set_index("bar")

    CONFIRM_MS = 60_000
    rr_policy = cfg["reverse_reentry"]

    bids = ticks["bid"].values
    ofrs = ticks["ofr"].values
    utms = ticks["utm"].values
    tz = ZoneInfo(cfg["tz"])

    # Build 1-min bars for BAR1 gate decisions
    bars1_ind = bars1.set_index("bar")
    bar1_keys = list(bars1_ind.index)
    bar5_keys = list(bars5_ind.index)

    for i in range(len(ticks)):
        bid, ofr, utm = float(bids[i]), float(ofrs[i]), int(utms[i])
        dt = pd.Timestamp(utm, unit="ms", tz="UTC").tz_convert(tz)
        bar1 = dt.floor("1min")
        bar5 = dt.floor("5min")

        # --- Trail update on new 5-min bar ---
        if position is not None and last_bar5 is not None and bar5 != last_bar5:
            if last_bar5 in bars5_ind.index:
                prev_c = float(bars5_ind.loc[last_bar5]["Close"])
                if position == "LONG":
                    unr = prev_c - entry_price
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry_price - cfg["be_buf"]
                        if new_be > current_stop: current_stop = new_be; stop_breach_since = 0
                    if prev_c > current_stop: current_stop = round(prev_c, 1); stop_breach_since = 0
                else:
                    unr = entry_price - prev_c
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry_price + cfg["be_buf"]
                        if new_be < current_stop: current_stop = new_be; stop_breach_since = 0
                    if prev_c < current_stop: current_stop = round(prev_c, 1); stop_breach_since = 0

        # --- Pending-gate resolution on bar close (BAR1 and BAR5) ---
        gate_bar = bar1 if gate == "BAR1" else (bar5 if gate == "BAR5" else None)
        if gate in ("BAR1", "BAR5") and pending_bar is not None and gate_bar != pending_bar:
            # The pending bar has just completed — check its close vs level
            bars_use = bars1_ind if gate == "BAR1" else bars5_ind
            if pending_bar in bars_use.index:
                pb_close = float(bars_use.loc[pending_bar]["Close"])
                if pending_long and pb_close >= buy_level:
                    # Trigger LONG at close price
                    position = "LONG"; entry_price = pb_close; current_stop = sig_l
                    be_hit = False; entries_used += 1; stop_breach_since = 0
                    if first_direction is None: first_direction = "LONG"
                elif pending_short and pb_close <= sell_level:
                    position = "SHORT"; entry_price = pb_close; current_stop = sig_h
                    be_hit = False; entries_used += 1; stop_breach_since = 0
                    if first_direction is None: first_direction = "SHORT"
            pending_long = False; pending_short = False; pending_bar = None

        last_bar1 = bar1; last_bar5 = bar5

        # --- Stop check with 60s timer (Variant B exit) ---
        if position is not None:
            breached = (position == "LONG" and bid <= current_stop) or \
                       (position == "SHORT" and ofr >= current_stop)
            if breached:
                if stop_breach_since == 0:
                    stop_breach_since = utm
                elif (utm - stop_breach_since) >= CONFIRM_MS:
                    exit_price = bid if position == "LONG" else ofr
                    pnl = (exit_price - entry_price) if position == "LONG" else (entry_price - exit_price)
                    trades.append({"direction": position, "entry": round(entry_price,1),
                                   "exit": round(exit_price,1), "pnl_pts": round(pnl,1),
                                   "reason": "STOP"})
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

        # --- Re-entry gate (return to bracket zone) ---
        if waiting_reentry and not re_gate_cleared:
            mid = (bid + ofr) / 2
            if (sell_level - 2) <= mid <= (buy_level + 2):
                re_gate_cleared = True; waiting_reentry = False

        # --- Entry logic (respecting gate + reverse_reentry) ---
        if position is None and entries_used < cfg["max_entries"] and re_gate_cleared:
            # Determine direction allowed by reverse_reentry
            long_ok = short_ok = True
            if entries_used > 0:
                first_won = (first_pnl is not None and first_pnl > 0)
                block = rr_policy == "never" or (rr_policy == "after_loss" and first_won)
                if block:
                    long_ok = (first_direction == "LONG")
                    short_ok = (first_direction == "SHORT")

            if gate == "TICK":
                # Fire immediately on tick crossing
                if ofr >= buy_level and long_ok:
                    position = "LONG"; entry_price = buy_level; current_stop = sig_l
                    be_hit = False; entries_used += 1; stop_breach_since = 0
                    if first_direction is None: first_direction = "LONG"
                elif bid <= sell_level and short_ok:
                    position = "SHORT"; entry_price = sell_level; current_stop = sig_h
                    be_hit = False; entries_used += 1; stop_breach_since = 0
                    if first_direction is None: first_direction = "SHORT"
            else:  # BAR1 or BAR5
                # Detect pending cross — set flag to check at bar close
                if pending_long or pending_short:
                    pass  # already pending
                else:
                    if ofr >= buy_level and long_ok:
                        pending_long = True; pending_bar = gate_bar
                    elif bid <= sell_level and short_ok:
                        pending_short = True; pending_bar = gate_bar

    # Session end: force exit
    if position is not None:
        exit_price = float(bids[-1]) if position == "LONG" else float(ofrs[-1])
        pnl = (exit_price - entry_price) if position == "LONG" else (entry_price - exit_price)
        trades.append({"direction": position, "entry": round(entry_price,1),
                       "exit": round(exit_price,1), "pnl_pts": round(pnl,1),
                       "reason": "EOD"})
    return trades


def main():
    # Load live trades
    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        "SELECT id, instrument, date, direction, entry_price, entry_intended, "
        "exit_price, pnl_pts, entry_time, bar_range "
        "FROM trades WHERE mode='live' ORDER BY date, entry_time",
        conn
    )
    conn.close()
    live_rows = live.to_dict("records")

    # Group by (instrument, date, session) based on entry_time
    sessions = defaultdict(list)
    for tr in live_rows:
        sess = infer_session(tr["entry_time"], CFG[tr["instrument"]])
        sessions[(tr["instrument"], tr["date"], sess)].append(tr)

    print(f"Sessions with live trades: {len(sessions)}")

    # Simulate each session under each gate
    totals = {g: defaultdict(lambda: {"n": 0, "net": 0.0}) for g in ["LIVE", "TICK", "BAR1", "BAR5"]}

    skipped = 0
    for (inst, date_str, sess_name), trs in sessions.items():
        cfg = CFG[inst]
        ticks = load_ticks(cfg["epic"], date_str)
        if ticks is None:
            skipped += 1
            continue

        # Narrow ticks to session hours
        tz = ZoneInfo(cfg["tz"])
        # Session start: first live trade's session_open
        sess_open_time = None
        for name, oh, om in cfg["sessions"]:
            if name == sess_name:
                sess_open_time = (oh, om); break
        sess_end_time = cfg["eod"]
        session_start_ts = pd.Timestamp(date_str, tz=tz).replace(
            hour=sess_open_time[0], minute=sess_open_time[1]).tz_convert("UTC").timestamp() * 1000
        session_end_ts = pd.Timestamp(date_str, tz=tz).replace(
            hour=sess_end_time[0], minute=sess_end_time[1]).tz_convert("UTC").timestamp() * 1000
        # Use ticks from 20 min before session open (for bar 4 formation) to eod
        tick_start_ts = session_start_ts - 20*60*1000
        sess_ticks = ticks[(ticks["utm"] >= tick_start_ts) & (ticks["utm"] < session_end_ts)].reset_index(drop=True)
        if len(sess_ticks) == 0:
            skipped += 1
            continue

        bars1 = build_bars(sess_ticks, cfg["tz"], "1min")
        bars5 = build_bars(sess_ticks, cfg["tz"], "5min")

        # Levels from first live trade of this session
        first_tr = trs[0]
        buf = cfg["buffer"]
        bar_range = float(first_tr.get("bar_range") or 0)
        entry_intended = float(first_tr.get("entry_intended") or first_tr["entry_price"])
        if first_tr["direction"] == "LONG":
            sig_h = entry_intended - buf; sig_l = sig_h - bar_range
        else:
            sig_l = entry_intended + buf; sig_h = sig_l + bar_range
        levels_info = {"buy_level": round(sig_h+buf,1), "sell_level": round(sig_l-buf,1),
                       "sig_h": sig_h, "sig_l": sig_l}

        # Keep only ticks from session open onwards for simulation (levels already known)
        sim_ticks = sess_ticks[sess_ticks["utm"] >= session_start_ts].reset_index(drop=True)
        if len(sim_ticks) == 0:
            skipped += 1
            continue

        # Live actuals (from DB)
        live_net = sum(float(t["pnl_pts"]) for t in trs)
        totals["LIVE"][inst]["n"] += len(trs)
        totals["LIVE"][inst]["net"] += live_net

        for gate in ["TICK", "BAR1", "BAR5"]:
            sim_trades = simulate_session(sim_ticks, bars1, bars5, None, cfg,
                                          levels_info, sess_name, gate)
            totals[gate][inst]["n"] += len(sim_trades)
            totals[gate][inst]["net"] += sum(t["pnl_pts"] for t in sim_trades)

    print(f"\nSkipped {skipped} sessions (no tick data)")
    print(f"\n{'='*80}")
    print(f"  ENTRY GATE COMPARISON (all exits use Variant B + 60s timer)")
    print(f"{'='*80}")
    print(f"  {'inst':<10} {'gate':<8} {'n':<6} {'net_pts':<10} {'Δ vs LIVE':<10}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        live_net = totals["LIVE"][inst]["net"]
        live_n = totals["LIVE"][inst]["n"]
        print(f"  {inst:<10} LIVE     {live_n:<6} {live_net:>+7.1f}")
        for gate in ["TICK", "BAR1", "BAR5"]:
            g = totals[gate][inst]
            delta = g["net"] - live_net
            print(f"  {'':<10} {gate:<8} {g['n']:<6} {g['net']:>+7.1f}    {delta:>+7.1f}")

    # Grand totals
    print(f"\n{'='*80}\n  ALL INSTRUMENTS\n{'='*80}")
    for gate in ["LIVE", "TICK", "BAR1", "BAR5"]:
        n = sum(totals[gate][i]["n"] for i in ["DAX","US30","NIKKEI"])
        net = sum(totals[gate][i]["net"] for i in ["DAX","US30","NIKKEI"])
        print(f"  {gate:<8} n={n:<5} net={net:>+8.1f}")


if __name__ == "__main__":
    main()
