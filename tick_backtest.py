"""
tick_backtest.py — Run ASRS strategy on real IG DFB tick data.

Reads tick CSVs from /app/data/ticks/, reconstructs 5-min bars from mid price
(matches the live bot's tick-bar builder), then simulates the strategy:
- Entry/stop checks at TICK level (bid for LONG stops, ofr for SHORT stops)
- Trail updates on 5-min bar close (matches live bot monitor_cycle semantics)
- Bar 4/5 hybrid signal detection
- Max 3 entries per session with re-entry gate

This is the most faithful backtest possible for the live bot on IG DFB.

Usage:
    python3 tick_backtest.py              # all instruments, all available days
    python3 tick_backtest.py DAX          # one instrument
    python3 tick_backtest.py --date 2026-04-10
"""
import argparse
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TICK_DIR = Path("/app/data/ticks")
# Fall back to host path if not in container
if not TICK_DIR.exists():
    TICK_DIR = Path("/root/asrs-bot/data/ticks")

INSTRUMENTS = {
    "DAX": {
        "epic": "IX.D.DAX.DAILY.IP",
        "tz": "Europe/Berlin",
        "buffer": 2.0, "narrow": 15, "wide": 40, "max_range": 120,
        "be_pts": 15.0, "be_buf": 5.0, "tight": 100.0, "max_entries": 3,
        "sessions": [(9, 0), (14, 0)], "eod": (17, 30),
        "reverse_reentry": "never",  # matches live config
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
        "reverse_reentry": "after_loss",  # block opposite only after winning trade #1
    },
}


def load_ticks(epic: str, date_str: str, tz: str) -> pd.DataFrame | None:
    """Load tick CSV for a given date."""
    path = TICK_DIR / f"{epic}_{date_str}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["utm", "bid", "ofr"])
    if len(df) == 0:
        return None
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["bid", "ofr", "mid"])
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["utm"])
    df["dt"] = pd.to_datetime(df["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    df = df.sort_values("dt").reset_index(drop=True)
    return df


IG_BARS_DIR = Path("/app/data/ig_bars")
if not IG_BARS_DIR.exists():
    IG_BARS_DIR = Path("/root/asrs-bot/data/ig_bars")


def load_ig_bars(epic: str, date_str: str, tz: str) -> pd.DataFrame | None:
    """Load IG-native 5min bars (LTP-based) from sidecar CSV if available.
    Matches what the live bot actually received from IG's CHART:5MIN stream.
    Returns None if no sidecar exists — caller should fall back to tick bars.
    """
    p = IG_BARS_DIR / f"{epic}_{date_str}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["bar_start"] = pd.to_datetime(df["bar_start_cet"]).dt.tz_convert(ZoneInfo(tz))
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close"})
    df["n_ticks"] = 1
    return df[["bar_start", "Open", "High", "Low", "Close", "n_ticks"]].reset_index(drop=True)


def build_5min_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticks into 5-min OHLC bars using mid price.
    Fallback when no IG sidecar bars are available for this date.
    """
    ticks = ticks.copy()
    ticks["bar_start"] = ticks["dt"].dt.floor("5min")
    bars = ticks.groupby("bar_start").agg(
        Open=("mid", "first"),
        High=("mid", "max"),
        Low=("mid", "min"),
        Close=("mid", "last"),
        n_ticks=("mid", "count"),
    ).reset_index()
    return bars


def simulate_session(cfg: dict, ticks: pd.DataFrame, bars: pd.DataFrame,
                     session_open: tuple, session_num: int) -> list:
    """Simulate one session (e.g., DAX_S1 on one day)."""
    trades = []
    oh, om = session_open
    open_m = oh * 60 + om
    eod_m = cfg["eod"][0] * 60 + cfg["eod"][1]

    # Filter bars to this session's window
    bars["hm"] = bars["bar_start"].dt.hour * 60 + bars["bar_start"].dt.minute
    session_bars = bars[(bars["hm"] >= open_m) & (bars["hm"] < eod_m)].reset_index(drop=True)

    if len(session_bars) < 5:
        return trades  # Need at least bar 4 and bar 5

    bar4 = session_bars.iloc[3]
    bar4_range = bar4["High"] - bar4["Low"]

    # Signal bar selection (matches asrs/strategy.py hybrid logic)
    use_bar4 = False
    if bar4_range < cfg["narrow"] and bar4_range > 0:
        sig_h, sig_l = bar4["High"], bar4["Low"]
        use_bar4 = True
        signal_bar_start = bar4["bar_start"]
    else:
        if len(session_bars) < 5:
            return trades
        bar5 = session_bars.iloc[4]
        sig_h, sig_l = bar5["High"], bar5["Low"]
        signal_bar_start = bar5["bar_start"]
        if sig_h - sig_l > cfg["max_range"] or sig_h - sig_l <= 0:
            return trades

    if sig_h - sig_l > cfg["max_range"] or sig_h - sig_l <= 0:
        return trades

    buy_lv = round(float(sig_h) + cfg["buffer"], 1)
    sell_lv = round(float(sig_l) - cfg["buffer"], 1)
    sig_h_f, sig_l_f = float(sig_h), float(sig_l)

    # Trading starts after the signal bar closes
    tick_start = signal_bar_start + pd.Timedelta(minutes=5)

    # Session end time (same date, eod hour/min in session tz)
    session_tz = signal_bar_start.tz
    sess_date = signal_bar_start.date()
    session_end = pd.Timestamp(sess_date, tz=session_tz) + pd.Timedelta(
        hours=cfg["eod"][0], minutes=cfg["eod"][1]
    )

    # Get ticks in trading window
    trading = ticks[(ticks["dt"] >= tick_start) & (ticks["dt"] < session_end)].reset_index(drop=True)
    if len(trading) == 0:
        return trades

    # Pre-index bars by bar_start for trail lookups
    bars_idx = bars.set_index("bar_start")

    # State machine
    entries_used = 0
    max_entries = cfg["max_entries"]
    position = None
    entry_price = 0.0
    current_stop = 0.0
    be_hit = False
    waiting_reentry = False
    re_gate_cleared = True
    last_bar_processed = None
    stop_breach_since = 0.0  # timestamp of first breach (0 = none)
    STOP_CONFIRM_SECS = 60   # 1-minute confirmation timer
    first_direction = None   # direction of trade #1 (for reverse-reentry filter)
    first_pnl = None         # pnl of trade #1 (for "after_loss" policy)
    rr_policy = cfg.get("reverse_reentry", "always")
    # MFE hybrid trail (only used when exit_mode="mfe_hybrid")
    exit_mode = cfg.get("exit_mode", "candle")
    mfe_retrace = cfg.get("mfe_retrace", 0.5)
    mfe_activate_mult = cfg.get("mfe_activate_mult", 0.5)
    mfe_peak = 0.0  # peak high (LONG) or trough low (SHORT) since entry
    initial_stop_dist = 0.0  # set at each entry

    bids = trading["bid"].values
    ofrs = trading["ofr"].values
    dts = trading["dt"].values

    for i in range(len(trading)):
        bid = bids[i]
        ofr = ofrs[i]
        dt = dts[i]
        bar_start = pd.Timestamp(dt).floor("5min").tz_localize(
            bars_idx.index.tz) if pd.Timestamp(dt).tz is None else pd.Timestamp(dt).floor("5min")

        # On NEW 5-min bar: apply trail update using previous bar's OHLC
        if position is not None and last_bar_processed is not None and bar_start != last_bar_processed:
            if last_bar_processed in bars_idx.index:
                pb = bars_idx.loc[last_bar_processed]
                prev_h = float(pb["High"])
                prev_l = float(pb["Low"])
                prev_c = float(pb["Close"])

                if position == "LONG":
                    unr = prev_c - entry_price
                    # Track MFE peak using prev bar's high
                    if prev_h > mfe_peak:
                        mfe_peak = prev_h
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry_price - cfg["be_buf"]
                        if new_be > current_stop:
                            current_stop = new_be
                            stop_breach_since = 0
                    prof = prev_c - entry_price
                    if cfg.get("trail_close_always", False):
                        ns = prev_c
                    else:
                        ns = prev_c if prof >= cfg["tight"] else prev_l
                    if ns > current_stop:
                        current_stop = round(ns, 1)
                        stop_breach_since = 0  # reset timer on trail tighten
                    # MFE hybrid trail — take max of candle trail and MFE trail
                    if exit_mode == "mfe_hybrid":
                        mfe = mfe_peak - entry_price
                        if mfe >= mfe_activate_mult * initial_stop_dist:
                            mfe_stop = mfe_peak - mfe_retrace * mfe
                            if mfe_stop > current_stop:
                                current_stop = round(mfe_stop, 1)
                                stop_breach_since = 0
                else:  # SHORT
                    unr = entry_price - prev_c
                    if mfe_peak == 0 or prev_l < mfe_peak:
                        mfe_peak = prev_l
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry_price + cfg["be_buf"]
                        if new_be < current_stop:
                            current_stop = new_be
                            stop_breach_since = 0
                    prof = entry_price - prev_c
                    if cfg.get("trail_close_always", False):
                        ns = prev_c
                    else:
                        ns = prev_c if prof >= cfg["tight"] else prev_h
                    if ns < current_stop:
                        current_stop = round(ns, 1)
                        stop_breach_since = 0  # reset timer on trail tighten
                    if exit_mode == "mfe_hybrid":
                        mfe = entry_price - mfe_peak
                        if mfe >= mfe_activate_mult * initial_stop_dist:
                            mfe_stop = mfe_peak + mfe_retrace * mfe
                            if mfe_stop < current_stop:
                                current_stop = round(mfe_stop, 1)
                                stop_breach_since = 0

        last_bar_processed = bar_start

        # Tick-level stop check with 60-second confirmation timer.
        # Filters DFB spread noise: breach must persist for STOP_CONFIRM_SECS.
        if position is not None:
            breached = False
            if position == "LONG" and bid <= current_stop:
                breached = True
            elif position == "SHORT" and ofr >= current_stop:
                breached = True

            tick_ts = pd.Timestamp(dt).timestamp()

            if breached:
                if stop_breach_since == 0:
                    stop_breach_since = tick_ts  # start timer
                elif (tick_ts - stop_breach_since) >= STOP_CONFIRM_SECS:
                    # Breach confirmed — live bot sends a market order, which
                    # fills at the current bid (LONG close) or offer (SHORT
                    # close). Use the tick bid/ofr at this moment, not the
                    # stop level — after 60s of persistent breach, price has
                    # typically drifted further adverse than the stop itself.
                    exit_price = float(bid) if position == "LONG" else float(ofr)
                    if position == "LONG":
                        trades.append({
                            "pnl_pts": round(exit_price - entry_price, 1),
                            "direction": "LONG", "entry": entry_price, "exit": exit_price,
                            "reason": "STOP", "session": f"S{session_num}",
                        })
                    else:
                        trades.append({
                            "pnl_pts": round(entry_price - exit_price, 1),
                            "direction": "SHORT", "entry": entry_price, "exit": exit_price,
                            "reason": "STOP", "session": f"S{session_num}",
                        })
                    if first_pnl is None:
                        first_pnl = trades[-1]["pnl_pts"]
                    position = None
                    stop_breach_since = 0
                    if entries_used < max_entries:
                        waiting_reentry = True
                        re_gate_cleared = False
                    continue
            else:
                stop_breach_since = 0  # price recovered, reset timer

        # Re-entry gate: price must be inside signal range before re-arming.
        # Use 2pt tolerance on boundaries to match live bot's streaming
        # price precision (mid from tick CSV can differ by 1-2pt from
        # the bot's real-time mid due to rounding/timing).
        if waiting_reentry and not re_gate_cleared:
            mid = (bid + ofr) / 2
            if (sell_lv - 2) <= mid <= (buy_lv + 2):
                re_gate_cleared = True
                waiting_reentry = False

        # Entry check (LONG priority matches Python v2).
        # Reverse-reentry filter (3 modes: always/never/after_loss).
        if position is None and entries_used < max_entries and re_gate_cleared:
            if entries_used == 0:
                long_ok = short_ok = True
            else:
                first_won = (first_pnl is not None and first_pnl > 0)
                block_reverse = (
                    rr_policy == "never"
                    or (rr_policy == "after_loss" and first_won)
                )
                if block_reverse:
                    long_ok  = (first_direction == "LONG")
                    short_ok = (first_direction == "SHORT")
                else:
                    long_ok = short_ok = True
            if ofr >= buy_lv and long_ok:
                position = "LONG"
                entry_price = buy_lv
                current_stop = sig_l_f
                be_hit = False
                entries_used += 1
                if first_direction is None:
                    first_direction = "LONG"
                stop_breach_since = 0
                mfe_peak = entry_price
                initial_stop_dist = entry_price - sig_l_f
            elif bid <= sell_lv and short_ok:
                position = "SHORT"
                entry_price = sell_lv
                current_stop = sig_h_f
                be_hit = False
                entries_used += 1
                if first_direction is None:
                    first_direction = "SHORT"
                stop_breach_since = 0
                mfe_peak = entry_price
                initial_stop_dist = sig_h_f - entry_price

    # EOD force close
    if position is not None:
        last_bid = bids[-1]
        last_ofr = ofrs[-1]
        exit_price = last_bid if position == "LONG" else last_ofr
        pnl = (exit_price - entry_price) if position == "LONG" else (entry_price - exit_price)
        trades.append({
            "pnl_pts": round(pnl, 1),
            "direction": position, "entry": entry_price, "exit": float(exit_price),
            "reason": "EOD", "session": f"S{session_num}",
        })
        if first_pnl is None:
            first_pnl = trades[-1]["pnl_pts"]

    return trades


def run_instrument(inst_name: str, cfg: dict, date_filter: str | None = None) -> list:
    """Run backtest for all (or one) available tick days."""
    epic = cfg["epic"]
    pattern = f"{epic}_*.csv"
    files = sorted(TICK_DIR.glob(pattern))

    if not files:
        print(f"  {inst_name}: NO TICK DATA")
        return []

    if date_filter:
        files = [f for f in files if date_filter in f.name]

    all_trades = []
    for f in files:
        date_str = f.stem.split("_")[-1]
        ticks = load_ticks(epic, date_str, cfg["tz"])
        if ticks is None or len(ticks) < 100:
            continue
        # Tick-reconstructed bars match what live bot uses (both use mid price).
        # (Earlier idea of IG sidecar turned out to be same data with different
        # labels — not a separate source.)
        bars = build_5min_bars(ticks)
        if len(bars) < 10:
            continue

        for sn, session in enumerate(cfg["sessions"], start=1):
            trades = simulate_session(cfg, ticks, bars, session, sn)
            for t in trades:
                t["date"] = date_str
                t["instrument"] = inst_name
            all_trades.extend(trades)

    return all_trades


def summarise(trades: list, label: str) -> None:
    if not trades:
        print(f"  {label:<20}  no trades")
        return
    wins = [t for t in trades if t["pnl_pts"] > 0]
    losses = [t for t in trades if t["pnl_pts"] < 0]
    w_sum = sum(t["pnl_pts"] for t in wins)
    l_sum = abs(sum(t["pnl_pts"] for t in losses))
    pf = w_sum / l_sum if l_sum > 0 else float("inf")
    net = sum(t["pnl_pts"] for t in trades)
    wr = len(wins) / len(trades) * 100
    avg_w = w_sum / len(wins) if wins else 0
    avg_l = l_sum / len(losses) if losses else 0
    print(f"  {label:<20}  {len(trades):>4}  PF={pf:>5.2f}  net={net:>+8,.0f}  "
          f"W/L={len(wins)}/{len(losses)} ({wr:.0f}%)  avgW={avg_w:.0f} avgL={avg_l:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", nargs="?", choices=list(INSTRUMENTS.keys()) + ["ALL"], default="ALL")
    ap.add_argument("--date", help="Only this date (YYYY-MM-DD)")
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable reverse-reentry filter (baseline comparison)")
    ap.add_argument("--exit-mode", choices=["candle", "mfe_hybrid"], default="candle",
                    help="candle (live config) or mfe_hybrid (shadow upgrade)")
    ap.add_argument("--trail-close", action="store_true",
                    help="Variant B: always trail to prev bar close (candidate)")
    ap.add_argument("--stress", action="store_true",
                    help="Apply 10%% haircut to wins, 10%% amplifier to losses "
                         "(models 60s timer + spread friction)")
    args = ap.parse_args()

    # Filter toggle — when --no-filter, allow opposite re-entries on all instruments
    if args.no_filter:
        for _cfg in INSTRUMENTS.values():
            _cfg["reverse_reentry"] = "always"

    # Exit mode — apply to all instruments
    for _cfg in INSTRUMENTS.values():
        _cfg["exit_mode"] = args.exit_mode
        _cfg["mfe_retrace"] = 0.5
        _cfg["mfe_activate_mult"] = 0.5
        _cfg["trail_close_always"] = args.trail_close

    mode_label = args.exit_mode.upper()
    filt_label = f"{'UNFILTERED' if args.no_filter else 'FILTERED'} | EXIT={mode_label}" + (
        " | TRAIL=CLOSE" if args.trail_close else "") + (
        " | STRESS(0.9W/1.1L)" if args.stress else "")
    print("=" * 80)
    print(f"  ASRS TICK-LEVEL BACKTEST — {filt_label}")
    print(f"  tick dir: {TICK_DIR}")
    print("=" * 80)
    print(f"{'':20}  {'n':>4}  {'PF':>5}  {'net':>9}  {'W/L':<15} stats")
    print("-" * 80)

    all_trades = {}
    instruments = [args.instrument] if args.instrument != "ALL" else list(INSTRUMENTS.keys())
    for inst in instruments:
        cfg = INSTRUMENTS[inst]
        trades = run_instrument(inst, cfg, args.date)
        if args.stress:
            # 10% win haircut, 10% loss amplifier — models 60s timer + fill slip
            for t in trades:
                if t["pnl_pts"] > 0:
                    t["pnl_pts"] = round(t["pnl_pts"] * 0.9, 1)
                elif t["pnl_pts"] < 0:
                    t["pnl_pts"] = round(t["pnl_pts"] * 1.1, 1)
        all_trades[inst] = trades
        summarise(trades, inst)

    # Combined + per-session breakdown
    combined = [t for trades in all_trades.values() for t in trades]
    if combined:
        print("-" * 80)
        summarise(combined, "COMBINED")

        # Per-session
        print()
        print("  PER SESSION:")
        sessions = sorted(set((t["instrument"], t["session"]) for t in combined))
        for inst, sess in sessions:
            sess_trades = [t for t in combined if t["instrument"] == inst and t["session"] == sess]
            summarise(sess_trades, f"    {inst}_{sess}")

        # Per-day
        print()
        print("  PER DAY:")
        dates = sorted(set(t["date"] for t in combined))
        for d in dates:
            day_trades = [t for t in combined if t["date"] == d]
            summarise(day_trades, f"    {d}")


if __name__ == "__main__":
    main()
