"""
replay_today_variant_b.py — re-simulate today's live trades under Variant B logic.

For each completed trade in the journal (mode=live, date=today):
  1. Pull ticks between entry_time and session_end.
  2. Re-run exit logic with prev_close trail (Variant B) + 60s stop-confirm timer.
  3. Compare Variant B exit vs actual live exit.
"""
import os, sys, pandas as pd, sqlite3
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TODAY = "2026-04-16"
TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"

INST = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",
               "be_pts": 15.0, "be_buf": 5.0, "eod": (17, 30),
               "trail_min_move": 3.0},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York",
               "be_pts": 20.0, "be_buf": 5.0, "eod": (16, 0),
               "trail_min_move": 5.0},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",
               "be_pts": 50.0, "be_buf": 10.0, "eod": (15, 0),
               "trail_min_move": 5.0},
}


def load_ticks(epic, d):
    df = pd.read_csv(f"{TICK_DIR}/{epic}_{d}.csv")
    df = df.dropna(subset=["utm", "bid", "ofr"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["bid", "ofr", "mid"])
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    return df.dropna(subset=["utm"]).sort_values("utm").reset_index(drop=True)


def build_5m_bars(ticks, tz):
    t = ticks.copy()
    t["dt"] = pd.to_datetime(t["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    t["bar"] = t["dt"].dt.floor("5min")
    bars = t.groupby("bar").agg(High=("mid","max"), Low=("mid","min"),
                                 Close=("mid","last"), Open=("mid","first")).reset_index()
    return bars


def simulate_b(trade, ticks, bars, cfg):
    """Return (exit_price, exit_reason, exit_utm) for this trade under Variant B."""
    direction = trade["direction"]
    entry = trade["entry_price"]
    entry_utm = trade["entry_utm"]
    sig_bar_low = trade["signal_low"]
    sig_bar_high = trade["signal_high"]
    tz = ZoneInfo(cfg["tz"])

    session_end = pd.Timestamp(TODAY, tz=tz).replace(
        hour=cfg["eod"][0], minute=cfg["eod"][1]).tz_convert("UTC").timestamp()*1000

    # Initial stop = signal low (LONG) / high (SHORT)
    current_stop = float(sig_bar_low) if direction == "LONG" else float(sig_bar_high)
    be_hit = False
    stop_breach_since = 0  # ms — 0 means timer inactive
    CONFIRM_MS = 60_000

    # Trail on each new 5-min bar close
    bars_ind = bars.set_index("bar")
    last_bar = None

    after = ticks[ticks["utm"] >= entry_utm].reset_index(drop=True)
    after = after[after["utm"] < session_end].reset_index(drop=True)
    if len(after) == 0:
        return (float(trade["exit_price"]), "NO_TICKS", 0)

    bids = after["bid"].values; ofrs = after["ofr"].values; utms = after["utm"].values

    for i in range(len(after)):
        bid, ofr, utm = float(bids[i]), float(ofrs[i]), int(utms[i])
        dt = pd.Timestamp(utm, unit="ms", tz="UTC").tz_convert(tz)
        bar = dt.floor("5min")

        # New bar → apply trail from the just-completed previous bar
        if last_bar is not None and bar != last_bar:
            if last_bar in bars_ind.index:
                pb = bars_ind.loc[last_bar]
                prev_c = float(pb["Close"])
                # Breakeven
                if direction == "LONG":
                    unr = prev_c - entry
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry - cfg["be_buf"]
                        if new_be > current_stop:
                            current_stop = new_be; stop_breach_since = 0
                else:
                    unr = entry - prev_c
                    if not be_hit and unr >= cfg["be_pts"]:
                        be_hit = True
                        new_be = entry + cfg["be_buf"]
                        if new_be < current_stop:
                            current_stop = new_be; stop_breach_since = 0
                # Variant B trail = always prev_close
                if direction == "LONG":
                    if prev_c > current_stop:
                        current_stop = round(prev_c, 1); stop_breach_since = 0
                else:
                    if prev_c < current_stop:
                        current_stop = round(prev_c, 1); stop_breach_since = 0
        last_bar = bar

        # Stop check with 60s confirm timer
        breached = (direction == "LONG" and bid <= current_stop) or \
                   (direction == "SHORT" and ofr >= current_stop)
        if breached:
            if stop_breach_since == 0:
                stop_breach_since = utm
            elif (utm - stop_breach_since) >= CONFIRM_MS:
                exit_price = bid if direction == "LONG" else ofr
                return (round(exit_price, 1), "STOP_B", utm)
        else:
            stop_breach_since = 0

    # Session end — exit at last price
    last_bid = float(bids[-1]); last_ofr = float(ofrs[-1])
    return (round(last_bid if direction == "LONG" else last_ofr, 1), "EOD", int(utms[-1]))


def get_signal_levels(inst, entry_time, tick_ticks, bars, cfg):
    """
    Derive the signal bar's H/L from bars preceding the entry time, matching
    the bot's bar4/bar5 logic. Simplification: use the bar immediately before
    entry_time as the signal bar's stop reference — accurate enough for replay.
    """
    tz = ZoneInfo(cfg["tz"])
    et = pd.Timestamp(f"{TODAY} {entry_time}", tz=tz)
    entry_bar = et.floor("5min")
    # Signal bar is typically the last bar BEFORE the 5-min window when entry fired
    prev_bars = bars[bars["bar"] < entry_bar]
    if len(prev_bars) == 0:
        return None, None, None
    # Walk back to find a bar that "looks like" the signal — closest to entry
    sig = prev_bars.iloc[-1]
    return float(sig["High"]), float(sig["Low"]), et.timestamp() * 1000


def main():
    conn = sqlite3.connect(DB)
    trades = pd.read_sql("""SELECT * FROM trades WHERE date=? AND mode='live' ORDER BY id""",
                          conn, params=(TODAY,))
    conn.close()
    if len(trades) == 0:
        print("No live trades today"); return

    print(f"{'id':>4} {'inst':<7} {'dir':<5} {'entry':>9} {'ACT exit':>9} {'ACT pnl':>7} "
          f"{'ACT reason':<14} {'B exit':>9} {'B pnl':>7} {'B reason':<10} {'delta_pts':>9}")
    print("-" * 120)

    total_actual = 0.0; total_b = 0.0
    for _, tr in trades.iterrows():
        inst = tr["instrument"]; cfg = INST[inst]
        ticks = load_ticks(cfg["epic"], TODAY)
        bars = build_5m_bars(ticks, cfg["tz"])
        sig_h, sig_l, entry_utm = get_signal_levels(inst, tr["entry_time"], ticks, bars, cfg)
        if sig_h is None:
            print(f"  skip {tr['id']}: no signal levels"); continue

        payload = {
            "direction": tr["direction"],
            "entry_price": float(tr["entry_price"]),
            "entry_utm": entry_utm,
            "signal_high": sig_h,
            "signal_low": sig_l,
            "exit_price": float(tr["exit_price"]),
        }
        b_exit, b_reason, _ = simulate_b(payload, ticks, bars, cfg)
        if tr["direction"] == "LONG":
            b_pnl = b_exit - float(tr["entry_price"])
        else:
            b_pnl = float(tr["entry_price"]) - b_exit
        delta = b_pnl - float(tr["pnl_pts"])

        total_actual += float(tr["pnl_pts"])
        total_b += b_pnl

        print(f"{tr['id']:>4} {inst:<7} {tr['direction']:<5} {float(tr['entry_price']):>9.1f} "
              f"{float(tr['exit_price']):>9.1f} {float(tr['pnl_pts']):>+7.1f} "
              f"{tr['exit_reason']:<14} {b_exit:>9.1f} {b_pnl:>+7.1f} "
              f"{b_reason:<10} {delta:>+9.1f}")

    print("-" * 120)
    print(f"  TOTAL (live/A trail):   {total_actual:+.1f} pts")
    print(f"  TOTAL (replay/B trail): {total_b:+.1f} pts")
    print(f"  DELTA:                  {total_b - total_actual:+.1f} pts")


if __name__ == "__main__":
    main()
