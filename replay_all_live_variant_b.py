"""
replay_all_live_variant_b.py — replay EVERY live trade under Variant B exits.

Uses live entries (same time, same price) and replays only the EXIT logic
under Variant B trail (prev_close always) + 60s confirm timer on IG tick data.

Isolates the trail-mechanism effect from firstrate-vs-IG bar divergence.
"""
import os, sys, sqlite3, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"

INST = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",
               "be_pts": 15.0, "be_buf": 5.0, "eod": (17, 30),
               "trail_min_move": 3.0, "buffer": 2.0},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York",
               "be_pts": 20.0, "be_buf": 5.0, "eod": (16, 0),
               "trail_min_move": 5.0, "buffer": 5.0},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",
               "be_pts": 50.0, "be_buf": 10.0, "eod": (15, 0),
               "trail_min_move": 5.0, "buffer": 2.0},
}


def load_ticks(epic, d):
    path = f"{TICK_DIR}/{epic}_{d}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["utm", "bid", "ofr"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.dropna(subset=["bid", "ofr", "mid"])
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    return df.dropna(subset=["utm"]).sort_values("utm").reset_index(drop=True)


def build_bars(ticks, tz):
    t = ticks.copy()
    t["dt"] = pd.to_datetime(t["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    t["bar"] = t["dt"].dt.floor("5min")
    return t.groupby("bar").agg(
        High=("mid","max"), Low=("mid","min"),
        Close=("mid","last"), Open=("mid","first")
    ).reset_index()


def simulate_b_exit(trade, ticks, bars, cfg, date_str):
    """Replay exit under Variant B: prev_close trail + 60s confirm timer."""
    direction = trade["direction"]; entry = float(trade["entry_price"])
    tz = ZoneInfo(cfg["tz"])
    t_entry = pd.Timestamp(f"{date_str} {trade['entry_time']}", tz=tz)
    entry_utm = int(t_entry.timestamp() * 1000)
    # Reconstruct signal bar precisely from bot's entry_intended + bar_range + buffer.
    # LONG:  entry_intended = sig_h + buffer  → sig_h = entry_intended - buffer
    #        sig_l = sig_h - bar_range
    # SHORT: entry_intended = sig_l - buffer  → sig_l = entry_intended + buffer
    #        sig_h = sig_l + bar_range
    buf = cfg["buffer"]
    bar_range = float(trade.get("bar_range", 0) or 0)
    entry_intended = float(trade.get("entry_intended", 0) or entry)
    if direction == "LONG":
        sig_h = entry_intended - buf
        sig_l = sig_h - bar_range
    else:
        sig_l = entry_intended + buf
        sig_h = sig_l + bar_range
    current_stop = sig_l if direction == "LONG" else sig_h

    be_hit = False; stop_breach_since = 0
    CONFIRM_MS = 60_000
    bars_ind = bars.set_index("bar")
    session_end_dt = pd.Timestamp(date_str, tz=tz).replace(
        hour=cfg["eod"][0], minute=cfg["eod"][1])
    session_end_utm = int(session_end_dt.tz_convert("UTC").timestamp() * 1000)

    after = ticks[(ticks["utm"] >= entry_utm) & (ticks["utm"] < session_end_utm)].reset_index(drop=True)
    if len(after) == 0:
        return {"exit_price": float(trade["exit_price"]), "exit_reason": "NO_TICKS"}

    bids = after["bid"].values; ofrs = after["ofr"].values; utms = after["utm"].values
    last_bar = None
    for i in range(len(after)):
        bid, ofr, utm = float(bids[i]), float(ofrs[i]), int(utms[i])
        bar = pd.Timestamp(utm, unit="ms", tz="UTC").tz_convert(tz).floor("5min")
        if last_bar is not None and bar != last_bar and last_bar in bars_ind.index:
            prev_c = float(bars_ind.loc[last_bar]["Close"])
            if direction == "LONG":
                unr = prev_c - entry
                if not be_hit and unr >= cfg["be_pts"]:
                    be_hit = True
                    new_be = entry - cfg["be_buf"]
                    if new_be > current_stop: current_stop = new_be; stop_breach_since = 0
                if prev_c > current_stop: current_stop = round(prev_c, 1); stop_breach_since = 0
            else:
                unr = entry - prev_c
                if not be_hit and unr >= cfg["be_pts"]:
                    be_hit = True
                    new_be = entry + cfg["be_buf"]
                    if new_be < current_stop: current_stop = new_be; stop_breach_since = 0
                if prev_c < current_stop: current_stop = round(prev_c, 1); stop_breach_since = 0
        last_bar = bar

        breached = (direction == "LONG" and bid <= current_stop) or \
                   (direction == "SHORT" and ofr >= current_stop)
        if breached:
            if stop_breach_since == 0:
                stop_breach_since = utm
            elif (utm - stop_breach_since) >= CONFIRM_MS:
                exit_price = bid if direction == "LONG" else ofr
                return {"exit_price": round(exit_price, 1), "exit_reason": "STOP_B"}
        else:
            stop_breach_since = 0

    last_bid = float(bids[-1]); last_ofr = float(ofrs[-1])
    exit_price = last_bid if direction == "LONG" else last_ofr
    return {"exit_price": round(exit_price, 1), "exit_reason": "EOD"}


def main():
    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        "SELECT id, instrument, date, direction, entry_price, exit_price, "
        "pnl_pts, exit_reason, entry_time, exit_time, bar_range, entry_intended, signal_bar "
        "FROM trades WHERE mode='live' ORDER BY date, entry_time",
        conn
    )
    conn.close()

    # Pre-cache ticks and bars per (epic, date)
    cache = {}
    def get(epic, date_str, tz):
        key = (epic, date_str)
        if key not in cache:
            ticks = load_ticks(epic, date_str)
            if ticks is None:
                cache[key] = None
            else:
                cache[key] = (ticks, build_bars(ticks, tz))
        return cache[key]

    print(f"{'date':<12} {'inst':<7} {'dir':<5} {'entry':>9} "
          f"{'LIVE exit':>9} {'LIVE pnl':>8} {'B exit':>9} {'B pnl':>8} {'Δ':>8} {'exit_reason':<14}")
    print("-" * 130)

    totals_live = {"DAX": 0, "US30": 0, "NIKKEI": 0}
    totals_b = {"DAX": 0, "US30": 0, "NIKKEI": 0}
    counts = {"DAX": 0, "US30": 0, "NIKKEI": 0}
    skipped = 0

    for _, tr in live.iterrows():
        inst = tr["instrument"]; cfg = INST[inst]
        date_str = tr["date"]
        cached = get(cfg["epic"], date_str, cfg["tz"])
        if cached is None:
            skipped += 1
            print(f"  skip {tr['id']}: no ticks for {cfg['epic']}_{date_str}")
            continue
        ticks, bars = cached
        payload = {
            "direction": tr["direction"],
            "entry_price": float(tr["entry_price"]),
            "entry_time": tr["entry_time"],
            "exit_price": float(tr["exit_price"]),
            "bar_range": float(tr.get("bar_range", 0) or 0),
            "entry_intended": float(tr.get("entry_intended", 0) or tr["entry_price"]),
            "signal_bar": int(tr.get("signal_bar", 4) or 4),
        }
        res = simulate_b_exit(payload, ticks, bars, cfg, date_str)
        if res is None:
            skipped += 1; continue
        b_exit = res["exit_price"]
        b_pnl = (b_exit - float(tr["entry_price"])) if tr["direction"] == "LONG" \
                else (float(tr["entry_price"]) - b_exit)
        delta = b_pnl - float(tr["pnl_pts"])

        totals_live[inst] += float(tr["pnl_pts"])
        totals_b[inst] += b_pnl
        counts[inst] += 1

        print(f"  {date_str:<10} {inst:<7} {tr['direction']:<5} {float(tr['entry_price']):>9.1f} "
              f"{float(tr['exit_price']):>9.1f} {float(tr['pnl_pts']):>+7.1f} "
              f"{b_exit:>9.1f} {b_pnl:>+7.1f} {delta:>+7.1f} "
              f"{tr['exit_reason']:<14}")

    print("-" * 130)
    print(f"\n  {'INSTRUMENT':<10} {'n':<5} {'LIVE net':<12} {'B net':<12} {'DELTA':<10} {'Δ/trade':<10}")
    grand_live = grand_b = grand_n = 0
    for inst in ["DAX", "US30", "NIKKEI"]:
        n = counts[inst]
        if n == 0:
            print(f"  {inst:<10} skipped")
            continue
        delta = totals_b[inst] - totals_live[inst]
        print(f"  {inst:<10} {n:<5} {totals_live[inst]:>+8.1f}     "
              f"{totals_b[inst]:>+8.1f}     {delta:>+7.1f}    {delta/n:>+5.1f}")
        grand_live += totals_live[inst]
        grand_b += totals_b[inst]
        grand_n += n
    print(f"  {'TOTAL':<10} {grand_n:<5} {grand_live:>+8.1f}     {grand_b:>+8.1f}     "
          f"{grand_b-grand_live:>+7.1f}    {(grand_b-grand_live)/max(grand_n,1):>+5.1f}")
    if skipped:
        print(f"\n  Skipped {skipped} trades (no tick data)")


if __name__ == "__main__":
    main()
