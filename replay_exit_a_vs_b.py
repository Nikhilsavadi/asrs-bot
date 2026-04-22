"""
replay_exit_a_vs_b.py — same live entries, replay under A and B exits.

For each live trade: keep entry (price, time, direction) as-is, replay the
EXIT under both A (prev_low with tight_threshold) and B (prev_close always),
using the exact IG tick data the bot saw.

Compares:
  LIVE actual exit (A trail pre-Wed, might differ due to 60s timer edge cases)
  A-REPLAY (simulated A trail on ticks)
  B-REPLAY (simulated B trail on ticks)

Goal: validate that A-REPLAY matches LIVE (exit logic is faithful).
Then B-REPLAY shows isolated A-vs-B effect.
"""
import os, sys, sqlite3, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"

CFG = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",
               "buffer": 2.0, "be_pts": 15.0, "be_buf": 5.0,
               "tight": 100.0, "eod": (17, 30), "trail_min_move": 3.0},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York",
               "buffer": 5.0, "be_pts": 20.0, "be_buf": 5.0,
               "tight": 80.0, "eod": (16, 0), "trail_min_move": 5.0},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",
               "buffer": 2.0, "be_pts": 50.0, "be_buf": 10.0,
               "tight": 300.0, "eod": (15, 0), "trail_min_move": 5.0},
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


def build_5min_bars(ticks, tz):
    t = ticks.copy()
    t["dt"] = pd.to_datetime(t["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(ZoneInfo(tz))
    t["bar"] = t["dt"].dt.floor("5min")
    return t.groupby("bar").agg(
        High=("mid","max"), Low=("mid","min"),
        Close=("mid","last"), Open=("mid","first")
    ).reset_index()


def simulate_exit(trade, ticks, bars, cfg, date_str, variant):
    """
    variant: "A" (prev_low/high with tight_threshold) or "B" (prev_close always)
    Uses same initial stop (sig_l/sig_h) as live bot would have set.
    Exit = 60s stop-confirm timer, matching live.
    """
    direction = trade["direction"]
    entry = float(trade["entry_price"])
    buf = cfg["buffer"]
    bar_range = float(trade.get("bar_range") or 0)
    entry_intended = float(trade.get("entry_intended") or entry)
    if direction == "LONG":
        sig_h = entry_intended - buf; sig_l = sig_h - bar_range
    else:
        sig_l = entry_intended + buf; sig_h = sig_l + bar_range

    current_stop = sig_l if direction == "LONG" else sig_h
    be_hit = False
    stop_breach_since = 0
    CONFIRM_MS = 60_000

    tz = ZoneInfo(cfg["tz"])
    t_entry = pd.Timestamp(f"{date_str} {trade['entry_time']}", tz=tz)
    entry_utm = int(t_entry.timestamp() * 1000)
    session_end = pd.Timestamp(date_str, tz=tz).replace(
        hour=cfg["eod"][0], minute=cfg["eod"][1])
    session_end_utm = int(session_end.tz_convert("UTC").timestamp() * 1000)

    after = ticks[(ticks["utm"] >= entry_utm) & (ticks["utm"] < session_end_utm)].reset_index(drop=True)
    if len(after) == 0:
        return {"exit_price": float(trade["exit_price"]), "reason": "NO_TICKS"}

    bars_ind = bars.set_index("bar")
    bids = after["bid"].values; ofrs = after["ofr"].values; utms = after["utm"].values
    last_bar = None

    for i in range(len(after)):
        bid, ofr, utm = float(bids[i]), float(ofrs[i]), int(utms[i])
        bar = pd.Timestamp(utm, unit="ms", tz="UTC").tz_convert(tz).floor("5min")

        if last_bar is not None and bar != last_bar and last_bar in bars_ind.index:
            pb = bars_ind.loc[last_bar]
            prev_h, prev_l, prev_c = float(pb["High"]), float(pb["Low"]), float(pb["Close"])
            if direction == "LONG":
                unr = prev_c - entry
                if not be_hit and unr >= cfg["be_pts"]:
                    be_hit = True
                    new_be = entry - cfg["be_buf"]
                    if new_be > current_stop: current_stop = new_be; stop_breach_since = 0
                # Variant A: prev_low, or prev_close if profit >= tight_threshold
                # Variant B: prev_close always
                if variant == "B":
                    ns = prev_c
                else:  # A
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight"] else prev_l
                if ns > current_stop: current_stop = round(ns, 1); stop_breach_since = 0
            else:
                unr = entry - prev_c
                if not be_hit and unr >= cfg["be_pts"]:
                    be_hit = True
                    new_be = entry + cfg["be_buf"]
                    if new_be < current_stop: current_stop = new_be; stop_breach_since = 0
                if variant == "B":
                    ns = prev_c
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight"] else prev_h
                if ns < current_stop: current_stop = round(ns, 1); stop_breach_since = 0
        last_bar = bar

        breached = (direction == "LONG" and bid <= current_stop) or \
                   (direction == "SHORT" and ofr >= current_stop)
        if breached:
            if stop_breach_since == 0:
                stop_breach_since = utm
            elif (utm - stop_breach_since) >= CONFIRM_MS:
                exit_price = bid if direction == "LONG" else ofr
                return {"exit_price": round(exit_price, 1), "reason": "STOP"}
        else:
            stop_breach_since = 0

    exit_price = float(bids[-1]) if direction == "LONG" else float(ofrs[-1])
    return {"exit_price": round(exit_price, 1), "reason": "EOD"}


def main():
    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        "SELECT id, instrument, date, direction, entry_price, entry_intended, "
        "exit_price, pnl_pts, exit_reason, entry_time, bar_range "
        "FROM trades WHERE mode='live' ORDER BY date, entry_time",
        conn
    )
    conn.close()

    cache = {}
    def get(epic, ds, tz):
        key = (epic, ds)
        if key not in cache:
            ticks = load_ticks(epic, ds)
            cache[key] = (ticks, build_5min_bars(ticks, tz)) if ticks is not None else None
        return cache[key]

    totals = {"LIVE": {}, "A": {}, "B": {}}
    for lbl in totals:
        for inst in ["DAX","US30","NIKKEI"]:
            totals[lbl][inst] = {"n": 0, "net": 0.0}
    skipped = 0

    print(f"{'date':<12} {'inst':<7} {'dir':<5} {'entry':>9} "
          f"{'LIVE':>7} {'A':>7} {'B':>7} {'|A-L|':>6} {'|B-A|':>6}")
    print("-" * 85)
    match_exact_a = 0; total_a = 0
    abs_diff_a = 0.0
    for _, tr in live.iterrows():
        inst = tr["instrument"]; cfg = CFG[inst]
        date_str = tr["date"]
        cached = get(cfg["epic"], date_str, cfg["tz"])
        if cached is None:
            skipped += 1; continue
        ticks, bars = cached

        payload = dict(tr)
        res_a = simulate_exit(payload, ticks, bars, cfg, date_str, variant="A")
        res_b = simulate_exit(payload, ticks, bars, cfg, date_str, variant="B")
        if tr["direction"] == "LONG":
            pnl_a = res_a["exit_price"] - float(tr["entry_price"])
            pnl_b = res_b["exit_price"] - float(tr["entry_price"])
        else:
            pnl_a = float(tr["entry_price"]) - res_a["exit_price"]
            pnl_b = float(tr["entry_price"]) - res_b["exit_price"]
        live_pnl = float(tr["pnl_pts"])
        a_vs_live = abs(pnl_a - live_pnl)
        b_vs_a = abs(pnl_b - pnl_a)

        totals["LIVE"][inst]["n"] += 1; totals["LIVE"][inst]["net"] += live_pnl
        totals["A"][inst]["n"] += 1; totals["A"][inst]["net"] += pnl_a
        totals["B"][inst]["n"] += 1; totals["B"][inst]["net"] += pnl_b

        match_exact_a += (a_vs_live < 0.5)
        total_a += 1
        abs_diff_a += a_vs_live

        print(f"  {date_str:<10} {inst:<7} {tr['direction']:<5} {float(tr['entry_price']):>9.1f} "
              f"{live_pnl:>+6.1f} {pnl_a:>+6.1f} {pnl_b:>+6.1f} "
              f"{a_vs_live:>5.1f} {b_vs_a:>5.1f}")

    print(f"\n{'='*85}\n  FIDELITY CHECK\n{'='*85}")
    print(f"  Live trades replayed: {total_a}")
    print(f"  A-replay exact-match to live (<0.5pt): {match_exact_a}/{total_a} ({match_exact_a/max(total_a,1)*100:.1f}%)")
    print(f"  Avg |A - live| per trade: {abs_diff_a/max(total_a,1):.2f}pt")

    print(f"\n{'='*85}\n  AGGREGATE\n{'='*85}")
    print(f"  {'instrument':<10} {'n':<5} {'LIVE':<10} {'A-replay':<10} {'B-replay':<10} {'B-A delta':<10}")
    gl=ga=gb=gn=0
    for inst in ["DAX","US30","NIKKEI"]:
        n = totals["LIVE"][inst]["n"]
        if n == 0: continue
        l = totals["LIVE"][inst]["net"]; a = totals["A"][inst]["net"]; b = totals["B"][inst]["net"]
        print(f"  {inst:<10} {n:<5} {l:>+7.1f}   {a:>+7.1f}   {b:>+7.1f}   {b-a:>+7.1f}")
        gl += l; ga += a; gb += b; gn += n
    print(f"  {'TOTAL':<10} {gn:<5} {gl:>+7.1f}   {ga:>+7.1f}   {gb:>+7.1f}   {gb-ga:>+7.1f}")
    if skipped:
        print(f"\n  Skipped {skipped} trades (no tick data)")


if __name__ == "__main__":
    main()
