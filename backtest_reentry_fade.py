"""
backtest_reentry_fade.py — test: after winning trail-stop, immediately fade.

For each live TRAIL_STOP winner:
  1. At exit price, open opposing position
  2. Target = bar 4/5 opposite extreme (sig_h for LONG-won → SHORT fade;
     sig_l for SHORT-won → LONG fade)
  3. Stop = 50pt from entry (configurable)
  4. Walk ticks forward, hit first of: target / stop / session end

Tests multiple stop sizes + target variants.
"""
import os, sys, sqlite3, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"

CFG = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",
               "buffer": 2.0, "eod": (17, 30)},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York",
               "buffer": 5.0, "eod": (16, 0)},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",
               "buffer": 2.0, "eod": (15, 0)},
}


def load_ticks(epic, date_str):
    path = f"{TICK_DIR}/{epic}_{date_str}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["utm", "bid", "ofr"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df = df.dropna(subset=["bid", "ofr"])
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    return df.dropna(subset=["utm"]).sort_values("utm").reset_index(drop=True)


def simulate_fade(ticks, cfg, entry_utm, entry_price, orig_dir, sig_h, sig_l,
                  stop_pts, target_type):
    """Open opposing fade at entry_price + stop_pts."""
    tz = ZoneInfo(cfg["tz"])
    fade_dir = "SHORT" if orig_dir == "LONG" else "LONG"
    session_end = pd.Timestamp(
        pd.Timestamp(entry_utm, unit="ms", tz="UTC").tz_convert(tz).date(),
        tz=tz).replace(hour=cfg["eod"][0], minute=cfg["eod"][1])
    session_end_utm = int(session_end.tz_convert("UTC").timestamp() * 1000)

    if target_type == "extreme":
        target = sig_h if fade_dir == "LONG" else sig_l
    elif target_type == "near":
        # Near side: sig_h for SHORT-fade, sig_l for LONG-fade (entering the bar range)
        target = sig_l if fade_dir == "LONG" else sig_h
    elif target_type == "mid":
        target = (sig_h + sig_l) / 2
    else:
        raise ValueError

    stop = entry_price + stop_pts if fade_dir == "SHORT" else entry_price - stop_pts

    after = ticks[(ticks["utm"] > entry_utm) & (ticks["utm"] < session_end_utm)].reset_index(drop=True)
    if len(after) == 0:
        return {"reason": "NO_TICKS", "pnl": 0, "exit_price": entry_price}

    bids = after["bid"].values
    ofrs = after["ofr"].values
    for i in range(len(after)):
        bid, ofr = float(bids[i]), float(ofrs[i])
        if fade_dir == "SHORT":
            # Target hit (bid ≤ target) OR stop hit (ofr ≥ stop)
            if bid <= target:
                pnl = entry_price - bid
                return {"reason": "TARGET", "pnl": round(pnl, 1), "exit_price": round(bid, 1)}
            if ofr >= stop:
                pnl = entry_price - ofr
                return {"reason": "STOP", "pnl": round(pnl, 1), "exit_price": round(ofr, 1)}
        else:  # LONG fade
            if ofr >= target:
                pnl = ofr - entry_price
                return {"reason": "TARGET", "pnl": round(pnl, 1), "exit_price": round(ofr, 1)}
            if bid <= stop:
                pnl = bid - entry_price
                return {"reason": "STOP", "pnl": round(pnl, 1), "exit_price": round(bid, 1)}

    # Session end
    last_bid = float(bids[-1]); last_ofr = float(ofrs[-1])
    if fade_dir == "SHORT":
        pnl = entry_price - last_ofr
    else:
        pnl = last_bid - entry_price
    return {"reason": "EOD", "pnl": round(pnl, 1), "exit_price": round(last_bid if fade_dir=="LONG" else last_ofr, 1)}


def main():
    conn = sqlite3.connect(DB)
    winners = pd.read_sql(
        "SELECT instrument, date, direction, entry_price, exit_price, pnl_pts, "
        "entry_time, exit_time, bar_range, entry_intended, exit_reason "
        "FROM trades WHERE mode='live' AND exit_reason='TRAIL_STOP' AND pnl_pts>0 "
        "ORDER BY date, entry_time",
        conn
    )
    conn.close()
    print(f"Winning TRAIL_STOP trades: {len(winners)}")

    # Pre-load ticks
    tick_cache = {}
    for _, tr in winners.iterrows():
        epic = CFG[tr["instrument"]]["epic"]
        key = (epic, tr["date"])
        if key not in tick_cache:
            tick_cache[key] = load_ticks(epic, tr["date"])

    print(f"\n{'='*110}")
    print(f"  FADE TEST: immediately open opposing position at TRAIL_STOP winner's exit")
    print(f"{'='*110}")

    scenarios = []
    for stop_pts in [30, 50, 75, 100]:
        for target_type in ["extreme", "near", "mid"]:
            scenarios.append((stop_pts, target_type))

    for stop_pts, target_type in scenarios:
        results = []
        skipped = 0
        for _, tr in winners.iterrows():
            inst = tr["instrument"]; cfg = CFG[inst]
            ticks = tick_cache.get((cfg["epic"], tr["date"]))
            if ticks is None:
                skipped += 1; continue
            # Reconstruct sig_h/sig_l from live trade fields
            buf = cfg["buffer"]
            bar_range = float(tr["bar_range"] or 0)
            entry_intended = float(tr["entry_intended"] or tr["entry_price"])
            if tr["direction"] == "LONG":
                sig_h = entry_intended - buf; sig_l = sig_h - bar_range
            else:
                sig_l = entry_intended + buf; sig_h = sig_l + bar_range
            # Entry time/utm
            tz = ZoneInfo(cfg["tz"])
            exit_ts = pd.Timestamp(f"{tr['date']} {tr['exit_time']}", tz=tz)
            exit_utm = int(exit_ts.tz_convert("UTC").timestamp() * 1000)
            # Fade entry price = live trade's exit price
            fade_entry = float(tr["exit_price"])
            res = simulate_fade(ticks, cfg, exit_utm, fade_entry,
                                 tr["direction"], sig_h, sig_l,
                                 stop_pts, target_type)
            results.append({
                "inst": inst, "orig_dir": tr["direction"],
                "orig_pnl": float(tr["pnl_pts"]),
                "fade_pnl": res["pnl"], "reason": res["reason"],
            })

        if not results:
            continue
        df = pd.DataFrame(results)
        wins = df[df["fade_pnl"] > 0]
        losses = df[df["fade_pnl"] < 0]
        pf = wins["fade_pnl"].sum() / max(abs(losses["fade_pnl"].sum()), 0.001)
        label = f"stop={stop_pts}pt tgt={target_type}"
        print(f"\n  {label:<25} n={len(df)} "
              f"WR={len(wins)/max(len(df),1)*100:.1f}% "
              f"net={df['fade_pnl'].sum():+.1f}pt "
              f"PF={pf:.2f} "
              f"avgW={wins['fade_pnl'].mean() if len(wins) else 0:+.1f} "
              f"avgL={losses['fade_pnl'].mean() if len(losses) else 0:+.1f} "
              f"target-hit={len(df[df['reason']=='TARGET'])}/{len(df)} "
              f"stop-hit={len(df[df['reason']=='STOP'])}/{len(df)} "
              f"EOD={len(df[df['reason']=='EOD'])}/{len(df)}")

    # Best scenario: details per instrument
    print(f"\n{'='*110}")
    print(f"  PER-INSTRUMENT (best variant: 50pt stop, extreme target)")
    print(f"{'='*110}")
    results = []
    for _, tr in winners.iterrows():
        inst = tr["instrument"]; cfg = CFG[inst]
        ticks = tick_cache.get((cfg["epic"], tr["date"]))
        if ticks is None: continue
        buf = cfg["buffer"]
        bar_range = float(tr["bar_range"] or 0)
        entry_intended = float(tr["entry_intended"] or tr["entry_price"])
        if tr["direction"] == "LONG":
            sig_h = entry_intended - buf; sig_l = sig_h - bar_range
        else:
            sig_l = entry_intended + buf; sig_h = sig_l + bar_range
        tz = ZoneInfo(cfg["tz"])
        exit_ts = pd.Timestamp(f"{tr['date']} {tr['exit_time']}", tz=tz)
        exit_utm = int(exit_ts.tz_convert("UTC").timestamp() * 1000)
        fade_entry = float(tr["exit_price"])
        res = simulate_fade(ticks, cfg, exit_utm, fade_entry,
                             tr["direction"], sig_h, sig_l, 50, "extreme")
        results.append({"inst": inst, "date": tr["date"],
                        "orig_dir": tr["direction"], "orig_pnl": float(tr["pnl_pts"]),
                        "fade_pnl": res["pnl"], "reason": res["reason"]})

    df = pd.DataFrame(results)
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = df[df["inst"] == inst]
        if len(d) == 0: continue
        wins = d[d["fade_pnl"] > 0]; losses = d[d["fade_pnl"] < 0]
        print(f"\n  {inst}: n={len(d)} "
              f"WR={len(wins)/len(d)*100:.1f}% "
              f"net={d['fade_pnl'].sum():+.1f}pt "
              f"(orig wins contributed {d['orig_pnl'].sum():+.1f}pt; "
              f"combined={d['orig_pnl'].sum()+d['fade_pnl'].sum():+.1f}pt)")


if __name__ == "__main__":
    main()
