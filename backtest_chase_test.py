"""
backtest_chase_test.py — after winning trail-stop, add SAME direction.

Inverted hypothesis: since 63% of winners see continuation (not reversal),
add a momentum trade in the same direction after each trail-stop winner.
"""
import os, sys, sqlite3, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICK_DIR = "/root/asrs-bot/data/ticks"
DB = "/root/asrs-bot/data/trade_journal.db"

CFG = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",    "eod": (17, 30)},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York", "eod": (16, 0)},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",       "eod": (15, 0)},
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


def simulate_chase(ticks, cfg, entry_utm, entry_price, orig_dir, stop_pts, target_pts):
    tz = ZoneInfo(cfg["tz"])
    session_end = pd.Timestamp(
        pd.Timestamp(entry_utm, unit="ms", tz="UTC").tz_convert(tz).date(),
        tz=tz).replace(hour=cfg["eod"][0], minute=cfg["eod"][1])
    session_end_utm = int(session_end.tz_convert("UTC").timestamp() * 1000)

    if orig_dir == "LONG":
        target = entry_price + target_pts; stop = entry_price - stop_pts
    else:
        target = entry_price - target_pts; stop = entry_price + stop_pts

    after = ticks[(ticks["utm"] > entry_utm) & (ticks["utm"] < session_end_utm)].reset_index(drop=True)
    if len(after) == 0: return {"reason": "NO_TICKS", "pnl": 0}
    bids = after["bid"].values; ofrs = after["ofr"].values
    for i in range(len(after)):
        bid, ofr = float(bids[i]), float(ofrs[i])
        if orig_dir == "LONG":
            if ofr >= target: return {"reason": "TARGET", "pnl": round(target - entry_price, 1)}
            if bid <= stop:   return {"reason": "STOP",   "pnl": round(bid - entry_price, 1)}
        else:
            if bid <= target: return {"reason": "TARGET", "pnl": round(entry_price - target, 1)}
            if ofr >= stop:   return {"reason": "STOP",   "pnl": round(entry_price - ofr, 1)}
    last_bid = float(bids[-1]); last_ofr = float(ofrs[-1])
    pnl = (last_bid - entry_price) if orig_dir == "LONG" else (entry_price - last_ofr)
    return {"reason": "EOD", "pnl": round(pnl, 1)}


def main():
    conn = sqlite3.connect(DB)
    winners = pd.read_sql(
        "SELECT instrument, date, direction, entry_price, exit_price, pnl_pts, "
        "entry_time, exit_time, bar_range "
        "FROM trades WHERE mode='live' AND exit_reason='TRAIL_STOP' AND pnl_pts>0 "
        "ORDER BY date, entry_time", conn
    )
    conn.close()
    print(f"Winning TRAIL_STOP trades: {len(winners)}")

    tick_cache = {}
    for _, tr in winners.iterrows():
        k = (CFG[tr["instrument"]]["epic"], tr["date"])
        if k not in tick_cache: tick_cache[k] = load_ticks(*k)

    print(f"\n{'='*100}\n  CHASE: add SAME direction after TRAIL_STOP winner\n{'='*100}")
    for stop_pts in [30, 50, 75]:
        for target_pts in [30, 50, 75, 100, 150]:
            results = []
            for _, tr in winners.iterrows():
                inst = tr["instrument"]; cfg = CFG[inst]
                ticks = tick_cache.get((cfg["epic"], tr["date"]))
                if ticks is None: continue
                tz = ZoneInfo(cfg["tz"])
                exit_ts = pd.Timestamp(f"{tr['date']} {tr['exit_time']}", tz=tz)
                exit_utm = int(exit_ts.tz_convert("UTC").timestamp() * 1000)
                res = simulate_chase(ticks, cfg, exit_utm, float(tr["exit_price"]),
                                      tr["direction"], stop_pts, target_pts)
                results.append({"inst": inst, "orig_pnl": float(tr["pnl_pts"]),
                                 "chase_pnl": res["pnl"], "reason": res["reason"]})
            if not results: continue
            df = pd.DataFrame(results)
            wins = df[df["chase_pnl"] > 0]; losses = df[df["chase_pnl"] < 0]
            pf = wins["chase_pnl"].sum() / max(abs(losses["chase_pnl"].sum()), 0.001)
            print(f"  stop={stop_pts:>3} tgt={target_pts:>3}  n={len(df)} "
                  f"WR={len(wins)/len(df)*100:>5.1f}% net={df['chase_pnl'].sum():>+7.1f}pt "
                  f"PF={pf:.2f} "
                  f"target={len(df[df['reason']=='TARGET']):>2} "
                  f"stop={len(df[df['reason']=='STOP']):>2} "
                  f"EOD={len(df[df['reason']=='EOD']):>2}")

    # Per-instrument at stop=50, tgt=50
    print(f"\n{'='*100}\n  PER-INSTRUMENT (stop=50, tgt=50)\n{'='*100}")
    results = []
    for _, tr in winners.iterrows():
        inst = tr["instrument"]; cfg = CFG[inst]
        ticks = tick_cache.get((cfg["epic"], tr["date"]))
        if ticks is None: continue
        tz = ZoneInfo(cfg["tz"])
        exit_ts = pd.Timestamp(f"{tr['date']} {tr['exit_time']}", tz=tz)
        exit_utm = int(exit_ts.tz_convert("UTC").timestamp() * 1000)
        res = simulate_chase(ticks, cfg, exit_utm, float(tr["exit_price"]),
                              tr["direction"], 50, 50)
        results.append({"inst": inst, "orig_pnl": float(tr["pnl_pts"]),
                        "chase_pnl": res["pnl"], "reason": res["reason"]})
    df = pd.DataFrame(results)
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = df[df["inst"] == inst]
        if len(d) == 0: continue
        wins = d[d["chase_pnl"] > 0]
        print(f"  {inst}: n={len(d)} WR={len(wins)/len(d)*100:.1f}% "
              f"chase={d['chase_pnl'].sum():+.1f}pt "
              f"orig+chase={d['orig_pnl'].sum()+d['chase_pnl'].sum():+.1f}pt "
              f"(orig alone {d['orig_pnl'].sum():+.1f})")


if __name__ == "__main__":
    main()
