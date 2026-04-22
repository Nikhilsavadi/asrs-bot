"""
backtest_wick_confirmation.py — does entry confirmation help on NIKKEI wicks?

For each NIKKEI trading day (Apr 10-22, IG tick CSVs), identify bar 4/5 signal
bar levels, then simulate entries under different confirmation policies:

  A: immediate (current)
  B: require price to hold at/beyond level for 5s before entry
  C: 10s hold
  D: 30s hold
  E: 60s hold

Uses Variant B trail (prev_close) + max_risk stop cap. Compares total PnL.

Applies same config as live:
  narrow_range=50, wide_range=150, max_bar_range=250, max_entries=3,
  max_risk_gbp=50, reverse_reentry=after_loss, fade=DISABLED on NIKKEI.
"""
import os, sys, pandas as pd, numpy as np
from zoneinfo import ZoneInfo

TICK_DIR = "/root/asrs-bot/data/ticks"
EPIC = "IX.D.NIKKEI.DAILY.IP"
TZ = "Asia/Tokyo"
SPREAD = 7.0
SLIP = 4.0
UPLIFT = 1.06

# NIKKEI config (from asrs/config.py)
CFG = {
    "narrow_range": 50, "wide_range": 150, "max_bar_range": 250,
    "max_entries": 3, "buffer": 2.0, "max_risk_gbp": 50.0,
    "breakeven_pts": 50.0, "max_slippage_pct": 0.5,
    "s1_h": 10, "s1_m": 0, "s2_h": 12, "s2_m": 0, "s3_h": 13, "s3_m": 0,
    "eod_h": 15, "eod_m": 0,
}
BAR5_RULES = {"WIDE"}   # bar 4 for NARROW+NORMAL, bar 5 only for WIDE


def load_day_ticks(date: str) -> pd.DataFrame:
    p = f"{TICK_DIR}/{EPIC}_{date}.csv"
    if not os.path.exists(p): return pd.DataFrame()
    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(df["utm"], unit="ms", utc=True)
    df = df.set_index("dt")
    df.index = df.index.tz_convert(ZoneInfo(TZ))
    return df


def build_5min_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty: return pd.DataFrame()
    mid = ticks["mid"].resample("5min", label="left", closed="left").ohlc()
    mid.columns = ["Open", "High", "Low", "Close"]
    return mid.dropna().round(1)


def session_bar_levels(bars5: pd.DataFrame, session_h: int, session_m: int) -> dict | None:
    """Return signal bar levels for a session, mimicking live BAR5_RULES=WIDE."""
    open_mins = session_h * 60 + session_m
    sess = bars5[(bars5.index.hour * 60 + bars5.index.minute) >= open_mins]
    if len(sess) < 5:
        return None
    bar4 = sess.iloc[3]
    bar4_range = bar4.High - bar4.Low
    if bar4_range < CFG["narrow_range"]:
        flag = "NARROW"
    elif bar4_range > CFG["wide_range"]:
        flag = "WIDE"
    else:
        flag = "NORMAL"
    sig = bar4
    bar_num = 4
    if flag in BAR5_RULES and len(sess) >= 5:
        sig = sess.iloc[4]
        bar_num = 5
    sig_range = sig.High - sig.Low
    if sig_range > CFG["max_bar_range"] or sig_range <= 0:
        return None
    buy = round(sig.High + CFG["buffer"], 1)
    sell = round(sig.Low - CFG["buffer"], 1)
    return {
        "buy": buy, "sell": sell, "sig_high": sig.High, "sig_low": sig.Low,
        "range": sig_range, "flag": flag, "bar_num": bar_num,
        "sig_close_time": sig.name,
    }


def simulate_entries(ticks: pd.DataFrame, bars5: pd.DataFrame,
                     levels: dict, eod_h: int, eod_m: int,
                     confirm_secs: float) -> list:
    """Find entries with given confirmation delay. Simulate exits with
    Variant B trail (prev_close) + max_risk cap. Returns list of trades."""
    if not levels: return []
    eod_time = ticks.index[-1].replace(hour=eod_h, minute=eod_m, second=0, microsecond=0)
    # Scan ticks after signal bar close
    start = levels["sig_close_time"] + pd.Timedelta(minutes=5)
    scan = ticks[(ticks.index >= start) & (ticks.index < eod_time)]
    if scan.empty: return []

    trades = []
    first_direction = None
    first_pnl = None
    entries = 0
    max_e = CFG["max_entries"]
    buy_level = levels["buy"]; sell_level = levels["sell"]
    bar_range = levels["range"]
    # risk cap
    risk_pts = bar_range + CFG["buffer"] * 2
    if risk_pts > CFG["max_risk_gbp"]:
        risk_pts = CFG["max_risk_gbp"]

    cursor = scan.index[0]
    while entries < max_e and cursor < eod_time:
        subset = scan[scan.index >= cursor]
        if subset.empty: break
        # Direction filter (reverse-reentry AFTER_LOSS, NIKKEI rule)
        allow_long = True; allow_short = True
        if first_direction is not None and entries > 0:
            first_won = first_pnl and first_pnl > 0
            if first_won:
                if first_direction == "LONG": allow_short = False
                else: allow_long = False
        # Find first cross of either allowed level
        # Use OFR for LONG trigger, BID for SHORT trigger (mirrors live R5)
        first_cross_idx = None
        first_cross_dir = None
        first_cross_time = None
        for tstamp, row in subset.iterrows():
            if allow_long and row["ofr"] >= buy_level:
                first_cross_idx = tstamp; first_cross_dir = "LONG"; first_cross_time = tstamp; break
            if allow_short and row["bid"] <= sell_level:
                first_cross_idx = tstamp; first_cross_dir = "SHORT"; first_cross_time = tstamp; break
        if first_cross_idx is None:
            break

        # Apply confirmation: require price to stay at/beyond level for confirm_secs
        if confirm_secs > 0:
            end_confirm = first_cross_time + pd.Timedelta(seconds=confirm_secs)
            window = subset[(subset.index >= first_cross_time) & (subset.index <= end_confirm)]
            # Require ALL ticks in window to still be at/beyond level
            if first_cross_dir == "LONG":
                held = (window["ofr"] >= buy_level).all()
            else:
                held = (window["bid"] <= sell_level).all()
            if not held:
                # Cross failed confirmation — advance cursor past this cross,
                # try again from window end
                cursor = window.index[-1] + pd.Timedelta(seconds=1)
                continue

        # Entry fills at end of confirmation window (or immediately if 0s)
        fill_ts = first_cross_time + pd.Timedelta(seconds=confirm_secs) if confirm_secs > 0 else first_cross_time
        fill_candidates = subset[subset.index <= fill_ts]
        if fill_candidates.empty: break
        fill_row = fill_candidates.iloc[-1]
        fill_price = float(fill_row["ofr"]) if first_cross_dir == "LONG" else float(fill_row["bid"])
        direction = first_cross_dir
        entry_price = fill_price
        # Initial stop = opposite level, tightened to max risk
        if direction == "LONG":
            stop = max(sell_level, entry_price - risk_pts)
        else:
            stop = min(buy_level, entry_price + risk_pts)

        # Walk forward minute by minute with Variant B trail
        post = ticks[(ticks.index > fill_ts) & (ticks.index < eod_time)]
        if post.empty: break
        exit_price = None; exit_ts = None; exit_reason = "EOD"
        # Rebuild prev-close minute markers
        post_5m = post["mid"].resample("5min", label="left", closed="left").ohlc()
        prev_close = None
        for bar_start, bar in post_5m.iterrows():
            if pd.isna(bar["close"]): continue
            # Check stop within bar
            bar_ticks = post[(post.index >= bar_start) & (post.index < bar_start + pd.Timedelta(minutes=5))]
            if bar_ticks.empty: continue
            if direction == "LONG":
                below = bar_ticks[bar_ticks["bid"] <= stop]
                if not below.empty:
                    exit_ts = below.index[0]; exit_price = stop; exit_reason = "STOP"; break
                if bar["close"] - entry_price >= CFG["breakeven_pts"] and stop < entry_price:
                    stop = entry_price
                if prev_close is not None and prev_close > stop:
                    stop = round(prev_close, 1)
            else:  # SHORT
                above = bar_ticks[bar_ticks["ofr"] >= stop]
                if not above.empty:
                    exit_ts = above.index[0]; exit_price = stop; exit_reason = "STOP"; break
                if entry_price - bar["close"] >= CFG["breakeven_pts"] and stop > entry_price:
                    stop = entry_price
                if prev_close is not None and prev_close < stop:
                    stop = round(prev_close, 1)
            prev_close = bar["close"]
        if exit_price is None:
            # EOD close at last tick
            exit_price = ticks.iloc[-1]["mid"]; exit_ts = ticks.index[-1]

        pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        trades.append({
            "direction": direction, "entry": round(entry_price, 1),
            "exit": round(exit_price, 1), "pnl_pts": round(pnl, 1),
            "entry_time": fill_ts.strftime("%H:%M:%S"),
            "exit_time": exit_ts.strftime("%H:%M:%S") if exit_ts else "",
            "reason": exit_reason,
        })
        if first_direction is None:
            first_direction = direction; first_pnl = pnl
        entries += 1
        # Move cursor past exit to look for re-entries
        cursor = exit_ts + pd.Timedelta(seconds=1) if exit_ts else post.index[0]

    return trades


def apply_friction(trades: list) -> list:
    out = []
    for t in trades:
        friction = SPREAD + (SLIP if t["pnl_pts"] < 0 else 0)
        t = dict(t)
        t["pnl_gross"] = t["pnl_pts"]
        t["pnl_net"] = (t["pnl_pts"] - friction) * UPLIFT
        out.append(t)
    return out


def run_day(date: str, confirm_secs: float) -> dict:
    ticks = load_day_ticks(date)
    if ticks.empty: return {"trades": [], "net": 0, "note": "no ticks"}
    bars5 = build_5min_bars(ticks)
    sessions = [(CFG["s1_h"], CFG["s1_m"], "S1"),
                (CFG["s2_h"], CFG["s2_m"], "S2"),
                (CFG["s3_h"], CFG["s3_m"], "S3")]
    all_trades = []
    for h, m, name in sessions:
        lvl = session_bar_levels(bars5, h, m)
        if lvl is None: continue
        trades = simulate_entries(ticks, bars5, lvl, CFG["eod_h"], CFG["eod_m"], confirm_secs)
        for t in trades: t["session"] = name
        all_trades.extend(trades)
    all_trades = apply_friction(all_trades)
    return {
        "trades": all_trades,
        "net_gross": sum(t["pnl_gross"] for t in all_trades),
        "net_friction": sum(t["pnl_net"] for t in all_trades),
    }


def main():
    dates = sorted(
        f.replace(f"{EPIC}_", "").replace(".csv", "")
        for f in os.listdir(TICK_DIR)
        if f.startswith(f"{EPIC}_") and f.endswith(".csv")
    )
    print(f"Days available: {dates}\n")
    variants = [
        ("A 0s  (current)",  0.0),
        ("B 5s  confirm",    5.0),
        ("C 10s confirm",   10.0),
        ("D 30s confirm",   30.0),
        ("E 60s confirm",   60.0),
    ]

    summary = {lbl: {"trades": 0, "wins": 0, "losses": 0,
                     "net_gross": 0, "net_friction": 0, "per_day": {}}
               for lbl, _ in variants}

    for date in dates:
        for lbl, sec in variants:
            r = run_day(date, sec)
            summary[lbl]["trades"] += len(r["trades"])
            summary[lbl]["wins"] += sum(1 for t in r["trades"] if t["pnl_gross"] > 0)
            summary[lbl]["losses"] += sum(1 for t in r["trades"] if t["pnl_gross"] < 0)
            summary[lbl]["net_gross"] += r.get("net_gross", 0)
            summary[lbl]["net_friction"] += r.get("net_friction", 0)
            summary[lbl]["per_day"][date] = r.get("net_friction", 0)

    print(f"{'='*100}")
    print(f"  NIKKEI entry-confirmation comparison — {len(dates)} days of IG tick data")
    print(f"{'='*100}")
    print(f"  {'Variant':<22} {'trades':>8} {'wins':>6} {'losses':>7} "
          f"{'gross':>10} {'post-friction':>14}  {'vs A':>10}")
    baseline = summary[variants[0][0]]["net_friction"]
    for lbl, sec in variants:
        s = summary[lbl]
        wr = s["wins"] / max(s["wins"] + s["losses"], 1) * 100
        delta = s["net_friction"] - baseline
        print(f"  {lbl:<22} {s['trades']:>8} {s['wins']:>6} {s['losses']:>7} "
              f"{s['net_gross']:>+10,.0f} {s['net_friction']:>+14,.0f}  "
              f"{delta:>+10,.0f}  WR {wr:.0f}%")

    print(f"\n  Per-day net (post-friction) by variant:")
    print(f"  {'date':<12} " + " ".join(f"{lbl[:5]:>8}" for lbl, _ in variants))
    for date in dates:
        vals = [f"{summary[lbl]['per_day'].get(date, 0):>+8.0f}" for lbl, _ in variants]
        print(f"  {date:<12} {' '.join(vals)}")


if __name__ == "__main__":
    main()
