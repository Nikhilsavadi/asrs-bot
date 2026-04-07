"""
diag_bars.py — Dump bar 4/5 detection from firstrate data alongside live trades
on the divergent days, so we can see exactly where the bot's view of the
market differed from the validated backtest.
"""
import sqlite3
import pandas as pd
from zoneinfo import ZoneInfo
import backtest as bt
from backtest_firstrate import load_firstrate, FIRSTRATE_FILES

# Match live config
for _cfg in bt.INSTRUMENTS.values():
    _cfg["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0


def dump_bars_for_session(df, day, oh, om, eod_h, eod_m, cfg, label):
    sub = df[df["_date"] == day]
    if sub.empty:
        print(f"  {label}: NO DATA")
        return
    open_mins = oh * 60 + om
    eod_mins = eod_h * 60 + eod_m
    sub = sub.copy()
    sub["mins"] = sub["_hour"] * 60 + sub["_minute"]
    sub["mfo"] = sub["mins"] - open_mins
    sub["bar_num"] = sub["mfo"] // 5 + 1
    bars14 = sub[(sub["mfo"] >= 0) & (sub["bar_num"].between(1, 6))]
    if bars14.empty:
        print(f"  {label}: no bars in session window")
        return

    print(f"  {label}: open={oh:02d}:{om:02d}")
    print(f"    {'time':<8}{'bar':>4}{'open':>10}{'high':>10}{'low':>10}{'close':>10}{'range':>8}")
    for _, r in bars14.iterrows():
        rng = r["High"] - r["Low"]
        print(f"    {r.name.strftime('%H:%M'):<8}{int(r['bar_num']):>4}"
              f"{r['Open']:>10.1f}{r['High']:>10.1f}{r['Low']:>10.1f}{r['Close']:>10.1f}{rng:>8.1f}")

    # Apply ASRS bar 4/5 logic
    b4 = bars14[bars14["bar_num"] == 4]
    if b4.empty:
        print("    -> no bar 4")
        return
    bar4_h, bar4_l = b4.iloc[0]["High"], b4.iloc[0]["Low"]
    bar4_range = bar4_h - bar4_l
    if bar4_range < cfg["narrow_range"]: rf = "NARROW"
    elif bar4_range > cfg["wide_range"]: rf = "WIDE"
    else: rf = "NORMAL"

    sig_h, sig_l, used = bar4_h, bar4_l, "bar4"
    if rf in ("NORMAL", "WIDE"):
        b5 = bars14[bars14["bar_num"] == 5]
        if not b5.empty:
            sig_h = b5.iloc[0]["High"]; sig_l = b5.iloc[0]["Low"]
            used = "bar5"
    sig_range = sig_h - sig_l
    if sig_range < cfg["narrow_range"]: rf2 = "NARROW"
    elif sig_range > cfg["wide_range"]: rf2 = "WIDE"
    else: rf2 = "NORMAL"

    capped = ""
    risk_pts = sig_range + cfg["buffer"] * 2
    if risk_pts > cfg["max_risk_gbp"]:
        capped = f"  [RISK CAP applied: {risk_pts:.1f} → {cfg['max_risk_gbp']}]"
    if sig_range > cfg["max_bar_range"]:
        print(f"    -> SIG {used} range {sig_range:.1f} > max_bar_range {cfg['max_bar_range']} → SKIP DAY")
        return
    buy = round(sig_h + cfg["buffer"], 1)
    sell = round(sig_l - cfg["buffer"], 1)
    print(f"    -> SIG {used}: H={sig_h:.1f} L={sig_l:.1f} range={sig_range:.1f} ({rf2})  "
          f"BUY={buy} SELL={sell}{capped}")


def main():
    cases = [
        ("NIKKEI", "2026-04-06"),
        ("US30",   "2026-04-06"),
        ("DAX",    "2026-04-07"),
        ("NIKKEI", "2026-04-07"),
    ]

    for inst, date in cases:
        cfg = bt.INSTRUMENTS[inst]
        fr = FIRSTRATE_FILES[inst]
        df = load_firstrate(fr["file"], fr["src_tz"], cfg["timezone"])
        target_day = pd.to_datetime(date).date()

        print(f"\n{'=' * 70}")
        print(f"  {inst}  {date}  (firstrate: {fr['file'].split('/')[-1]})")
        print(f"{'=' * 70}")

        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        for s in sessions:
            dump_bars_for_session(
                df, target_day,
                cfg[f"s{s}_open_hour"], cfg[f"s{s}_open_minute"],
                cfg["session_end_hour"], cfg["session_end_minute"],
                cfg, f"S{s}"
            )

        # Live trades for that day
        c = sqlite3.connect("/root/asrs-bot/data/trade_journal.db")
        rows = c.execute(
            "SELECT entry_time, direction, entry_price, exit_price, pnl_pts, "
            "exit_reason, bar_range, range_flag FROM trades "
            "WHERE mode='live' AND instrument=? AND date=? ORDER BY entry_time",
            (inst, date)
        ).fetchall()
        c.close()
        if rows:
            print(f"\n  LIVE TRADES (instrument tz):")
            print(f"    {'time':<8}{'dir':<6}{'entry':>11}{'exit':>11}{'pnl':>8}  reason")
            for r in rows:
                print(f"    {r[0]:<8}{r[1]:<6}{r[2]:>11.1f}{r[3]:>11.1f}"
                      f"{r[4]:>+8.1f}  {r[5]} (br={r[6]} {r[7]})")


if __name__ == "__main__":
    main()
