"""
nightly_backtest_vs_live.py — proper A/B comparison.

Uses the REAL backtest engine (sim_with_fade: Variant B trail, honest re-entry,
fade layer, milestone_lock, range filter) against IG historical 5-min + 1-min
bars for a specific date. Compares per-instrument net to live journal.

Usage:
    python3 nightly_backtest_vs_live.py               # today (UTC)
    python3 nightly_backtest_vs_live.py 2026-04-21    # specific date

Intended for cron at 22:30 UTC.
"""
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from shared.ig_session import IGSharedSession
import backtest as bt
from backtest_v2_fade import sim_with_fade


def filter_reverse_reentry(df: pd.DataFrame) -> pd.DataFrame:
    """Drop opposite-direction re-entries per instrument policy.
    DAX/US30: never  |  NIKKEI: after_loss (keep opposite only if first LOST).
    Orders trades by exit_idx (chronological), not entry price."""
    df = df.sort_values(["date", "signal", "exit_idx"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"})
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    opp = df["is_re"] & ~df["same"]
    drop = opp & (df["instrument"].isin(["DAX", "US30"])
                  | ((df["instrument"] == "NIKKEI") & df["first_won"]))
    return df[~drop].drop(columns=["tn", "fd", "fp", "first_won", "is_re", "same"])

for _c in bt.INSTRUMENTS.values():
    _c["max_entries"] = 3
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["session_end_hour"] = 15
bt.INSTRUMENTS["NIKKEI"]["session_end_minute"] = 0

# Mirror live config: fade currently only on US30 (canary 2026-04-20)
bt.INSTRUMENTS["DAX"]["fade_on_trail_win"] = False
bt.INSTRUMENTS["US30"]["fade_on_trail_win"] = True
bt.INSTRUMENTS["NIKKEI"]["fade_on_trail_win"] = False

INST_META = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin"},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York"},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo"},
}
SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP   = {"DAX": 1.0, "US30": 3.0,  "NIKKEI": 4.0}

DB = "/root/asrs-bot/data/trade_journal.db"


TICK_DIR = "/root/asrs-bot/data/ticks"


def load_ticks_for_date(epic: str, date: str, tz_str: str) -> pd.DataFrame:
    """Load tick CSV for the date and return with local-tz datetime index."""
    path = f"{TICK_DIR}/{epic}_{date}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["utm"], unit="ms", utc=True)
    df = df.set_index("dt")
    df.index = df.index.tz_convert(ZoneInfo(tz_str))
    return df


def build_bars_from_ticks(ticks: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate ticks to OHLC bars at given pandas rule ('5min' / '1min').
    Uses mid price."""
    if ticks.empty: return pd.DataFrame()
    mid = ticks["mid"].resample(rule, label="left", closed="left").ohlc()
    mid.columns = ["Open", "High", "Low", "Close"]
    mid = mid.dropna()
    return mid.round(1)


def fetch_bars_local(epic: str, date: str, tz_str: str, resolution: str) -> pd.DataFrame:
    """Build OHLC bars from locally-recorded IG tick CSVs (no allowance hit)."""
    ticks = load_ticks_for_date(epic, date, tz_str)
    if ticks.empty:
        return pd.DataFrame()
    rule = "5min" if resolution == "MINUTE_5" else "1min"
    return build_bars_from_ticks(ticks, rule)


def bars_to_numpy(df: pd.DataFrame):
    """Return (ohlc, hours, minutes) numpy arrays expected by sim_with_fade."""
    if df.empty: return None, None, None
    ohlc = df[["Open", "High", "Low", "Close"]].values.astype(float)
    hours = df.index.hour.values.astype(int)
    minutes = df.index.minute.values.astype(int)
    return ohlc, hours, minutes


def apply_friction(trades: list, inst: str) -> list:
    """Subtract spread + slippage cost per trade."""
    out = []
    spread = SPREAD[inst]; slip = SLIP[inst]
    for t in trades:
        t = dict(t)
        t["pnl_gross"] = t["pnl_pts"]
        friction = spread + (slip if t["pnl_pts"] < 0 else 0)
        t["pnl_net"] = t["pnl_pts"] - friction
        out.append(t)
    return out


def run_backtest_for_date(date: str, bars5: dict, bars1: dict) -> dict:
    """Run sim_with_fade for all instruments + sessions on given date bars."""
    results = {}
    for inst, meta in INST_META.items():
        cfg = bt.INSTRUMENTS[inst]
        sessions = [s for s in (1, 2, 3) if f"s{s}_open_hour" in cfg]
        trades_all = []
        d5 = bars5.get(inst)
        d1 = bars1.get(inst)
        if d5 is None or d5.empty:
            results[inst] = []
            continue
        o5, h5, m5 = bars_to_numpy(d5)
        o1 = h1 = m1 = None
        if d1 is not None and not d1.empty:
            o1, h1, m1 = bars_to_numpy(d1)
            m1_abs = h1 * 60 + m1 if h1 is not None else None
        else:
            m1_abs = None
        for s in sessions:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            ts = sim_with_fade(
                o5, h5, m5, oh, om, eh, em, cfg,
                day_1min=o1, one_min_hours=h1, one_min_minutes=m1_abs,
                fade_stop=50, fade_target_mode="extreme",
            )
            for t in ts:
                t["session"] = s
                t["signal"] = f"{inst}_S{s}_FADE" if t.get("is_fade") else f"{inst}_S{s}"
            trades_all.extend(ts)
        base = [t for t in trades_all if not t.get("is_fade")]
        fade_raw = [t for t in trades_all if t.get("is_fade")]

        # Apply reverse-reentry filter via validated filter_combined.
        # Without this, backtest includes phantom SHORTs the live bot blocks.
        if base:
            df_base = pd.DataFrame(base)
            df_base["instrument"] = inst
            df_base["signal"] = df_base["session"].apply(lambda s: f"{inst}_S{s}")
            df_base["date"] = date
            df_base["entry"] = df_base["entry"].astype(float)
            df_filt = filter_reverse_reentry(df_base)
            base = df_filt.to_dict("records")

        fade_enabled_live = bt.INSTRUMENTS[inst].get("fade_on_trail_win", False)
        fade_f = []
        if fade_enabled_live:
            base_wins = [t["pnl_pts"] for t in base
                         if t.get("reason") == "TRAIL_WIN" and t["pnl_pts"] >= 20]
            if base_wins:
                fade_f = fade_raw
        results[inst] = apply_friction(base + fade_f, inst)
    return results


def load_live_trades(date: str) -> dict:
    """Pull live trades from journal for a given date."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        """SELECT instrument, direction, entry_price, exit_price,
                  pnl_pts, exit_reason, stake_per_point, signal_type
           FROM trades WHERE mode='live' AND date=?
           ORDER BY entry_time""",
        conn, params=(date,),
    )
    conn.close()
    out = {"DAX": [], "US30": [], "NIKKEI": []}
    for _, r in df.iterrows():
        out.setdefault(r["instrument"], []).append({
            "direction": r["direction"],
            "entry": r["entry_price"], "exit": r["exit_price"],
            "pnl_pts": r["pnl_pts"], "reason": r["exit_reason"],
            "stake": r.get("stake_per_point", 0.5),
            "is_fade": (r.get("signal_type") == "FADE"
                        or (isinstance(r.get("exit_reason"), str) and r["exit_reason"].startswith("FADE"))),
        })
    return out


def fmt_trades(trades: list, label: str) -> str:
    if not trades: return f"  {label}: (no trades)\n"
    lines = [f"  {label}: {len(trades)} trades  "
             f"net_gross={sum(t['pnl_pts'] for t in trades):+.0f}pt  "
             f"net_after_friction={sum(t.get('pnl_net', t['pnl_pts']) for t in trades):+.0f}pt"]
    for t in trades:
        fade_tag = " [FADE]" if t.get("is_fade") else ""
        reason = t.get("reason", "")
        dir_str = t.get("direction", "")
        entry = t.get("entry", 0)
        exit_ = t.get("exit", 0)
        pnl = t.get("pnl_pts", 0)
        lines.append(f"    {dir_str:<5}  {entry:>9.1f} → {exit_:>9.1f}  {pnl:>+6.0f}pt  {reason}{fade_tag}")
    return "\n".join(lines) + "\n"


async def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Backtest vs Live — {date}\n" + "=" * 80)

    bars5 = {}
    bars1 = {}
    for inst, meta in INST_META.items():
        print(f"  loading {inst} ticks ...", flush=True)
        bars5[inst] = fetch_bars_local(meta["epic"], date, meta["tz"], "MINUTE_5")
        bars1[inst] = fetch_bars_local(meta["epic"], date, meta["tz"], "MINUTE")
        print(f"    5m bars: {len(bars5[inst])}  1m bars: {len(bars1[inst])}")

    bt_results = run_backtest_for_date(date, bars5, bars1)
    live_results = load_live_trades(date)

    tot_bt_gross = tot_bt_net = tot_live = 0.0
    summary_lines = []
    for inst in ["DAX", "US30", "NIKKEI"]:
        bt_trades = bt_results.get(inst, [])
        live_trades = live_results.get(inst, [])
        bt_gross = sum(t["pnl_pts"] for t in bt_trades)
        bt_net = sum(t.get("pnl_net", t["pnl_pts"]) for t in bt_trades)
        live_net = sum(t["pnl_pts"] for t in live_trades)
        delta = live_net - bt_net
        tot_bt_gross += bt_gross; tot_bt_net += bt_net; tot_live += live_net

        print(f"\n---- {inst} ----")
        print(fmt_trades(bt_trades, "BACKTEST (post-friction)"))
        print(fmt_trades(live_trades, "LIVE     "))
        print(f"  BT gross {bt_gross:+.0f}pt → BT net {bt_net:+.0f}pt  |  Live {live_net:+.0f}pt  |  "
              f"Delta (Live−BT) {delta:+.0f}pt")
        summary_lines.append(f"{inst}: BT {bt_net:+.0f} | Live {live_net:+.0f} | Δ {delta:+.0f}")

    total_delta = tot_live - tot_bt_net
    print("\n" + "=" * 80)
    print(f"  TOTAL  BT gross {tot_bt_gross:+.0f}  BT net {tot_bt_net:+.0f}  "
          f"Live {tot_live:+.0f}  Delta (Live−BT) {total_delta:+.0f}")
    print("=" * 80)

    # Telegram
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        import requests
        msg = (
            f"📊 <b>Nightly Backtest vs Live — {date}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(summary_lines) +
            f"\n━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Total: BT {tot_bt_net:+.0f} | Live {tot_live:+.0f} | Δ {total_delta:+.0f}pt</b>\n"
            f"<i>BT = Variant B + fade + milestone, post-friction (real engine)</i>"
        )
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                data={"chat_id": tg_chat, "text": msg, "parse_mode": "HTML"},
                timeout=10,
            )
            if r.status_code == 200:
                print("  Telegram posted ✓")
            else:
                print(f"  Telegram failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"  Telegram error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
