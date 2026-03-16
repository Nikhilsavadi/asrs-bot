"""
Backtest: Gold 15-min ORB with breakeven + candle trail
Uses xauusd_m5.csv (5-min bars), aggregates to 15-min, applies ORB strategy.
Tests with and without breakeven/trail to show impact.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

# ── Config (matching gold_bot/config.py) ──────────────────────────
CANDLE_TF = 15          # Aggregate to 15-min
RANGE_BARS = 4          # 4 x 15-min = 1 hour opening range
CONFIRMS_REQUIRED = 2
CONFIRM_BODY_RATIO = 0.6
CONFIRM_RANGE_MULT = 1.5
TARGET_R = 2.0
MAX_TRADES_PER_SESSION = 2
EXCLUSION_BARS = 1
SPREAD = 0.4
MIN_RANGE = 2.0
MAX_RANGE = 50.0
BASE_RISK = 25.0        # GBP
BREAKEVEN_R = 0.5       # Move stop to entry at 0.5R profit

SESSIONS = {
    "ASIAN":  {"start": (0, 0),  "end": (7, 0)},
    "LONDON": {"start": (7, 0),  "end": (12, 0)},
    "US":     {"start": (13, 0), "end": (18, 0)},
}
SESSION_OPENS = [(0, 0), (7, 0), (13, 0)]


def load_data(path="gold_bot/xauusd_m5.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.columns = ["Open", "High", "Low", "Close"]
    # Filter out flat bars (market closed)
    df = df[df["High"] != df["Low"]]
    return df


def aggregate_15min(df_5min):
    """Aggregate 5-min bars to 15-min bars aligned to clock boundaries."""
    df = df_5min.resample("15min", closed="left", label="left").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()
    return df


def compute_ema(closes, period=20):
    if len(closes) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = alpha * c + (1 - alpha) * ema
    return ema


def run_backtest(df_15, use_breakeven=True, use_trail=True):
    """Run ORB backtest. Returns list of trade dicts."""
    trades = []
    bars_list = []
    range_avg_window = []

    for i, (ts, bar) in enumerate(df_15.iterrows()):
        bars_list.append(bar)
        if len(bars_list) > 200:
            bars_list = bars_list[-200:]

        # Track range average
        bar_range = bar["High"] - bar["Low"]
        range_avg_window.append(bar_range)
        if len(range_avg_window) > 20:
            range_avg_window = range_avg_window[-20:]

        h, m = ts.hour, ts.minute
        t_mins = h * 60 + m

        # Find active session
        active_session = None
        sess_start = sess_end = 0
        for sname, sconf in SESSIONS.items():
            sh, sm = sconf["start"]
            eh, em = sconf["end"]
            start = sh * 60 + sm
            end = eh * 60 + em
            if start <= t_mins < end:
                active_session = sname
                sess_start = start
                sess_end = end
                break

        if active_session is None:
            continue

        date_str = ts.strftime("%Y-%m-%d")
        sess_key = date_str + "_" + active_session

        # Initialize session tracking
        if not hasattr(run_backtest, "_sessions"):
            run_backtest._sessions = {}

        if sess_key not in run_backtest._sessions:
            run_backtest._sessions[sess_key] = {
                "state": "IDLE", "range_high": 0, "range_low": 0,
                "range_bars": 0, "trades_taken": 0, "daily_pnl_r": 0,
                "direction": None, "entry_price": 0, "stop_level": 0,
                "target_level": 0, "initial_risk": 0, "entry_time": None,
                "breakeven_moved": False, "stake": 0,
            }

        s = run_backtest._sessions[sess_key]
        bars_into = (t_mins - sess_start) // CANDLE_TF

        # ── State machine ──
        if s["state"] == "IDLE":
            if bars_into == 0:
                s["state"] = "BUILDING"
                s["range_high"] = bar["High"]
                s["range_low"] = bar["Low"]
                s["range_bars"] = 1

        elif s["state"] == "BUILDING":
            s["range_high"] = max(s["range_high"], bar["High"])
            s["range_low"] = min(s["range_low"], bar["Low"])
            s["range_bars"] += 1
            if s["range_bars"] >= RANGE_BARS:
                rng = s["range_high"] - s["range_low"]
                if MIN_RANGE <= rng <= MAX_RANGE:
                    s["state"] = "WATCHING"
                else:
                    s["state"] = "IDLE"

        elif s["state"] == "WATCHING":
            # Don't enter last 30 mins
            if t_mins >= sess_end - 30:
                s["state"] = "IDLE"
                continue

            if s["trades_taken"] >= MAX_TRADES_PER_SESSION:
                continue

            # Exclusion bars
            skip = False
            for oh, om in SESSION_OPENS:
                bar_mins = (ts.hour - oh) * 60 + (ts.minute - om)
                if 0 <= bar_mins < EXCLUSION_BARS * CANDLE_TF:
                    skip = True
            if skip:
                continue

            # Breakout detection
            direction = None
            if bar["High"] > s["range_high"]:
                direction = "BUY"
            elif bar["Low"] < s["range_low"]:
                direction = "SELL"

            if direction is None:
                continue

            # Confirmation (2 of 3)
            confirms = 0
            closes = [b["Close"] for b in bars_list[-20:]]
            ema = compute_ema(closes) if len(closes) >= 20 else None
            if ema:
                if (direction == "BUY" and bar["Close"] > ema) or \
                   (direction == "SELL" and bar["Close"] < ema):
                    confirms += 1

            range_avg = np.mean(range_avg_window) if range_avg_window else 0
            if range_avg > 0 and bar_range > CONFIRM_RANGE_MULT * range_avg:
                confirms += 1

            body = abs(bar["Close"] - bar["Open"])
            if bar_range > 0 and body / bar_range > CONFIRM_BODY_RATIO:
                confirms += 1

            if confirms < CONFIRMS_REQUIRED:
                continue

            # Entry
            entry_price = bar["Close"]
            if direction == "BUY":
                stop_level = s["range_low"] - SPREAD
                initial_risk = entry_price - stop_level
            else:
                stop_level = s["range_high"] + SPREAD
                initial_risk = stop_level - entry_price

            if initial_risk <= 0:
                continue

            raw_stake = BASE_RISK / initial_risk
            stake = max(0.5, min(raw_stake, 50.0))

            s["state"] = "ENTERED"
            s["direction"] = direction
            s["entry_price"] = entry_price
            s["stop_level"] = stop_level
            s["target_level"] = (entry_price + initial_risk * TARGET_R
                                 if direction == "BUY"
                                 else entry_price - initial_risk * TARGET_R)
            s["initial_risk"] = initial_risk
            s["entry_time"] = ts
            s["breakeven_moved"] = False
            s["stake"] = stake
            s["trades_taken"] += 1
            s["mfe"] = 0

        elif s["state"] == "ENTERED":
            # Track MFE
            if s["direction"] == "BUY":
                unrealized = bar["High"] - s["entry_price"]
            else:
                unrealized = s["entry_price"] - bar["Low"]
            s["mfe"] = max(s.get("mfe", 0), unrealized)

            exit_price = None
            exit_reason = None

            # Check stop hit
            if s["direction"] == "BUY" and bar["Low"] <= s["stop_level"]:
                exit_price = s["stop_level"]
                exit_reason = "STOP" if not s["breakeven_moved"] else "BREAKEVEN_STOP"
            elif s["direction"] == "SELL" and bar["High"] >= s["stop_level"]:
                exit_price = s["stop_level"]
                exit_reason = "STOP" if not s["breakeven_moved"] else "BREAKEVEN_STOP"

            # Check target hit
            if exit_price is None:
                if s["direction"] == "BUY" and bar["High"] >= s["target_level"]:
                    exit_price = s["target_level"]
                    exit_reason = "TARGET"
                elif s["direction"] == "SELL" and bar["Low"] <= s["target_level"]:
                    exit_price = s["target_level"]
                    exit_reason = "TARGET"

            # Session end close
            if exit_price is None and t_mins >= sess_end - 15:
                exit_price = bar["Close"]
                exit_reason = "SESSION_CLOSE"

            # Breakeven check (before trail)
            if exit_price is None and use_breakeven and not s["breakeven_moved"]:
                if s["direction"] == "BUY":
                    profit = bar["High"] - s["entry_price"]
                else:
                    profit = s["entry_price"] - bar["Low"]

                if profit >= BREAKEVEN_R * s["initial_risk"]:
                    s["breakeven_moved"] = True
                    s["stop_level"] = s["entry_price"]

            # Candle trail (after breakeven)
            if exit_price is None and use_trail and s["breakeven_moved"] and len(bars_list) >= 2:
                prev = bars_list[-2]
                if s["direction"] == "BUY" and prev["Low"] > s["stop_level"]:
                    s["stop_level"] = prev["Low"]
                elif s["direction"] == "SELL" and prev["High"] < s["stop_level"]:
                    s["stop_level"] = prev["High"]

            if exit_price is not None:
                if s["direction"] == "BUY":
                    pnl = exit_price - s["entry_price"]
                else:
                    pnl = s["entry_price"] - exit_price

                pnl_gbp = pnl * s["stake"]
                r_mult = pnl / s["initial_risk"] if s["initial_risk"] > 0 else 0

                trades.append({
                    "date": date_str,
                    "session": active_session,
                    "direction": s["direction"],
                    "entry": s["entry_price"],
                    "exit": exit_price,
                    "pnl_pts": round(pnl, 2),
                    "pnl_gbp": round(pnl_gbp, 2),
                    "r_mult": round(r_mult, 2),
                    "mfe": round(s["mfe"], 2),
                    "exit_reason": exit_reason,
                    "stake": round(s["stake"], 2),
                })
                s["state"] = "WATCHING"  # Can re-enter
                s["breakeven_moved"] = False

    # Clean up
    run_backtest._sessions = {}
    return trades


def print_results(trades, label):
    if not trades:
        print(f"\n{'='*60}")
        print(f"  {label}: NO TRADES")
        return

    df = pd.DataFrame(trades)
    wins = df[df["pnl_pts"] > 0]
    losses = df[df["pnl_pts"] < 0]
    be = df[df["pnl_pts"] == 0]

    total_pnl = df["pnl_gbp"].sum()
    gross_win = wins["pnl_gbp"].sum() if len(wins) else 0
    gross_loss = abs(losses["pnl_gbp"].sum()) if len(losses) else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(df) * 100

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(df)}  |  Wins: {len(wins)}  |  Losses: {len(losses)}  |  BE: {len(be)}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  PF: {pf:.2f}")
    print(f"  Total PnL: GBP {total_pnl:,.2f}")
    print(f"  Avg PnL/trade: GBP {df['pnl_gbp'].mean():.2f}")
    print(f"  Avg Winner: {wins['pnl_pts'].mean():.2f} pts" if len(wins) else "  No winners")
    print(f"  Avg Loser: {losses['pnl_pts'].mean():.2f} pts" if len(losses) else "  No losers")
    print(f"  Avg MFE: {df['mfe'].mean():.2f} pts")
    print(f"  Best: GBP {df['pnl_gbp'].max():.2f}  |  Worst: GBP {df['pnl_gbp'].min():.2f}")

    # By session
    print(f"\n  By Session:")
    for sess in ["ASIAN", "LONDON", "US"]:
        sdf = df[df["session"] == sess]
        if len(sdf):
            sw = sdf[sdf["pnl_pts"] > 0]
            sl = sdf[sdf["pnl_pts"] < 0]
            gw = sw["pnl_gbp"].sum() if len(sw) else 0
            gl = abs(sl["pnl_gbp"].sum()) if len(sl) else 1
            spf = gw / gl if gl > 0 else float("inf")
            print(f"    {sess:8s}: {len(sdf):3d} trades | WR {len(sw)/len(sdf)*100:5.1f}% | PF {spf:.2f} | GBP {sdf['pnl_gbp'].sum():,.2f}")

    # By exit reason
    print(f"\n  By Exit Reason:")
    for reason in df["exit_reason"].unique():
        rdf = df[df["exit_reason"] == reason]
        print(f"    {reason:16s}: {len(rdf):3d} trades | GBP {rdf['pnl_gbp'].sum():,.2f}")


if __name__ == "__main__":
    print("Loading Gold 5-min data...")
    df_5 = load_data()
    print(f"  {len(df_5)} bars: {df_5.index[0]} to {df_5.index[-1]}")

    print("Aggregating to 15-min...")
    df_15 = aggregate_15min(df_5)
    print(f"  {len(df_15)} bars")

    # Run three variants
    print("\nRunning backtests...")

    trades_base = run_backtest(df_15, use_breakeven=False, use_trail=False)
    print_results(trades_base, "BASELINE (no breakeven, no trail)")

    trades_be = run_backtest(df_15, use_breakeven=True, use_trail=False)
    print_results(trades_be, "BREAKEVEN ONLY (0.5R)")

    trades_full = run_backtest(df_15, use_breakeven=True, use_trail=True)
    print_results(trades_full, "BREAKEVEN + CANDLE TRAIL")
