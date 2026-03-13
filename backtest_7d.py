"""
backtest_7d.py — 7-day DAX ASRS backtest using IG historical data
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()
os.chdir("/root/asrs-bot")

TZ_CET = ZoneInfo("Europe/Berlin")
TZ_UK = ZoneInfo("Europe/London")

BUFFER_PTS = 2
NARROW_RANGE = 15
WIDE_RANGE = 40
SESSION_END_H = 17
SESSION_END_M = 30


def candle_number(ts):
    open_time = ts.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((ts - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def get_bar(day_df, n):
    for idx, row in day_df.iterrows():
        cn = candle_number(idx)
        if cn == n:
            return {
                "high": round(row["High"], 1),
                "low": round(row["Low"], 1),
                "open": round(row["Open"], 1),
                "close": round(row["Close"], 1),
                "range": round(row["High"] - row["Low"], 1),
                "time": idx,
            }
    return None


def simulate_day(day_df, prev_close=None):
    """
    Simulate one trading day:
    1. Get bar 4 → set BUY/SELL levels
    2. Walk through subsequent bars checking for triggers
    3. Once triggered, trail stop using previous completed bar low/high
    4. Session close at 17:30 CET (16:30 UK) if still open
    """
    result = {
        "date": str(day_df.index[0].date()),
        "bar4": None,
        "range_flag": "",
        "buy_level": 0, "sell_level": 0,
        "direction": "", "entry_price": 0, "entry_time": "",
        "exit_price": 0, "exit_time": "", "exit_reason": "",
        "pnl": 0, "mfe": 0, "mae": 0,
        "trail_stops": [],
        "flip": None,
    }

    bar4 = get_bar(day_df, 4)
    if not bar4:
        result["exit_reason"] = "NO_BAR4"
        return result

    result["bar4"] = bar4
    rng = bar4["range"]
    if rng < NARROW_RANGE:
        result["range_flag"] = "NARROW"
    elif rng > WIDE_RANGE:
        result["range_flag"] = "WIDE"
    else:
        result["range_flag"] = "NORMAL"

    buy_level = round(bar4["high"] + BUFFER_PTS, 1)
    sell_level = round(bar4["low"] - BUFFER_PTS, 1)
    result["buy_level"] = buy_level
    result["sell_level"] = sell_level

    # Walk bars after bar 4
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    max_adv = 0  # MAE
    prev_bar = None

    # Session end in CET = 17:30
    session_end = day_df.index[0].replace(hour=17, minute=30, second=0, microsecond=0)

    for idx, row in day_df.iterrows():
        cn = candle_number(idx)
        if cn <= 4:
            if cn >= 1:
                prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        if not direction:
            # Check for trigger
            if h >= buy_level:
                direction = "LONG"
                entry_price = buy_level  # assume fill at stop level
                trailing_stop = sell_level  # initial stop = opposite level
                max_fav = entry_price
                result["direction"] = "LONG"
                result["entry_price"] = entry_price
                result["entry_time"] = str(idx)
                prev_bar = row
                continue
            elif l <= sell_level:
                direction = "SHORT"
                entry_price = sell_level
                trailing_stop = buy_level
                max_fav = entry_price
                result["direction"] = "SHORT"
                result["entry_price"] = entry_price
                result["entry_time"] = str(idx)
                prev_bar = row
                continue
            prev_bar = row
            continue

        # We have a position — check stop hit on THIS bar
        if direction == "LONG":
            if l <= trailing_stop:
                result["exit_price"] = trailing_stop
                result["exit_time"] = str(idx)
                result["exit_reason"] = "STOPPED"
                result["pnl"] = round(trailing_stop - entry_price, 1)
                result["mfe"] = round(max_fav - entry_price, 1)
                result["mae"] = round(max_adv, 1)

                # Can we flip? (entry 2)
                flip = simulate_flip(day_df, idx, buy_level, sell_level, "SHORT", session_end)
                result["flip"] = flip
                return result

            # Update MFE
            if h > max_fav:
                max_fav = h
            if entry_price - l > max_adv:
                max_adv = entry_price - l

            # Trail: use previous bar's low (only ratchet up)
            if prev_bar is not None:
                new_stop = round(prev_bar["Low"], 1)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop
                    result["trail_stops"].append((str(idx), trailing_stop))

        elif direction == "SHORT":
            if h >= trailing_stop:
                result["exit_price"] = trailing_stop
                result["exit_time"] = str(idx)
                result["exit_reason"] = "STOPPED"
                result["pnl"] = round(entry_price - trailing_stop, 1)
                result["mfe"] = round(entry_price - max_fav, 1)
                result["mae"] = round(max_adv, 1)

                flip = simulate_flip(day_df, idx, buy_level, sell_level, "LONG", session_end)
                result["flip"] = flip
                return result

            if l < max_fav:
                max_fav = l
            if h - entry_price > max_adv:
                max_adv = h - entry_price

            if prev_bar is not None:
                new_stop = round(prev_bar["High"], 1)
                if new_stop < trailing_stop:
                    trailing_stop = new_stop
                    result["trail_stops"].append((str(idx), trailing_stop))

        # Session close check
        if idx >= session_end:
            result["exit_price"] = round(c, 1)
            result["exit_time"] = str(idx)
            result["exit_reason"] = "SESSION_CLOSE"
            if direction == "LONG":
                result["pnl"] = round(c - entry_price, 1)
                result["mfe"] = round(max_fav - entry_price, 1)
            else:
                result["pnl"] = round(entry_price - c, 1)
                result["mfe"] = round(entry_price - max_fav, 1)
            result["mae"] = round(max_adv, 1)
            return result

        prev_bar = row

    # End of data without exit
    if direction:
        last_c = day_df["Close"].iloc[-1]
        result["exit_price"] = round(last_c, 1)
        result["exit_time"] = str(day_df.index[-1])
        result["exit_reason"] = "EOD_DATA"
        if direction == "LONG":
            result["pnl"] = round(last_c - entry_price, 1)
            result["mfe"] = round(max_fav - entry_price, 1)
        else:
            result["pnl"] = round(entry_price - last_c, 1)
            result["mfe"] = round(entry_price - max_fav, 1)
        result["mae"] = round(max_adv, 1)
    else:
        result["exit_reason"] = "NO_TRIGGER"

    return result


def simulate_flip(day_df, from_idx, buy_level, sell_level, flip_dir, session_end):
    """After a stop-out, simulate the flip entry (re-entry in opposite direction)."""
    result = {
        "direction": flip_dir,
        "entry_price": buy_level if flip_dir == "LONG" else sell_level,
        "exit_price": 0, "exit_reason": "", "pnl": 0, "mfe": 0,
    }

    entry_price = result["entry_price"]
    trailing_stop = sell_level if flip_dir == "LONG" else buy_level
    max_fav = entry_price
    direction = flip_dir
    entered = False
    prev_bar = None

    for idx, row in day_df.iterrows():
        if idx <= from_idx:
            prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        if not entered:
            # Check for re-trigger
            if direction == "LONG" and h >= buy_level:
                entered = True
            elif direction == "SHORT" and l <= sell_level:
                entered = True
            else:
                prev_bar = row
                continue

        # Position active
        if direction == "LONG":
            if l <= trailing_stop:
                result["exit_price"] = trailing_stop
                result["exit_reason"] = "STOPPED"
                result["pnl"] = round(trailing_stop - entry_price, 1)
                result["mfe"] = round(max_fav - entry_price, 1)
                return result
            if h > max_fav:
                max_fav = h
            if prev_bar is not None:
                new_stop = round(prev_bar["Low"], 1)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop
        else:
            if h >= trailing_stop:
                result["exit_price"] = trailing_stop
                result["exit_reason"] = "STOPPED"
                result["pnl"] = round(entry_price - trailing_stop, 1)
                result["mfe"] = round(entry_price - max_fav, 1)
                return result
            if l < max_fav:
                max_fav = l
            if prev_bar is not None:
                new_stop = round(prev_bar["High"], 1)
                if new_stop < trailing_stop:
                    trailing_stop = new_stop

        if idx >= session_end:
            result["exit_price"] = round(c, 1)
            result["exit_reason"] = "SESSION_CLOSE"
            if direction == "LONG":
                result["pnl"] = round(c - entry_price, 1)
                result["mfe"] = round(max_fav - entry_price, 1)
            else:
                result["pnl"] = round(entry_price - c, 1)
                result["mfe"] = round(entry_price - max_fav, 1)
            return result

        prev_bar = row

    if entered:
        last_c = day_df["Close"].iloc[-1]
        result["exit_price"] = round(last_c, 1)
        result["exit_reason"] = "EOD_DATA"
        if direction == "LONG":
            result["pnl"] = round(last_c - entry_price, 1)
        else:
            result["pnl"] = round(entry_price - last_c, 1)
    else:
        result["exit_reason"] = "NO_TRIGGER"

    return result


async def main():
    import pandas as pd

    # Load from cached parquet (already in CET, 624 trading days)
    CSV_PATH = "/root/asrs-bot/data/dax_5min_cache.csv"
    print("Loading cached data...")
    out = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    # Index is already CET (naive) — treat as CET

    print(f"Fetched {len(out)} bars ({out.index[0].date()} to {out.index[-1].date()})")

    # Group by trading day
    trading_days = sorted(set(out.index.date))
    # Only keep weekdays
    trading_days = [d for d in trading_days if d.weekday() < 5]
    # Last 7
    trading_days = trading_days[-7:]

    print(f"Backtesting {len(trading_days)} trading days\n")
    print("=" * 90)

    total_pnl = 0
    total_trades = 0
    wins = 0
    losses = 0
    results = []

    prev_close = None
    for day in trading_days:
        day_df = out[out.index.date == day]
        if day_df.empty:
            continue

        r = simulate_day(day_df, prev_close)
        results.append(r)

        bar4 = r["bar4"]
        if not bar4:
            print(f"\n{day} | NO BAR 4 — skipped")
            prev_close = day_df["Close"].iloc[-1] if not day_df.empty else prev_close
            continue

        day_pnl = r["pnl"]
        flip_pnl = 0
        flip_str = ""

        if r["flip"] and r["flip"]["exit_reason"] not in ("NO_TRIGGER", ""):
            flip_pnl = r["flip"]["pnl"]
            flip_str = (
                f"  FLIP → {r['flip']['direction']} @ {r['flip']['entry_price']} "
                f"→ {r['flip']['exit_price']} ({r['flip']['exit_reason']}) "
                f"= {'+' if flip_pnl >= 0 else ''}{flip_pnl} pts"
            )

        combined = day_pnl + flip_pnl
        total_pnl += combined

        if r["direction"]:
            total_trades += 1
            if day_pnl >= 0:
                wins += 1
            else:
                losses += 1

        if r["flip"] and r["flip"]["exit_reason"] not in ("NO_TRIGGER", ""):
            total_trades += 1
            if flip_pnl >= 0:
                wins += 1
            else:
                losses += 1

        icon = "✅" if combined >= 0 else "❌" if r["direction"] else "⏸️"
        pnl_sign = "+" if combined >= 0 else ""

        print(f"\n{'─' * 90}")
        print(f"{icon} {day} | Bar4: {bar4['high']}/{bar4['low']} ({bar4['range']}pts {r['range_flag']})")
        print(f"  Levels: BUY {r['buy_level']} / SELL {r['sell_level']}")

        if r["direction"]:
            print(
                f"  Entry #{1}: {r['direction']} @ {r['entry_price']} "
                f"→ {r['exit_price']} ({r['exit_reason']}) "
                f"= {'+' if day_pnl >= 0 else ''}{day_pnl} pts "
                f"| MFE: {r['mfe']} | MAE: {r['mae']}"
            )
            if r["trail_stops"]:
                stops_str = " → ".join([f"{s[1]}" for s in r["trail_stops"][-5:]])
                print(f"  Trail: {stops_str}")
        else:
            print(f"  {r['exit_reason']}")

        if flip_str:
            print(flip_str)

        print(f"  Day total: {pnl_sign}{combined} pts | Running: {'+' if total_pnl >= 0 else ''}{total_pnl} pts")

        prev_close = day_df["Close"].iloc[-1]

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  7-DAY BACKTEST SUMMARY — DAX ASRS")
    print(f"{'=' * 90}")
    print(f"  Days:       {len(trading_days)}")
    print(f"  Trades:     {total_trades}")
    print(f"  Wins:       {wins}")
    print(f"  Losses:     {losses}")
    wr = round(wins / total_trades * 100, 1) if total_trades else 0
    print(f"  Win rate:   {wr}%")
    print(f"  Net P&L:    {'+' if total_pnl >= 0 else ''}{total_pnl} pts")
    print(f"  Per trade:  {'+' if total_pnl >= 0 else ''}{round(total_pnl / total_trades, 1) if total_trades else 0} pts")
    print(f"  At €3/pt:   {'+' if total_pnl >= 0 else ''}€{round(total_pnl * 3, 1)}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    asyncio.run(main())
