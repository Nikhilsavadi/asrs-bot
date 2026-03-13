"""
backtest_ftse_on_us.py — FTSE 1BN/1BP Strategy on SPY & DIA via Polygon.io
═══════════════════════════════════════════════════════════════════════════════

Adapts the FTSE first-bar fade strategy to US ETFs:
  - RTH: 09:30-16:00 ET
  - Bar 1 = 09:30-09:35 ET (first 5-min candle)
  - 1BN (close < open): BUY below + SELL above
  - 1BP (close > open): SELL above only
  - DOJI: skip
  - Candle trail, add-to-winners, flip after stop-out
  - Thresholds scaled for US price levels

Usage:
    python backtest_ftse_on_us.py          # Both SPY and DIA
    python backtest_ftse_on_us.py SPY      # SPY only
    python backtest_ftse_on_us.py DIA      # DIA only
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta, date
from collections import defaultdict

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(H:%M:%S)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("FTSE_US")

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "arL6Kqp4GoBiLF_x97ovrFeHYS7ilN80")

# ── Config (adapted from FTSE strategy) ─────────────────────────────────────
# FTSE ~8200: BUFFER=1pt (0.012%), BAR_WIDTH_THRESHOLD=30pt (0.37%)
# SPY  ~500:  scale proportionally
# DIA  ~420:  similar scale
BUFFER_PTS        = 0.05     # $0.05 buffer above/below bar 1
BAR_WIDTH_THRESHOLD = 2.00   # Halve stake if bar > $2.00 (≈30pt FTSE scaled)
MAX_ENTRIES       = 2        # Max entries per day (entry + flip)
NUM_CONTRACTS     = 3        # Matches FTSE setup
STAKE_PER_POINT   = 1.0      # $1 per point per contract
DOJI_ACTION       = "SKIP"

# Add-to-winners
ADD_STRENGTH_ENABLED = True
ADD_STRENGTH_TRIGGER = 1.50  # $1.50 ≈ 25pts FTSE scaled
ADD_STRENGTH_MAX     = 2

# Slippage
SLIPPAGE_PER_FILL = 0.02     # $0.02 per fill

TRAIN_RATIO = 0.70
START_DATE = "2024-03-10"
END_DATE = "2026-03-10"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_5min(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(__file__), "data", ticker.lower())
    cache_name = f"{ticker.lower()}_5min_rth_{start_date}_{end_date}.parquet"
    cache_path = os.path.join(data_dir, cache_name)
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded cached {ticker} data: {len(df)} bars")
        return df

    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}

    async with httpx.AsyncClient(timeout=60) as client:
        page = 0
        while url:
            page += 1
            logger.info(f"Fetching {ticker} page {page}...")
            r = await client.get(url, params=params if page == 1 else {"apiKey": POLYGON_KEY})
            if r.status_code == 429:
                logger.warning("Rate limited, waiting 15s...")
                await asyncio.sleep(15)
                continue
            if r.status_code != 200:
                logger.error(f"Polygon error: {r.status_code} {r.text}")
                break
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            logger.info(f"  Got {len(results)} bars (total: {len(all_results)})")
            url = data.get("next_url")
            if url:
                url = url + f"&apiKey={POLYGON_KEY}"
                params = {}
                await asyncio.sleep(0.5)
            else:
                break

    if not all_results:
        logger.error(f"No data for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert("US/Eastern")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df.set_index("datetime_et")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.between_time("09:30", "15:59")

    logger.info(f"Total: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    os.makedirs(data_dir, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"Cached to {cache_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FTSE STRATEGY ENGINE (adapted for US)
# ══════════════════════════════════════════════════════════════════════════════

def get_first_bar(day_df):
    """Get the 09:30 bar (first 5-min candle at open)."""
    bar = day_df[(day_df.index.hour == 9) & (day_df.index.minute == 30)]
    if bar.empty:
        return None
    first = bar.iloc[0]
    return {
        "open":  round(first["Open"], 2),
        "high":  round(first["High"], 2),
        "low":   round(first["Low"], 2),
        "close": round(first["Close"], 2),
        "range": round(first["High"] - first["Low"], 2),
    }


def classify_bar(bar):
    """1BN, 1BP, or DOJI."""
    if bar["close"] < bar["open"]:
        return "1BN"
    elif bar["close"] > bar["open"]:
        return "1BP"
    return "DOJI"


def simulate_entry(day_df, start_time, buy_level, sell_level, direction_hint,
                   session_end, stake, bar_width):
    direction = ""
    entry_price = 0
    trailing_stop = 0
    max_fav = 0
    prev_bar = None

    adds_used = 0
    add_entries = []
    last_add_ref = 0

    trade = {
        "direction": "", "entry": 0, "entry_time": "",
        "exit": 0, "exit_time": "", "exit_reason": "",
        "pnl_pts": 0, "pnl_usd": 0, "mfe": 0,
        "stake": stake, "adds": 0, "slippage_pts": 0,
    }

    for idx, row in day_df.iterrows():
        if idx <= start_time:
            prev_bar = row
            continue

        h, l, c = row["High"], row["Low"], row["Close"]

        # ── Not yet triggered ──
        if not direction:
            if direction_hint in ("LONG", "BOTH") and l <= buy_level:
                direction = "LONG"
                entry_price = buy_level - SLIPPAGE_PER_FILL
                trailing_stop = round(entry_price - bar_width, 2)
            elif direction_hint in ("SHORT", "BOTH") and h >= sell_level:
                direction = "SHORT"
                entry_price = sell_level + SLIPPAGE_PER_FILL
                trailing_stop = round(entry_price + bar_width, 2)
            else:
                prev_bar = row
                continue

            max_fav = entry_price
            last_add_ref = entry_price
            trade["direction"] = direction
            trade["entry"] = entry_price
            trade["entry_time"] = str(idx)
            trade["slippage_pts"] = SLIPPAGE_PER_FILL
            prev_bar = row
            continue

        # ── Position active ──
        # Add-to-winners
        if ADD_STRENGTH_ENABLED and adds_used < ADD_STRENGTH_MAX:
            if direction == "LONG":
                profit_from_ref = h - last_add_ref
            else:
                profit_from_ref = last_add_ref - l

            if profit_from_ref >= ADD_STRENGTH_TRIGGER:
                adds_used += 1
                if direction == "LONG":
                    add_price = round(last_add_ref + ADD_STRENGTH_TRIGGER + SLIPPAGE_PER_FILL, 2)
                else:
                    add_price = round(last_add_ref - ADD_STRENGTH_TRIGGER - SLIPPAGE_PER_FILL, 2)
                add_entries.append(add_price)
                last_add_ref = add_price
                trade["adds"] = adds_used
                trade["slippage_pts"] += SLIPPAGE_PER_FILL

        # MFE
        if direction == "LONG" and h > max_fav:
            max_fav = h
        elif direction == "SHORT" and l < max_fav:
            max_fav = l

        # Check trailing stop
        stopped = False
        if direction == "LONG" and l <= trailing_stop:
            stopped = True
        elif direction == "SHORT" and h >= trailing_stop:
            stopped = True

        if stopped:
            if direction == "LONG":
                exit_price = trailing_stop - SLIPPAGE_PER_FILL
                pnl_pts = round(exit_price - entry_price, 2)
            else:
                exit_price = trailing_stop + SLIPPAGE_PER_FILL
                pnl_pts = round(entry_price - exit_price, 2)

            add_pnl = 0.0
            for ap in add_entries:
                if direction == "LONG":
                    add_pnl += round(exit_price - ap, 2)
                else:
                    add_pnl += round(ap - exit_price, 2)

            total_pnl_pts = round(pnl_pts * NUM_CONTRACTS + add_pnl, 2)
            pnl_usd = round(total_pnl_pts * stake, 2)

            trade["slippage_pts"] += SLIPPAGE_PER_FILL
            mfe = round((max_fav - entry_price if direction == "LONG" else entry_price - max_fav), 2)
            trade.update({"exit": exit_price, "exit_time": str(idx),
                          "exit_reason": "STOPPED", "pnl_pts": total_pnl_pts,
                          "pnl_usd": pnl_usd, "mfe": mfe})
            return trade

        # Candle trail
        if prev_bar is not None:
            if direction == "LONG":
                new_stop = round(prev_bar["Low"], 2)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop
            else:
                new_stop = round(prev_bar["High"], 2)
                if new_stop < trailing_stop:
                    trailing_stop = new_stop

        # Session close (16:00 ET)
        if idx >= session_end:
            if direction == "LONG":
                exit_price = round(c - SLIPPAGE_PER_FILL, 2)
                pnl_pts = round(exit_price - entry_price, 2)
            else:
                exit_price = round(c + SLIPPAGE_PER_FILL, 2)
                pnl_pts = round(entry_price - exit_price, 2)

            add_pnl = 0.0
            for ap in add_entries:
                if direction == "LONG":
                    add_pnl += round(exit_price - ap, 2)
                else:
                    add_pnl += round(ap - exit_price, 2)

            total_pnl_pts = round(pnl_pts * NUM_CONTRACTS + add_pnl, 2)
            pnl_usd = round(total_pnl_pts * stake, 2)

            trade["slippage_pts"] += SLIPPAGE_PER_FILL
            mfe = round((max_fav - entry_price if direction == "LONG" else entry_price - max_fav), 2)
            trade.update({"exit": exit_price, "exit_time": str(idx),
                          "exit_reason": "SESSION_CLOSE", "pnl_pts": total_pnl_pts,
                          "pnl_usd": pnl_usd, "mfe": mfe})
            return trade

        prev_bar = row

    # EOD fallback
    if direction and not trade.get("exit_reason"):
        last_c = day_df["Close"].iloc[-1]
        if direction == "LONG":
            exit_price = round(last_c - SLIPPAGE_PER_FILL, 2)
            pnl_pts = round(exit_price - entry_price, 2)
        else:
            exit_price = round(last_c + SLIPPAGE_PER_FILL, 2)
            pnl_pts = round(entry_price - exit_price, 2)

        add_pnl = 0.0
        for ap in add_entries:
            if direction == "LONG":
                add_pnl += round(exit_price - ap, 2)
            else:
                add_pnl += round(ap - exit_price, 2)

        total_pnl_pts = round(pnl_pts * NUM_CONTRACTS + add_pnl, 2)
        pnl_usd = round(total_pnl_pts * stake, 2)
        mfe = round((max_fav - entry_price if direction == "LONG" else entry_price - max_fav), 2)
        trade.update({"exit": exit_price, "exit_time": str(day_df.index[-1]),
                      "exit_reason": "EOD", "pnl_pts": total_pnl_pts,
                      "pnl_usd": pnl_usd, "mfe": mfe})
        return trade

    if not direction:
        trade["exit_reason"] = "NO_TRIGGER"
    return trade


def simulate_day(day_df):
    first_bar = get_first_bar(day_df)
    if not first_bar:
        return None

    bar_type = classify_bar(first_bar)
    if bar_type == "DOJI":
        return {
            "date": str(day_df.index[0].date()),
            "bar_type": bar_type, "bar": first_bar,
            "trades": [], "total_pnl_usd": 0, "skip_reason": "DOJI",
        }

    bar_width = first_bar["range"]

    stake = STAKE_PER_POINT
    stake_halved = False
    if bar_width > BAR_WIDTH_THRESHOLD:
        stake = round(STAKE_PER_POINT / 2, 2)
        stake_halved = True

    buy_level = round(first_bar["low"] - BUFFER_PTS, 2)
    sell_level = round(first_bar["high"] + BUFFER_PTS, 2)

    # 1BN: BUY + SELL, 1BP: SELL only
    if bar_type == "1BN":
        direction_hint = "BOTH"
    else:
        direction_hint = "SHORT"

    session_end = day_df.index[0].replace(hour=15, minute=55, second=0, microsecond=0)
    bar_time = day_df.index[0].replace(hour=9, minute=30, second=0, microsecond=0)

    trades = []

    t1 = simulate_entry(day_df, bar_time, buy_level, sell_level, direction_hint,
                         session_end, stake, bar_width)
    if t1["direction"]:
        trades.append(t1)

        if t1["exit_reason"] == "STOPPED" and len(trades) < MAX_ENTRIES:
            flip_dir = "SHORT" if t1["direction"] == "LONG" else "LONG"
            exit_time = pd.Timestamp(t1["exit_time"])
            t2 = simulate_entry(day_df, exit_time, buy_level, sell_level,
                                flip_dir, session_end, stake, bar_width)
            if t2["direction"]:
                trades.append(t2)

    total_pnl_usd = sum(t.get("pnl_usd", 0) for t in trades)

    return {
        "date": str(day_df.index[0].date()),
        "bar_type": bar_type,
        "bar": first_bar,
        "bar_width": bar_width,
        "stake": stake,
        "stake_halved": stake_halved,
        "buy_level": buy_level,
        "sell_level": sell_level,
        "trades": trades,
        "total_pnl_usd": round(total_pnl_usd, 2),
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def calc_stats(results):
    all_trades = []
    daily_pnl = []

    for r in results:
        if r is None:
            daily_pnl.append(0)
            continue
        daily_pnl.append(r.get("total_pnl_usd", 0))
        all_trades.extend(r.get("trades", []))

    completed = [t for t in all_trades if t.get("exit_reason") not in ("", "NO_TRIGGER")]
    total_pnl = sum(t.get("pnl_usd", 0) for t in completed)
    wins = [t for t in completed if t.get("pnl_usd", 0) >= 0]
    losses = [t for t in completed if t.get("pnl_usd", 0) < 0]

    daily_pnl = np.array(daily_pnl)
    cum_pnl = np.cumsum(daily_pnl)
    max_dd = 0
    peak = 0
    for p in cum_pnl:
        if p > peak:
            peak = p
        dd = peak - p
        if dd > max_dd:
            max_dd = dd

    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    avg_win = round(np.mean([t["pnl_usd"] for t in wins]), 2) if wins else 0
    avg_loss = round(np.mean([t["pnl_usd"] for t in losses]), 2) if losses else 0

    return {
        "total_trades": len(completed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(completed) * 100, 1) if completed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "max_drawdown": round(max_dd, 2),
        "daily_pnl": daily_pnl,
    }


def print_stats(label, stats, num_days):
    print(f"\n{'━' * 65}")
    print(f"  {label}")
    print(f"{'━' * 65}")
    print(f"  Days:           {num_days}")
    print(f"  Trades:         {stats['total_trades']}")
    print(f"  Wins:           {stats['wins']}")
    print(f"  Losses:         {stats['losses']}")
    print(f"  Win rate:       {stats['win_rate']}%")
    ps = '+' if stats['total_pnl'] >= 0 else ''
    print(f"  Net P&L:        {ps}${stats['total_pnl']:,.2f}")
    per_day = round(stats['total_pnl'] / num_days, 2) if num_days > 0 else 0
    print(f"  Per day avg:    {'+' if per_day >= 0 else ''}${per_day:,.2f}")
    print(f"  Avg winner:     +${stats['avg_win']:,.2f}")
    print(f"  Avg loser:      ${stats['avg_loss']:,.2f}")
    print(f"  Profit factor:  {stats['profit_factor']}")
    print(f"  Max drawdown:   ${stats['max_drawdown']:,.2f}")
    print(f"{'━' * 65}")


async def run_backtest(ticker: str):
    print(f"\n{'═' * 70}")
    print(f"  FTSE 1BN/1BP STRATEGY on {ticker}")
    print(f"{'═' * 70}")

    df = await fetch_5min(ticker, START_DATE, END_DATE)
    if df.empty:
        print(f"No data for {ticker}")
        return

    trading_days = sorted(set(df.index.date))
    trading_days = [d for d in trading_days if d.weekday() < 5]
    print(f"Total trading days: {len(trading_days)}")

    split_idx = int(len(trading_days) * TRAIN_RATIO)
    train_days = trading_days[:split_idx]
    test_days = trading_days[split_idx:]

    print(f"Train: {len(train_days)} days ({train_days[0]} to {train_days[-1]})")
    print(f"Test:  {len(test_days)} days ({test_days[0]} to {test_days[-1]})")
    print(f"\nConfig: {NUM_CONTRACTS} contracts × ${STAKE_PER_POINT}/pt")
    print(f"  Buffer: ${BUFFER_PTS} | Halve stake if bar > ${BAR_WIDTH_THRESHOLD}")
    print(f"  1BN: BUY + SELL | 1BP: SELL only | DOJI: SKIP")
    print(f"  Add +1 every +${ADD_STRENGTH_TRIGGER} (max {ADD_STRENGTH_MAX})")
    print(f"  Slippage: ${SLIPPAGE_PER_FILL}/fill")
    print(f"  Session: 09:30-16:00 ET")

    all_results = []
    train_results = []
    test_results = []
    dojis = bn_count = bp_count = 0

    for i, day in enumerate(trading_days):
        day_df = df[df.index.date == day]
        if day_df.empty:
            continue

        r = simulate_day(day_df)
        all_results.append(r)

        if r:
            if r.get("skip_reason") == "DOJI":
                dojis += 1
            if r.get("bar_type") == "1BN":
                bn_count += 1
            elif r.get("bar_type") == "1BP":
                bp_count += 1

        if day in train_days:
            train_results.append(r)
        else:
            test_results.append(r)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(trading_days)} days")

    print(f"\nBar types: {bn_count} 1BN, {bp_count} 1BP, {dojis} DOJI (skipped)")

    all_stats = calc_stats(all_results)
    print_stats(f"{ticker} — ALL ({len(trading_days)} days)", all_stats, len(trading_days))

    train_stats = calc_stats(train_results)
    print_stats(f"{ticker} — TRAIN ({len(train_days)} days)", train_stats, len(train_days))

    test_stats = calc_stats(test_results)
    print_stats(f"{ticker} — TEST ({len(test_days)} days)", test_stats, len(test_days))

    # ── Bar type breakdown ─────────────────────────────────────────────────
    print(f"\n  P&L BY BAR TYPE:")
    for bt in ["1BN", "1BP"]:
        bt_results = [r for r in all_results if r and r.get("bar_type") == bt]
        bt_pnl = sum(r.get("total_pnl_usd", 0) for r in bt_results)
        bt_trades = sum(len(r.get("trades", [])) for r in bt_results)
        bt_wins = sum(1 for r in bt_results for t in r.get("trades", []) if t.get("pnl_usd", 0) >= 0)
        wr = round(bt_wins / bt_trades * 100, 1) if bt_trades > 0 else 0
        ps = "+" if bt_pnl >= 0 else ""
        print(f"    {bt}: {len(bt_results)} days, {bt_trades} trades, WR={wr}%, P&L={ps}${bt_pnl:,.2f}")

    # ── Monthly breakdown ──────────────────────────────────────────────────
    print(f"\n  {'Month':<10} {'P&L ($)':>12} {'Trades':>8} {'WR':>6} {'Set':<6}")
    print(f"  {'─'*10} {'─'*12} {'─'*8} {'─'*6} {'─'*6}")

    monthly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    monthly_set = {}
    for r in all_results:
        if r is None:
            continue
        month = r["date"][:7]
        monthly[month]["pnl"] += r.get("total_pnl_usd", 0)
        monthly[month]["trades"] += len(r.get("trades", []))
        monthly[month]["wins"] += sum(1 for t in r.get("trades", []) if t.get("pnl_usd", 0) >= 0)

    for day in trading_days:
        month = str(day)[:7]
        monthly_set[month] = "TRAIN" if day in train_days else "TEST"

    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = round(m["wins"] / m["trades"] * 100, 0) if m["trades"] > 0 else 0
        ps = "+" if m["pnl"] >= 0 else ""
        set_label = monthly_set.get(month, "?")
        print(f"  {month:<10} {ps}${round(m['pnl'], 2):>10,.2f} {m['trades']:>8} {wr:>5}% {set_label:<6}")


async def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["SPY", "DIA"]
    for ticker in tickers:
        await run_backtest(ticker.upper())


if __name__ == "__main__":
    asyncio.run(main())
