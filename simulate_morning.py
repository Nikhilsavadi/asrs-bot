#!/usr/bin/env python3
"""
simulate_morning.py — Connect to IG, fetch live bars, calculate levels,
print the exact Telegram messages for both DAX and FTSE morning routines.

Uses cached data for bar calculation + live IG market snapshot for current price.
READ-ONLY: Does NOT place any orders.
"""

import asyncio
import sys
import os
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from shared.ig_session import IGSharedSession

TZ_CET = ZoneInfo("Europe/Berlin")
TZ_UK  = ZoneInfo("Europe/London")


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_dax_cache() -> pd.DataFrame:
    """Load DAX 5-min cache (timezone-naive, CET-based)."""
    df = pd.read_csv("data/dax_5min_cache.csv", index_col=0, parse_dates=True)
    return df


def load_ftse_cache() -> pd.DataFrame:
    """Load FTSE 5-min parquet data."""
    t = pq.read_table("data/ftse/ftse_all.parquet")
    idx = t.schema.get_field_index("date")
    naive = t.column("date").cast(pa.timestamp("us"))
    t = t.set_column(idx, pa.field("date", pa.timestamp("us")), naive)
    df = pd.DataFrame({c: t.column(c).to_pylist() for c in t.column_names})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


async def get_ig_snapshot(session: IGSharedSession, epic: str) -> dict:
    """Get live market snapshot from IG."""
    try:
        market = await session.rest_call(session.ig.fetch_market_by_epic, epic)
        snap = market.get("snapshot", {})
        name = market.get("instrument", {}).get("name", epic)
        return {
            "name": name,
            "bid": snap.get("bid", 0),
            "offer": snap.get("offer", 0),
            "mid": round((snap.get("bid", 0) + snap.get("offer", 0)) / 2, 1),
            "high": snap.get("high", 0),
            "low": snap.get("low", 0),
            "status": snap.get("marketStatus", "UNKNOWN"),
            "update_time": snap.get("updateTime", ""),
        }
    except Exception as e:
        return {"name": epic, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  DAX Logic (from dax_bot/strategy.py)
# ═══════════════════════════════════════════════════════════════════════════

def dax_candle_number(timestamp) -> int:
    """Which 5-min candle from 09:00 CET open."""
    open_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def dax_get_bar(df: pd.DataFrame, n: int) -> dict | None:
    for idx, row in df.iterrows():
        cn = dax_candle_number(idx)
        if cn == n:
            return {
                "high": round(row["High"], 1), "low": round(row["Low"], 1),
                "open": round(row["Open"], 1), "close": round(row["Close"], 1),
                "range": round(row["High"] - row["Low"], 1),
                "bullish": row["Close"] > row["Open"], "time": idx,
            }
    return None


def dax_analyse_context(df: pd.DataFrame) -> dict:
    bars = []
    for idx, row in df.iterrows():
        cn = dax_candle_number(idx)
        if 1 <= cn <= 3:
            body = abs(row["Close"] - row["Open"])
            rng = row["High"] - row["Low"]
            bars.append({
                "high": row["High"], "low": row["Low"],
                "wick_pct": round((rng - body) / rng * 100, 1) if rng > 0 else 0,
                "bullish": row["Close"] > row["Open"],
            })
    if len(bars) < 3:
        return {"overlap": False, "choppy": False, "directional": False}

    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    total_rng = max(highs) - min(lows)
    avg_rng = float(np.mean([b["high"] - b["low"] for b in bars]))

    return {
        "overlap": bool(total_rng < avg_rng * 2),
        "choppy": bool(np.mean([b["wick_pct"] for b in bars]) > 50),
        "directional": bool(all(b["bullish"] for b in bars) or all(not b["bullish"] for b in bars)),
    }


def dax_classify_gap(prev_close: float, today_open: float) -> tuple:
    gap = round(today_open - prev_close, 1)
    if gap > 10:
        return "GAP_UP", gap
    elif gap < -10:
        return "GAP_DOWN", gap
    return "FLAT", gap


def dax_overnight_bias(overnight_df: pd.DataFrame, bar4_high: float, bar4_low: float) -> dict:
    result = {
        "range_high": 0, "range_low": 0, "range_size": 0,
        "bar4_vs_range": "N/A", "bias": "NO_DATA", "emoji": "❓",
    }
    if overnight_df.empty:
        return result

    result["range_high"] = round(overnight_df["High"].max(), 1)
    result["range_low"] = round(overnight_df["Low"].min(), 1)
    result["range_size"] = round(result["range_high"] - result["range_low"], 1)

    if result["range_size"] <= 0:
        return result

    bar4_range = bar4_high - bar4_low
    if bar4_range <= 0:
        result["bias"] = "STANDARD"; result["emoji"] = "⚪"
        return result

    tolerance_pct = 0.25

    if bar4_low >= result["range_high"]:
        result["bar4_vs_range"] = "ABOVE"
        result["bias"] = "SHORT_ONLY"; result["emoji"] = "🔴"
    elif bar4_high <= result["range_low"]:
        result["bar4_vs_range"] = "BELOW"
        result["bias"] = "LONG_ONLY"; result["emoji"] = "🟢"
    elif bar4_low > result["range_low"] and bar4_high > result["range_high"]:
        above_pct = (bar4_high - result["range_high"]) / bar4_range * 100
        result["bar4_vs_range"] = "PARTIAL_ABOVE"
        if above_pct > (1 - tolerance_pct) * 100:
            result["bias"] = "SHORT_ONLY"; result["emoji"] = "🔴"
        else:
            result["bias"] = "STANDARD"; result["emoji"] = "⚪"
    elif bar4_high < result["range_high"] and bar4_low < result["range_low"]:
        below_pct = (result["range_low"] - bar4_low) / bar4_range * 100
        result["bar4_vs_range"] = "PARTIAL_BELOW"
        if below_pct > (1 - tolerance_pct) * 100:
            result["bias"] = "LONG_ONLY"; result["emoji"] = "🟢"
        else:
            result["bias"] = "STANDARD"; result["emoji"] = "⚪"
    else:
        result["bar4_vs_range"] = "INSIDE"
        result["bias"] = "STANDARD"; result["emoji"] = "⚪"

    return result


async def simulate_dax(session: IGSharedSession):
    """Full DAX morning routine simulation."""
    from dax_bot import config as cfg

    # Load cached data
    df = load_dax_cache()
    available_dates = sorted(set(df.index.date))
    today_date = available_dates[-1]  # Most recent day in cache

    # Get live snapshot
    snap = await get_ig_snapshot(session, cfg.IG_EPIC)
    print(f"  Live: {snap['name']} — Bid {snap['bid']} / Offer {snap['offer']} "
          f"({snap['status']}) @ {snap['update_time']}")
    print(f"  Day range: {snap['low']} – {snap['high']}")
    print(f"  Cache data: using {today_date} (most recent in cache)")

    today_bars = df[df.index.date == today_date]
    prev_date = available_dates[-2] if len(available_dates) > 1 else None
    prev_bars = df[df.index.date == prev_date] if prev_date else pd.DataFrame()

    # Show opening bars
    print(f"\n  {today_date} opening bars (CET):")
    for idx, row in today_bars.iterrows():
        cn = dax_candle_number(idx)
        if 1 <= cn <= 8:
            marker = " ◀ SIGNAL" if cn == 4 else ""
            print(f"    Bar {cn:2d} ({idx.strftime('%H:%M')} CET): "
                  f"O={row['Open']:.1f} H={row['High']:.1f} "
                  f"L={row['Low']:.1f} C={row['Close']:.1f}  "
                  f"Range={row['High']-row['Low']:.1f}{marker}")

    # Context (bars 1-3)
    ctx = dax_analyse_context(today_bars)

    # Bar 4
    bar4 = dax_get_bar(today_bars, 4)
    if not bar4:
        print("\n  ERROR: Bar 4 not available")
        return

    # Range classification
    NARROW, WIDE, BUFFER = cfg.NARROW_RANGE, cfg.WIDE_RANGE, cfg.BUFFER_PTS
    if bar4["range"] < NARROW:
        range_flag = "NARROW"
    elif bar4["range"] > WIDE:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    # Gap
    gap_dir, gap_size = "FLAT", 0.0
    if not prev_bars.empty and not today_bars.empty:
        gap_dir, gap_size = dax_classify_gap(prev_bars["Close"].iloc[-1], today_bars["Open"].iloc[0])

    # Overnight range
    overnight_df = today_bars.between_time("00:00", "06:00")
    overnight = dax_overnight_bias(overnight_df, bar4["high"], bar4["low"])

    # Levels
    buy_level = round(bar4["high"] + BUFFER, 1)
    sell_level = round(bar4["low"] - BUFFER, 1)
    risk_pts = round(buy_level - sell_level, 1)
    max_per_pt = round(cfg.RISK_GBP / risk_pts, 2) if risk_pts > 0 else 0

    # Context string
    ctx_str = ""
    if ctx["directional"]:
        ctx_str = "\n📈 Opening: Directional"
    elif ctx["choppy"]:
        ctx_str = "\n🔀 Opening: Choppy"
    elif ctx["overlap"]:
        ctx_str = "\n📊 Opening: Overlapping"

    flag_icon = {"NARROW": "⚠️", "WIDE": "🔶", "NORMAL": "✅"}.get(range_flag, "")
    mode = "📄 DEMO" if cfg.IG_DEMO else "🔴 LIVE"
    bar_label = "Bar #4 (default)"

    # Order action based on V58 bias
    bias = overnight["bias"]
    if bias == "SHORT_ONLY":
        order_action = f"🔴 V58 SHORT ONLY → placing SELL stop only at {sell_level}"
    elif bias == "LONG_ONLY":
        order_action = f"🟢 V58 LONG ONLY → placing BUY stop only at {buy_level}"
    else:
        order_action = f"⚪ V58 STANDARD → OCA bracket: BUY {buy_level} / SELL {sell_level}"

    # ═══ Print Telegram Message ═══
    print("\n" + "═" * 60)
    print("  DAX TELEGRAM MESSAGE (exact bot output)")
    print("═" * 60)
    print(
        f"📐 ASRS — DAX 40 [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {today_date}\n"
        f"🕐 {bar_label}: {bar4['high']} / {bar4['low']}\n"
        f"   Range: {bar4['range']} pts {flag_icon}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟩 BUY stop: {buy_level}\n"
        f"🟥 SELL stop: {sell_level}\n"
        f"   Risk: {risk_pts} pts | OCA: auto-cancel\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 £{cfg.RISK_GBP} risk → max £{max_per_pt}/pt\n"
        f"📦 Contracts: {cfg.NUM_CONTRACTS}x Micro DAX (€1/pt)\n"
        f"   Risk = €{round(risk_pts * cfg.NUM_CONTRACTS, 1)}"
        f"{ctx_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Orders placed on IG automatically ✅\n"
        f"Max {cfg.MAX_ENTRIES} entries | Trail: 5-min low/high"
    )

    # Overnight message
    print(
        f"\n🌙 Overnight Range: {overnight['range_low']:.0f}–"
        f"{overnight['range_high']:.0f} ({overnight['range_size']:.0f}pts)\n"
        f"Bar 4 vs range: {overnight['bar4_vs_range']}\n"
        f"V58 Bias: {overnight['emoji']} {overnight['bias']}"
    )

    print(f"\n📊 Gap: {gap_dir} ({'+' if gap_size > 0 else ''}{gap_size} pts)")
    print(f"\n🤖 {order_action}")

    # Show what would happen with current live price
    mid = snap["mid"]
    if mid > 0:
        if mid >= buy_level:
            print(f"\n⚡ CURRENT PRICE {mid} is ABOVE buy stop {buy_level} — would have triggered BUY")
        elif mid <= sell_level:
            print(f"\n⚡ CURRENT PRICE {mid} is BELOW sell stop {sell_level} — would have triggered SELL")
        else:
            dist_buy = round(buy_level - mid, 1)
            dist_sell = round(mid - sell_level, 1)
            print(f"\n⏳ CURRENT PRICE {mid} — {dist_buy} pts from BUY, {dist_sell} pts from SELL")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  FTSE Logic (from ftse_bot/strategy.py)
# ═══════════════════════════════════════════════════════════════════════════

async def simulate_ftse(session: IGSharedSession):
    """Full FTSE morning routine simulation."""
    from ftse_bot import config as cfg

    # Load cached data
    df = load_ftse_cache()
    available_dates = sorted(set(df.index.date))
    today_date = available_dates[-1]

    # Get live snapshot
    snap = await get_ig_snapshot(session, cfg.IG_EPIC)
    print(f"  Live: {snap['name']} — Bid {snap['bid']} / Offer {snap['offer']} "
          f"({snap['status']}) @ {snap['update_time']}")
    print(f"  Day range: {snap['low']} – {snap['high']}")
    print(f"  Cache data: using {today_date} (most recent in cache)")

    today_bars = df[df.index.date == today_date]

    # Show bars around 08:00
    print(f"\n  {today_date} morning bars (UK time):")
    for idx, row in today_bars.iterrows():
        if 7 <= idx.hour <= 9:
            marker = " ◀ SIGNAL" if idx.hour == 8 and idx.minute == 0 else ""
            print(f"    {idx.strftime('%H:%M')} UK: "
                  f"O={row['Open']:.1f} H={row['High']:.1f} "
                  f"L={row['Low']:.1f} C={row['Close']:.1f}  "
                  f"Range={row['High']-row['Low']:.1f}{marker}")

    # Find 08:00 bar
    bar_0800 = None
    for idx, row in today_bars.iterrows():
        if idx.hour == 8 and idx.minute == 0:
            bar_0800 = row
            break
    if bar_0800 is None:
        for idx, row in today_bars.iterrows():
            if idx.hour == 8 and idx.minute == 5:
                bar_0800 = row
                break
    if bar_0800 is None:
        print("  ERROR: Cannot find 08:00 bar")
        return

    o, h, l, c = bar_0800["Open"], bar_0800["High"], bar_0800["Low"], bar_0800["Close"]
    bar_width = round(h - l, 1)

    # Classify
    if c < o:
        bar_type, bar_dir, icon = "1BN", "Bearish", "🔴"
        directions = ["BUY", "SELL"]
    elif c > o:
        bar_type, bar_dir, icon = "1BP", "Bullish", "🟢"
        directions = ["SELL"]
    else:
        bar_type, bar_dir, icon = "DOJI", "Doji", "⚪"
        directions = []

    # Stake
    stake = cfg.STAKE_PER_POINT
    stake_halved = False
    if bar_width > cfg.BAR_WIDTH_THRESHOLD:
        stake = round(stake / 2, 2)
        stake_halved = True
    stake_note = f" (halved: width > {cfg.BAR_WIDTH_THRESHOLD})" if stake_halved else ""

    # Levels
    buy_level = round(l - cfg.BUFFER_PTS, 1)
    sell_level = round(h + cfg.BUFFER_PTS, 1)
    order_qty = int(cfg.NUM_CONTRACTS * stake)

    mode = "DEMO" if cfg.IG_DEMO else "LIVE"

    # ═══ Print Telegram Messages ═══
    print("\n" + "═" * 60)
    print("  FTSE TELEGRAM MESSAGE (exact bot output)")
    print("═" * 60)

    # Bar detected message
    print(
        f"{icon} FTSE 1st Bar: {bar_type} ({bar_dir}) [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Open:  {o:.1f}\n"
        f"High:  {h:.1f}\n"
        f"Low:   {l:.1f}\n"
        f"Close: {c:.1f}\n"
        f"Width: {bar_width} pts\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Stake: {cfg.CURRENCY}{stake}/pt{stake_note}"
    )

    # Orders message
    print(f"\n📥 FTSE ORDERS PLACED\n━━━━━━━━━━━━━━━━━━━━━━")
    if "BUY" in directions:
        print(f"🟩 Buy stop: {buy_level}\n"
              f"   Stop if filled: {round(buy_level - bar_width, 1)}")
    if "SELL" in directions:
        print(f"🟥 Sell stop: {sell_level}\n"
              f"   Stop if filled: {round(sell_level + bar_width, 1)}")
    if not directions:
        print(f"⚪ DOJI — {cfg.DOJI_ACTION} (no orders)")

    print(f"━━━━━━━━━━━━━━━━━━━━━━\n"
          f"Stake: {cfg.CURRENCY}{stake}/pt × {cfg.NUM_CONTRACTS} contracts = "
          f"{cfg.CURRENCY}{order_qty}/pt total\n"
          f"Session ends 16:30 UK")

    # Show what would happen with current live price
    mid = snap["mid"]
    if mid > 0 and directions:
        if "BUY" in directions and mid <= buy_level:
            print(f"\n⚡ CURRENT PRICE {mid} is BELOW buy stop {buy_level} — would have triggered BUY")
        elif "SELL" in directions and mid >= sell_level:
            print(f"\n⚡ CURRENT PRICE {mid} is ABOVE sell stop {sell_level} — would have triggered SELL")
        else:
            parts = []
            if "BUY" in directions:
                parts.append(f"{round(mid - buy_level, 1)} pts from BUY")
            if "SELL" in directions:
                parts.append(f"{round(sell_level - mid, 1)} pts from SELL")
            print(f"\n⏳ CURRENT PRICE {mid} — {', '.join(parts)}")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    now_uk = datetime.now(TZ_UK)
    print("═" * 60)
    print("  MORNING ROUTINE SIMULATION")
    print(f"  {now_uk.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("═" * 60)

    session = IGSharedSession()
    ok = await session.connect()
    if not ok:
        print("FATAL: Cannot connect to IG")
        sys.exit(1)
    mode = "DEMO" if session._demo else "LIVE"
    print(f"  IG Connected ({mode}) — account {session._acc_number}")
    print(f"  (Historical data quota exceeded — using cache + live snapshots)")

    print("\n" + "━" * 60)
    print("  1. DAX 40 (ASRS Strategy)")
    print("━" * 60)
    await simulate_dax(session)

    print("\n" + "━" * 60)
    print("  2. FTSE 100 (1BN/1BP Strategy)")
    print("━" * 60)
    await simulate_ftse(session)

    await session.disconnect()
    print(f"\n  IG Disconnected. Simulation complete.")


if __name__ == "__main__":
    asyncio.run(main())
