"""
Full-rules DAX ASRS backtest matching EXACT live bot logic.
Uses ger40_m5.csv (5-min bars, 2020-2026).

Rules implemented:
1. Bar 4 signal (08:20 CET = 4th 5-min bar from 08:00)
2. Buffer +2pts above/below bar 4 high/low
3. Overnight bias filter (V58): bar 4 close vs overnight midpoint
4. OCA bracket: BUY and SELL levels, first triggered wins
5. Breakeven stop at +15pts
6. Candle trail: 5-min candle low (LONG) / high (SHORT)
7. Tight trail at +100pts: switch to candle close
8. Add-to-winners at +25pts from last entry (max 2 adds)
9. Profit lock on add: stop moves to entry + 50% of profit at add time
10. Flip on loss stop (entry 2)
11. Re-entry on profitable trail stop in same direction (entry 3)
12. MAX_ENTRIES = 3
13. Narrow range 2x sizing
14. Spread + slippage costs
15. Daily loss cap £200
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ── Config (matching live) ────────────────────────────────────────────────
BUFFER_PTS = 2.0
NARROW_RANGE = 15
WIDE_RANGE = 40
MAX_ENTRIES = 3
NUM_CONTRACTS = 1
NARROW_MULTIPLIER = 2

TRAIL_BREAKEVEN_PTS = 15.0
TRAIL_TIGHT_THRESHOLD = 100.0

ADD_ENABLED = True
ADD_TRIGGER = 25.0
ADD_MAX = 2

RISK_GBP = 25.0
MAX_DAILY_LOSS_GBP = 200.0

# Cost assumptions
SPREAD_COST = 0.0   # IG data has spread baked in; set >0 for independent data
SLIPPAGE_PTS = 0.0  # Set >0 for realistic slippage simulation

# Session times (CET)
OPEN_H, OPEN_M = 8, 0
BAR4_H, BAR4_M = 8, 20  # Bar 4 closes at 08:20
SESSION_END_H = 17
OVERNIGHT_START_H = 0


@dataclass
class Trade:
    date: str
    entry_num: int
    direction: str
    entry_price: float
    exit_price: float = 0.0
    pnl_pts: float = 0.0
    pnl_gbp: float = 0.0
    mfe: float = 0.0
    exit_reason: str = ""
    contracts: int = 1
    adds_used: int = 0
    add_pnl: float = 0.0
    bar4_range: float = 0.0
    range_flag: str = ""
    overnight_bias: str = ""
    entry_type: str = ""  # BRACKET, FLIP, REENTRY


def load_data() -> pd.DataFrame:
    df = pd.read_csv("gold_bot/ger40_m5.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.columns = ["Open", "High", "Low", "Close"]
    # Convert to CET
    df.index = df.index.tz_convert("Europe/Berlin")
    df = df[df["High"] != df["Low"]]
    return df


def get_overnight_range(df: pd.DataFrame, date):
    """Get high/low from 00:00 to 08:00 CET."""
    start = date.replace(hour=0, minute=0)
    end = date.replace(hour=8, minute=0)
    on = df[start:end]
    if len(on) < 5:
        return None, None, None
    h, l = on["High"].max(), on["Low"].min()
    mid = (h + l) / 2
    return h, l, mid


def simulate_day(df: pd.DataFrame, date, daily_pnl_gbp: float) -> list[Trade]:
    """Simulate one full trading day with all rules."""
    trades = []

    # Get bar 4
    bar4_start = date.replace(hour=BAR4_H, minute=BAR4_M - 5)
    bar4_end = date.replace(hour=BAR4_H, minute=BAR4_M)
    bars_open = date.replace(hour=OPEN_H, minute=OPEN_M)
    session_bars = df[bars_open:bar4_end]

    if len(session_bars) < 4:
        return trades

    bar4 = session_bars.iloc[-1]
    bar4_high = bar4["High"]
    bar4_low = bar4["Low"]
    bar4_range = bar4_high - bar4_low

    if bar4_range < 3 or bar4_range > 120:
        return trades

    # Range flag
    if bar4_range <= NARROW_RANGE:
        range_flag = "NARROW"
    elif bar4_range >= WIDE_RANGE:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    # Overnight bias
    on_h, on_l, on_mid = get_overnight_range(df, date)
    if on_mid is not None:
        if bar4["Close"] > on_mid:
            bias = "LONG_ONLY"  # Only take longs
        else:
            bias = "SHORT_ONLY"  # Only take shorts
    else:
        bias = "NO_DATA"

    # Levels
    buy_level = bar4_high + BUFFER_PTS
    sell_level = bar4_low - BUFFER_PTS

    # Position sizing
    base_qty = NUM_CONTRACTS
    if range_flag == "NARROW":
        base_qty *= NARROW_MULTIPLIER

    # Post-bar4 data
    session_end = date.replace(hour=SESSION_END_H, minute=30)
    post_bars = df[bar4_end:session_end]
    if len(post_bars) < 2:
        return trades

    entries_used = 0
    current_buy = buy_level
    current_sell = sell_level

    while entries_used < MAX_ENTRIES:
        # Daily loss check
        if daily_pnl_gbp <= -MAX_DAILY_LOSS_GBP:
            break

        trade = _simulate_entry(
            post_bars, current_buy, current_sell, bias, bar4_range, range_flag,
            base_qty, entries_used, date, bar4_high, bar4_low
        )

        if trade is None:
            break

        # Determine entry type
        if entries_used == 0:
            trade.entry_type = "BRACKET"
        elif trade.pnl_pts <= 0:
            trade.entry_type = "FLIP"
        else:
            trade.entry_type = "REENTRY"

        trade.overnight_bias = bias
        trades.append(trade)
        daily_pnl_gbp += trade.pnl_gbp
        entries_used += 1

        if entries_used >= MAX_ENTRIES:
            break

        # Decide next entry
        if trade.exit_reason == "TRAILED_STOP" and trade.pnl_pts > 0:
            # Re-entry: same direction at exit price
            if trade.direction == "LONG":
                current_buy = trade.exit_price
                current_sell = 0.01  # Unreachable
            else:
                current_buy = 999999.0
                current_sell = trade.exit_price
            # Trim post_bars to after exit
            exit_idx = _find_exit_bar_idx(post_bars, trade)
            if exit_idx is not None and exit_idx + 1 < len(post_bars):
                post_bars = post_bars.iloc[exit_idx + 1:]
            else:
                break
        elif trade.pnl_pts <= 0:
            # Flip: opposite direction at original levels
            if trade.direction == "LONG":
                current_buy = 999999.0
                current_sell = sell_level
            else:
                current_buy = buy_level
                current_sell = 0.01
            # Check flip slippage
            exit_idx = _find_exit_bar_idx(post_bars, trade)
            if exit_idx is not None and exit_idx + 1 < len(post_bars):
                post_bars = post_bars.iloc[exit_idx + 1:]
                # Check if price already overshot flip level by >10pts
                first_bar = post_bars.iloc[0] if len(post_bars) > 0 else None
                if first_bar is not None:
                    if trade.direction == "LONG" and current_sell < 999999:
                        if first_bar["Low"] < current_sell - 10:
                            break  # Flip slippage too high
                    elif trade.direction == "SHORT" and current_buy < 999999:
                        if first_bar["High"] > current_buy + 10:
                            break
            else:
                break
        else:
            break  # Profitable initial stop / breakeven — no action

    return trades


def _find_exit_bar_idx(post_bars, trade):
    """Find approximate bar index where trade exited."""
    # Use the accumulated bar count from simulation
    return getattr(trade, "_exit_bar_idx", None)


def _simulate_entry(post_bars, buy_level, sell_level, bias, bar4_range,
                     range_flag, base_qty, entry_num, date,
                     bar4_high, bar4_low) -> Optional[Trade]:
    """Simulate a single entry with full management rules."""

    entry_price = None
    direction = None
    stop_level = None
    initial_risk = None
    breakeven_hit = False
    mfe = 0.0
    contracts = base_qty
    adds_used = 0
    last_add_price = 0.0
    add_entries = []

    for j in range(len(post_bars)):
        b = post_bars.iloc[j]

        if entry_price is None:
            # Check triggers
            trig_buy = buy_level < 999999 and b["High"] >= buy_level
            trig_sell = sell_level > 0.02 and b["Low"] <= sell_level

            if trig_buy and trig_sell:
                continue
            if trig_buy:
                if bias == "SHORT_ONLY":
                    continue
                direction = "LONG"
                entry_price = buy_level + SLIPPAGE_PTS
                stop_level = bar4_low - SPREAD_COST
                initial_risk = entry_price - stop_level
            elif trig_sell:
                if bias == "LONG_ONLY":
                    continue
                direction = "SHORT"
                entry_price = sell_level - SLIPPAGE_PTS
                stop_level = bar4_high + SPREAD_COST
                initial_risk = stop_level - entry_price
            else:
                continue

            if initial_risk <= 0:
                entry_price = None
                continue

            last_add_price = entry_price
            continue  # Entry bar — don't manage on same bar

        # ── Position management ──────────────────────────────────────
        if direction == "LONG":
            bar_mfe = b["High"] - entry_price
            mfe = max(mfe, bar_mfe)
            current_profit = b["Close"] - entry_price

            # Stop check
            if b["Low"] <= stop_level:
                pnl = stop_level - entry_price
                reason = "INITIAL_STOP" if not breakeven_hit else "TRAILED_STOP"
                if breakeven_hit and stop_level == entry_price:
                    reason = "BREAKEVEN_STOP"
                return _build_trade(
                    date, entry_num, direction, entry_price, stop_level,
                    pnl, mfe, reason, contracts, adds_used, add_entries,
                    bar4_range, range_flag, initial_risk, j
                )

            # Breakeven
            if not breakeven_hit and current_profit >= TRAIL_BREAKEVEN_PTS:
                breakeven_hit = True
                if stop_level < entry_price:
                    stop_level = entry_price

            # Add-to-winners
            if ADD_ENABLED and adds_used < ADD_MAX:
                profit_from_ref = b["Close"] - last_add_price
                if profit_from_ref >= ADD_TRIGGER:
                    adds_used += 1
                    add_price = last_add_price + ADD_TRIGGER + SLIPPAGE_PTS
                    add_entries.append(add_price)
                    last_add_price = add_price
                    contracts += 1
                    # Lock profit: move stop to entry + 50% of profit
                    lock_stop = entry_price + (add_price - entry_price) * 0.5
                    if lock_stop > stop_level:
                        stop_level = lock_stop
                        breakeven_hit = True

            # Candle trail
            use_tight = current_profit >= TRAIL_TIGHT_THRESHOLD
            new_stop = b["Close"] if use_tight else b["Low"]
            if j > 0:  # Use previous candle for trail
                prev = post_bars.iloc[j - 1]
                new_stop = prev["Close"] if use_tight else prev["Low"]
            if new_stop > stop_level:
                stop_level = new_stop

        else:  # SHORT
            bar_mfe = entry_price - b["Low"]
            mfe = max(mfe, bar_mfe)
            current_profit = entry_price - b["Close"]

            if b["High"] >= stop_level:
                pnl = entry_price - stop_level
                reason = "INITIAL_STOP" if not breakeven_hit else "TRAILED_STOP"
                if breakeven_hit and stop_level == entry_price:
                    reason = "BREAKEVEN_STOP"
                return _build_trade(
                    date, entry_num, direction, entry_price, stop_level,
                    pnl, mfe, reason, contracts, adds_used, add_entries,
                    bar4_range, range_flag, initial_risk, j
                )

            if not breakeven_hit and current_profit >= TRAIL_BREAKEVEN_PTS:
                breakeven_hit = True
                if stop_level > entry_price:
                    stop_level = entry_price

            if ADD_ENABLED and adds_used < ADD_MAX:
                profit_from_ref = last_add_price - b["Close"]
                if profit_from_ref >= ADD_TRIGGER:
                    adds_used += 1
                    add_price = last_add_price - ADD_TRIGGER - SLIPPAGE_PTS
                    add_entries.append(add_price)
                    last_add_price = add_price
                    contracts += 1
                    # Lock profit: move stop to entry - 50% of profit
                    lock_stop = entry_price - (entry_price - add_price) * 0.5
                    if lock_stop < stop_level:
                        stop_level = lock_stop
                        breakeven_hit = True

            use_tight = current_profit >= TRAIL_TIGHT_THRESHOLD
            new_stop = b["Close"] if use_tight else b["High"]
            if j > 0:
                prev = post_bars.iloc[j - 1]
                new_stop = prev["Close"] if use_tight else prev["High"]
            if new_stop < stop_level:
                stop_level = new_stop

    # Session end — close at last bar close
    if entry_price is not None:
        last_close = post_bars.iloc[-1]["Close"]
        if direction == "LONG":
            pnl = last_close - entry_price
        else:
            pnl = entry_price - last_close
        return _build_trade(
            date, entry_num, direction, entry_price, last_close,
            pnl, mfe, "SESSION_CLOSE", contracts, adds_used, add_entries,
            bar4_range, range_flag, initial_risk, len(post_bars) - 1
        )

    return None


def _build_trade(date, entry_num, direction, entry_price, exit_price,
                  pnl_pts, mfe, reason, contracts, adds_used, add_entries,
                  bar4_range, range_flag, initial_risk, exit_bar_idx):
    """Build Trade object with full P&L calculation."""
    # P&L for original contracts
    orig_contracts = contracts - adds_used
    total_pnl = pnl_pts * orig_contracts

    # P&L for each add
    add_pnl = 0.0
    for ap in add_entries:
        if direction == "LONG":
            add_pnl += exit_price - ap
        else:
            add_pnl += ap - exit_price
    total_pnl += add_pnl

    # Slippage on all contracts
    total_pnl -= SLIPPAGE_PTS * contracts
    total_pnl -= SPREAD_COST * contracts

    # £/pt based on risk
    stake = max(0.5, min(RISK_GBP / initial_risk, 50.0)) if initial_risk > 0 else 1.0
    pnl_gbp = total_pnl * stake

    t = Trade(
        date=date.strftime("%Y-%m-%d"),
        entry_num=entry_num,
        direction=direction,
        entry_price=round(entry_price, 1),
        exit_price=round(exit_price, 1),
        pnl_pts=round(total_pnl, 1),
        pnl_gbp=round(pnl_gbp, 2),
        mfe=round(mfe, 1),
        exit_reason=reason,
        contracts=contracts,
        adds_used=adds_used,
        add_pnl=round(add_pnl, 1),
        bar4_range=round(bar4_range, 1),
        range_flag=range_flag,
    )
    t._exit_bar_idx = exit_bar_idx
    return t


def run_backtest(df: pd.DataFrame) -> list[Trade]:
    dates = df.index.normalize().unique()
    all_trades = []

    for date in dates:
        if date.weekday() >= 5:
            continue

        # Track daily P&L for loss cap
        daily_pnl = sum(t.pnl_gbp for t in all_trades if t.date == date.strftime("%Y-%m-%d"))
        day_trades = simulate_day(df, date, daily_pnl)
        all_trades.extend(day_trades)

    return all_trades


def print_results(trades: list[Trade], label: str = ""):
    df = pd.DataFrame([t.__dict__ for t in trades])
    # Remove internal fields
    df = df.drop(columns=["_exit_bar_idx"], errors="ignore")

    wins = df[df["pnl_pts"] > 0]
    losses = df[df["pnl_pts"] < 0]
    be = df[df["pnl_pts"] == 0]
    gw = wins["pnl_gbp"].sum() if len(wins) else 0
    gl = abs(losses["pnl_gbp"].sum()) if len(losses) else 1
    pf = gw / gl if gl > 0 else float("inf")

    print()
    print("=" * 70)
    print(f"  {label or 'DAX ASRS FULL RULES BACKTEST'}")
    print("=" * 70)
    print(f"  Period: {df['date'].min()} to {df['date'].max()}")
    print(f"  Trades: {len(df)}  |  Wins: {len(wins)}  |  Losses: {len(losses)}  |  BE: {len(be)}")
    print(f"  Win Rate: {len(wins)/len(df)*100:.1f}%")
    print(f"  PF: {pf:.2f}")
    print(f"  Total PnL: GBP {df['pnl_gbp'].sum():>10,.2f}")
    print(f"  Avg PnL/trade: GBP {df['pnl_gbp'].mean():.2f}")
    if len(wins):
        print(f"  Avg Winner: {wins['pnl_pts'].mean():.1f} pts (GBP {wins['pnl_gbp'].mean():.2f})")
    if len(losses):
        print(f"  Avg Loser: {losses['pnl_pts'].mean():.1f} pts (GBP {losses['pnl_gbp'].mean():.2f})")
    print(f"  Avg MFE: {df['mfe'].mean():.1f} pts")
    print(f"  Best: GBP {df['pnl_gbp'].max():.2f}  |  Worst: GBP {df['pnl_gbp'].min():.2f}")

    print()
    print("  By Exit Reason:")
    for r in sorted(df["exit_reason"].unique()):
        rdf = df[df["exit_reason"] == r]
        print(f"    {r:18s}: {len(rdf):4d} trades | GBP {rdf['pnl_gbp'].sum():>10,.2f}")

    print()
    print("  By Entry Type:")
    for et in ["BRACKET", "FLIP", "REENTRY"]:
        edf = df[df["entry_type"] == et]
        if len(edf) == 0:
            continue
        ew = edf[edf["pnl_pts"] > 0]
        egw = ew["pnl_gbp"].sum() if len(ew) else 0
        egl = abs(edf[edf["pnl_pts"] < 0]["pnl_gbp"].sum()) or 1
        epf = egw / egl
        print(f"    {et:10s}: {len(edf):4d} trades | WR {len(ew)/len(edf)*100:5.1f}% | PF {epf:.2f} | GBP {edf['pnl_gbp'].sum():>10,.2f}")

    print()
    print("  By Range Flag:")
    for rf in ["NARROW", "NORMAL", "WIDE"]:
        rdf = df[df["range_flag"] == rf]
        if len(rdf) == 0:
            continue
        rw = rdf[rdf["pnl_pts"] > 0]
        rgw = rw["pnl_gbp"].sum() if len(rw) else 0
        rgl = abs(rdf[rdf["pnl_pts"] < 0]["pnl_gbp"].sum()) or 1
        rpf = rgw / rgl
        print(f"    {rf:10s}: {len(rdf):4d} trades | WR {len(rw)/len(rdf)*100:5.1f}% | PF {rpf:.2f} | GBP {rdf['pnl_gbp'].sum():>10,.2f}")

    print()
    print("  By Overnight Bias:")
    for b in sorted(df["overnight_bias"].unique()):
        bdf = df[df["overnight_bias"] == b]
        bw = bdf[bdf["pnl_pts"] > 0]
        bgw = bw["pnl_gbp"].sum() if len(bw) else 0
        bgl = abs(bdf[bdf["pnl_pts"] < 0]["pnl_gbp"].sum()) or 1
        bpf = bgw / bgl
        print(f"    {b:12s}: {len(bdf):4d} trades | WR {len(bw)/len(bdf)*100:5.1f}% | PF {bpf:.2f} | GBP {bdf['pnl_gbp'].sum():>10,.2f}")

    print()
    print("  By Year:")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    for yr in sorted(df["year"].unique()):
        ydf = df[df["year"] == yr]
        yw = ydf[ydf["pnl_pts"] > 0]
        ygw = yw["pnl_gbp"].sum() if len(yw) else 0
        ygl = abs(ydf[ydf["pnl_pts"] < 0]["pnl_gbp"].sum()) or 1
        ypf = ygw / ygl
        print(f"    {yr}: {len(ydf):4d} trades | WR {len(yw)/len(ydf)*100:5.1f}% | PF {ypf:.2f} | GBP {ydf['pnl_gbp'].sum():>8,.2f}")

    # MFE capture analysis
    print()
    profitable = df[df["pnl_pts"] > 0]
    if len(profitable):
        avg_mfe = profitable["mfe"].mean()
        avg_pnl = profitable["pnl_pts"].mean()
        capture = (avg_pnl / avg_mfe * 100) if avg_mfe > 0 else 0
        print(f"  MFE Capture (winners): {capture:.0f}% (avg PnL {avg_pnl:.1f} / avg MFE {avg_mfe:.1f})")

    big_movers = df[df["mfe"] >= 100]
    if len(big_movers):
        avg_mfe_big = big_movers["mfe"].mean()
        avg_pnl_big = big_movers["pnl_pts"].mean()
        capture_big = (avg_pnl_big / avg_mfe_big * 100) if avg_mfe_big > 0 else 0
        print(f"  MFE Capture (MFE>=100): {capture_big:.0f}% (avg PnL {avg_pnl_big:.1f} / avg MFE {avg_mfe_big:.1f}) [{len(big_movers)} trades]")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"{len(df)} bars: {df.index[0]} to {df.index[-1]}")

    print("\nRunning backtest (zero costs)...")
    SPREAD_COST = 0.0
    SLIPPAGE_PTS = 0.0
    trades = run_backtest(df)
    print_results(trades, "DAX ASRS FULL RULES — ZERO COSTS")

    print("\n\nRunning backtest (with costs: 1.4pt spread + 2pt slippage)...")
    # Re-import to reset globals
    SPREAD_COST = 1.4
    SLIPPAGE_PTS = 2.0
    trades_cost = run_backtest(df)
    print_results(trades_cost, "DAX ASRS FULL RULES — WITH COSTS (spread=1.4 + slip=2.0)")
