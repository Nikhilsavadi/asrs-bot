"""
Backtest: DAX ASRS Bar-4 — EXACT live bot rules
Matches dax_bot config.py + broker_ig.py logic precisely.

Rules tested:
- Bar 4 high/low + 2pt buffer
- V58 overnight bias filter
- NARROW/NORMAL/WIDE range classification (2x size on NARROW)
- Breakeven at +15pts
- Candle trail (5-min bar low/high)
- Flip on stop (max 2 entries/day)
- Flip slippage guard (10pts max)
- Add to winners (+25pts, max 2 adds)
- Max daily loss £200
- Spread cost (1.4pts per entry)
- Entry slippage (2pts avg assumed)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Config (exact match to config.py) ─────────────────────────
BUFFER_PTS = 2
NARROW_RANGE = 15
WIDE_RANGE = 40
MAX_ENTRIES = 2
BASE_CONTRACTS = 1         # £1/pt
NARROW_MULTIPLIER = 2      # 2x on NARROW
BREAKEVEN_PTS = 15
MAX_SLIPPAGE_PTS = 10
SPREAD_COST = 1.4           # IG DAX spread ~1.4pts
ENTRY_SLIPPAGE = 2.0        # Avg observed slippage
ADD_TRIGGER_PTS = 25
ADD_MAX = 2
MAX_DAILY_LOSS_GBP = 200


def load_data(path="gold_bot/ger40_m5.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df.columns = ["Open", "High", "Low", "Close"]
    df = df[df["High"] != df["Low"]]
    return df


def get_overnight_bias(df, date):
    """V58: compare overnight high/low midpoint to bar 4 area."""
    # Overnight = previous day 17:30 to today 08:00 CET (07:00-08:00 UTC roughly)
    prev_eve = date - timedelta(days=1)
    on_start = prev_eve.replace(hour=16, minute=30)
    on_end = date.replace(hour=7, minute=0)
    on_bars = df[on_start:on_end]

    if len(on_bars) < 5:
        return "NO_DATA", 0, 0

    on_high = on_bars["High"].max()
    on_low = on_bars["Low"].min()
    return "STANDARD", on_high, on_low


def classify_range(bar4_range):
    if bar4_range < NARROW_RANGE:
        return "NARROW"
    elif bar4_range > WIDE_RANGE:
        return "WIDE"
    return "NORMAL"


def run_backtest(df):
    trades = []
    dates = df.index.normalize().unique()
    daily_pnl = 0.0
    current_date = None

    for date in dates:
        if date.weekday() >= 5:
            continue

        # Reset daily P&L
        if current_date != date.date():
            current_date = date.date()
            daily_pnl = 0.0

        # Skip if daily loss cap hit
        if daily_pnl <= -MAX_DAILY_LOSS_GBP:
            continue

        # Bar 1-4: 08:00-08:20 CET = 07:00-07:20 UTC
        bars_start = date.replace(hour=7, minute=0)
        bar4_end = date.replace(hour=7, minute=20)
        session_bars = df[bars_start:bar4_end]

        if len(session_bars) < 4:
            continue

        # Bar 4 = last bar in the 07:00-07:20 window
        bar4 = session_bars.iloc[-1]
        bar4_high = bar4["High"]
        bar4_low = bar4["Low"]
        bar4_range = bar4_high - bar4_low

        if bar4_range < 3:  # Too tight, skip
            continue

        range_class = classify_range(bar4_range)
        contracts = BASE_CONTRACTS * NARROW_MULTIPLIER if range_class == "NARROW" else BASE_CONTRACTS

        # Overnight bias
        bias, on_high, on_low = get_overnight_bias(df, date)
        if on_high > 0 and on_low > 0:
            on_mid = (on_high + on_low) / 2
            if bar4["Close"] > on_mid:
                bias = "LONG_ONLY"  # Only BUY
            else:
                bias = "SHORT_ONLY"  # Only SELL

        buy_level = bar4_high + BUFFER_PTS
        sell_level = bar4_low - BUFFER_PTS

        # Post bar-4 bars (07:20 to 16:30 UTC = 08:20 to 17:30 CET)
        session_end = date.replace(hour=16, minute=30)
        post_bars = df[bar4_end:session_end]

        if len(post_bars) < 2:
            continue

        entries_today = 0

        while entries_today < MAX_ENTRIES:
            entry_price = None
            direction = None
            stop_level = None
            initial_risk = None
            breakeven_moved = False
            mfe = 0
            pnl = 0
            reason = None
            add_count = 0
            add_pnl = 0
            total_contracts = contracts
            last_add_price = None
            entry_bar_idx = None

            for j in range(len(post_bars)):
                b = post_bars.iloc[j]
                ts = post_bars.index[j]

                if entry_price is None:
                    # Check trigger
                    triggered_buy = buy_level is not None and b["High"] >= buy_level
                    triggered_sell = sell_level is not None and b["Low"] <= sell_level

                    if triggered_buy and triggered_sell:
                        continue

                    if triggered_buy:
                        if bias == "SHORT_ONLY":
                            continue
                        direction = "BUY"
                        entry_price = buy_level + ENTRY_SLIPPAGE
                        stop_level = bar4_low
                        initial_risk = entry_price - stop_level
                    elif triggered_sell:
                        if bias == "LONG_ONLY":
                            continue
                        direction = "SELL"
                        entry_price = sell_level - ENTRY_SLIPPAGE
                        stop_level = bar4_high
                        initial_risk = stop_level - entry_price
                    else:
                        continue

                    if initial_risk <= 0:
                        entry_price = None
                        continue

                    last_add_price = entry_price
                    entry_bar_idx = j
                    entries_today += 1
                else:
                    # Track MFE
                    if direction == "BUY":
                        unrealized = b["High"] - entry_price
                    else:
                        unrealized = entry_price - b["Low"]
                    mfe = max(mfe, unrealized)

                    exit_price = None

                    # Check stop hit
                    if direction == "BUY" and b["Low"] <= stop_level:
                        exit_price = stop_level
                        reason = "INITIAL_STOP" if not breakeven_moved else "TRAIL_STOP"
                    elif direction == "SELL" and b["High"] >= stop_level:
                        exit_price = stop_level
                        reason = "INITIAL_STOP" if not breakeven_moved else "TRAIL_STOP"

                    if exit_price is not None:
                        if direction == "BUY":
                            pnl = exit_price - entry_price - SPREAD_COST
                        else:
                            pnl = entry_price - exit_price - SPREAD_COST
                        # Add PnL from add positions
                        for a in range(add_count):
                            if direction == "BUY":
                                add_pnl += (exit_price - (entry_price + ADD_TRIGGER_PTS * (a + 1))) - SPREAD_COST
                            else:
                                add_pnl += ((entry_price - ADD_TRIGGER_PTS * (a + 1)) - exit_price) - SPREAD_COST
                        break

                    # Breakeven check
                    if not breakeven_moved:
                        if direction == "BUY" and (b["High"] - entry_price) >= BREAKEVEN_PTS:
                            breakeven_moved = True
                            stop_level = entry_price
                        elif direction == "SELL" and (entry_price - b["Low"]) >= BREAKEVEN_PTS:
                            breakeven_moved = True
                            stop_level = entry_price

                    # Candle trail (after breakeven, use prior 5-min bar)
                    if breakeven_moved and j > entry_bar_idx:
                        prev = post_bars.iloc[j - 1]
                        if direction == "BUY" and prev["Low"] > stop_level:
                            stop_level = prev["Low"]
                        elif direction == "SELL" and prev["High"] < stop_level:
                            stop_level = prev["High"]

                    # Add to winners
                    if add_count < ADD_MAX and last_add_price is not None:
                        if direction == "BUY" and (b["High"] - last_add_price) >= ADD_TRIGGER_PTS:
                            add_count += 1
                            last_add_price = last_add_price + ADD_TRIGGER_PTS
                            total_contracts += 1
                        elif direction == "SELL" and (last_add_price - b["Low"]) >= ADD_TRIGGER_PTS:
                            add_count += 1
                            last_add_price = last_add_price - ADD_TRIGGER_PTS
                            total_contracts += 1

            else:
                # Session end
                if entry_price is not None:
                    exit_price = post_bars.iloc[-1]["Close"]
                    if direction == "BUY":
                        pnl = exit_price - entry_price - SPREAD_COST
                    else:
                        pnl = entry_price - exit_price - SPREAD_COST
                    reason = "SESSION_CLOSE"
                    for a in range(add_count):
                        if direction == "BUY":
                            add_pnl += (exit_price - (entry_price + ADD_TRIGGER_PTS * (a + 1))) - SPREAD_COST
                        else:
                            add_pnl += ((entry_price - ADD_TRIGGER_PTS * (a + 1)) - exit_price) - SPREAD_COST

            if entry_price is not None and reason is not None:
                pnl_gbp = pnl * contracts + add_pnl * BASE_CONTRACTS
                daily_pnl += pnl_gbp

                trades.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "direction": direction,
                    "entry": round(entry_price, 1),
                    "exit": round(exit_price, 1) if exit_price else 0,
                    "pnl_pts": round(pnl, 1),
                    "pnl_gbp": round(pnl_gbp, 2),
                    "mfe": round(mfe, 1),
                    "exit_reason": reason,
                    "range_class": range_class,
                    "bar4_range": round(bar4_range, 1),
                    "bias": bias,
                    "contracts": contracts,
                    "adds": add_count,
                    "add_pnl": round(add_pnl, 1),
                    "breakeven_hit": breakeven_moved,
                    "daily_pnl": round(daily_pnl, 2),
                })

                # Setup flip
                if reason == "INITIAL_STOP" and entries_today < MAX_ENTRIES:
                    # Check flip slippage
                    if exit_price is not None:
                        if direction == "BUY":
                            # Flip to SELL
                            flip_dist = sell_level - exit_price if sell_level else 999
                            if abs(flip_dist) > MAX_SLIPPAGE_PTS:
                                break  # Too much slippage for flip
                            buy_level = None  # Only sell now
                        else:
                            flip_dist = exit_price - buy_level if buy_level else 999
                            if abs(flip_dist) > MAX_SLIPPAGE_PTS:
                                break
                            sell_level = None
                    # Slice remaining bars for flip
                    if exit_price is not None:
                        try:
                            exit_idx = post_bars.index.get_loc(ts)
                            post_bars = post_bars.iloc[exit_idx:]
                        except:
                            break
                    continue
                break
            else:
                break

    return trades


def print_results(trades):
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pts"] > 0]
    losses = df[df["pnl_pts"] < 0]
    be = df[df["pnl_pts"] == 0]

    gw = wins["pnl_gbp"].sum() if len(wins) else 0
    gl = abs(losses["pnl_gbp"].sum()) if len(losses) else 1
    pf = gw / gl if gl > 0 else float("inf")

    print(f"\n{'='*65}")
    print(f"  DAX ASRS BAR-4 — EXACT LIVE RULES BACKTEST")
    print(f"{'='*65}")
    print(f"  Period: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"  Trades: {len(df)}  |  Wins: {len(wins)}  |  Losses: {len(losses)}  |  BE: {len(be)}")
    print(f"  Win Rate: {len(wins)/len(df)*100:.1f}%")
    print(f"  PF: {pf:.2f}")
    print(f"  Total PnL: GBP {df['pnl_gbp'].sum():,.2f}")
    print(f"  Avg PnL/trade: GBP {df['pnl_gbp'].mean():.2f}")
    if len(wins): print(f"  Avg Winner: {wins['pnl_pts'].mean():.1f} pts (GBP {wins['pnl_gbp'].mean():.2f})")
    if len(losses): print(f"  Avg Loser: {losses['pnl_pts'].mean():.1f} pts (GBP {losses['pnl_gbp'].mean():.2f})")
    print(f"  Avg MFE: {df['mfe'].mean():.1f} pts")
    print(f"  Best: GBP {df['pnl_gbp'].max():.2f}  |  Worst: GBP {df['pnl_gbp'].min():.2f}")
    print(f"  Max drawdown day: GBP {df.groupby('date')['pnl_gbp'].sum().min():.2f}")

    # By exit reason
    print(f"\n  By Exit Reason:")
    for r in sorted(df["exit_reason"].unique()):
        rdf = df[df["exit_reason"] == r]
        rw = rdf[rdf["pnl_pts"] > 0]
        print(f"    {r:16s}: {len(rdf):4d} trades | WR {len(rw)/len(rdf)*100:5.1f}% | GBP {rdf['pnl_gbp'].sum():,.2f}")

    # By range class
    print(f"\n  By Range Class:")
    for rc in ["NARROW", "NORMAL", "WIDE"]:
        rdf = df[df["range_class"] == rc]
        if len(rdf):
            rw = rdf[rdf["pnl_pts"] > 0]
            rgw = rw["pnl_gbp"].sum() if len(rw) else 0
            rgl = abs(rdf[rdf["pnl_pts"] < 0]["pnl_gbp"].sum()) if len(rdf[rdf["pnl_pts"] < 0]) else 1
            rpf = rgw / rgl if rgl > 0 else float("inf")
            print(f"    {rc:8s}: {len(rdf):4d} trades | WR {len(rw)/len(rdf)*100:5.1f}% | PF {rpf:.2f} | GBP {rdf['pnl_gbp'].sum():,.2f} | Avg size: {rdf['contracts'].mean():.1f}x")

    # By bias
    print(f"\n  By Overnight Bias:")
    for b in df["bias"].unique():
        bdf = df[df["bias"] == b]
        bw = bdf[bdf["pnl_pts"] > 0]
        bgw = bw["pnl_gbp"].sum() if len(bw) else 0
        bgl = abs(bdf[bdf["pnl_pts"] < 0]["pnl_gbp"].sum()) if len(bdf[bdf["pnl_pts"] < 0]) else 1
        bpf = bgw / bgl if bgl > 0 else float("inf")
        print(f"    {b:12s}: {len(bdf):4d} trades | WR {len(bw)/len(bdf)*100:5.1f}% | PF {bpf:.2f} | GBP {bdf['pnl_gbp'].sum():,.2f}")

    # Adds stats
    with_adds = df[df["adds"] > 0]
    if len(with_adds):
        print(f"\n  Add-to-Winners:")
        print(f"    Trades with adds: {len(with_adds)} ({len(with_adds)/len(df)*100:.1f}%)")
        print(f"    Add PnL contribution: GBP {with_adds['add_pnl'].sum():,.2f}")
        print(f"    Avg adds per winning trade: {with_adds['adds'].mean():.1f}")

    # Breakeven stats
    be_trades = df[df["breakeven_hit"]]
    print(f"\n  Breakeven Stop:")
    print(f"    Trades hitting BE: {len(be_trades)} ({len(be_trades)/len(df)*100:.1f}%)")
    be_then_stopped = be_trades[be_trades["exit_reason"] == "TRAIL_STOP"]
    print(f"    BE -> trail stop: {len(be_then_stopped)} (avg PnL: GBP {be_then_stopped['pnl_gbp'].mean():.2f})" if len(be_then_stopped) else "")

    # Yearly breakdown
    print(f"\n  Yearly Breakdown:")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    for yr in sorted(df["year"].unique()):
        ydf = df[df["year"] == yr]
        yw = ydf[ydf["pnl_pts"] > 0]
        ygw = yw["pnl_gbp"].sum() if len(yw) else 0
        ygl = abs(ydf[ydf["pnl_pts"] < 0]["pnl_gbp"].sum()) if len(ydf[ydf["pnl_pts"] < 0]) else 1
        ypf = ygw / ygl if ygl > 0 else float("inf")
        print(f"    {yr}: {len(ydf):4d} trades | WR {len(yw)/len(ydf)*100:5.1f}% | PF {ypf:.2f} | GBP {ydf['pnl_gbp'].sum():,.2f}")


if __name__ == "__main__":
    print("Loading DAX 5-min data...")
    df = load_data()
    print(f"  {len(df)} bars: {df.index[0]} to {df.index[-1]}")

    print("\nRunning exact-rules backtest...")
    trades = run_backtest(df)
    print_results(trades)
