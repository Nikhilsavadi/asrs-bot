"""
shadow.py -- Bar 5 Shadow Tracker for A/B Comparison
=====================================================

Runs alongside the real bar 4 strategy WITHOUT placing orders.
Tracks what bar 5 levels would have done, using price data only.

State persists in data/shadow_state.json (reset daily like DailyState).
"""

import json
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict

import pandas as pd

from dax_bot import config
from dax_bot.strategy import candle_number, get_bar, Phase

logger = logging.getLogger(__name__)

SHADOW_FILE = os.path.join(os.path.dirname(__file__), "data", "shadow_state.json")


@dataclass
class ShadowState:
    date:            str = ""
    phase:           str = Phase.IDLE

    # Bar 5 levels
    bar5_high:       float = 0.0
    bar5_low:        float = 0.0
    bar5_range:      float = 0.0
    buy_level:       float = 0.0
    sell_level:      float = 0.0

    # Shadow position
    direction:       str = ""
    entry_price:     float = 0.0
    trailing_stop:   float = 0.0
    max_favourable:  float = 0.0
    entries_used:    int = 0

    # Previous candle for trailing
    prev_candle_high: float = 0.0
    prev_candle_low:  float = 0.0

    # Trade log
    trades:          list = field(default_factory=list)

    def save(self):
        os.makedirs(os.path.dirname(SHADOW_FILE), exist_ok=True)
        with open(SHADOW_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "ShadowState":
        today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
        try:
            if os.path.exists(SHADOW_FILE):
                with open(SHADOW_FILE) as f:
                    data = json.load(f)
                if data.get("date") == today:
                    s = cls()
                    for k, v in data.items():
                        if hasattr(s, k):
                            setattr(s, k, v)
                    return s
        except Exception as e:
            logger.error(f"Shadow state load error: {e}")
        return cls(date=today)

    @property
    def total_pnl(self) -> float:
        return round(sum(t.get("pnl_pts", 0) for t in self.trades), 1)


def calculate_shadow_levels(shadow: ShadowState, df: pd.DataFrame) -> bool:
    """
    Calculate bar 5 levels. Called after bar 5 closes (~09:25 CET).
    Returns True if levels were set.
    """
    today = df[df.index.date == datetime.now(config.TZ_CET).date()]
    if today.empty:
        return False

    bar5 = get_bar(today, 5)
    if not bar5:
        logger.info("Shadow: bar 5 not available yet")
        return False

    shadow.bar5_high = bar5["high"]
    shadow.bar5_low = bar5["low"]
    shadow.bar5_range = bar5["range"]
    shadow.buy_level = round(bar5["high"] + config.BUFFER_PTS, 1)
    shadow.sell_level = round(bar5["low"] - config.BUFFER_PTS, 1)
    shadow.phase = Phase.LEVELS_SET
    shadow.save()

    logger.info(f"Shadow bar 5 levels: buy {shadow.buy_level} / sell {shadow.sell_level} "
                f"(range {shadow.bar5_range})")
    return True


def update_shadow(shadow: ShadowState, df: pd.DataFrame) -> list[str]:
    """
    Called every 5 min (same as monitor_cycle).
    Checks for entries, trails stops, detects exits — all from price data.
    Returns list of events for alerting.
    """
    events = []

    if shadow.phase in (Phase.IDLE, Phase.DONE):
        return events

    today = df[df.index.date == datetime.now(config.TZ_CET).date()]
    if today.empty:
        return events

    # Get candles after bar 5
    post_bars = []
    for idx, row in today.iterrows():
        if candle_number(idx) > 5:
            post_bars.append((idx, row))

    if not post_bars:
        return events

    # Use latest candle for entry/exit checks
    latest_idx, latest = post_bars[-1]

    # -- Entry check --
    if shadow.phase == Phase.LEVELS_SET and shadow.entries_used < config.MAX_ENTRIES:
        if latest["High"] >= shadow.buy_level and shadow.direction == "":
            shadow.direction = "LONG"
            shadow.entry_price = shadow.buy_level
            shadow.trailing_stop = shadow.sell_level
            shadow.max_favourable = latest["High"]
            shadow.entries_used += 1
            shadow.phase = Phase.LONG_ACTIVE
            shadow.trades.append({
                "num": shadow.entries_used,
                "direction": "LONG",
                "entry": shadow.buy_level,
                "time": datetime.now(config.TZ_UK).strftime("%H:%M"),
            })
            events.append("SHADOW_LONG_ENTRY")
            logger.info(f"Shadow LONG entry @ {shadow.buy_level}")

        elif latest["Low"] <= shadow.sell_level and shadow.direction == "":
            shadow.direction = "SHORT"
            shadow.entry_price = shadow.sell_level
            shadow.trailing_stop = shadow.buy_level
            shadow.max_favourable = latest["Low"]
            shadow.entries_used += 1
            shadow.phase = Phase.SHORT_ACTIVE
            shadow.trades.append({
                "num": shadow.entries_used,
                "direction": "SHORT",
                "entry": shadow.sell_level,
                "time": datetime.now(config.TZ_UK).strftime("%H:%M"),
            })
            events.append("SHADOW_SHORT_ENTRY")
            logger.info(f"Shadow SHORT entry @ {shadow.sell_level}")

    # -- Trail & exit --
    if shadow.phase == Phase.LONG_ACTIVE:
        # MFE
        if latest["High"] > shadow.max_favourable:
            shadow.max_favourable = latest["High"]

        # Trail: use previous candle low (if we have 2+ post-bar5 candles)
        if len(post_bars) >= 2:
            prev_low = round(post_bars[-2][1]["Low"], 1)
            if prev_low > shadow.trailing_stop:
                old = shadow.trailing_stop
                shadow.trailing_stop = prev_low
                events.append(f"SHADOW_TRAIL:{old}->{prev_low}")

        # Stop hit?
        if latest["Low"] <= shadow.trailing_stop:
            pnl = round(shadow.trailing_stop - shadow.entry_price, 1)
            mfe = round(shadow.max_favourable - shadow.entry_price, 1)
            if shadow.trades:
                shadow.trades[-1]["exit"] = shadow.trailing_stop
                shadow.trades[-1]["pnl_pts"] = pnl
                shadow.trades[-1]["mfe"] = mfe
                shadow.trades[-1]["exit_time"] = datetime.now(config.TZ_UK).strftime("%H:%M")

            events.append(f"SHADOW_LONG_EXIT:{pnl:+.1f}")
            logger.info(f"Shadow LONG exit @ {shadow.trailing_stop} ({pnl:+.1f} pts)")

            shadow.direction = ""
            if shadow.entries_used < config.MAX_ENTRIES:
                shadow.phase = Phase.LEVELS_SET
            else:
                shadow.phase = Phase.DONE

    elif shadow.phase == Phase.SHORT_ACTIVE:
        if latest["Low"] < shadow.max_favourable:
            shadow.max_favourable = latest["Low"]

        if len(post_bars) >= 2:
            prev_high = round(post_bars[-2][1]["High"], 1)
            if prev_high < shadow.trailing_stop:
                old = shadow.trailing_stop
                shadow.trailing_stop = prev_high
                events.append(f"SHADOW_TRAIL:{old}->{prev_high}")

        if latest["High"] >= shadow.trailing_stop:
            pnl = round(shadow.entry_price - shadow.trailing_stop, 1)
            mfe = round(shadow.entry_price - shadow.max_favourable, 1)
            if shadow.trades:
                shadow.trades[-1]["exit"] = shadow.trailing_stop
                shadow.trades[-1]["pnl_pts"] = pnl
                shadow.trades[-1]["mfe"] = mfe
                shadow.trades[-1]["exit_time"] = datetime.now(config.TZ_UK).strftime("%H:%M")

            events.append(f"SHADOW_SHORT_EXIT:{pnl:+.1f}")
            logger.info(f"Shadow SHORT exit @ {shadow.trailing_stop} ({pnl:+.1f} pts)")

            shadow.direction = ""
            if shadow.entries_used < config.MAX_ENTRIES:
                shadow.phase = Phase.LEVELS_SET
            else:
                shadow.phase = Phase.DONE

    shadow.save()
    return events


def close_shadow_eod(shadow: ShadowState, current_price: float) -> list[str]:
    """Force-close any open shadow position at EOD."""
    events = []
    if shadow.direction and current_price:
        if shadow.direction == "LONG":
            pnl = round(current_price - shadow.entry_price, 1)
        else:
            pnl = round(shadow.entry_price - current_price, 1)

        if shadow.trades:
            shadow.trades[-1]["exit"] = current_price
            shadow.trades[-1]["pnl_pts"] = pnl
            shadow.trades[-1]["exit_time"] = "EOD"

        events.append(f"SHADOW_EOD_CLOSE:{pnl:+.1f}")
        shadow.direction = ""

    shadow.phase = Phase.DONE
    shadow.save()
    return events


def format_comparison(bar4_state, shadow: ShadowState) -> str:
    """Format Telegram comparison message: bar 4 vs bar 5."""
    b4_trades = bar4_state.trades or []
    b5_trades = shadow.trades or []

    b4_pnl = sum(t.get("pnl_pts", 0) for t in b4_trades)
    b5_pnl = shadow.total_pnl

    b4_detail = " | ".join(
        f"{t.get('direction', '?')[0]}:{t.get('pnl_pts', 0):+.0f}"
        for t in b4_trades
    ) if b4_trades else "no trades"

    b5_detail = " | ".join(
        f"{t.get('direction', '?')[0]}:{t.get('pnl_pts', 0):+.0f}"
        for t in b5_trades
    ) if b5_trades else "no trades"

    diff = round(b5_pnl - b4_pnl, 1)
    winner = "BAR 5" if diff > 0 else ("BAR 4" if diff < 0 else "TIE")

    return (
        f"<b>BAR 4 vs BAR 5 — Daily Comparison</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Bar 4:</b> buy {bar4_state.buy_level} / sell {bar4_state.sell_level}\n"
        f"  Trades: {b4_detail}\n"
        f"  P&L: <b>{b4_pnl:+.1f} pts</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Bar 5:</b> buy {shadow.buy_level} / sell {shadow.sell_level}\n"
        f"  Trades: {b5_detail}\n"
        f"  P&L: <b>{b5_pnl:+.1f} pts</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Winner: <b>{winner}</b> ({diff:+.1f} pts)\n"
    )
