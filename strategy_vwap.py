"""
strategy_vwap.py — VWAP Bounce Strategy for DAX
═══════════════════════════════════════════════════════════════════════════════

Core concept:
    VWAP acts as dynamic support/resistance. Institutions buy at VWAP because
    it represents "fair value" for the day. When price pulls back to VWAP and
    bounces, we enter in the direction of the bounce.

    This is NON-CORRELATED to ASRS:
    - ASRS = fixed-time breakout (08:20 UK, momentum)
    - VWAP Bounce = dynamic pullback entry (throughout the day, trend continuation)

Rules (fully mechanical):
    1. Calculate rolling VWAP from 09:00 CET
    2. Determine TREND: is price predominantly above or below VWAP?
       - "Above bias" = 4+ of last 6 closes above VWAP
       - "Below bias" = 4+ of last 6 closes below VWAP
       - "Choppy"     = mixed → skip (no clear trend to ride)
    3. ENTRY — price pulls back to VWAP and bounces:
       LONG:  Above bias + bar low touches/dips below VWAP + bar closes ABOVE VWAP
       SHORT: Below bias + bar high touches/pokes above VWAP + bar closes BELOW VWAP
    4. Confirmation: bounce bar closes in the direction of the trend
    5. STOP: other side of VWAP + buffer (VWAP ∓ buffer pts)
    6. TARGET: 1σ band in trend direction (or previous swing high/low)
    7. TRAIL: once past 1σ, move stop up to VWAP
    8. TIME FILTER: only enter 09:30–15:30 CET
    9. CHOP FILTER: skip if price crossed VWAP > 4 times in last 12 bars
    10. MAX 3 entries per day
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime
from enum import Enum

import numpy as np
import pandas as pd

from dax_bot import config

logger = logging.getLogger(__name__)

VWAP_STATE_FILE = os.path.join(os.path.dirname(__file__), "data", "vwap_state.json")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Trend detection
TREND_LOOKBACK   = int(os.getenv("VWAP_TREND_LOOKBACK", "6"))    # Bars to assess
TREND_THRESHOLD  = int(os.getenv("VWAP_TREND_THRESHOLD", "4"))   # Bars on one side

# Chop filter
CHOP_LOOKBACK    = int(os.getenv("VWAP_CHOP_LOOKBACK", "12"))    # Bars to check
CHOP_MAX_CROSSES = int(os.getenv("VWAP_CHOP_MAX_CROSSES", "4"))  # Max crosses allowed

# Stop / target
STOP_BUFFER_PTS  = float(os.getenv("VWAP_STOP_BUFFER", "5"))     # Pts beyond VWAP for stop
MAX_STOP_PTS     = float(os.getenv("VWAP_MAX_STOP_PTS", "25"))   # Hard cap

# Target: 1σ band in trend direction, or fixed pts
TARGET_MODE      = os.getenv("VWAP_TARGET_MODE", "band")         # "band" or "fixed"
FIXED_TARGET_PTS = float(os.getenv("VWAP_FIXED_TARGET", "20"))

# Trail: once past 1σ, move stop to VWAP
TRAIL_ACTIVATION = float(os.getenv("VWAP_TRAIL_ACTIVATION", "1.0"))  # σ multiplier

# Bounce bar quality — close must be in this % of bar in bounce direction
MIN_CLOSE_POSITION = float(os.getenv("VWAP_MIN_CLOSE_POS", "0.4"))

# Time filters (CET)
ENTRY_START      = dtime(9, 30)
ENTRY_END        = dtime(15, 30)
FORCE_CLOSE      = dtime(17, 25)

# Max entries
VWAP_MAX_ENTRIES = int(os.getenv("VWAP_MAX_ENTRIES", "3"))


# ══════════════════════════════════════════════════════════════════════════════
#  VWAP CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VWAP + σ bands from intraday bars.
    Expects: Open, High, Low, Close, Volume columns. Index = datetime (CET).
    """
    df = df.copy()
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3

    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        df["Volume"] = 1

    df["cum_tp_vol"] = (df["typical_price"] * df["Volume"]).cumsum()
    df["cum_vol"] = df["Volume"].cumsum()
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]

    df["sq_diff"] = (df["typical_price"] - df["vwap"]) ** 2
    df["cum_sq_diff_vol"] = (df["sq_diff"] * df["Volume"]).cumsum()
    df["vwap_var"] = df["cum_sq_diff_vol"] / df["cum_vol"]
    df["vwap_std"] = np.sqrt(df["vwap_var"])

    df["upper_1s"] = df["vwap"] + df["vwap_std"]
    df["lower_1s"] = df["vwap"] - df["vwap_std"]
    df["upper_2s"] = df["vwap"] + (2 * df["vwap_std"])
    df["lower_2s"] = df["vwap"] - (2 * df["vwap_std"])

    df["above_vwap"] = df["Close"] > df["vwap"]
    df["z_score"] = np.where(
        df["vwap_std"] > 0,
        (df["Close"] - df["vwap"]) / df["vwap_std"],
        0
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TREND & CHOP DETECTION
# ══════════════════════════════════════════════════════════════════════════════

class VwapBias(str, Enum):
    ABOVE  = "ABOVE"      # Bullish — look for long bounces off VWAP support
    BELOW  = "BELOW"      # Bearish — look for short bounces off VWAP resistance
    CHOPPY = "CHOPPY"     # No clear trend — skip


def detect_trend(closes_above_vwap: list[bool],
                 lookback: int = TREND_LOOKBACK,
                 threshold: int = TREND_THRESHOLD) -> VwapBias:
    """Determine trend from recent closes vs VWAP."""
    if len(closes_above_vwap) < lookback:
        return VwapBias.CHOPPY

    recent = closes_above_vwap[-lookback:]
    above_count = sum(recent)
    below_count = lookback - above_count

    if above_count >= threshold:
        return VwapBias.ABOVE
    elif below_count >= threshold:
        return VwapBias.BELOW
    return VwapBias.CHOPPY


def count_vwap_crosses(above_series: list[bool]) -> int:
    """Count VWAP crosses in a series of above/below flags."""
    if len(above_series) < 2:
        return 0
    crosses = 0
    for i in range(1, len(above_series)):
        if above_series[i] != above_series[i - 1]:
            crosses += 1
    return crosses


def is_bounce_bar(high: float, low: float, close: float, open_: float,
                  vwap: float, bias: VwapBias) -> bool:
    """
    Check if this bar is a valid bounce off VWAP.

    LONG bounce (bias=ABOVE):
        - Bar low touches or dips below VWAP (probes the support)
        - Bar closes ABOVE VWAP (bounces back up)
        - Close in upper portion of bar range (shows buying strength)

    SHORT bounce (bias=BELOW):
        - Bar high touches or pokes above VWAP (probes the resistance)
        - Bar closes BELOW VWAP (bounces back down)
        - Close in lower portion of bar range (shows selling strength)
    """
    bar_range = high - low
    if bar_range == 0:
        return False

    if bias == VwapBias.ABOVE:
        touched = low <= vwap
        closed_above = close > vwap
        close_position = (close - low) / bar_range  # 1.0 = closed at high
        return touched and closed_above and close_position >= MIN_CLOSE_POSITION

    elif bias == VwapBias.BELOW:
        touched = high >= vwap
        closed_below = close < vwap
        close_position = (high - close) / bar_range  # 1.0 = closed at low
        return touched and closed_below and close_position >= MIN_CLOSE_POSITION

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  STOP / TARGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calc_stop(vwap: float, direction: str, entry_price: float) -> float:
    """Stop on the wrong side of VWAP + buffer, capped at MAX_STOP_PTS."""
    if direction == "LONG":
        raw = round(vwap - STOP_BUFFER_PTS, 1)
        return round(max(raw, entry_price - MAX_STOP_PTS), 1)
    else:
        raw = round(vwap + STOP_BUFFER_PTS, 1)
        return round(min(raw, entry_price + MAX_STOP_PTS), 1)


def calc_target(vwap: float, vwap_std: float, bias: VwapBias,
                entry_price: float, swing_price: float = 0) -> float:
    """Target at 1σ band in trend direction, or fixed pts."""
    if TARGET_MODE == "fixed":
        offset = FIXED_TARGET_PTS
        return round(entry_price + offset, 1) if bias == VwapBias.ABOVE \
            else round(entry_price - offset, 1)

    # Band mode — 1σ in trend direction
    if bias == VwapBias.ABOVE:
        band = round(vwap + max(vwap_std, 5), 1)  # At least 5pts target
        if swing_price and swing_price > band:
            return swing_price
        return max(band, entry_price + 5)
    else:
        band = round(vwap - max(vwap_std, 5), 1)
        if swing_price and swing_price < band:
            return swing_price
        return min(band, entry_price - 5)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

class VwapPhase(str, Enum):
    IDLE          = "IDLE"
    MONITORING    = "MONITORING"
    LONG_ACTIVE   = "LONG_ACTIVE"
    SHORT_ACTIVE  = "SHORT_ACTIVE"
    DONE          = "DONE"


@dataclass
class VwapState:
    date:             str = ""
    phase:            str = VwapPhase.IDLE
    entries_used:     int = 0
    vwap:             float = 0.0
    vwap_std:         float = 0.0
    bias:             str = ""
    vwap_crosses:     int = 0
    direction:        str = ""
    entry_price:      float = 0.0
    entry_time:       str = ""
    stop_price:       float = 0.0
    target_price:     float = 0.0
    trailing:         bool = False
    trail_stop:       float = 0.0
    entry_order_id:   int = 0
    stop_order_id:    int = 0
    target_order_id:  int = 0
    trades:           list = field(default_factory=list)

    def save(self):
        import json
        os.makedirs(os.path.dirname(VWAP_STATE_FILE), exist_ok=True)
        with open(VWAP_STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "VwapState":
        import json
        today = datetime.now(config.TZ_CET).strftime("%Y-%m-%d")
        try:
            with open(VWAP_STATE_FILE) as f:
                data = json.load(f)
            state = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            if state.date != today:
                state = cls(date=today)
                state.save()
            return state
        except (FileNotFoundError, json.JSONDecodeError):
            state = cls(date=today)
            state.save()
            return state


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE PROCESSING (for live bot)
# ══════════════════════════════════════════════════════════════════════════════

def process_vwap_entry(state: VwapState, direction: str, price: float,
                       stop: float, target: float, bias: str, vwap: float) -> list[str]:
    """Process a new VWAP bounce entry."""
    state.direction = direction
    state.entry_price = price
    state.entry_time = datetime.now(config.TZ_CET).strftime("%H:%M")
    state.stop_price = stop
    state.target_price = target
    state.trailing = False
    state.trail_stop = 0
    state.entries_used += 1
    state.bias = bias
    state.vwap = vwap
    state.phase = VwapPhase.LONG_ACTIVE if direction == "LONG" else VwapPhase.SHORT_ACTIVE
    state.save()
    logger.info(f"VWAP bounce: {direction} @ {price}, stop: {stop}, target: {target} ({bias})")
    return [f"VWAP_{direction}_ENTRY"]


def process_vwap_exit(state: VwapState, exit_price: float, reason: str) -> dict:
    """Process exit. Returns trade dict."""
    pnl = round(exit_price - state.entry_price, 1) if state.direction == "LONG" \
        else round(state.entry_price - exit_price, 1)

    trade = {
        "num": state.entries_used, "direction": state.direction,
        "entry": state.entry_price, "entry_time": state.entry_time,
        "exit": round(exit_price, 1),
        "exit_time": datetime.now(config.TZ_CET).strftime("%H:%M"),
        "pnl_pts": pnl, "reason": reason,
        "bias": state.bias, "vwap_at_entry": state.vwap,
    }
    state.trades.append(trade)
    state.phase = VwapPhase.DONE if state.entries_used >= VWAP_MAX_ENTRIES \
        else VwapPhase.MONITORING
    state.direction = ""
    state.entry_price = state.stop_price = state.target_price = 0
    state.trailing = False
    state.trail_stop = 0
    state.save()
    logger.info(f"VWAP exit: {trade['direction']} @ {exit_price} ({reason}), P&L: {pnl:+.1f}")
    return trade


def process_vwap_trail(state: VwapState, new_stop: float):
    """Move trailing stop."""
    state.trailing = True
    if state.direction == "LONG":
        state.stop_price = max(state.stop_price, new_stop)
    else:
        state.stop_price = min(state.stop_price, new_stop)
    state.save()
    logger.info(f"VWAP trail: stop → {state.stop_price}")
