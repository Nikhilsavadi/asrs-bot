"""
strategy.py -- FTSE 1BN/1BP Strategy Logic
==============================================================================

Tom Hougaard's First Bar Negative / First Bar Positive strategy.

Bar classification:
  1BN: close < open  -> buy stop below bar low AND sell stop above bar high
  1BP: close > open  -> sell stop below bar low only
  DOJI: close == open -> configurable (SKIP / TREAT_AS_1BN / TREAT_AS_1BP)

Exit strategy: Candle Trail (previous candle low/high as trailing stop)
  - Same approach as DAX bot — no breakeven/phases, just trail
  - Initial stop: entry_price - bar_width (LONG) or + bar_width (SHORT)
  - After first bar: trail to previous candle low (LONG) or high (SHORT)

Add-to-winners: Strength mode (S25_A2)
  - When trade moves +TRIGGER pts from last entry, add 1 more contract
  - All positions share same trail stop
  - Max ADD_STRENGTH_MAX extra positions
"""

import json
import os
import logging
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime

from ftse_bot import config

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ftse", "ftse_state.json")


class Phase(str, Enum):
    IDLE            = "IDLE"
    ORDERS_PLACED   = "ORDERS_PLACED"
    LONG_ACTIVE     = "LONG_ACTIVE"
    SHORT_ACTIVE    = "SHORT_ACTIVE"
    DONE            = "DONE"


class StopPhase(str, Enum):
    """Kept for compatibility but candle trail only uses CANDLE_TRAIL."""
    INITIAL       = "INITIAL"
    CANDLE_TRAIL  = "CANDLE_TRAIL"


class BarType(str, Enum):
    ONE_BN = "1BN"
    ONE_BP = "1BP"
    DOJI   = "DOJI"


@dataclass
class AddPosition:
    """Tracks an add-to-winners position."""
    entry_price: float = 0.0
    order_id: int = 0
    stop_order_id: int = 0
    filled: bool = False


@dataclass
class DailyState:
    date:            str = ""
    phase:           Phase = Phase.IDLE

    # First bar data
    bar_open:        float = 0.0
    bar_high:        float = 0.0
    bar_low:         float = 0.0
    bar_close:       float = 0.0
    bar_width:       float = 0.0
    bar_type:        str = ""        # 1BN, 1BP, DOJI

    # Orders
    buy_order_id:    int = 0
    sell_order_id:   int = 0
    buy_level:       float = 0.0     # bar_low - buffer
    sell_level:      float = 0.0     # bar_high + buffer
    oca_group:       str = ""

    # Position
    direction:       str = ""        # LONG or SHORT
    entry_price:     float = 0.0
    initial_stop:    float = 0.0
    trailing_stop:   float = 0.0
    stop_phase:      str = StopPhase.INITIAL.value
    stop_order_id:   int = 0
    max_favourable:  float = 0.0     # MFE tracking
    entries_used:    int = 0         # Re-entry counter

    # Multi-contract
    contracts_active: int = 0        # Total active contracts (initial + adds)
    total_stake:     float = 0.0     # contracts_active * stake_per_point

    # Sizing
    stake:           float = config.STAKE_PER_POINT
    stake_halved:    bool = False

    # Add-to-winners
    adds_used:       int = 0
    add_positions:   list = field(default_factory=list)  # list of AddPosition dicts
    last_add_price:  float = 0.0     # Price of last entry (initial or add)

    # Previous candle for trail
    prev_candle_low:  float = 0.0
    prev_candle_high: float = 0.0

    # Result
    exit_price:      float = 0.0
    exit_reason:     str = ""        # "STOPPED", "SESSION_CLOSE"
    pnl_pts:         float = 0.0
    pnl_gbp:         float = 0.0

    # Trail alert tracking
    last_trail_alert_stop: float = 0.0

    def save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls) -> "DailyState":
        if not os.path.exists(STATE_FILE):
            return cls()
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            # Filter to known fields
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            state = cls(**known)
            # Reset if new day
            today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
            if state.date != today:
                return cls()
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return cls()


def classify_bar(open_: float, high: float, low: float, close: float) -> BarType:
    if close < open_:
        return BarType.ONE_BN
    elif close > open_:
        return BarType.ONE_BP
    return BarType.DOJI


def resolve_doji(bar_type: BarType) -> BarType | None:
    if bar_type != BarType.DOJI:
        return bar_type
    if config.DOJI_ACTION == "TREAT_AS_1BN":
        return BarType.ONE_BN
    elif config.DOJI_ACTION == "TREAT_AS_1BP":
        return BarType.ONE_BP
    return None  # SKIP


def process_bar(state: DailyState, open_: float, high: float, low: float, close: float) -> list[str]:
    """Process the 08:00-08:05 bar. Sets levels and bar type on state."""
    events = []
    today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")

    state.date = today
    state.bar_open = open_
    state.bar_high = high
    state.bar_low = low
    state.bar_close = close
    state.bar_width = round(high - low, 1)

    raw_type = classify_bar(open_, high, low, close)
    state.bar_type = raw_type.value
    events.append("BAR_CLASSIFIED")

    resolved = resolve_doji(raw_type)
    if resolved is None:
        state.phase = Phase.DONE
        state.save()
        events.append("DOJI_SKIP")
        return events

    # Calculate stake (halve if bar > threshold)
    state.stake = config.STAKE_PER_POINT
    state.stake_halved = False
    if state.bar_width > config.BAR_WIDTH_THRESHOLD:
        state.stake = round(config.STAKE_PER_POINT / 2, 2)
        state.stake_halved = True

    # Set entry levels
    state.buy_level = round(low - config.BUFFER_PTS, 1)
    state.sell_level = round(high + config.BUFFER_PTS, 1)
    state.oca_group = f"FTSE_{today}_{resolved.value}"

    state.save()
    events.append("LEVELS_SET")
    return events


def get_order_directions(bar_type_str: str) -> list[str]:
    """
    Return which order directions to place.
    1BN: buy stop below low AND sell stop above high
    1BP: sell stop below low only
    """
    if bar_type_str == BarType.ONE_BN.value:
        return ["BUY", "SELL"]
    elif bar_type_str == BarType.ONE_BP.value:
        return ["SELL"]
    return []


def process_fill(state: DailyState, direction: str, fill_price: float) -> list[str]:
    """Process an entry fill."""
    events = []
    state.direction = direction
    state.entry_price = fill_price
    state.last_add_price = fill_price
    state.max_favourable = fill_price
    state.entries_used += 1

    # Initial stop = bar width on opposite side
    if direction == "LONG":
        state.initial_stop = round(fill_price - state.bar_width, 1)
        state.phase = Phase.LONG_ACTIVE
    else:
        state.initial_stop = round(fill_price + state.bar_width, 1)
        state.phase = Phase.SHORT_ACTIVE

    state.trailing_stop = state.initial_stop
    state.stop_phase = StopPhase.INITIAL.value
    state.last_trail_alert_stop = state.initial_stop

    # Multi-contract: initial position
    state.contracts_active = config.NUM_CONTRACTS
    state.total_stake = round(config.NUM_CONTRACTS * state.stake, 2)
    state.adds_used = 0
    state.add_positions = []

    state.save()
    events.append("ENTRY_FILLED")
    return events


def update_candle_trail(state: DailyState, prev_low: float, prev_high: float) -> list[str]:
    """
    Update trailing stop based on previous candle's low/high.
    Called each bar with the previous bar's data.
    """
    events = []
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return events

    state.prev_candle_low = prev_low
    state.prev_candle_high = prev_high

    old_stop = state.trailing_stop

    if state.direction == "LONG":
        new_stop = round(prev_low, 1)
        if new_stop > state.trailing_stop:
            state.trailing_stop = new_stop
            state.stop_phase = StopPhase.CANDLE_TRAIL.value
            events.append("TRAIL_UPDATED")
    else:
        new_stop = round(prev_high, 1)
        if new_stop < state.trailing_stop:
            state.trailing_stop = new_stop
            state.stop_phase = StopPhase.CANDLE_TRAIL.value
            events.append("TRAIL_UPDATED")

    if events:
        state.save()

    return events


def check_add_trigger(state: DailyState, current_price: float) -> bool:
    """Check if add-to-winners should trigger based on current price."""
    if not config.ADD_STRENGTH_ENABLED:
        return False
    if state.adds_used >= config.ADD_STRENGTH_MAX:
        return False
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return False

    trigger = config.ADD_STRENGTH_TRIGGER

    if state.direction == "LONG":
        profit_from_last = current_price - state.last_add_price
        return profit_from_last >= trigger
    else:
        profit_from_last = state.last_add_price - current_price
        return profit_from_last >= trigger


def process_add(state: DailyState, fill_price: float, order_id: int = 0) -> dict:
    """Record an add-to-winners fill."""
    state.adds_used += 1
    state.contracts_active += 1
    state.total_stake = round(state.contracts_active * state.stake, 2)
    state.last_add_price = fill_price
    add = {"entry_price": fill_price, "order_id": order_id, "filled": True}
    state.add_positions.append(add)
    state.save()
    return add


def update_stop(state: DailyState, current_price: float) -> list[str]:
    """
    Legacy update_stop kept for compatibility with monitor_cycle.
    With candle trail, this just tracks MFE. Trail updates happen
    via update_candle_trail() when we have previous bar data.
    """
    events = []
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return events

    if state.direction == "LONG":
        state.max_favourable = max(state.max_favourable, current_price)
    else:
        state.max_favourable = min(state.max_favourable, current_price)

    return events


def process_exit(state: DailyState, exit_price: float, reason: str) -> dict:
    """Process position exit. Returns trade summary dict."""
    if state.direction == "LONG":
        pnl_pts = round(exit_price - state.entry_price, 1)
    else:
        pnl_pts = round(state.entry_price - exit_price, 1)

    # Add P&L from add positions
    add_pnl = 0.0
    for add in state.add_positions:
        ap = add.get("entry_price", 0)
        if ap:
            if state.direction == "LONG":
                add_pnl += round(exit_price - ap, 1)
            else:
                add_pnl += round(ap - exit_price, 1)

    # Total P&L: initial contracts + adds
    total_pnl_pts = round(pnl_pts * config.NUM_CONTRACTS + add_pnl, 1)
    pnl_gbp = round(total_pnl_pts * state.stake, 2)

    # MFE
    if state.direction == "LONG":
        mfe = round(state.max_favourable - state.entry_price, 1)
    else:
        mfe = round(state.entry_price - state.max_favourable, 1)

    state.exit_price = exit_price
    state.exit_reason = reason
    state.pnl_pts = total_pnl_pts
    state.pnl_gbp = pnl_gbp
    state.phase = Phase.DONE
    state.contracts_active = 0
    state.save()

    return {
        "date": state.date,
        "bar_type": state.bar_type,
        "direction": state.direction,
        "entry": state.entry_price,
        "exit": exit_price,
        "pnl_pts": total_pnl_pts,
        "pnl_per_contract": pnl_pts,
        "add_pnl": add_pnl,
        "adds_used": state.adds_used,
        "pnl_gbp": pnl_gbp,
        "mfe": mfe,
        "bar_width": state.bar_width,
        "stake": state.stake,
        "contracts": config.NUM_CONTRACTS,
        "stop_phase": state.stop_phase,
        "exit_reason": reason,
        "bar_open": state.bar_open,
        "bar_high": state.bar_high,
        "bar_low": state.bar_low,
        "bar_close": state.bar_close,
    }
