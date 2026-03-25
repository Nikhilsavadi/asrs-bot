"""
strategy.py — ASRS State Machine & Logic
═══════════════════════════════════════════════════════════════════════════════

State Machine:
    IDLE → LEVELS_SET → ORDERS_PLACED → LONG/SHORT_ACTIVE → DONE
                                     ↗ (flip if entries < MAX)
    LONG/SHORT_ACTIVE → STOPPED_OUT ─┤
                                     ↘ DONE (if entries = MAX)
"""

import json
import os
import logging
from datetime import datetime, time as dtime
from dataclasses import dataclass, field, asdict
from enum import Enum

import pandas as pd
import numpy as np

from dax_bot import config

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "data", "dax", "daily_state.json")


class Phase(str, Enum):
    IDLE           = "IDLE"
    LEVELS_SET     = "LEVELS_SET"
    ORDERS_PLACED  = "ORDERS_PLACED"
    LONG_ACTIVE    = "LONG_ACTIVE"
    SHORT_ACTIVE   = "SHORT_ACTIVE"
    STOPPED_OUT    = "STOPPED_OUT"
    DONE           = "DONE"


@dataclass
class DailyState:
    date:              str = ""
    phase:             str = Phase.IDLE
    entries_used:      int = 0

    # ASRS bar
    bar_number:        int = 0
    bar_high:          float = 0.0
    bar_low:           float = 0.0
    bar_range:         float = 0.0
    range_flag:        str = ""

    # Entry levels
    buy_level:         float = 0.0
    sell_level:        float = 0.0

    # Active position
    direction:         str = ""
    entry_price:       float = 0.0
    initial_stop:      float = 0.0
    trailing_stop:     float = 0.0
    max_favourable:    float = 0.0

    # IBKR order IDs
    oca_group:         str = ""
    buy_order_id:      int = 0
    sell_order_id:     int = 0
    stop_order_id:     int = 0

    # 3-contract partial exit tracking
    # contracts_active: how many of the 3 contracts are still open
    contracts_active:  int = 0

    # Add-to-winners tracking
    adds_used:         int = 0
    add_positions:     list = field(default_factory=list)  # [{entry_price, order_id}]
    last_add_price:    float = 0.0   # Price of last add entry (for next trigger calc)
    tp1_order_id:      int = 0
    tp2_order_id:      int = 0
    tp1_filled:        bool = False
    tp2_filled:        bool = False
    tp1_price:         float = 0.0
    tp2_price:         float = 0.0
    tp1_replaces:      int = 0       # How many times TP1 was re-placed after cancel
    tp2_replaces:      int = 0       # How many times TP2 was re-placed after cancel
    breakeven_hit:     bool = False  # True once stop moved to entry price
    position_size:     int = 0      # Contracts for this day (0 = use NUM_CONTRACTS default)

    # Context
    use_5th_bar:       bool = False
    context_overlap:   bool = False
    context_choppy:    bool = False
    context_directional: bool = False
    gap_dir:           str = ""         # GAP_UP, GAP_DOWN, FLAT
    gap_size:          float = 0.0
    bar5_rule_matched: str = ""         # Which BAR5_RULE triggered bar 5 (e.g. "OVERLAP+WIDE")

    # Overnight range (V58 theory)
    overnight_high:    float = 0.0
    overnight_low:     float = 0.0
    overnight_range:   float = 0.0
    overnight_bias:    str = ""       # SHORT_ONLY, LONG_ONLY, STANDARD, NO_DATA
    bar4_vs_overnight: str = ""       # ABOVE, BELOW, INSIDE, PARTIAL_ABOVE, PARTIAL_BELOW

    # Re-entry after profitable exit
    reentry_direction: str = ""        # LONG or SHORT — same as profitable exit
    reentry_price:     float = 0.0     # Exit price — re-enter if price resumes through here

    # Session 2 (11:00 CET continuation)
    s2_phase:          str = "IDLE"    # Independent phase for session 2
    s2_bar_high:       float = 0.0
    s2_bar_low:        float = 0.0
    s2_bar_range:      float = 0.0
    s2_buy_level:      float = 0.0
    s2_sell_level:     float = 0.0
    s2_direction:      str = ""
    s2_entry_price:    float = 0.0
    s2_initial_stop:   float = 0.0
    s2_trailing_stop:  float = 0.0
    s2_max_favourable: float = 0.0
    s2_contracts_active: int = 0
    s2_adds_used:      int = 0
    s2_last_add_price: float = 0.0
    s2_breakeven_hit:  bool = False
    s2_entries_used:   int = 0

    # Trade log
    trades:            list = field(default_factory=list)

    def save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "DailyState":
        today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE) as f:
                    data = json.load(f)
                if data.get("date") == today:
                    s = cls()
                    for k, v in data.items():
                        if hasattr(s, k):
                            setattr(s, k, v)
                    return s
        except Exception as e:
            logger.error(f"State load error: {e}")
        return cls(date=today)


# ── Bar Identification ─────────────────────────────────────────────────────────

def candle_number(timestamp: pd.Timestamp) -> int:
    """Which 5-min candle from 09:00 CET open."""
    open_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def get_bar(df: pd.DataFrame, n: int) -> dict | None:
    """Extract nth 5-min candle."""
    for idx, row in df.iterrows():
        if candle_number(idx) == n:
            return {
                "high":    round(row["High"], 1),
                "low":     round(row["Low"], 1),
                "open":    round(row["Open"], 1),
                "close":   round(row["Close"], 1),
                "range":   round(row["High"] - row["Low"], 1),
                "bullish": row["Close"] > row["Open"],
                "time":    idx,
            }
    return None


def analyse_context(df: pd.DataFrame) -> dict:
    """Analyse bars 1-3 for choppiness / overlap."""
    bars = []
    for idx, row in df.iterrows():
        cn = candle_number(idx)
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
    avg_rng = np.mean([b["high"] - b["low"] for b in bars])

    return {
        "overlap":     bool(total_rng < avg_rng * 2),
        "choppy":      bool(np.mean([b["wick_pct"] for b in bars]) > 50),
        "directional": bool(all(b["bullish"] for b in bars) or all(not b["bullish"] for b in bars)),
    }


def get_last_completed_candle(df: pd.DataFrame) -> dict | None:
    """Most recent completed candle for trailing stop."""
    now = datetime.now(config.TZ_CET)
    completed = df[df.index < now - pd.Timedelta(seconds=30)]
    if completed.empty:
        return None
    last = completed.iloc[-1]
    return {
        "high":  round(last["High"], 1),
        "low":   round(last["Low"], 1),
        "close": round(last["Close"], 1),
        "time":  completed.index[-1],
    }


# ── Gap Classification ────────────────────────────────────────────────────────

def classify_gap(prev_close: float, today_open: float) -> tuple[str, float]:
    """Classify opening gap direction and size."""
    if prev_close == 0:
        return "FLAT", 0
    gap = round(today_open - prev_close, 1)
    if gap > 10:
        return "GAP_UP", gap
    elif gap < -10:
        return "GAP_DOWN", gap
    return "FLAT", gap


# ── Signal Bar Selection ─────────────────────────────────────────────────────

def should_use_bar5(state: DailyState) -> str:
    """
    Check if current conditions match any BAR5_RULES.
    Rules are "CONDITION1+CONDITION2" combos from config.BAR5_RULES.

    Returns the matched rule string (e.g. "OVERLAP+WIDE") or "" if no match.

    Supported condition tokens:
        Context:   OVERLAP, CHOPPY, DIRECTIONAL
        Range:     WIDE, NARROW, NORMAL
        Gap:       GAP_UP, GAP_DOWN, FLAT
        Overnight: LONG_ONLY, SHORT_ONLY, STANDARD
    """
    # Build set of active condition tags
    tags = set()

    # Context
    if state.context_overlap:
        tags.add("OVERLAP")
    if state.context_choppy:
        tags.add("CHOPPY")
    if state.context_directional:
        tags.add("DIRECTIONAL")

    # Range (from bar 4, computed before bar selection)
    tags.add(state.range_flag)          # WIDE, NARROW, NORMAL

    # Gap direction
    if state.gap_dir:
        tags.add(state.gap_dir)         # GAP_UP, GAP_DOWN, FLAT

    # Overnight bias
    if state.overnight_bias:
        tags.add(state.overnight_bias)  # LONG_ONLY, SHORT_ONLY, STANDARD

    # Check each rule: all tokens in the rule must be present
    for rule in config.BAR5_RULES:
        tokens = rule.split("+")
        if all(t in tags for t in tokens):
            logger.info(f"BAR5 rule matched: {rule} (tags: {tags})")
            return rule

    return ""


# ── Level Calculation ──────────────────────────────────────────────────────────

def calculate_levels(state: DailyState, df: pd.DataFrame) -> list[str]:
    """
    Called at 08:20 UK. Calculate ASRS levels from bar 4 (default) or bar 5
    (when BAR5_RULES conditions are met).
    Returns list of events.
    """
    events = []
    today = df[df.index.date == datetime.now(config.TZ_CET).date()]
    if today.empty:
        events.append("NO_DATA")
        return events

    # Context (bars 1-3)
    ctx = analyse_context(today)
    state.context_overlap = ctx["overlap"]
    state.context_choppy = ctx["choppy"]
    state.context_directional = ctx["directional"]

    # Bar 4 (always needed for range classification)
    bar4 = get_bar(today, 4)
    if not bar4:
        events.append("NO_BAR4")
        return events

    # Classify bar 4 range (needed for should_use_bar5 check)
    bar4_range = bar4["range"]
    if bar4_range < config.NARROW_RANGE:
        state.range_flag = "NARROW"
    elif bar4_range > config.WIDE_RANGE:
        state.range_flag = "WIDE"
    else:
        state.range_flag = "NORMAL"

    # Determine signal bar
    matched_rule = should_use_bar5(state)
    state.use_5th_bar = bool(matched_rule)
    state.bar5_rule_matched = matched_rule

    if matched_rule:
        bar5 = get_bar(today, 5)
        if bar5:
            state.bar_number = 5
            state.bar_high = bar5["high"]
            state.bar_low = bar5["low"]
            state.bar_range = bar5["range"]
            events.append("USING_5TH_BAR")
        else:
            # Bar 5 not yet available — use bar 4 temporarily, bot.py will wait
            state.bar_number = 4
            state.bar_high = bar4["high"]
            state.bar_low = bar4["low"]
            state.bar_range = bar4["range"]
            events.append("WAITING_FOR_BAR5")
    else:
        # Default: use bar 4
        state.bar_number = 4
        state.bar_high = bar4["high"]
        state.bar_low = bar4["low"]
        state.bar_range = bar4["range"]
        events.append("USING_4TH_BAR")

    # Recalculate range flag from the actual signal bar
    if state.bar_range < config.NARROW_RANGE:
        state.range_flag = "NARROW"
    elif state.bar_range > config.WIDE_RANGE:
        state.range_flag = "WIDE"
    else:
        state.range_flag = "NORMAL"

    # Max bar range check
    max_range = getattr(config, 'MAX_BAR_RANGE', 999)
    if state.bar_range > max_range:
        events.append(f"SKIP_WIDE_RANGE_{state.bar_range:.0f}")
        state.save()
        return events

    # Max risk check at minimum stake
    max_risk_gbp = getattr(config, 'MAX_RISK_GBP', 999)
    min_stake = 0.5
    risk_gbp = state.bar_range * min_stake
    if risk_gbp > max_risk_gbp:
        events.append(f"SKIP_RISK_{risk_gbp:.0f}")
        state.save()
        return events

    # Set levels
    state.buy_level = round(state.bar_high + config.BUFFER_PTS, 1)
    state.sell_level = round(state.bar_low - config.BUFFER_PTS, 1)
    state.phase = Phase.LEVELS_SET

    # OCA group name for this day
    state.oca_group = f"ASRS_{state.date}_{state.entries_used + 1}"

    events.append("LEVELS_SET")
    state.save()
    return events


# ── Position Management ────────────────────────────────────────────────────────

def process_fill(state: DailyState, direction: str, fill_price: float) -> list[str]:
    """
    Called when an OCA order fills (entry triggered).
    """
    events = []

    state.direction = direction
    state.entry_price = fill_price
    state.entries_used += 1
    state.contracts_active = state.position_size or config.NUM_CONTRACTS
    state.last_add_price = 0.0  # Reset for add-to-winners (will use entry_price as ref)
    state.adds_used = 0
    state.add_positions = []

    if direction == "LONG":
        state.phase = Phase.LONG_ACTIVE
        state.initial_stop = state.sell_level
        state.trailing_stop = state.sell_level
        state.max_favourable = fill_price
    else:
        state.phase = Phase.SHORT_ACTIVE
        state.initial_stop = state.buy_level
        state.trailing_stop = state.buy_level
        state.max_favourable = fill_price

    # Calculate TP levels for partial exits (3-contract mode)
    if config.PARTIAL_EXIT:
        state.tp1_filled = False
        state.tp2_filled = False
        if direction == "LONG":
            state.tp1_price = round(fill_price + config.TP1_PTS, 1)
            state.tp2_price = round(fill_price + config.TP2_PTS, 1)
        else:
            state.tp1_price = round(fill_price - config.TP1_PTS, 1)
            state.tp2_price = round(fill_price - config.TP2_PTS, 1)

    # Entry slippage: how much worse we got filled vs intended stop level
    intended = state.buy_level if direction == "LONG" else state.sell_level
    # Positive = unfavourable (paid more for LONG, received less for SHORT)
    entry_slip = round(fill_price - intended, 1) if direction == "LONG" else round(intended - fill_price, 1)

    state.trades.append({
        "num":       state.entries_used,
        "direction": direction,
        "entry":     fill_price,
        "entry_intended": intended,
        "entry_slippage": entry_slip,
        "stop":      state.initial_stop,
        "time":      datetime.now(config.TZ_UK).strftime("%H:%M"),
        "slippage_total": entry_slip,  # Running total, updated on exit
        "signal_bar": state.bar_number,
        "bar5_rule":  state.bar5_rule_matched,
        "gap_dir":    state.gap_dir,
    })

    events.append(f"{direction}_ENTRY")
    state.save()
    return events


def update_trail(state: DailyState, df: pd.DataFrame) -> list[str]:
    """
    Update trailing stop based on last completed 5-min candle.
    Returns events if trail moved.
    """
    events = []
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return events

    candle = get_last_completed_candle(df)
    if not candle:
        return events

    old_stop = state.trailing_stop

    # ── Breakeven: lock stop at entry after +N pts profit ──
    if not state.breakeven_hit:
        if state.direction == "LONG":
            current_profit = candle["close"] - state.entry_price
        else:
            current_profit = state.entry_price - candle["close"]

        if current_profit >= config.TRAIL_BREAKEVEN_PTS:
            state.breakeven_hit = True
            if state.direction == "LONG" and state.trailing_stop < state.entry_price:
                state.trailing_stop = state.entry_price
                events.append("BREAKEVEN_HIT")
            elif state.direction == "SHORT" and state.trailing_stop > state.entry_price:
                state.trailing_stop = state.entry_price
                events.append("BREAKEVEN_HIT")
            else:
                events.append("BREAKEVEN_HIT")

    # ── Candle trail: ratchet stop to candle low/high ──
    # After +TRAIL_TIGHT_THRESHOLD pts, use candle close (tighter) instead of low/high
    if state.direction == "LONG":
        current_profit = candle["close"] - state.entry_price
        use_tight = current_profit >= config.TRAIL_TIGHT_THRESHOLD
        new_stop = candle["close"] if use_tight else candle["low"]
        if new_stop > state.trailing_stop:
            state.trailing_stop = new_stop
            events.append("TRAIL_TIGHT" if use_tight else "TRAIL_UPDATED")

        if candle["high"] > state.max_favourable:
            state.max_favourable = candle["high"]

    elif state.direction == "SHORT":
        current_profit = state.entry_price - candle["close"]
        use_tight = current_profit >= config.TRAIL_TIGHT_THRESHOLD
        new_stop = candle["close"] if use_tight else candle["high"]
        if new_stop < state.trailing_stop:
            state.trailing_stop = new_stop
            events.append("TRAIL_TIGHT" if use_tight else "TRAIL_UPDATED")

        if candle["low"] < state.max_favourable:
            state.max_favourable = candle["low"]

    if events:
        state.save()

    return events


def check_add_to_winners(state: DailyState, current_price: float) -> list[str]:
    """
    Check if we should add to the current winning position (strength mode).
    Triggers when price extends +ADD_STRENGTH_TRIGGER pts from the last entry.
    Returns events list with "ADD_TRIGGERED" if an add should be placed.
    """
    events = []
    if not config.ADD_STRENGTH_ENABLED:
        return events
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return events
    if state.adds_used >= config.ADD_STRENGTH_MAX:
        return events

    # Reference price: original entry or last add entry
    ref_price = state.last_add_price if state.last_add_price > 0 else state.entry_price

    if state.direction == "LONG":
        profit_from_ref = current_price - ref_price
    else:
        profit_from_ref = ref_price - current_price

    if profit_from_ref >= config.ADD_STRENGTH_TRIGGER:
        events.append("ADD_TRIGGERED")
        logger.info(f"Add-to-winners triggered: {state.direction} profit={profit_from_ref:.1f} pts "
                     f"from ref={ref_price}, current={current_price}")

    return events


def process_add_fill(state: DailyState, fill_price: float) -> list[str]:
    """Called when an add-to-winners market order fills.
    Also ratchets trailing stop to lock profit on existing contracts.
    """
    events = []
    state.adds_used += 1
    state.contracts_active += 1
    state.last_add_price = fill_price
    state.add_positions.append({
        "entry_price": fill_price,
        "time": datetime.now(config.TZ_UK).strftime("%H:%M"),
    })
    events.append("ADD_FILLED")

    # Lock profit: move stop to entry + half the profit at add time
    # e.g. entry 23100, add at 23075 (SHORT +25pts) → stop moves to 23088 (lock ~12pts)
    if state.direction == "LONG":
        lock_stop = state.entry_price + (fill_price - state.entry_price) * 0.5
        if lock_stop > state.trailing_stop:
            state.trailing_stop = round(lock_stop, 1)
            events.append("ADD_STOP_RATCHETED")
    elif state.direction == "SHORT":
        lock_stop = state.entry_price - (state.entry_price - fill_price) * 0.5
        if lock_stop < state.trailing_stop:
            state.trailing_stop = round(lock_stop, 1)
            events.append("ADD_STOP_RATCHETED")

    state.breakeven_hit = True  # Force breakeven since we're locking profit
    state.save()
    return events


def process_partial_fill(state: DailyState, tp_num: int, fill_price: float) -> list[str]:
    """
    Called when a TP limit order fills (partial exit of 1 contract).
    """
    events = []
    if tp_num == 1:
        state.tp1_filled = True
        state.contracts_active -= 1
        # TP slippage: positive = worse (got less than intended for exit)
        intended = state.tp1_price
        if state.direction == "LONG":
            slip = round(intended - fill_price, 1)  # Wanted higher, got lower = bad
        else:
            slip = round(fill_price - intended, 1)  # Wanted lower, got higher = bad
        if state.trades:
            state.trades[-1]["tp1_slippage"] = slip
            state.trades[-1]["slippage_total"] = round(
                state.trades[-1].get("slippage_total", 0) + slip, 1)
        events.append("TP1_FILLED")
    elif tp_num == 2:
        state.tp2_filled = True
        state.contracts_active -= 1
        intended = state.tp2_price
        if state.direction == "LONG":
            slip = round(intended - fill_price, 1)
        else:
            slip = round(fill_price - intended, 1)
        if state.trades:
            state.trades[-1]["tp2_slippage"] = slip
            state.trades[-1]["slippage_total"] = round(
                state.trades[-1].get("slippage_total", 0) + slip, 1)
        events.append("TP2_FILLED")

    state.save()
    return events


def process_stop_hit(state: DailyState, exit_price: float) -> list[str]:
    """
    Called when trailing stop is hit (exits all remaining contracts).
    """
    events = []

    if state.direction == "LONG":
        pnl_per_contract = round(exit_price - state.entry_price, 1)
        mfe = round(state.max_favourable - state.entry_price, 1)
    else:
        pnl_per_contract = round(state.entry_price - exit_price, 1)
        mfe = round(state.entry_price - state.max_favourable, 1)

    # Calculate total P&L across all contracts (original + adds)
    # pnl_per_contract is for original entry — adds have different entries
    total_pnl = pnl_per_contract * (state.contracts_active - len(state.add_positions))
    # Add P&L for each add position
    add_pnl = 0.0
    for add in state.add_positions:
        if state.direction == "LONG":
            ap = round(exit_price - add["entry_price"], 1)
        else:
            ap = round(add["entry_price"] - exit_price, 1)
        add_pnl += ap
    total_pnl += add_pnl

    if config.PARTIAL_EXIT:
        if state.tp1_filled:
            total_pnl += config.TP1_PTS
        if state.tp2_filled:
            total_pnl += config.TP2_PTS

    # Exit slippage: how much worse than intended stop price
    exit_intended = state.trailing_stop
    if state.direction == "LONG":
        exit_slip = round(exit_intended - exit_price, 1)  # Wanted higher, got lower = bad
    else:
        exit_slip = round(exit_price - exit_intended, 1)  # Wanted lower, got higher = bad

    # Determine exit reason
    if pnl_per_contract == 0.0 and state.breakeven_hit:
        exit_reason = "BREAKEVEN_STOP"
    elif state.breakeven_hit and state.trailing_stop != state.entry_price:
        exit_reason = "TRAILED_STOP"
    else:
        exit_reason = "INITIAL_STOP"

    # Update trade log
    if state.trades:
        state.trades[-1]["exit"] = exit_price
        state.trades[-1]["exit_intended"] = exit_intended
        state.trades[-1]["exit_slippage"] = exit_slip
        state.trades[-1]["slippage_total"] = round(
            state.trades[-1].get("slippage_total", 0) + (exit_slip * state.contracts_active), 1)
        state.trades[-1]["pnl_pts"] = round(total_pnl, 1)
        state.trades[-1]["pnl_per_contract"] = pnl_per_contract
        state.trades[-1]["contracts_stopped"] = state.contracts_active
        state.trades[-1]["tp1_filled"] = state.tp1_filled
        state.trades[-1]["tp2_filled"] = state.tp2_filled
        state.trades[-1]["mfe"] = mfe
        state.trades[-1]["exit_time"] = datetime.now(config.TZ_UK).strftime("%H:%M")
        state.trades[-1]["exit_reason"] = exit_reason

    state.contracts_active = 0
    prev_direction = state.direction
    events.append(f"{state.direction}_STOPPED")

    # Can we re-enter or flip?
    if state.entries_used < config.MAX_ENTRIES:
        state.phase = Phase.LEVELS_SET
        state.direction = ""
        state.breakeven_hit = False
        state.tp1_filled = False
        state.tp2_filled = False
        state.tp1_order_id = 0
        state.tp2_order_id = 0
        state.adds_used = 0
        state.add_positions = []
        state.last_add_price = 0.0
        state.oca_group = f"ASRS_{state.date}_{state.entries_used + 1}"

        # Profitable exit → re-enter same direction on trend continuation
        # Loss exit → re-enter same direction (second attempt at same levels)
        # Flip to opposite direction only if ENABLE_FLIPS is on
        if pnl_per_contract > 0 and exit_reason == "TRAILED_STOP":
            state.reentry_direction = prev_direction
            state.reentry_price = exit_price
            events.append("CAN_REENTER")
        elif config.ENABLE_FLIPS:
            events.append("CAN_FLIP")
        else:
            # Re-enter same direction — second attempt with same bracket
            state.reentry_direction = prev_direction
            state.reentry_price = exit_price
            events.append("CAN_REENTER_SAME")
    else:
        state.phase = Phase.DONE
        state.direction = ""
        state.adds_used = 0
        state.add_positions = []
        state.last_add_price = 0.0
        events.append("MAX_ENTRIES")

    state.save()
    return events


# ── EMA Trail Logic ───────────────────────────────────────────────────────────

class TrailPhase(str, Enum):
    UNDERWATER = "UNDERWATER"
    BREAKEVEN  = "BREAKEVEN"
    EMA_TRAIL  = "EMA_TRAIL"


def calc_ema(closes: list[float], period: int) -> float | None:
    """Calculate EMA from a list of close prices. Returns None if insufficient data."""
    if len(closes) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period  # SMA seed
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return round(ema, 1)


def determine_trail_phase(
    direction: str,
    entry_price: float,
    current_high: float,
    current_low: float,
    ema_value: float | None,
) -> TrailPhase:
    """Determine which trail phase we're in based on price vs entry."""
    if direction == "LONG":
        favour = current_high - entry_price
        above_ema = ema_value is not None and current_low > ema_value
    else:
        favour = entry_price - current_low
        above_ema = ema_value is not None and current_high < ema_value

    if ema_value is not None and favour >= config.TRAIL_EMA_TRIGGER and above_ema:
        return TrailPhase.EMA_TRAIL
    elif favour >= config.TRAIL_BREAKEVEN_TRIGGER:
        return TrailPhase.BREAKEVEN
    return TrailPhase.UNDERWATER


def calc_ema_trail_stop(
    direction: str,
    ema_value: float,
    entry_price: float,
    current_stop: float,
) -> float:
    """Calculate the EMA-based trailing stop with buffer. Never moves backwards."""
    if direction == "LONG":
        raw_stop = round(ema_value * (1 - config.TRAIL_EMA_BUFFER), 1)
        # Never below breakeven once we're in EMA phase
        raw_stop = max(raw_stop, entry_price)
        # Never move backwards
        return max(raw_stop, current_stop)
    else:
        raw_stop = round(ema_value * (1 + config.TRAIL_EMA_BUFFER), 1)
        raw_stop = min(raw_stop, entry_price)
        return min(raw_stop, current_stop)


def check_ema_bounce(
    direction: str,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    ema_value: float,
) -> bool:
    """
    Check if a bar touches the EMA zone and bounces back.
    Touch = within ADD_EMA_TOUCH_ZONE of EMA. Bounce = close back in trend direction.
    """
    touch_zone = ema_value * config.ADD_EMA_TOUCH_ZONE
    if direction == "LONG":
        touched = bar_low <= ema_value + touch_zone
        bounced = bar_close > ema_value
        return touched and bounced
    else:
        touched = bar_high >= ema_value - touch_zone
        bounced = bar_close < ema_value
        return touched and bounced


def update_trail_ema(state: DailyState, df: pd.DataFrame) -> list[str]:
    """
    EMA-based trailing stop update for live trading.
    3 phases: Underwater → Breakeven → EMA Trail.
    Returns events list.
    """
    events = []
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return events

    # Get today's close prices for EMA
    today = df[df.index.date == datetime.now(config.TZ_CET).date()]
    if today.empty:
        return events

    closes = today["Close"].tolist()
    ema = calc_ema(closes, config.TRAIL_EMA_PERIOD)

    candle = get_last_completed_candle(df)
    if not candle:
        return events

    # Track MFE
    if state.direction == "LONG":
        if candle["high"] > state.max_favourable:
            state.max_favourable = candle["high"]
    else:
        if candle["low"] < state.max_favourable:
            state.max_favourable = candle["low"]

    phase = determine_trail_phase(
        state.direction, state.entry_price,
        state.max_favourable,
        candle["low"] if state.direction == "LONG" else candle["high"],
        ema,
    )

    old_stop = state.trailing_stop

    if phase == TrailPhase.UNDERWATER:
        pass  # Keep original stop

    elif phase == TrailPhase.BREAKEVEN:
        if state.direction == "LONG":
            if state.entry_price > state.trailing_stop:
                state.trailing_stop = state.entry_price
                events.append("TRAIL_BREAKEVEN")
        else:
            if state.entry_price < state.trailing_stop:
                state.trailing_stop = state.entry_price
                events.append("TRAIL_BREAKEVEN")

    elif phase == TrailPhase.EMA_TRAIL and ema is not None:
        new_stop = calc_ema_trail_stop(
            state.direction, ema, state.entry_price, state.trailing_stop
        )
        if new_stop != state.trailing_stop:
            state.trailing_stop = new_stop
            events.append("TRAIL_UPDATED")

    if state.trailing_stop != old_stop:
        state.save()

    return events


# ── Day Summary ────────────────────────────────────────────────────────────────

def day_pnl(state: DailyState) -> float:
    """Total P&L for the day in points."""
    return sum(t.get("pnl_pts", 0) for t in state.trades)
