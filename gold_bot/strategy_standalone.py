"""
Strategy 3 — Variant B: Opening Range Breakout
Live trading implementation with state machine.

State machine per instrument per session:
  IDLE → BUILDING_RANGE → WATCHING → ENTERED → IDLE

Each session (London/US for Gold, London/Overlap for EURUSD) is
an independent cycle. All positions closed by session end.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from enum import Enum

import pandas as pd
import numpy as np

from config import InstrumentConfig, StrategyParams, STRATEGY
from indicators import add_all_indicators, classify_regime

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "IDLE"
    BUILDING_RANGE = "BUILDING_RANGE"
    WATCHING = "WATCHING"
    ENTERED = "ENTERED"


@dataclass
class SessionState:
    """Tracks state for one instrument in one session on one day."""
    instrument: str
    session_name: str
    date: str               # YYYY-MM-DD
    state: State = State.IDLE

    # Opening range
    range_high: float = 0.0
    range_low: float = 0.0
    range_bars_collected: int = 0

    # Trade tracking
    trades_taken: int = 0
    daily_pnl_r: float = 0.0

    # Active position
    deal_id: Optional[str] = None
    direction: Optional[str] = None  # BUY or SELL
    entry_price: float = 0.0
    stop_level: float = 0.0
    target_level: float = 0.0
    stake: float = 0.0
    initial_risk: float = 0.0
    entry_time: Optional[datetime] = None


class VariantBStrategy:
    """
    Opening Range Breakout strategy engine.

    On each 5-min bar close:
      1. Check if we're in a session window
      2. If BUILDING_RANGE: accumulate range bars
      3. If WATCHING: check for breakout + confirmation
      4. If ENTERED: manage trailing stop, check session end
    """

    def __init__(self, instrument_config: InstrumentConfig,
                 params: StrategyParams = STRATEGY):
        self.config = instrument_config
        self.params = params
        self.sessions: Dict[str, SessionState] = {}
        self._bar_history: pd.DataFrame = pd.DataFrame()

    def get_session_key(self, ts: datetime, session_name: str) -> str:
        return f"{ts.strftime('%Y-%m-%d')}_{session_name}"

    def get_active_session(self, ts: datetime) -> Optional[tuple]:
        """Return (session_name, start_mins, end_mins) if ts is in a session."""
        h, m = ts.hour, ts.minute
        t_mins = h * 60 + m

        for name, (sh, sm, eh, em) in self.config.sessions.items():
            start = sh * 60 + sm
            end = eh * 60 + em
            if start <= t_mins < end:
                return name, start, end
        return None

    def is_session_open_exclusion(self, ts: datetime) -> bool:
        """First 2 bars (10 mins) of each session = no entry."""
        for open_h, open_m in self.config.session_opens:
            bar_mins = (ts.hour - open_h) * 60 + (ts.minute - open_m)
            if 0 <= bar_mins < self.params.session_open_exclusion_bars * 5:
                return True
        return False

    def on_bar(self, bars_df: pd.DataFrame) -> Optional[dict]:
        """
        Called on each 5-min bar close with full bar history.
        Returns action dict or None.

        Actions:
          {"action": "OPEN", "direction": "BUY"/"SELL",
           "stop_distance": float, "limit_distance": float,
           "stake": float, "reason": str}
          {"action": "CLOSE", "deal_id": str, "reason": str}
          {"action": "UPDATE_STOP", "deal_id": str, "new_stop": float}
          None = no action
        """
        if len(bars_df) < 30:
            return None

        self._bar_history = bars_df
        latest = bars_df.iloc[-1]
        ts = bars_df.index[-1]

        if not isinstance(ts, datetime):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Which session are we in?
        session_info = self.get_active_session(ts)

        # If not in any session, check if we need to close positions
        if session_info is None:
            return self._handle_out_of_session(ts)

        session_name, sess_start_mins, sess_end_mins = session_info
        sess_key = self.get_session_key(ts, session_name)

        # Initialize session state if new
        if sess_key not in self.sessions:
            self.sessions[sess_key] = SessionState(
                instrument=self.config.name,
                session_name=session_name,
                date=ts.strftime("%Y-%m-%d"),
            )
            # Clean up old session states (keep last 5)
            keys = sorted(self.sessions.keys())
            for old_key in keys[:-5]:
                del self.sessions[old_key]

        state = self.sessions[sess_key]
        t_mins = ts.hour * 60 + ts.minute

        # Time within session
        bars_into_session = (t_mins - sess_start_mins) // 5

        # ── State machine ────────────────────────────────────────

        if state.state == State.IDLE:
            # Start building range at session open
            if bars_into_session == 0:
                state.state = State.BUILDING_RANGE
                state.range_high = latest["High"]
                state.range_low = latest["Low"]
                state.range_bars_collected = 1
                logger.info(
                    "[%s/%s] Session started, building range. Bar 1: H=%.2f L=%.2f",
                    self.config.name, session_name, latest["High"], latest["Low"],
                )
            return None

        elif state.state == State.BUILDING_RANGE:
            # Accumulate range bars
            state.range_high = max(state.range_high, latest["High"])
            state.range_low = min(state.range_low, latest["Low"])
            state.range_bars_collected += 1

            if state.range_bars_collected >= self.params.range_bars:
                range_size = state.range_high - state.range_low

                # Validate range
                if range_size < self.config.min_range:
                    logger.info(
                        "[%s/%s] Range too tight (%.4f < %.4f), skipping session",
                        self.config.name, session_name,
                        range_size, self.config.min_range,
                    )
                    state.state = State.IDLE
                    return None

                if range_size > self.config.max_range:
                    logger.info(
                        "[%s/%s] Range too wide (%.4f > %.4f), skipping session",
                        self.config.name, session_name,
                        range_size, self.config.max_range,
                    )
                    state.state = State.IDLE
                    return None

                state.state = State.WATCHING
                logger.info(
                    "[%s/%s] Range set: HIGH=%.4f LOW=%.4f SIZE=%.4f",
                    self.config.name, session_name,
                    state.range_high, state.range_low, range_size,
                )
            return None

        elif state.state == State.WATCHING:
            return self._check_breakout(bars_df, state, ts, sess_end_mins)

        elif state.state == State.ENTERED:
            return self._manage_position(bars_df, state, ts, sess_end_mins)

        return None

    def _check_breakout(self, bars_df: pd.DataFrame, state: SessionState,
                        ts: datetime, sess_end_mins: int) -> Optional[dict]:
        """Check for opening range breakout with confirmation."""
        latest = bars_df.iloc[-1]
        t_mins = ts.hour * 60 + ts.minute

        # Session end approaching — stop looking
        if t_mins >= sess_end_mins - 15:  # stop 15 mins before session end
            state.state = State.IDLE
            return None

        # Trade limit
        if state.trades_taken >= self.params.max_trades_per_session:
            return None

        # Session open exclusion
        if self.is_session_open_exclusion(ts):
            return None

        # Daily loss limit
        if state.daily_pnl_r <= -self.params.max_daily_loss_r:
            logger.info(
                "[%s/%s] Daily loss limit hit (%.1fR), stopping",
                self.config.name, state.session_name, state.daily_pnl_r,
            )
            return None

        # Check breakout
        direction = None
        if latest["High"] > state.range_high:
            direction = "BUY"
        elif latest["Low"] < state.range_low:
            direction = "SELL"

        if direction is None:
            return None

        # Confirmation filters (need 2 of 3)
        confirms = 0

        # Filter A: close vs EMA(20)
        ema20 = latest.get("EMA20")
        if ema20 is not None and not pd.isna(ema20):
            if (direction == "BUY" and latest["Close"] > ema20) or \
               (direction == "SELL" and latest["Close"] < ema20):
                confirms += 1

        # Filter B: range > 1.5× average (volume proxy)
        range_avg = latest.get("range_avg")
        if range_avg is not None and not pd.isna(range_avg) and range_avg > 0:
            if latest["range"] > self.params.confirm_range_mult * range_avg:
                confirms += 1

        # Filter C: body > 60% of bar range
        bar_range = latest["range"]
        if bar_range > 0:
            if latest["body"] / bar_range > self.params.confirm_body_ratio:
                confirms += 1

        if confirms < self.params.confirms_required:
            logger.debug(
                "[%s/%s] Breakout %s but only %d/3 confirms",
                self.config.name, state.session_name, direction, confirms,
            )
            return None

        # Calculate entry, stop, target
        spread = self.config.spread
        if direction == "BUY":
            entry_ref = state.range_high
            stop_level = state.range_low - spread
            initial_risk = entry_ref - stop_level + spread
        else:
            entry_ref = state.range_low
            stop_level = state.range_high + spread
            initial_risk = stop_level - entry_ref + spread

        if initial_risk <= 0:
            return None

        # Position sizing
        raw_stake = self.params.base_risk_gbp / initial_risk
        stake = max(self.config.min_stake,
                    min(raw_stake, self.config.max_stake))
        stake = round(stake, 2)

        stop_distance = initial_risk
        limit_distance = initial_risk * self.params.target_r

        # Store state
        state.direction = direction
        state.entry_price = entry_ref  # approximate — actual from IG confirm
        state.stop_level = stop_level
        state.target_level = (entry_ref + limit_distance if direction == "BUY"
                              else entry_ref - limit_distance)
        state.stake = stake
        state.initial_risk = initial_risk
        state.entry_time = ts
        state.state = State.ENTERED
        state.trades_taken += 1

        logger.info(
            "[%s/%s] ENTRY SIGNAL: %s | range H=%.2f L=%.2f | "
            "stop=%.2f target=%.2f risk=%.2f stake=%.2f | confirms=%d",
            self.config.name, state.session_name, direction,
            state.range_high, state.range_low,
            stop_level, state.target_level, initial_risk, stake, confirms,
        )

        return {
            "action": "OPEN",
            "instrument": self.config.name,
            "epic": self.config.ig_epic,
            "direction": direction,
            "stop_distance": round(stop_distance, 2),
            "limit_distance": round(limit_distance, 2),
            "stake": stake,
            "session": state.session_name,
            "reason": f"ORB breakout {direction} confirms={confirms}",
        }

    def _manage_position(self, bars_df: pd.DataFrame, state: SessionState,
                         ts: datetime, sess_end_mins: int) -> Optional[dict]:
        """Manage open position: session close check."""
        t_mins = ts.hour * 60 + ts.minute

        # Session end — close position
        if t_mins >= sess_end_mins - 5:  # close 5 mins before session end
            logger.info(
                "[%s/%s] Session ending — closing position",
                self.config.name, state.session_name,
            )
            deal_id = state.deal_id
            direction = "SELL" if state.direction == "BUY" else "BUY"
            state.state = State.IDLE
            state.deal_id = None

            if deal_id:
                return {
                    "action": "CLOSE",
                    "instrument": self.config.name,
                    "deal_id": deal_id,
                    "direction": direction,
                    "stake": state.stake,
                    "reason": "SESSION_CLOSE",
                }

        return None

    def _handle_out_of_session(self, ts: datetime) -> Optional[dict]:
        """Close any position that's still open outside session hours."""
        for key, state in self.sessions.items():
            if state.state == State.ENTERED and state.deal_id:
                logger.warning(
                    "[%s/%s] Position open outside session — force closing",
                    self.config.name, state.session_name,
                )
                direction = "SELL" if state.direction == "BUY" else "BUY"
                deal_id = state.deal_id
                state.state = State.IDLE
                state.deal_id = None
                return {
                    "action": "CLOSE",
                    "instrument": self.config.name,
                    "deal_id": deal_id,
                    "direction": direction,
                    "stake": state.stake,
                    "reason": "OUT_OF_SESSION",
                }
        return None

    def register_fill(self, session_key: str, deal_id: str,
                      fill_price: float):
        """Called after IG confirms an order fill."""
        if session_key in self.sessions:
            state = self.sessions[session_key]
            state.deal_id = deal_id
            state.entry_price = fill_price
            logger.info(
                "[%s/%s] Fill registered: deal=%s price=%.2f",
                self.config.name, state.session_name, deal_id, fill_price,
            )

    def register_close(self, session_key: str, exit_price: float,
                       reason: str):
        """Called after a position is closed — update P&L tracking."""
        if session_key in self.sessions:
            state = self.sessions[session_key]
            if state.direction == "BUY":
                pnl_pts = exit_price - state.entry_price
            else:
                pnl_pts = state.entry_price - exit_price

            r_mult = pnl_pts / state.initial_risk if state.initial_risk > 0 else 0
            state.daily_pnl_r += r_mult
            state.state = State.WATCHING  # can take another trade if limit allows
            state.deal_id = None

            logger.info(
                "[%s/%s] Position closed: pnl=%.2f pts (%.2fR) reason=%s daily=%.1fR",
                self.config.name, state.session_name,
                pnl_pts, r_mult, reason, state.daily_pnl_r,
            )
            return {
                "r_multiple": r_mult,
                "pnl_pts": pnl_pts,
                "pnl_gbp": pnl_pts * state.stake,
            }
        return None
