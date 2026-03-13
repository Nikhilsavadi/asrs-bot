"""
strategy.py -- Gold 15-min ORB + Weekly ORB
State machine per session. Receives 15-min aggregated bars.
Also tracks daily bars for weekly ORB (Monday range -> trade Tue-Fri).
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from enum import Enum

import numpy as np

from gold_bot import config

_WEEKLY_STATE_FILE = os.path.join(os.getenv("LOG_DIR", "/data"), "gold_weekly_state.json")

logger = logging.getLogger("GOLD_ORB")


class State(Enum):
    IDLE = "IDLE"
    BUILDING_RANGE = "BUILDING_RANGE"
    WATCHING = "WATCHING"
    ENTERED = "ENTERED"


@dataclass
class SessionState:
    instrument: str
    session_name: str
    date: str
    state: State = State.IDLE
    range_high: float = 0.0
    range_low: float = 0.0
    range_bars_collected: int = 0
    trades_taken: int = 0
    daily_pnl_r: float = 0.0
    deal_id: Optional[str] = None
    direction: Optional[str] = None
    entry_price: float = 0.0
    stop_level: float = 0.0
    target_level: float = 0.0
    stake: float = 0.0
    initial_risk: float = 0.0
    entry_time: Optional[datetime] = None


@dataclass
class WeeklyState:
    """Tracks Monday's range and weekly ORB position."""
    week_key: str = ""
    range_high: float = 0.0
    range_low: float = 0.0
    range_set: bool = False
    traded: bool = False
    deal_id: Optional[str] = None
    direction: Optional[str] = None
    entry_price: float = 0.0
    stop_level: float = 0.0
    target_level: float = 0.0
    stake: float = 0.0
    initial_risk: float = 0.0

    def save(self):
        """Persist to JSON so weekly range survives bot restarts."""
        try:
            with open(_WEEKLY_STATE_FILE, "w") as f:
                json.dump(asdict(self), f)
        except Exception as e:
            logger.error("Failed to save weekly state: %s", e)

    @classmethod
    def load(cls) -> "WeeklyState":
        """Load from JSON, return default if missing/corrupt."""
        try:
            with open(_WEEKLY_STATE_FILE) as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()


class ORBStrategy:
    """
    15-min Opening Range Breakout for Gold.
    Called on each 15-min aggregated bar close.
    """

    def __init__(self, instrument_name: str):
        self.name = instrument_name
        self.cfg = config.INSTRUMENTS[instrument_name]
        self.sessions: Dict[str, SessionState] = {}
        self._bars: list = []
        self._daily_pnl_gbp: float = 0.0  # Reset each day, hard GBP cap
        self._pnl_date: str = ""           # Track which day we're on
        self._ema20: float = 0.0
        self._range_avg: float = 0.0

        # Weekly ORB (load persisted state so range survives restarts)
        self._daily_bars: list = []  # {date, Open, High, Low, Close}
        self._current_day_bar: dict = {}
        self.weekly: WeeklyState = WeeklyState.load()
        if self.weekly.range_set:
            logger.info("[%s/WEEKLY] Restored from disk: %s H=%.2f L=%.2f traded=%s deal=%s",
                        instrument_name, self.weekly.week_key,
                        self.weekly.range_high, self.weekly.range_low,
                        self.weekly.traded, self.weekly.deal_id)

    def on_bar(self, bar: dict) -> Optional[dict]:
        """
        Called on each 15-min bar close.
        bar: {Open, High, Low, Close, timestamp}
        Returns action dict or None.
        """
        self._bars.append(bar)
        if len(self._bars) > 200:
            self._bars = self._bars[-200:]

        self._update_indicators()
        self._update_daily_bar(bar)

        ts = bar["timestamp"]
        if not isinstance(ts, datetime):
            from pandas import Timestamp
            ts = Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Check 15-min session ORB first
        action = self._process_bar(bar, ts)
        if action is not None:
            return action

        # Check weekly ORB
        if config.WEEKLY_ORB_ENABLED:
            action = self._process_weekly(bar, ts)
            if action is not None:
                return action

        return None

    def _update_indicators(self):
        if len(self._bars) < 20:
            return
        closes = [b["Close"] for b in self._bars[-20:]]
        alpha = 2.0 / 21.0
        ema = closes[0]
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        self._ema20 = ema
        ranges = [b["High"] - b["Low"] for b in self._bars[-20:]]
        self._range_avg = np.mean(ranges)

    def _update_daily_bar(self, bar):
        """Aggregate 15-min bars into daily bars for weekly ORB."""
        ts = bar["timestamp"]
        if not isinstance(ts, datetime):
            from pandas import Timestamp
            ts = Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        date_str = ts.strftime("%Y-%m-%d")

        if not self._current_day_bar or self._current_day_bar.get("date") != date_str:
            # Save previous day
            if self._current_day_bar:
                self._daily_bars.append(self._current_day_bar)
                if len(self._daily_bars) > 30:
                    self._daily_bars = self._daily_bars[-30:]
                # Check if Monday just closed -> set weekly range
                self._check_monday_close()
            # Start new day
            self._current_day_bar = {
                "date": date_str,
                "weekday": ts.weekday(),
                "Open": bar["Open"],
                "High": bar["High"],
                "Low": bar["Low"],
                "Close": bar["Close"],
            }
        else:
            self._current_day_bar["High"] = max(self._current_day_bar["High"], bar["High"])
            self._current_day_bar["Low"] = min(self._current_day_bar["Low"], bar["Low"])
            self._current_day_bar["Close"] = bar["Close"]

    def _check_monday_close(self):
        """After Monday's daily bar closes, set weekly range."""
        if not self._daily_bars:
            return
        last_day = self._daily_bars[-1]
        if last_day.get("weekday") != 0:  # Not Monday
            return

        week_key = last_day["date"]  # Use Monday's date as key
        rng = last_day["High"] - last_day["Low"]

        if rng < config.WEEKLY_MIN_RANGE or rng > config.WEEKLY_MAX_RANGE:
            logger.info("[%s/WEEKLY] Monday range %.2f out of bounds, skip", self.name, rng)
            return

        self.weekly = WeeklyState(
            week_key=week_key,
            range_high=last_day["High"],
            range_low=last_day["Low"],
            range_set=True,
        )
        self.weekly.save()
        logger.info("[%s/WEEKLY] Range set from Monday %s: H=%.2f L=%.2f size=%.2f",
                     self.name, week_key, last_day["High"], last_day["Low"], rng)

    # ── 15-min Session ORB ─────────────────────────────────────

    def _process_bar(self, bar, ts: datetime) -> Optional[dict]:
        h, m = ts.hour, ts.minute
        t_mins = h * 60 + m

        active_session = None
        sess_start = sess_end = 0
        for sname, sconf in self.cfg["sessions"].items():
            sh, sm = sconf["start"]
            eh, em = sconf["end"]
            start = sh * 60 + sm
            end = eh * 60 + em
            if start <= t_mins < end:
                active_session = sname
                sess_start = start
                sess_end = end
                break

        if active_session is None:
            return self._handle_out_of_session()

        sess_key = ts.strftime("%Y-%m-%d") + "_" + active_session
        if sess_key not in self.sessions:
            self.sessions[sess_key] = SessionState(
                instrument=self.name, session_name=active_session,
                date=ts.strftime("%Y-%m-%d"),
            )
            keys = sorted(self.sessions.keys())
            for old in keys[:-10]:
                del self.sessions[old]

        state = self.sessions[sess_key]
        bars_into = (t_mins - sess_start) // config.CANDLE_TF_MINUTES

        if state.state == State.IDLE:
            if bars_into == 0:
                state.state = State.BUILDING_RANGE
                state.range_high = bar["High"]
                state.range_low = bar["Low"]
                state.range_bars_collected = 1
                logger.info("[%s/%s] Building range. Bar 1: H=%.2f L=%.2f",
                            self.name, active_session, bar["High"], bar["Low"])
            return None

        elif state.state == State.BUILDING_RANGE:
            state.range_high = max(state.range_high, bar["High"])
            state.range_low = min(state.range_low, bar["Low"])
            state.range_bars_collected += 1

            if state.range_bars_collected >= config.RANGE_BARS:
                rng = state.range_high - state.range_low
                if rng < self.cfg["min_range"]:
                    logger.info("[%s/%s] Range too tight (%.2f), skip",
                                self.name, active_session, rng)
                    state.state = State.IDLE
                    return None
                if rng > self.cfg["max_range"]:
                    logger.info("[%s/%s] Range too wide (%.2f), skip",
                                self.name, active_session, rng)
                    state.state = State.IDLE
                    return None
                state.state = State.WATCHING
                logger.info("[%s/%s] Range set: H=%.2f L=%.2f size=%.2f",
                            self.name, active_session,
                            state.range_high, state.range_low, rng)
            return None

        elif state.state == State.WATCHING:
            return self._check_breakout(bar, state, ts, sess_end)

        elif state.state == State.ENTERED:
            return self._manage_position(bar, state, ts, sess_end)

        return None

    def _check_breakout(self, bar, state, ts, sess_end_mins):
        t_mins = ts.hour * 60 + ts.minute

        # Don't enter in last 30 mins of session
        if t_mins >= sess_end_mins - 30:
            state.state = State.IDLE
            logger.info("[%s/%s] Session ending (last 30 mins), going idle",
                        self.name, state.session_name)
            return None

        if state.trades_taken >= config.MAX_TRADES_PER_SESSION:
            return None

        # Session open exclusion
        for oh, om in self.cfg["session_opens"]:
            bar_mins = (ts.hour - oh) * 60 + (ts.minute - om)
            if 0 <= bar_mins < config.EXCLUSION_BARS * config.CANDLE_TF_MINUTES:
                return None

        if state.daily_pnl_r <= -config.MAX_DAILY_LOSS_R:
            return None

        # Hard GBP daily loss cap (reset on new day)
        today = ts.strftime("%Y-%m-%d")
        if self._pnl_date != today:
            self._pnl_date = today
            self._daily_pnl_gbp = 0.0
        if self._daily_pnl_gbp <= -config.MAX_DAILY_LOSS_GBP:
            logger.warning("[%s] Daily GBP loss cap hit: %.2f", self.name, self._daily_pnl_gbp)
            return None

        direction = None
        if bar["High"] > state.range_high:
            direction = "BUY"
        elif bar["Low"] < state.range_low:
            direction = "SELL"

        if direction is None:
            return None

        logger.info("[%s/%s] Breakout detected: %s | bar H=%.2f L=%.2f C=%.2f | "
                    "range H=%.2f L=%.2f",
                    self.name, state.session_name, direction,
                    bar["High"], bar["Low"], bar["Close"],
                    state.range_high, state.range_low)

        # Confirmation (2 of 3)
        confirms = 0
        if len(self._bars) >= 20:
            if (direction == "BUY" and bar["Close"] > self._ema20) or \
               (direction == "SELL" and bar["Close"] < self._ema20):
                confirms += 1
        bar_range = bar["High"] - bar["Low"]
        if self._range_avg > 0 and bar_range > config.CONFIRM_RANGE_MULT * self._range_avg:
            confirms += 1
        body = abs(bar["Close"] - bar["Open"])
        if bar_range > 0 and body / bar_range > config.CONFIRM_BODY_RATIO:
            confirms += 1

        if confirms < config.CONFIRMS_REQUIRED:
            logger.info("[%s/%s] Breakout %s rejected: confirms=%d/%d "
                        "(EMA=%.2f, range=%.2f/avg=%.2f, body%%=%.0f)",
                        self.name, state.session_name, direction, confirms,
                        config.CONFIRMS_REQUIRED, self._ema20,
                        bar_range, self._range_avg,
                        (body / bar_range * 100) if bar_range > 0 else 0)
            return None

        spread = self.cfg["spread"]
        if direction == "BUY":
            stop_level = state.range_low - spread
            initial_risk = bar["Close"] - stop_level
        else:
            stop_level = state.range_high + spread
            initial_risk = stop_level - bar["Close"]

        if initial_risk <= 0:
            return None

        raw_stake = config.BASE_RISK_GBP / initial_risk
        stake = max(config.MIN_STAKE, min(raw_stake, config.MAX_STAKE))
        stake = round(stake, 2)

        state.direction = direction
        state.stop_level = stop_level
        state.target_level = (bar["Close"] + initial_risk * config.TARGET_R
                              if direction == "BUY"
                              else bar["Close"] - initial_risk * config.TARGET_R)
        state.stake = stake
        state.initial_risk = initial_risk
        state.entry_time = ts
        state.state = State.ENTERED
        state.trades_taken += 1

        logger.info(
            "[%s/%s] ENTRY: %s | range H=%.2f L=%.2f | "
            "stop=%.2f tgt=%.2f risk=%.2f stake=%.2f confirms=%d",
            self.name, state.session_name, direction,
            state.range_high, state.range_low,
            stop_level, state.target_level, initial_risk, stake, confirms,
        )

        return {
            "action": "OPEN",
            "instrument": self.name,
            "epic": self.cfg["epic"],
            "direction": direction,
            "stop_distance": round(abs(bar["Close"] - stop_level), 2),
            "limit_distance": round(initial_risk * config.TARGET_R, 2),
            "stake": stake,
            "session": state.session_name,
            "sess_key": ts.strftime("%Y-%m-%d") + "_" + state.session_name,
        }

    def _manage_position(self, bar, state, ts, sess_end_mins):
        t_mins = ts.hour * 60 + ts.minute

        # Close 15 mins before session end
        if t_mins >= sess_end_mins - 15:
            logger.info("[%s/%s] Session ending -- closing",
                        self.name, state.session_name)
            close_dir = "SELL" if state.direction == "BUY" else "BUY"
            deal_id = state.deal_id
            state.state = State.IDLE
            state.deal_id = None
            if deal_id:
                return {
                    "action": "CLOSE",
                    "instrument": self.name,
                    "deal_id": deal_id,
                    "direction": close_dir,
                    "stake": state.stake,
                    "reason": "SESSION_CLOSE",
                    "sess_key": ts.strftime("%Y-%m-%d") + "_" + state.session_name,
                }
        return None

    def _handle_out_of_session(self):
        for key, state in self.sessions.items():
            if state.state == State.ENTERED and state.deal_id:
                logger.warning("[%s/%s] Position open outside session!",
                               self.name, state.session_name)
                close_dir = "SELL" if state.direction == "BUY" else "BUY"
                deal_id = state.deal_id
                state.state = State.IDLE
                state.deal_id = None
                return {
                    "action": "CLOSE", "instrument": self.name,
                    "deal_id": deal_id, "direction": close_dir,
                    "stake": state.stake, "reason": "OUT_OF_SESSION",
                    "sess_key": key,
                }
        return None

    # ── Weekly ORB ─────────────────────────────────────────────

    def _process_weekly(self, bar, ts: datetime) -> Optional[dict]:
        """Check weekly ORB on each 15-min bar. Monday = range, Tue-Fri = trade."""
        if not self.weekly.range_set:
            return None

        # Only trade Tue-Fri (weekday 1-4)
        if ts.weekday() == 0:
            return None
        if ts.weekday() >= 5:
            return None

        # If already traded this week
        if self.weekly.traded and not self.weekly.deal_id:
            return None

        # Manage existing weekly position
        if self.weekly.deal_id:
            # Close on Friday after 17:00 UTC
            if ts.weekday() == 4 and ts.hour >= 17:
                logger.info("[%s/WEEKLY] Friday close", self.name)
                close_dir = "SELL" if self.weekly.direction == "BUY" else "BUY"
                deal_id = self.weekly.deal_id
                self.weekly.deal_id = None
                self.weekly.save()
                return {
                    "action": "CLOSE", "instrument": self.name,
                    "deal_id": deal_id, "direction": close_dir,
                    "stake": self.weekly.stake, "reason": "WEEKLY_CLOSE",
                    "sess_key": "WEEKLY_" + self.weekly.week_key,
                }
            return None

        # Look for weekly breakout
        direction = None
        if bar["High"] > self.weekly.range_high:
            direction = "BUY"
        elif bar["Low"] < self.weekly.range_low:
            direction = "SELL"

        if direction is None:
            return None

        # Confirmation (2 of 3) on 15-min bar
        confirms = 0
        if len(self._bars) >= 20:
            if (direction == "BUY" and bar["Close"] > self._ema20) or \
               (direction == "SELL" and bar["Close"] < self._ema20):
                confirms += 1
        bar_range = bar["High"] - bar["Low"]
        if self._range_avg > 0 and bar_range > config.CONFIRM_RANGE_MULT * self._range_avg:
            confirms += 1
        body = abs(bar["Close"] - bar["Open"])
        if bar_range > 0 and body / bar_range > config.CONFIRM_BODY_RATIO:
            confirms += 1

        if confirms < config.WEEKLY_CONFIRMS:
            return None

        spread = self.cfg["spread"]
        if direction == "BUY":
            stop_level = self.weekly.range_low - spread
            initial_risk = bar["Close"] - stop_level
        else:
            stop_level = self.weekly.range_high + spread
            initial_risk = stop_level - bar["Close"]

        if initial_risk <= 0:
            return None

        raw_stake = config.BASE_RISK_GBP / initial_risk
        stake = max(config.MIN_STAKE, min(raw_stake, config.MAX_STAKE))
        stake = round(stake, 2)

        self.weekly.direction = direction
        self.weekly.stop_level = stop_level
        self.weekly.target_level = (bar["Close"] + initial_risk * config.WEEKLY_TARGET_R
                                    if direction == "BUY"
                                    else bar["Close"] - initial_risk * config.WEEKLY_TARGET_R)
        self.weekly.stake = stake
        self.weekly.initial_risk = initial_risk
        self.weekly.traded = True
        self.weekly.save()

        logger.info(
            "[%s/WEEKLY] ENTRY: %s | Mon H=%.2f L=%.2f | "
            "stop=%.2f tgt=%.2f risk=%.2f stake=%.2f",
            self.name, direction,
            self.weekly.range_high, self.weekly.range_low,
            stop_level, self.weekly.target_level, initial_risk, stake,
        )

        return {
            "action": "OPEN",
            "instrument": self.name,
            "epic": self.cfg["epic"],
            "direction": direction,
            "stop_distance": round(abs(bar["Close"] - stop_level), 2),
            "limit_distance": round(initial_risk * config.WEEKLY_TARGET_R, 2),
            "stake": stake,
            "session": "WEEKLY",
            "sess_key": "WEEKLY_" + self.weekly.week_key,
        }

    # ── Fill / Close registration ──────────────────────────────

    def register_fill(self, sess_key, deal_id, fill_price):
        # Weekly fill
        if sess_key.startswith("WEEKLY_"):
            self.weekly.deal_id = deal_id
            self.weekly.entry_price = fill_price
            self.weekly.save()
            logger.info("[%s/WEEKLY] Fill: deal=%s price=%.2f",
                        self.name, deal_id, fill_price)
            return

        if sess_key in self.sessions:
            s = self.sessions[sess_key]
            s.deal_id = deal_id
            s.entry_price = fill_price
            logger.info("[%s/%s] Fill: deal=%s price=%.2f",
                        self.name, s.session_name, deal_id, fill_price)

    def register_close(self, sess_key, exit_price, reason):
        # Weekly close
        if sess_key.startswith("WEEKLY_"):
            w = self.weekly
            if w.direction and w.entry_price > 0:
                pnl = (exit_price - w.entry_price) if w.direction == "BUY" \
                    else (w.entry_price - exit_price)
                r = pnl / w.initial_risk if w.initial_risk > 0 else 0
                pnl_gbp = pnl * w.stake
                self._daily_pnl_gbp += pnl_gbp
                w.deal_id = None
                w.save()
                logger.info("[%s/WEEKLY] Closed: %.2f pts (%.2fR) %s (GBP=%.2f)",
                            self.name, pnl, r, reason, pnl_gbp)
                return {"pnl_pts": pnl, "r_multiple": r, "pnl_gbp": pnl_gbp}
            return None

        if sess_key in self.sessions:
            s = self.sessions[sess_key]
            pnl = (exit_price - s.entry_price) if s.direction == "BUY" \
                else (s.entry_price - exit_price)
            r = pnl / s.initial_risk if s.initial_risk > 0 else 0
            s.daily_pnl_r += r
            pnl_gbp = pnl * s.stake
            self._daily_pnl_gbp += pnl_gbp
            s.state = State.WATCHING
            s.deal_id = None
            logger.info("[%s/%s] Closed: %.2f pts (%.2fR) %s daily=%.1fR (GBP=%.2f, day=%.2f)",
                        self.name, s.session_name, pnl, r, reason, s.daily_pnl_r,
                        pnl_gbp, self._daily_pnl_gbp)
            return {"pnl_pts": pnl, "r_multiple": r, "pnl_gbp": pnl_gbp}
        return None
