"""
strategy.py -- Signal class: one instance per signal (6 total)
=============================================================

State machine per signal:
    IDLE -> LEVELS_SET -> BRACKET_ARMED -> LONG/SHORT -> (exit) -> BRACKET_ARMED or DONE

Rules (30 rules from spec -- numbered in comments):
    R1-R5:   Entry (bar 4, hybrid bar 5, buy/sell levels, OCA)
    R6-R11:  Risk checks at entry
    R12-R16: Position management (stop, breakeven, trail, adds)
    R17-R19: Exit
    R20-R22: Re-entry
    R23-R25: S2 independence
"""

import json
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from zoneinfo import ZoneInfo

from asrs import config

logger = logging.getLogger(__name__)


# -- State machine phases -----------------------------------------------------

class Phase(str, Enum):
    IDLE           = "IDLE"
    LEVELS_SET     = "LEVELS_SET"
    BRACKET_ARMED  = "BRACKET_ARMED"
    LONG           = "LONG"
    SHORT          = "SHORT"
    DONE           = "DONE"


# -- Per-signal daily state ---------------------------------------------------

@dataclass
class SignalState:
    """Fresh state for each trading day. JSON-serialisable."""
    date:              str = ""
    phase:             str = Phase.IDLE

    # Bar data
    bar_number:        int = 0       # 4 or 5
    bar_high:          float = 0.0
    bar_low:           float = 0.0
    bar_range:         float = 0.0
    range_flag:        str = ""      # NARROW, NORMAL, WIDE

    # Entry levels (original, for re-entry)
    buy_level:         float = 0.0
    sell_level:        float = 0.0

    # Position
    direction:         str = ""
    entry_price:       float = 0.0
    initial_stop:      float = 0.0
    trailing_stop:     float = 0.0
    max_favourable:    float = 0.0
    breakeven_hit:     bool = False
    trail_moved:       bool = False
    entries_used:      int = 0

    # Adds
    adds_used:         int = 0
    add_positions:     list = field(default_factory=list)
    last_add_price:    float = 0.0

    # Deal tracking (IG)
    deal_ids:          list = field(default_factory=list)

    # Trade log (intra-day)
    trades:            list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalState":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


# =============================================================================
#  Signal class -- one per (instrument, session) pair
# =============================================================================

class Signal:
    """
    One instance per signal. 6 total: DAX_S1, DAX_S2, US30_S1, US30_S2,
    NIKKEI_S1, NIKKEI_S2.

    The Signal owns its state machine, level calculation, trailing logic,
    breakeven, adds, re-entry. It delegates order execution to broker.
    """

    def __init__(self, instrument: str, session_num: int, broker, stream, alert_fn):
        self.instrument = instrument                          # "DAX", "US30", "NIKKEI"
        self.session = session_num                            # 1 or 2
        self.name = f"{instrument}_S{session_num}"            # e.g. "DAX_S1"
        self.cfg = config.INSTRUMENTS[instrument]             # instrument config dict
        self.tz = ZoneInfo(self.cfg["timezone"])               # instrument timezone
        self.broker = broker                                  # IGBroker instance
        self.stream = stream                                  # IGStreamManager
        self.alert = alert_fn                                 # async fn(text)
        self.state = SignalState()                             # fresh each day
        self._bar4_triggered = False                           # prevent double trigger
        self._morning_running = False                          # mutex
        self._sibling: "Signal | None" = None                  # S1<->S2 link
        self._state_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "state"
        )
        os.makedirs(self._state_dir, exist_ok=True)

    def set_sibling(self, other: "Signal"):
        """Link S1 and S2 so S2 can cancel S1 bracket (R24)."""
        self._sibling = other

    # -- State persistence ----------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self._state_dir, f"{self.name}.json")

    def save_state(self):
        with open(self._state_path(), "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load_state(self):
        """Load state for today, or reset if stale."""
        today = datetime.now(self.tz).strftime("%Y-%m-%d")
        path = self._state_path()
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if data.get("date") == today:
                    self.state = SignalState.from_dict(data)
                    return
        except Exception as e:
            logger.error(f"[{self.name}] State load error: {e}")
        self.state = SignalState(date=today)
        self._bar4_triggered = False

    # -- Bar numbering --------------------------------------------------------

    def _bar_number(self, bar_time: datetime) -> int:
        """
        Which 5-min bar from session open. Bar 1 starts at session open.
        Uses hour/minute comparison, NOT timestamp arithmetic (R-impl-1).
        Converts bar_time to instrument timezone first (tick-bar builder uses CET).
        """
        open_h = self.cfg[f"s{self.session}_open_hour"]
        open_m = self.cfg[f"s{self.session}_open_minute"]
        # Convert to instrument timezone (bars are timestamped in CET by tick-bar builder)
        bar_local = bar_time.astimezone(self.tz) if bar_time.tzinfo else bar_time
        bar_h = bar_local.hour
        bar_m = bar_local.minute
        mins_from_open = (bar_h * 60 + bar_m) - (open_h * 60 + open_m)
        if mins_from_open < 0:
            return -1
        return mins_from_open // 5 + 1

    # =========================================================================
    #  on_bar_complete -- event-driven entry from tick-bar callback (R1, R2)
    # =========================================================================

    async def on_bar_complete(self, bar: dict):
        """
        Called when a 5-min bar completes via Lightstreamer tick-bar builder.
        Checks if it is our bar 4 (or bar 5 in hybrid mode).
        """
        bar_time = bar["time"]  # tz-aware datetime (CET from tick-bar builder)
        bar_local = bar_time.astimezone(self.tz) if bar_time.tzinfo else bar_time
        today = datetime.now(self.tz).date()
        if bar_local.date() != today:
            return

        bn = self._bar_number(bar_time)
        if bn <= 0:
            return

        # Log bar 1 for stream-alive confirmation
        if bn == 1:
            logger.info(f"[{self.name}] Bar 1 complete -- stream alive "
                        f"O={bar['Open']} H={bar['High']} L={bar['Low']} C={bar['Close']}")

        # Bar 4 trigger (R1)
        if bn == 4 and not self._bar4_triggered:
            self.load_state()
            if self.state.phase == Phase.IDLE:
                self._bar4_triggered = True
                import asyncio
                logger.info(f"[{self.name}] Bar 4 complete -- triggering morning routine")
                await asyncio.sleep(2)  # let bar store update before reading
                await self.morning_routine()

        # Bar 5 trigger -- only if we are waiting for bar 5 (R2)
        if bn == 5 and self.state.phase == Phase.IDLE and self.state.bar_number == 0:
            # morning_routine already ran and decided to wait for bar 5
            pass  # morning_routine handles bar 5 wait internally

    # =========================================================================
    #  morning_routine -- calculate levels, arm bracket (R1-R11)
    # =========================================================================

    async def morning_routine(self):
        """
        Cron-scheduled + event-driven fallback.
        Calculates bar 4 levels, checks bar 5 hybrid, arms OCA bracket.
        """
        if self._morning_running:
            logger.warning(f"[{self.name}] morning_routine already running -- skip")
            return
        self._morning_running = True
        try:
            await self._morning_routine_inner()
        except Exception as e:
            logger.error(f"[{self.name}] morning_routine error: {e}", exc_info=True)
            await self.alert(f"[{self.name}] Morning routine ERROR: {e}")
        finally:
            self._morning_running = False

    async def _morning_routine_inner(self):
        now = datetime.now(self.tz)
        if now.weekday() >= 5:
            return

        # Holiday check (R10)
        from shared.holidays import is_holiday
        if is_holiday(now.date(), self.instrument):
            logger.info(f"[{self.name}] Market holiday -- skip")
            await self.alert(f"[{self.name}] Market holiday today -- no trading")
            return

        self.load_state()
        if self.state.phase != Phase.IDLE:
            logger.info(f"[{self.name}] Already processed (phase={self.state.phase})")
            return

        # S2 cancels S1 bracket (R24)
        if self.session == 2 and self._sibling:
            sib = self._sibling
            if sib.state.phase == Phase.BRACKET_ARMED:
                sib.broker.deactivate_bracket()
                sib.state.phase = Phase.DONE
                sib.save_state()
                logger.info(f"[{self.name}] Cancelled S1 bracket")
                await self.alert(f"[{self.name}] S1 bracket cancelled -- S2 starting")

        # Status: morning routine starting
        bar_count = self.broker.get_streaming_bar_count()
        logger.info(f"[{self.name}] ═══ MORNING ROUTINE ═══ ({bar_count} bars)")
        await self.alert(
            f"[{self.name}] ═══ MORNING ROUTINE ═══\n"
            f"Streaming bars: {bar_count}\n"
            f"IG: {'✅' if await self.broker.ensure_connected() else '❌'}"
        )

        # Ensure broker connected
        if not await self.broker.ensure_connected():
            await self.alert(f"[{self.name}] IG connection failed -- cannot calculate levels")
            return

        # Get today's bars from streaming (with REST fallback)
        df = await self._get_bars_with_fallback()
        if df is None or df.empty:
            await self.alert(f"[{self.name}] No bar data available -- skipping")
            return

        # Find bar 4 using hour/minute matching (R-impl-1)
        bar4 = self._find_bar(df, 4)
        if bar4 is None:
            await self.alert(f"[{self.name}] Bar 4 not found in data -- skipping")
            return

        bar4_range = round(bar4["High"] - bar4["Low"], 1)

        # Range classification
        if bar4_range < self.cfg["narrow_range"]:
            range_flag = "NARROW"
        elif bar4_range > self.cfg["wide_range"]:
            range_flag = "WIDE"
        else:
            range_flag = "NORMAL"

        # Hybrid bar 5: if bar 4 is NORMAL or WIDE, wait for bar 5 (R2)
        use_bar5 = range_flag in config.BAR5_RULES
        signal_bar = bar4
        bar_num = 4

        if use_bar5:
            bar5 = self._find_bar(df, 5)
            if bar5 is not None:
                signal_bar = bar5
                bar_num = 5
                bar4_range = round(bar5["High"] - bar5["Low"], 1)
                logger.info(f"[{self.name}] Using bar 5 (bar 4 was {range_flag})")
            else:
                # Wait up to 5 minutes for bar 5
                logger.info(f"[{self.name}] Waiting for bar 5...")
                import asyncio
                for i in range(6):
                    await asyncio.sleep(50)
                    df = self.broker.get_streaming_bars_df()
                    bar5 = self._find_bar(df, 5) if df is not None and not df.empty else None
                    if bar5 is not None:
                        signal_bar = bar5
                        bar_num = 5
                        bar4_range = round(bar5["High"] - bar5["Low"], 1)
                        break
                if bar_num == 4:
                    logger.info(f"[{self.name}] Bar 5 not available -- using bar 4")

        # Reclassify range for actual signal bar
        if bar4_range < self.cfg["narrow_range"]:
            range_flag = "NARROW"
        elif bar4_range > self.cfg["wide_range"]:
            range_flag = "WIDE"
        else:
            range_flag = "NORMAL"

        bar_high = round(signal_bar["High"], 1)
        bar_low = round(signal_bar["Low"], 1)
        bar_range = round(bar_high - bar_low, 1)

        # R6: Max bar range check
        if bar_range > self.cfg["max_bar_range"]:
            logger.info(f"[{self.name}] Bar range {bar_range} > max {self.cfg['max_bar_range']} -- skip")
            await self.alert(f"[{self.name}] SKIP: bar range {bar_range} > {self.cfg['max_bar_range']}")
            self.state.phase = Phase.DONE
            self.save_state()
            return

        # Calculate levels (R3, R4)
        buffer = self.cfg["buffer"]
        buy_level = round(bar_high + buffer, 1)
        sell_level = round(bar_low - buffer, 1)

        # R7: Max risk check -- if risk > cap, TIGHTEN stop (don't skip)
        risk_pts = bar_range + buffer * 2
        risk_gbp = risk_pts * 1.0  # at minimum £1/pt stake
        if risk_gbp > self.cfg["max_risk_gbp"]:
            max_stop_distance = self.cfg["max_risk_gbp"]
            logger.info(f"[{self.name}] Risk cap: {risk_gbp:.0f} > {self.cfg['max_risk_gbp']:.0f} "
                        f"-- tightening to {max_stop_distance:.0f}pts")
            bar_high = round(sell_level + max_stop_distance, 1)
            bar_low = round(buy_level - max_stop_distance, 1)
            bar_range = max_stop_distance
            await self.alert(f"[{self.name}] Risk capped: {risk_pts:.0f}pts -> {max_stop_distance:.0f}pts "
                             f"(max {self.cfg['max_risk_gbp']:.0f} GBP)")

        # Store state
        self.state.bar_number = bar_num
        self.state.bar_high = bar_high
        self.state.bar_low = bar_low
        self.state.bar_range = bar_range
        self.state.range_flag = range_flag
        self.state.buy_level = buy_level
        self.state.sell_level = sell_level
        self.state.phase = Phase.LEVELS_SET
        self.save_state()

        # Send alert
        await self.alert(
            f"<b>{self.name} LEVELS SET</b>\n"
            f"Bar {bar_num}: H={bar_high} L={bar_low} ({range_flag})\n"
            f"BUY: {buy_level} | SELL: {sell_level}\n"
            f"Range: {bar_range}pts | Risk: {risk_pts:.0f}pts"
        )

        # Arm bracket (R4, R5)
        await self._arm_bracket()

    # =========================================================================
    #  Bracket arming and tick trigger (R4, R5, R8)
    # =========================================================================

    async def _arm_bracket(self):
        """Place OCA bracket: both directions armed, tick-triggered."""
        qty = config.NUM_CONTRACTS
        result = await self.broker.place_oca_bracket(
            buy_price=self.state.buy_level,
            sell_price=self.state.sell_level,
            qty=qty,
            oca_group=f"ASRS_{self.name}_{self.state.date}_{self.state.entries_used + 1}",
        )
        if "error" in result:
            await self.alert(f"[{self.name}] Bracket placement FAILED: {result['error']}")
            return

        self.state.phase = Phase.BRACKET_ARMED
        self.save_state()
        logger.info(f"[{self.name}] Bracket armed: BUY@{self.state.buy_level} SELL@{self.state.sell_level}")

    async def on_tick_trigger(self, trigger: dict):
        """
        Called by broker when tick crosses bracket level (sub-second).
        Handles fill processing, slippage check, stop placement.
        R5: BUY triggers on offer >= buy_level, SELL on bid <= sell_level.
        """
        self.load_state()
        if self.state.phase != Phase.BRACKET_ARMED:
            return

        direction = trigger["direction"]    # "LONG" or "SHORT"
        fill_price = trigger["fill_price"]
        deal_id = trigger.get("order_id", "")

        logger.info(f"[{self.name}] Tick trigger: {direction} @ {fill_price}")

        # R9: Proportional slippage check
        trigger_price = self.state.buy_level if direction == "LONG" else self.state.sell_level
        slippage = abs(fill_price - trigger_price)
        initial_risk = self.state.bar_range + self.cfg["buffer"] * 2
        max_slip = initial_risk * self.cfg["max_slippage_pct"]

        if slippage > max_slip:
            logger.error(f"[{self.name}] EXCESSIVE SLIPPAGE: {slippage:.1f} > {max_slip:.1f}")
            await self.alert(
                f"[{self.name}] SLIPPAGE CLOSE: fill={fill_price}, "
                f"trigger={trigger_price}, slip={slippage:.1f}pts > {max_slip:.1f}")
            await self.broker.close_position()
            self.broker.deactivate_bracket()
            self.state.entries_used += 1
            if self.state.entries_used < self.cfg["max_entries"]:
                # Keep levels but don't re-arm — wait for price to return
                self.state.phase = Phase.LEVELS_SET
                logger.info(f"[{self.name}] Slippage close: entry {self.state.entries_used}/{self.cfg['max_entries']}, "
                            f"bracket deactivated, levels kept")
            else:
                self.state.phase = Phase.DONE
                logger.info(f"[{self.name}] Slippage close: max entries reached, DONE")
            self.save_state()
            return

        # Process fill
        self._process_fill(direction, fill_price, deal_id)

        # R12: Set disaster stop on IG immediately
        if not await self._place_stop_with_retry():
            return  # emergency close happened

        await self.alert(
            f"<b>{self.name} {direction} ENTRY</b>\n"
            f"Price: {fill_price} | Stop: {self.state.trailing_stop}\n"
            f"Risk: {abs(fill_price - self.state.trailing_stop):.1f}pts\n"
            f"Entry {self.state.entries_used}/{self.cfg['max_entries']}"
        )

    def _process_fill(self, direction: str, fill_price: float, deal_id: str = ""):
        """Update state for a new entry fill."""
        self.state.direction = direction
        self.state.entry_price = fill_price
        self.state.entries_used += 1
        self.state.adds_used = 0
        self.state.add_positions = []
        self.state.last_add_price = 0.0
        self.state.breakeven_hit = False
        self.state.trail_moved = False
        self.state.max_favourable = fill_price

        if deal_id:
            self.state.deal_ids = [deal_id]

        if direction == "LONG":
            self.state.phase = Phase.LONG
            self.state.initial_stop = self.state.sell_level
            self.state.trailing_stop = self.state.sell_level
        else:
            self.state.phase = Phase.SHORT
            self.state.initial_stop = self.state.buy_level
            self.state.trailing_stop = self.state.buy_level

        # Slippage tracking
        intended = self.state.buy_level if direction == "LONG" else self.state.sell_level
        entry_slip = round(fill_price - intended, 1) if direction == "LONG" else round(intended - fill_price, 1)

        self.state.trades.append({
            "num": self.state.entries_used,
            "direction": direction,
            "entry": fill_price,
            "entry_intended": intended,
            "entry_slippage": entry_slip,
            "stop": self.state.initial_stop,
            "time": datetime.now(self.tz).strftime("%H:%M"),
            "signal_bar": self.state.bar_number,
            "session": f"S{self.session}",
        })
        self.save_state()

    # =========================================================================
    #  monitor_cycle -- every minute: trail, breakeven, adds, stop detect (R12-R18)
    # =========================================================================

    async def monitor_cycle(self):
        """
        Called every minute by scheduler.
        Handles: bracket trigger polling, trailing stop, breakeven, adds,
        stop/exit detection, re-entry.
        """
        try:
            self.load_state()

            # If bracket armed, poll for triggers (backup to tick trigger)
            if self.state.phase == Phase.BRACKET_ARMED:
                await self._poll_bracket_trigger()
                return

            # If levels set but bracket not armed (e.g. after slippage close),
            # re-arm once price is back between the levels
            if self.state.phase == Phase.LEVELS_SET and self.state.buy_level > 0:
                price = await self.broker.get_current_price()
                if price and self.state.sell_level < price < self.state.buy_level:
                    logger.info(f"[{self.name}] Price {price} back between levels — re-arming bracket")
                    await self._arm_bracket()
                return

            # If in active position, manage it
            if self.state.phase in (Phase.LONG, Phase.SHORT):
                await self._manage_position()
                return

        except Exception as e:
            logger.error(f"[{self.name}] monitor_cycle error: {e}", exc_info=True)

    async def _poll_bracket_trigger(self):
        """Backup polling for bracket trigger (tick trigger is primary)."""
        import asyncio
        if not await self.broker.ensure_connected():
            return

        for _ in range(12):  # 12 x 5s = 60s
            self.load_state()
            if self.state.phase != Phase.BRACKET_ARMED:
                break

            trigger = await self.broker.check_trigger_levels()
            if trigger:
                await self.on_tick_trigger(trigger)
                return

            await asyncio.sleep(5)

    async def _manage_position(self):
        """Active position management: trail, breakeven, adds, exit."""
        if not await self.broker.ensure_connected():
            return

        price = await self.broker.get_current_price()
        if price is None or price <= 0:
            return

        # Check stop hit (R18) — verify via IG position, not just price
        stopped = False
        if self.state.direction == "LONG" and price <= self.state.trailing_stop:
            stopped = True
        elif self.state.direction == "SHORT" and price >= self.state.trailing_stop:
            stopped = True

        if stopped:
            # Verify with IG — is the position actually closed?
            import asyncio
            await asyncio.sleep(2)  # give IG time to process the stop
            ig_pos = await self.broker.get_position()
            if ig_pos["direction"] == "FLAT":
                # Position closed — use current price as exit (closest to actual fill)
                exit_price = price
                logger.info(f"[{self.name}] Stop confirmed by IG (position FLAT), exit ~{exit_price}")
            else:
                # IG still shows position open — stop hasn't triggered on IG yet
                logger.warning(f"[{self.name}] Price crossed stop but IG position still open — skipping exit")
                return
            await self._process_exit(exit_price)
            return

        # Update MFE
        if self.state.direction == "LONG":
            if price > self.state.max_favourable:
                self.state.max_favourable = price
            unrealized = price - self.state.entry_price
        else:
            if price < self.state.max_favourable:
                self.state.max_favourable = price
            unrealized = self.state.entry_price - price

        # R13: Breakeven
        if not self.state.breakeven_hit and unrealized >= self.cfg["breakeven_pts"]:
            self.state.breakeven_hit = True
            if self.state.direction == "LONG" and self.state.trailing_stop < self.state.entry_price:
                self.state.trailing_stop = self.state.entry_price
            elif self.state.direction == "SHORT" and self.state.trailing_stop > self.state.entry_price:
                self.state.trailing_stop = self.state.entry_price
            self.save_state()
            await self._update_ig_stop()
            await self.alert(f"[{self.name}] BREAKEVEN -- stop -> {self.state.entry_price}")
            logger.info(f"[{self.name}] Breakeven hit, stop -> {self.state.entry_price}")

        # R14 + R15: Candle trail
        await self._update_candle_trail()

        # R16: Add to winners
        await self._check_add(price)

        self.save_state()

    async def _update_candle_trail(self):
        """
        R14: Ratchet stop to previous bar's low (LONG) or high (SHORT).
        R15: Switch to previous bar's CLOSE when profit >= tight_threshold.
        """
        df = self.broker.get_streaming_bars_df()
        if df is None or df.empty:
            return

        today = datetime.now(self.tz).date()
        today_bars = df[df.index.date == today]
        if len(today_bars) < 2:
            return

        # Use second-to-last bar (last completed)
        prev_bar = today_bars.iloc[-2]
        prev_high = prev_bar["High"]
        prev_low = prev_bar["Low"]
        prev_close = prev_bar["Close"]

        old_stop = self.state.trailing_stop

        if self.state.direction == "LONG":
            profit = prev_close - self.state.entry_price
            use_tight = profit >= self.cfg["tight_threshold"]  # R15
            new_stop = prev_close if use_tight else prev_low   # R14
            if new_stop > self.state.trailing_stop:
                self.state.trailing_stop = round(new_stop, 1)
        else:
            profit = self.state.entry_price - prev_close
            use_tight = profit >= self.cfg["tight_threshold"]
            new_stop = prev_close if use_tight else prev_high
            if new_stop < self.state.trailing_stop:
                self.state.trailing_stop = round(new_stop, 1)

        if self.state.trailing_stop != old_stop:
            self.state.trail_moved = True
            self.save_state()
            await self._update_ig_stop()
            move = abs(self.state.trailing_stop - old_stop)
            if move >= self.cfg["trail_min_move"]:
                label = "TIGHT" if (self.state.direction == "LONG" and prev_close - self.state.entry_price >= self.cfg["tight_threshold"]) or \
                                   (self.state.direction == "SHORT" and self.state.entry_price - prev_close >= self.cfg["tight_threshold"]) else "TRAIL"
                # Before breakeven: show risk reduction. After: show locked profit.
                risk_pts = abs(self.state.trailing_stop - self.state.entry_price)
                if self.state.breakeven_hit:
                    risk_line = f"Locked: {risk_pts:.1f}pts"
                else:
                    old_risk = abs(old_stop - self.state.entry_price)
                    risk_line = f"Risk: {old_risk:.1f} → {risk_pts:.1f}pts"
                await self.alert(
                    f"[{self.name}] {label}: stop {old_stop} -> {self.state.trailing_stop}\n"
                    f"Entry: {self.state.entry_price} | {risk_line}"
                )

    async def _check_add(self, current_price: float):
        """R16: Add to winners when profit >= add_trigger from last ref price."""
        if self.state.adds_used >= self.cfg["add_max"]:
            return
        if self.state.phase not in (Phase.LONG, Phase.SHORT):
            return

        ref_price = self.state.last_add_price if self.state.last_add_price > 0 else self.state.entry_price

        if self.state.direction == "LONG":
            profit_from_ref = current_price - ref_price
        else:
            profit_from_ref = ref_price - current_price

        if profit_from_ref < self.cfg["add_trigger"]:
            return

        # Place market order for add
        action = "BUY" if self.state.direction == "LONG" else "SELL"
        result = await self.broker.place_market_order(action=action, qty=config.NUM_CONTRACTS)

        if "error" in result:
            logger.error(f"[{self.name}] Add order failed: {result['error']}")
            return

        fill_price = result.get("avg_price", current_price)
        deal_id = result.get("order_id", "")

        self.state.adds_used += 1
        self.state.last_add_price = fill_price
        self.state.add_positions.append({
            "entry_price": fill_price,
            "time": datetime.now(self.tz).strftime("%H:%M"),
            "deal_id": deal_id,
        })
        if deal_id:
            self.state.deal_ids.append(deal_id)

        # R17: Set stop on new deal at current trailing_stop
        self.state.breakeven_hit = True  # Force breakeven since we are in profit
        self.save_state()
        await self._update_ig_stop()  # Updates ALL deal IDs (R-impl-4)

        await self.alert(
            f"[{self.name}] ADD #{self.state.adds_used} @ {fill_price}\n"
            f"Ref: {ref_price} | Profit: {profit_from_ref:.1f}pts\n"
            f"Stop on all deals: {self.state.trailing_stop}"
        )
        logger.info(f"[{self.name}] Add #{self.state.adds_used} @ {fill_price}, deal={deal_id}")

    # =========================================================================
    #  Exit processing (R18, R19, R20, R21, R22)
    # =========================================================================

    async def _process_exit(self, exit_price: float):
        """
        Stop hit or EOD close. Log trade, check re-entry.
        R18: Process exit, log to journal.
        R20: After ANY exit, re-arm BOTH directions at ORIGINAL bar levels.
        R21: MAX_ENTRIES check before re-entry.
        R22: Re-entry uses original bar_high/bar_low + buffer.
        """
        direction = self.state.direction
        entry = self.state.entry_price

        # P&L calculation
        if direction == "LONG":
            pnl_original = round(exit_price - entry, 1)
            mfe = round(self.state.max_favourable - entry, 1)
        else:
            pnl_original = round(entry - exit_price, 1)
            mfe = round(entry - self.state.max_favourable, 1)

        # Add P&L
        add_pnl = 0.0
        add_details = []
        for add in self.state.add_positions:
            if direction == "LONG":
                ap = round(exit_price - add["entry_price"], 1)
            else:
                ap = round(add["entry_price"] - exit_price, 1)
            add_pnl += ap
            add_details.append({"entry": add["entry_price"], "exit": exit_price, "pnl": ap})

        total_pnl = pnl_original + add_pnl

        # Exit reason
        if pnl_original == 0.0 and self.state.breakeven_hit:
            exit_reason = "BREAKEVEN_STOP"
        elif self.state.trail_moved and self.state.trailing_stop != self.state.initial_stop:
            exit_reason = "TRAIL_STOP"
        else:
            exit_reason = "INITIAL_STOP"

        # Update trade log
        if self.state.trades:
            t = self.state.trades[-1]
            t["exit"] = exit_price
            t["exit_time"] = datetime.now(self.tz).strftime("%H:%M")
            t["pnl_pts"] = round(total_pnl, 1)
            t["pnl_original"] = pnl_original
            t["pnl_adds"] = round(add_pnl, 1)
            t["add_details"] = add_details
            t["adds_used"] = self.state.adds_used
            t["mfe"] = mfe
            t["exit_reason"] = exit_reason
            t["contracts_stopped"] = 1 + len(self.state.add_positions)

        self.save_state()

        # Log to journal DB
        try:
            from asrs.journal import log_trade
            trade = self.state.trades[-1] if self.state.trades else {}
            log_trade(self.instrument, trade, self.state)
        except Exception as e:
            logger.error(f"[{self.name}] Journal log error: {e}")

        # Alert
        icon = "+" if total_pnl >= 0 else ""
        await self.alert(
            f"<b>{self.name} EXIT ({exit_reason})</b>\n"
            f"Direction: {direction} | Entry: {entry}\n"
            f"Exit: {exit_price} | P&L: {icon}{total_pnl:.1f}pts\n"
            f"MFE: {mfe:.1f}pts | Adds: {self.state.adds_used}"
        )
        logger.info(f"[{self.name}] Exit: {direction} {total_pnl:+.1f}pts ({exit_reason})")

        # R20-R22: Re-entry check
        self.state.direction = ""
        self.state.deal_ids = []
        self.state.add_positions = []
        self.state.adds_used = 0
        self.state.last_add_price = 0.0
        self.state.breakeven_hit = False
        self.state.trail_moved = False

        if self.state.entries_used < self.cfg["max_entries"]:
            # R20: Re-arm BOTH directions at ORIGINAL bar levels
            # R22: Uses original bar_high/bar_low + buffer
            self.state.buy_level = round(self.state.bar_high + self.cfg["buffer"], 1)
            self.state.sell_level = round(self.state.bar_low - self.cfg["buffer"], 1)
            self.state.phase = Phase.LEVELS_SET
            self.save_state()

            await self._arm_bracket()
            await self.alert(
                f"[{self.name}] RE-ENTRY ARMED (both sides)\n"
                f"BUY: {self.state.buy_level} | SELL: {self.state.sell_level}\n"
                f"Entry {self.state.entries_used}/{self.cfg['max_entries']}"
            )
        else:
            # R11: Max entries reached
            self.state.phase = Phase.DONE
            self.save_state()
            logger.info(f"[{self.name}] Max entries reached ({self.cfg['max_entries']})")

    # =========================================================================
    #  end_of_day -- force close (R19)
    # =========================================================================

    async def end_of_day(self):
        """Force close all positions at session end."""
        self.load_state()

        if self.state.phase in (Phase.LONG, Phase.SHORT):
            logger.info(f"[{self.name}] EOD force close")
            price = await self.broker.get_current_price()
            await self.broker.close_position()
            await self.broker.cancel_all_orders()
            if price:
                await self._process_exit(price)
            else:
                self.state.phase = Phase.DONE
                self.save_state()
            await self.alert(f"[{self.name}] EOD: position closed")

        elif self.state.phase == Phase.BRACKET_ARMED:
            self.broker.deactivate_bracket()
            await self.broker.cancel_all_orders()
            self.state.phase = Phase.DONE
            self.save_state()
            await self.alert(f"[{self.name}] EOD: bracket cancelled, no fill today")

        elif self.state.phase in (Phase.IDLE, Phase.LEVELS_SET):
            self.state.phase = Phase.DONE
            self.save_state()

        # Day summary
        total_pnl = sum(t.get("pnl_pts", 0) for t in self.state.trades)
        n_trades = len([t for t in self.state.trades if "exit" in t])
        if n_trades > 0:
            await self.alert(
                f"<b>{self.name} DAY SUMMARY</b>\n"
                f"Trades: {n_trades} | P&L: {total_pnl:+.1f}pts"
            )

        # Reset for tomorrow
        self._bar4_triggered = False
        logger.info(f"[{self.name}] EOD complete, P&L: {total_pnl:+.1f}pts")

    # =========================================================================
    #  Broker helpers
    # =========================================================================

    async def _place_stop_with_retry(self, max_attempts: int = 3) -> bool:
        """Place trailing stop after entry. Emergency close if all fail (R12)."""
        stop_action = "SELL" if self.state.direction == "LONG" else "BUY"
        import asyncio
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.broker.place_stop_order(
                    action=stop_action,
                    qty=config.NUM_CONTRACTS,
                    stop_price=self.state.trailing_stop,
                )
                if "order_id" in result:
                    if attempt > 1:
                        logger.info(f"[{self.name}] Stop placed on attempt {attempt}")
                    return True
            except Exception as e:
                logger.error(f"[{self.name}] Stop attempt {attempt}/{max_attempts}: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(2 * attempt)

        # Emergency close -- all retries failed
        logger.error(f"[{self.name}] EMERGENCY: stop placement failed {max_attempts}x -- closing")
        await self.alert(
            f"[{self.name}] EMERGENCY CLOSE: stop placement failed. "
            f"Closing position to prevent unprotected exposure."
        )
        try:
            await self.broker.close_position()
            price = await self.broker.get_current_price()
            if price:
                await self._process_exit(price)
        except Exception as e:
            logger.error(f"[{self.name}] Emergency close failed: {e}")
            await self.alert(f"[{self.name}] EMERGENCY CLOSE FAILED: {e}")
        return False

    async def _update_ig_stop(self):
        """Update stop on ALL deal IDs (R-impl-4: adds have separate deals)."""
        failed = []
        for deal_id in self.state.deal_ids:
            try:
                ok = await self.broker.modify_stop(deal_id, self.state.trailing_stop)
                if not ok:
                    failed.append(deal_id)
            except Exception as e:
                logger.error(f"[{self.name}] Stop update failed for {deal_id}: {e}")
                failed.append(deal_id)
        if failed:
            await self.alert(
                f"[{self.name}] STOP UPDATE FAILED on {len(failed)}/{len(self.state.deal_ids)} deals!\n"
                f"Target: {self.state.trailing_stop} | Failed: {failed}"
            )

    # =========================================================================
    #  Bar helpers
    # =========================================================================

    def _find_bar(self, df, bar_num: int) -> dict | None:
        """
        Find bar N from session open using hour/minute comparison.
        Bar 1 starts at session open, bar 2 at +5min, etc.
        Uses hour/minute, NOT timestamp arithmetic (R-impl-1).
        """
        if df is None or df.empty:
            return None

        open_h = self.cfg[f"s{self.session}_open_hour"]
        open_m = self.cfg[f"s{self.session}_open_minute"]
        today = datetime.now(self.tz).date()

        bars_found = []
        for idx, row in df.iterrows():
            # Convert to instrument timezone (bars may be timestamped in CET)
            local_idx = idx.astimezone(self.tz) if hasattr(idx, 'astimezone') and idx.tzinfo else idx
            if local_idx.date() != today:
                continue
            h = local_idx.hour
            m = local_idx.minute
            mins_from_open = (h * 60 + m) - (open_h * 60 + open_m)
            if mins_from_open < 0:
                continue
            bn = mins_from_open // 5 + 1
            bars_found.append(f"{h:02d}:{m:02d}=B{bn}")
            if bn == bar_num:
                return {
                    "High": round(row["High"], 1),
                    "Low": round(row["Low"], 1),
                    "Open": round(row["Open"], 1),
                    "Close": round(row["Close"], 1),
                }
        logger.warning(f"[{self.name}] _find_bar({bar_num}) NOT FOUND in {len(df)} rows. "
                       f"Bars: {bars_found} | open={open_h:02d}:{open_m:02d}")
        return None

    async def _get_bars_with_fallback(self):
        """Get today's bars: streaming first, REST fallback."""
        import asyncio
        import pandas as pd

        # Step 1: Streaming bars
        df = self.broker.get_streaming_bars_df()
        if df is not None and not df.empty:
            bar4 = self._find_bar(df, 4)
            if bar4 is not None:
                return df

        # Step 2: Wait up to 60s for streaming bar 4
        for i in range(12):
            await asyncio.sleep(5)
            df = self.broker.get_streaming_bars_df()
            if df is not None and not df.empty:
                if self._find_bar(df, 4) is not None:
                    return df

        # Step 3: REST fallback with retries
        logger.warning(f"[{self.name}] Stream failed -- REST fallback")
        for attempt in range(1, 7):  # 6 attempts x 15s = 90s window
            df = await self.broker.get_5min_bars("1 D")
            if df is not None and not df.empty:
                if self._find_bar(df, 4) is not None:
                    return df
            if attempt < 6:
                await asyncio.sleep(15)

        return pd.DataFrame()

    # =========================================================================
    #  Status (for Telegram /status)
    # =========================================================================

    def status_text(self) -> str:
        """Concise status string for Telegram."""
        s = self.state
        if s.phase in (Phase.IDLE, Phase.DONE):
            pnl = sum(t.get("pnl_pts", 0) for t in s.trades)
            n = len([t for t in s.trades if "exit" in t])
            return f"{self.name}: {s.phase} ({n} trades, {pnl:+.1f}pts)"

        if s.phase == Phase.BRACKET_ARMED:
            return (f"{self.name}: BRACKET BUY@{s.buy_level} SELL@{s.sell_level} "
                    f"(entry {s.entries_used}/{self.cfg['max_entries']})")

        if s.phase in (Phase.LONG, Phase.SHORT):
            price_str = f"{s.entry_price}"
            stop_str = f"{s.trailing_stop}"
            be = " [BE]" if s.breakeven_hit else ""
            adds = f" +{s.adds_used}" if s.adds_used > 0 else ""
            return (f"{self.name}: {s.direction} @{price_str} stop={stop_str}{be}{adds} "
                    f"(entry {s.entries_used}/{self.cfg['max_entries']})")

        return f"{self.name}: {s.phase}"
