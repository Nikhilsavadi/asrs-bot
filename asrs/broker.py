"""
broker.py -- IG Markets broker wrapper
=======================================
Thin adapter over shared IG session. One instance per epic.
Handles: bracket simulation (tick trigger), market orders,
stop management, position queries.

Identical interface to dax_bot/broker_ig.py but cleaned up.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from shared.ig_session import IGSharedSession
from shared.ig_stream import IGStreamManager
from asrs import audit_log

logger = logging.getLogger(__name__)


class IGBroker:
    """IG Markets broker -- one per epic. Shares session + stream manager."""

    # Per-epic entry lock shared across all signals on the same epic.
    # Without this, sibling signals (DAX_S1 + DAX_S2) can both fire on
    # the same tick and place duplicate market orders before either's
    # get_position() check reflects the other's fill.
    # Same pattern as IBBroker._contract_entry_locks.
    _epic_entry_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def _get_entry_lock(cls, epic: str) -> asyncio.Lock:
        if epic not in cls._epic_entry_locks:
            cls._epic_entry_locks[epic] = asyncio.Lock()
        return cls._epic_entry_locks[epic]

    def __init__(
        self,
        shared: IGSharedSession,
        stream: IGStreamManager,
        epic: str,
        currency: str,
        disaster_stop_pts: int = 200,
        max_spread_pts: float = 10.0,
    ):
        self._shared = shared
        self._stream = stream
        self.epic = epic
        self.currency = currency
        self._disaster_stop_pts = disaster_stop_pts
        self._max_spread_pts = max_spread_pts
        self.connected = False

        # Bracket simulation (IG rejects working orders near market)
        self._pending_bracket: dict | None = None
        self._tick_trigger_active = False
        self._on_trigger_callbacks: list = []

        # Local position mirror + sibling-broker list for deferred arming.
        # Updated on entry/exit so ticks can do a cheap sibling check without
        # hitting IG REST API on every crossing tick.
        self._local_in_position: bool = False
        self._sibling_brokers: list = []

        # Tick-based stop monitoring (exit via market order)
        self._stop_monitor: dict | None = None
        self._stop_exit_active = False
        self._on_stop_callbacks: list = []
        self._tick_rearm_callbacks: list = []

        # Stop breach confirmation timer (filters DFB spread noise).
        # When bid/ofr first breaches the stop level, start a timer.
        # Only execute stop exit if breach persists for STOP_CONFIRM_SECS.
        # Resets if price recovers above stop within the window.
        import time as _time
        self._stop_breach_since: float = 0  # timestamp of first breach (0 = no active breach)
        self.STOP_CONFIRM_SECS = 60  # 1 minute confirmation

        # Track open position deal IDs (for multi-deal stop updates)
        self._position_deal_ids: dict[str, dict] = {}

        # Captured event loop for sync→async scheduling from tick handlers.
        # Same pattern as IBBroker._loop.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Consecutive order-error counter (same pattern as IBBroker)
        import os
        self._consecutive_order_errors: int = 0
        self._max_consecutive_order_errors: int = int(
            os.getenv("MAX_CONSECUTIVE_ORDER_ERRORS", "3")
        )

        # Register tick callback for real-time entry detection
        self._stream.register_tick_callback(epic, self._on_tick)

    # -- Connection -----------------------------------------------------------

    async def connect(self) -> bool:
        """Connect using shared session. Subscribe to streaming."""
        # Capture loop for sync→async scheduling from tick handlers
        self._loop = asyncio.get_running_loop()
        try:
            ok = await self._shared.ensure_connected()
            if not ok:
                self.connected = False
                return False

            market = await self._shared.rest_call(
                self._shared.ig.fetch_market_by_epic, self.epic
            )
            self.connected = True

            name = market.get("instrument", {}).get("name", self.epic)
            logger.info(f"IG connected -- {name} ({self.epic})")

            await self._stream.subscribe_ticks(self.epic)
            await self._stream.subscribe_candles(self.epic)
            return True

        except Exception as e:
            logger.error(f"IG connection failed ({self.epic}): {e}")
            self.connected = False
            return False

    async def ensure_connected(self) -> bool:
        ok = await self._shared.ensure_connected()
        self.connected = ok
        return ok

    # -- Market Data ----------------------------------------------------------

    async def get_5min_bars(self, duration: str = "1 D") -> pd.DataFrame:
        """Fetch historical 5-min bars via REST."""
        if not await self.ensure_connected():
            return pd.DataFrame()
        try:
            parts = duration.strip().split()
            num_days = int(parts[0]) if parts else 1
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=num_days + 1)

            result = await self._shared.rest_call(
                self._shared.ig.fetch_historical_prices_by_epic,
                epic=self.epic, resolution="MINUTE_5",
                start_date=start.strftime("%Y-%m-%dT%H:%M:%S"),
                end_date=end.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            if result is None or "prices" not in result:
                return pd.DataFrame()

            df = result["prices"]
            if df.empty:
                return pd.DataFrame()

            out = pd.DataFrame(index=df.index)
            for col in ["Open", "High", "Low", "Close"]:
                if ("bid", col) in df.columns and ("ask", col) in df.columns:
                    out[col] = (df[("bid", col)] + df[("ask", col)]) / 2
                elif ("last", col) in df.columns:
                    out[col] = df[("last", col)]
                else:
                    out[col] = df[col] if col in df.columns else 0
            out.index = pd.to_datetime(out.index, utc=True)
            return out

        except Exception as e:
            logger.error(f"get_5min_bars failed ({self.epic}): {e}", exc_info=True)
            return pd.DataFrame()

    def get_streaming_bars_df(self) -> pd.DataFrame:
        """Today's 5-min bars from Lightstreamer (no REST call)."""
        return self._stream.get_today_bars_df(self.epic)

    def get_streaming_bar_count(self) -> int:
        return self._stream.get_bar_count_today(self.epic)

    async def get_current_price(self) -> float | None:
        """Streaming first, REST fallback. None if stale (>30s)."""
        age = self._stream.get_tick_age(self.epic)
        if age < 30:
            return self._stream._prices.get(self.epic)
        return await self._stream.get_price(self.epic)

    # -- Tick-based bracket trigger -------------------------------------------

    def register_trigger_callback(self, callback):
        """Register async callback for tick-triggered fills."""
        self._on_trigger_callbacks.append(callback)

    def deactivate_bracket(self, reason: str = "deactivated"):
        """Deactivate the pending bracket (e.g. S2 cancels S1)."""
        if self._pending_bracket:
            self._pending_bracket["active"] = False
            audit_log.cancel(signal=getattr(self, "_signal_name", "unknown"),
                             epic=self.epic, reason=reason)

    # -- Tick-based stop monitor (exits via market order, not IG stop) ---------

    def activate_stop_monitor(self, direction: str, stop_level: float):
        """Activate tick-level stop monitoring. Exit via market order on hit."""
        self._stop_breach_since = 0  # reset confirmation timer on new position
        self._stop_monitor = {
            "active": True,
            "direction": direction,
            "stop_level": stop_level,
        }
        logger.info(f"Stop monitor active ({self.epic}): {direction} stop={stop_level}")

    def update_stop_level(self, new_stop: float):
        """Update the monitored stop level (called on trail/breakeven)."""
        if self._stop_monitor and self._stop_monitor["active"]:
            old = self._stop_monitor["stop_level"]
            self._stop_monitor["stop_level"] = new_stop
            self._stop_breach_since = 0  # reset timer on new stop level
            if abs(new_stop - old) > 0.1:
                logger.info(f"Stop monitor updated ({self.epic}): {old} -> {new_stop}")

    def deactivate_stop_monitor(self):
        """Deactivate stop monitoring (after exit or EOD)."""
        if self._stop_monitor:
            self._stop_monitor["active"] = False
            self._stop_monitor = None

    def register_stop_callback(self, callback):
        """Register async callback for tick-triggered stop exits."""
        self._on_stop_callbacks.append(callback)

    def register_tick_rearm_callback(self, callback):
        """Register sync callback for tick-level re-entry gate check."""
        self._tick_rearm_callbacks.append(callback)

    async def _execute_stop_exit(self, direction: str, exit_price: float):
        """Close all positions via market order when stop is hit."""
        import time as _time
        t_detect = _time.time()
        try:
            self._stop_monitor["active"] = False  # prevent re-trigger

            # Close all individual deals — retry up to 3 times
            import asyncio
            t_send = _time.time()
            for attempt in range(1, 4):
                closed = await self.close_position()
                if closed:
                    break
                logger.error(f"Stop exit close attempt {attempt}/3 failed ({self.epic})")
                if attempt < 3:
                    await asyncio.sleep(2)
            t_filled = _time.time()

            # Verify position is flat and get actual fill
            pos = await self.get_position()
            fills = getattr(self, '_last_close_fills', [])
            actual_exit = sum(fills) / len(fills) if fills else exit_price
            if pos["direction"] != "FLAT":
                logger.error(f"STOP EXIT FAILED — position still open ({self.epic})")
                if self._stop_monitor:
                    self._stop_monitor["active"] = True
                self._stop_exit_active = False
                return
            self._local_in_position = False  # siblings can now arm/fire
            for sb in self._sibling_brokers:
                sb._local_in_position = False

            # Exit slippage: positive = WORSE than intended (LONG fill below stop, SHORT fill above stop)
            if direction == "LONG":
                exit_slip = exit_price - actual_exit  # paid less than expected = positive slip
            else:
                exit_slip = actual_exit - exit_price  # paid more than expected = positive slip

            send_lat_ms = (t_send - t_detect) * 1000
            fill_lat_ms = (t_filled - t_send) * 1000
            total_lat_ms = (t_filled - t_detect) * 1000

            logger.info(
                f"EXIT ({self.epic}): {direction} stop_intended={exit_price:.1f} actual={actual_exit:.1f} "
                f"slip={exit_slip:+.1f}pt | latency send={send_lat_ms:.0f}ms fill={fill_lat_ms:.0f}ms total={total_lat_ms:.0f}ms"
            )

            # Sanity check: alert on excessive exit slippage
            if abs(exit_slip) > 10:
                logger.error(
                    f"SANITY: excessive exit slippage on {self.epic} {direction}: "
                    f"intended={exit_price:.1f} actual={actual_exit:.1f} slip={exit_slip:+.1f}pt"
                )

            # Fire callbacks to strategy
            for cb in self._on_stop_callbacks:
                try:
                    await cb({
                        "exit_price": actual_exit,
                        "exit_intended": exit_price,  # tick price at detection
                        "exit_slippage_pts": round(exit_slip, 1),
                        "exit_latency_ms": round(total_lat_ms),
                        "direction": direction,
                    })
                except Exception as e:
                    logger.error(f"Stop exit callback error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Stop exit execution failed ({self.epic}): {e}", exc_info=True)
        finally:
            self._stop_exit_active = False

    def _on_tick(self, mid: float, bid: float = 0.0, ofr: float = 0.0):
        """
        Called on every tick from Lightstreamer.
        Handles: bracket entry triggers AND stop exit monitoring.
        Also logs ticks to DB during active trades for post-analysis.
        """
        # Tick logging during active trades (batched — every 10th tick)
        if self._stop_monitor and self._stop_monitor.get("active"):
            self._tick_log_counter = getattr(self, "_tick_log_counter", 0) + 1
            if self._tick_log_counter % 10 == 0:
                try:
                    from shared.journal_db import log_tick
                    from datetime import datetime
                    trade_id = getattr(self, "_active_trade_id", 0)
                    log_tick(trade_id, self.epic,
                             datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                             bid, ofr, mid)
                except Exception:
                    pass

        # Stop monitor: check if price hit trailing stop
        # Uses confirmation timer: breach must persist for STOP_CONFIRM_SECS
        # before exit fires. Filters DFB spread noise (90%+ of breaches are <30s).
        if self._stop_monitor and self._stop_monitor.get("active") and not self._stop_exit_active:
            import time as _time
            sm = self._stop_monitor
            stop = sm["stop_level"]
            breached = False
            if sm["direction"] == "LONG" and bid > 0 and bid <= stop:
                breached = True
            elif sm["direction"] == "SHORT" and ofr > 0 and ofr >= stop:
                breached = True

            if breached:
                if self._stop_breach_since == 0:
                    # First breach — start timer
                    self._stop_breach_since = _time.time()
                    logger.debug(f"Stop breach started ({self.epic}): {sm['direction']} stop={stop}")
                elif (_time.time() - self._stop_breach_since) >= self.STOP_CONFIRM_SECS:
                    # Breach persisted for confirmation period — execute exit
                    logger.info(
                        f"Stop CONFIRMED ({self.epic}): {sm['direction']} breached for "
                        f"{_time.time() - self._stop_breach_since:.0f}s (threshold {self.STOP_CONFIRM_SECS}s)"
                    )
                    self._stop_breach_since = 0
                    hit = True
                else:
                    hit = False  # still waiting for confirmation
            else:
                # Price recovered — reset timer
                if self._stop_breach_since > 0:
                    elapsed = _time.time() - self._stop_breach_since
                    logger.debug(f"Stop breach cleared ({self.epic}): recovered after {elapsed:.0f}s")
                    self._stop_breach_since = 0
                hit = False

            if hit:
                self._stop_exit_active = True
                exit_price = bid if sm["direction"] == "LONG" else ofr
                logger.info(f"Stop hit ({self.epic}): {sm['direction']} stop={stop} "
                            f"exit_price={exit_price:.1f} (bid={bid:.1f} ofr={ofr:.1f})")
                loop = self._loop
                if loop is None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        logger.error(f"{self.epic}: no loop for stop exit")
                        self._stop_exit_active = False
                        return
                asyncio.run_coroutine_threadsafe(
                    self._execute_stop_exit(sm["direction"], exit_price), loop
                )

        # Tick-level re-entry gate check (fire on every tick, sync)
        for cb in self._tick_rearm_callbacks:
            try:
                cb(mid, bid, ofr)
            except Exception:
                pass

        # Bracket trigger: entry signals
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return
        if self._tick_trigger_active:
            return

        # Deferred arming: if SELF or any sibling on the same epic currently
        # holds a position (base or fade), skip silently (don't consume the
        # bracket). When the holder closes, _local_in_position flips and the
        # very next crossing tick fires our entry normally. Local flag — no
        # IG REST per tick.
        if self._local_in_position or any(
            getattr(sb, "_local_in_position", False) for sb in self._sibling_brokers
        ):
            return

        if bid > 0 and ofr > 0:
            spread = ofr - bid
            if spread > self._max_spread_pts:
                return
        else:
            logger.warning(f"Spread check skipped ({self.epic}): bid={bid} ofr={ofr} — incomplete tick data")
            return  # refuse entry on incomplete data

        bracket = self._pending_bracket
        triggered_dir = None

        # R5: Use offer for BUY, bid for SELL
        if ofr > 0 and ofr >= bracket["buy_price"]:
            triggered_dir = "BUY"
        elif bid > 0 and bid <= bracket["sell_price"]:
            triggered_dir = "SELL"
        elif bid == 0 or ofr == 0:
            if mid >= bracket["buy_price"]:
                triggered_dir = "BUY"
            elif mid <= bracket["sell_price"]:
                triggered_dir = "SELL"

        if not triggered_dir:
            return

        self._tick_trigger_active = True
        bracket["active"] = False
        trigger_price = ofr if triggered_dir == "BUY" else bid
        # Capture microstructure at trigger (TCA)
        bracket["_trigger_bid"] = bid
        bracket["_trigger_ofr"] = ofr
        bracket["_trigger_spread"] = round(spread, 2)
        logger.info(f"Tick trigger ({self.epic}): {triggered_dir} @ {trigger_price:.1f} "
                     f"(bid={bid:.1f} ofr={ofr:.1f} spread={spread:.1f})")

        # Use captured loop (not get_event_loop which is deprecated 3.10+/broken 3.12)
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error(f"{self.epic}: no loop for tick trigger")
                self._tick_trigger_active = False
                return
        asyncio.run_coroutine_threadsafe(
            self._execute_tick_trigger(triggered_dir, trigger_price, bracket), loop
        )

    async def _execute_tick_trigger(self, direction: str, price: float, bracket: dict):
        """Execute market order triggered by tick."""
        # Per-epic lock prevents sibling bracket double-fill race
        lock = self._get_entry_lock(self.epic)
        # Latency tracking
        import time as _time
        t_trigger = _time.time()
        try:
            await lock.acquire()
            # Pre-entry safety: no existing position (final safety net — tick path
            # already skips when any sibling has _local_in_position, but this
            # catches races where IG reports a position we didn't track locally).
            try:
                pos = await self.get_position()
                if pos["direction"] != "FLAT":
                    logger.error(f"BLOCKED: existing {pos['direction']} on {self.epic}")
                    audit_log.trigger_blocked(
                        signal=getattr(self, "_signal_name", "unknown"),
                        epic=self.epic,
                        reason=f"existing {pos['direction']} (unexpected)",
                        direction=direction, price=price,
                    )
                    # Re-arm bracket so next tick can retry. Since the local
                    # sibling check runs first, we only hit this for unexpected
                    # (non-sibling) positions — rare, so re-arming is cheap.
                    if self._pending_bracket:
                        self._pending_bracket["active"] = True
                    self._tick_trigger_active = False
                    lock.release()
                    return
            except Exception:
                pass

            t_send = _time.time()
            result = await self.place_market_order(action=direction, qty=bracket["qty"])
            t_filled = _time.time()
            if "error" in result:
                logger.error(f"Tick-triggered order failed: {result['error']}")
                self._tick_trigger_active = False
                return

            # Latency + slippage logging
            fill_price = result.get("avg_price", price)
            slippage = abs(fill_price - price)
            send_lat_ms = (t_send - t_trigger) * 1000
            fill_lat_ms = (t_filled - t_send) * 1000
            total_lat_ms = (t_filled - t_trigger) * 1000
            logger.info(
                f"EXEC ({self.epic}): {direction} trigger={price:.1f} fill={fill_price:.1f} "
                f"slip={slippage:.1f}pt | latency send={send_lat_ms:.0f}ms fill={fill_lat_ms:.0f}ms total={total_lat_ms:.0f}ms"
            )
            audit_log.trigger_fired(
                signal=getattr(self, "_signal_name", self.epic),
                epic=self.epic, direction=direction,
                price=fill_price, slippage=round(slippage, 2),
            )
            self._local_in_position = True

            # Pre-fill sanity check: alert if slippage > 10pt
            if slippage > 10:
                logger.error(
                    f"SANITY: excessive slippage on {self.epic} {direction}: "
                    f"intended={price:.1f} filled={fill_price:.1f} slip={slippage:.1f}pt"
                )

            trigger_result = {
                "direction": "LONG" if direction == "BUY" else "SHORT",
                "fill_price": fill_price,
                "order_id": result.get("order_id", ""),
                "exec_latency_ms": total_lat_ms,
                "slippage_pts": round(slippage, 1),
            }

            for cb in self._on_trigger_callbacks:
                try:
                    await cb(trigger_result)
                except Exception as e:
                    logger.error(f"Trigger callback error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Tick trigger execution failed: {e}", exc_info=True)
        finally:
            self._tick_trigger_active = False
            try:
                if lock.locked():
                    lock.release()
            except Exception:
                pass

    # -- Order Management -----------------------------------------------------

    async def place_oca_bracket(
        self, buy_price: float, sell_price: float, qty: int, oca_group: str,
    ) -> dict:
        """
        Simulate OCA bracket. IG rejects working orders near market,
        so we store levels locally and trigger via _on_tick / check_trigger_levels.
        """
        self._pending_bracket = {
            "buy_price": buy_price, "sell_price": sell_price,
            "qty": qty, "oca_group": oca_group, "active": True,
        }
        logger.info(f"OCA bracket ({self.epic}): BUY@{buy_price} / SELL@{sell_price}")
        audit_log.arm_success(signal=getattr(self, "_signal_name", oca_group),
                              epic=self.epic, buy=buy_price, sell=sell_price, qty=qty)
        return {"buy_order_id": f"pending_buy_{oca_group}",
                "sell_order_id": f"pending_sell_{oca_group}",
                "oca_group": oca_group}

    async def check_trigger_levels(self) -> dict | None:
        """Backup polling: check if price crossed bracket levels."""
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return None
        if self._tick_trigger_active:
            return None

        # Deferred arming: skip if SELF or any sibling currently holds a
        # position (mirrors _on_tick guard — otherwise check_trigger_levels
        # bypasses sibling-block and creates orphan positions).
        if self._local_in_position or any(
            getattr(sb, "_local_in_position", False) for sb in self._sibling_brokers
        ):
            return None

        bid = self._stream._prices.get(f"{self.epic}_bid")
        ofr = self._stream._prices.get(f"{self.epic}_ofr")
        if bid is None or ofr is None:
            price = await self.get_current_price()
            if price is None:
                return None
            bid = ofr = price

        spread = ofr - bid
        if spread > self._max_spread_pts:
            return None

        bracket = self._pending_bracket
        triggered_dir = None
        if ofr >= bracket["buy_price"]:
            triggered_dir = "BUY"
        elif bid <= bracket["sell_price"]:
            triggered_dir = "SELL"

        if not triggered_dir:
            return None

        # Pre-entry safety
        try:
            pos = await self.get_position()
            if pos["direction"] != "FLAT":
                logger.error(f"BLOCKED: existing {pos['direction']} on {self.epic}")
                return None
        except Exception:
            pass

        trigger_price = ofr if triggered_dir == "BUY" else bid
        logger.info(f"Bracket triggered ({self.epic}): {triggered_dir} @ {trigger_price:.1f}")
        result = await self.place_market_order(action=triggered_dir, qty=bracket["qty"])

        if "error" in result:
            return None

        self._pending_bracket["active"] = False
        self._local_in_position = True
        audit_log.trigger_fired(
            signal=getattr(self, "_signal_name", self.epic),
            epic=self.epic, direction=triggered_dir,
            price=result.get("avg_price", trigger_price), slippage=0,
        )
        return {
            "direction": "LONG" if triggered_dir == "BUY" else "SHORT",
            "fill_price": result.get("avg_price", trigger_price),
            "order_id": result.get("order_id", ""),
        }

    async def place_market_order(self, action: str, qty: int) -> dict:
        """Place a market order with disaster stop."""
        if not await self.ensure_connected():
            return {"error": "Not connected"}
        try:
            direction = "BUY" if action == "BUY" else "SELL"
            result = await self._shared.rest_call(
                self._shared.ig.create_open_position,
                currency_code=self.currency, direction=direction,
                epic=self.epic, expiry="DFB", force_open=True,
                guaranteed_stop=True, level=None, limit_distance=None,
                limit_level=None, order_type="MARKET", quote_id=None,
                size=qty, stop_distance=self._disaster_stop_pts, stop_level=None,
                trailing_stop=False, trailing_stop_increment=None,
            )

            deal_ref = result.get("dealReference", "")
            confirm = await self._confirm_deal(deal_ref)

            if confirm.get("dealStatus") == "REJECTED":
                reason = confirm.get("reason", "Unknown")
                self._consecutive_order_errors += 1
                logger.error(
                    f"Order REJECTED: {reason} "
                    f"(consecutive errors: {self._consecutive_order_errors}/{self._max_consecutive_order_errors})"
                )
                try:
                    from asrs.alerts import send as _tg_send
                    await _tg_send(
                        f"⚠️ <b>Order REJECTED</b> ({self.epic})\n"
                        f"{direction} {qty} | reason: <code>{reason}</code>\n"
                        f"consecutive: {self._consecutive_order_errors}/{self._max_consecutive_order_errors}"
                    )
                except Exception as _e:
                    logger.error(f"tg_send on rejection failed: {_e}")
                if self._consecutive_order_errors >= self._max_consecutive_order_errors:
                    logger.critical(f"AUTO-PAUSE: {self._consecutive_order_errors} consecutive order rejections")
                    try:
                        from telegram_cmd import _set_paused
                        _set_paused(True)
                    except Exception:
                        pass
                return {"error": f"Rejected: {reason}"}

            deal_id = confirm.get("dealId", deal_ref)
            fill_level = confirm.get("level", 0)

            self._position_deal_ids[deal_id] = {
                "direction": direction, "size": qty, "level": fill_level,
            }

            logger.info(f"Market order ({self.epic}): {direction} {qty}, fill={fill_level} ({deal_id})")
            self._consecutive_order_errors = 0
            return {"order_id": deal_id, "avg_price": fill_level}

        except Exception as e:
            self._consecutive_order_errors += 1
            logger.error(
                f"Market order failed ({self.epic}): {e} "
                f"(consecutive errors: {self._consecutive_order_errors}/{self._max_consecutive_order_errors})"
            )
            if self._consecutive_order_errors >= self._max_consecutive_order_errors:
                logger.critical(f"AUTO-PAUSE: {self._consecutive_order_errors} consecutive order errors")
                try:
                    from telegram_cmd import _set_paused
                    _set_paused(True)
                except Exception:
                    pass
            return {"error": str(e)}

    async def place_stop_order(
        self, action: str, qty: int, stop_price: float,
    ) -> dict:
        """Set or update stop on all open position deals."""
        if not await self.ensure_connected():
            return {"error": "Not connected"}
        try:
            if self._position_deal_ids:
                updated = []
                for deal_id in list(self._position_deal_ids.keys()):
                    try:
                        await self._shared.rest_call(
                            self._shared.ig.update_open_position,
                            limit_level=None, stop_level=stop_price, deal_id=deal_id,
                        )
                        updated.append(deal_id)
                    except Exception as e:
                        logger.error(f"Stop set failed on {deal_id}: {e}")
                if updated:
                    return {"order_id": f"stop_{updated[0]}"}
                return {"error": "Failed to set stop on any deal"}
            return {"error": "No position deal IDs tracked"}
        except Exception as e:
            logger.error(f"place_stop_order failed: {e}")
            return {"error": str(e)}

    async def modify_stop(self, deal_id: str, new_stop: float) -> bool:
        """Modify stop on a specific deal ID (not all deals)."""
        if not await self.ensure_connected():
            return False
        try:
            clean_id = str(deal_id).replace("stop_", "")
            await self._shared.rest_call(
                self._shared.ig.update_open_position,
                limit_level=None, stop_level=new_stop, deal_id=clean_id,
            )
            logger.info(f"Stop modified on {clean_id} -> {new_stop}")
            return True
        except Exception as e:
            logger.error(f"modify_stop failed on {deal_id}: {e}")
            return False

    async def modify_stop_all(self, new_stop: float) -> bool:
        """Modify stop on ALL tracked position deals (for place_stop_order)."""
        if not await self.ensure_connected():
            return False
        updated = False
        for pos_id in list(self._position_deal_ids.keys()):
            try:
                await self._shared.rest_call(
                    self._shared.ig.update_open_position,
                    limit_level=None, stop_level=new_stop, deal_id=pos_id,
                )
                updated = True
            except Exception as e:
                logger.error(f"modify_stop_all failed on {pos_id}: {e}")
        return updated

    async def cancel_all_orders(self) -> int:
        """Cancel all working orders."""
        if not await self.ensure_connected():
            return 0
        try:
            orders = await self._shared.rest_call(self._shared.ig.fetch_working_orders)
            if orders is None or (hasattr(orders, "empty") and orders.empty):
                return 0

            order_list = (orders if isinstance(orders, list)
                          else orders.to_dict("records") if hasattr(orders, "to_dict")
                          else [])
            count = 0
            for order in order_list:
                deal_id = order.get("dealId", "")
                if deal_id:
                    try:
                        await self._shared.rest_call(
                            self._shared.ig.delete_working_order, deal_id=deal_id,
                        )
                        count += 1
                    except Exception:
                        pass

            if self._pending_bracket:
                self._pending_bracket["active"] = False
            return count
        except Exception as e:
            logger.error(f"cancel_all_orders failed: {e}")
            return 0

    async def close_position_by_deal_id(self, deal_id: str) -> bool:
        """Close a specific deal (used by fade so we don't close sibling base positions)."""
        if not await self.ensure_connected():
            return False
        if not deal_id:
            logger.error(f"close_position_by_deal_id: empty deal_id")
            return False
        try:
            positions = await self._shared.rest_call(self._shared.ig.fetch_open_positions)
            pos_list = (positions if isinstance(positions, list)
                        else positions.to_dict("records") if hasattr(positions, "to_dict")
                        else [])
            for pos in pos_list:
                if pos.get("dealId") == deal_id and pos.get("epic") == self.epic:
                    direction = pos.get("direction", "")
                    size = pos.get("dealSize") or pos.get("size", 0)
                    close_dir = "SELL" if direction == "BUY" else "BUY"
                    await self._shared.rest_call(
                        self._shared.ig.close_open_position,
                        deal_id=deal_id, direction=close_dir,
                        epic=None, expiry="DFB", level=None,
                        order_type="MARKET", quote_id=None, size=size,
                    )
                    self._position_deal_ids.pop(deal_id, None)
                    if not self._position_deal_ids:
                        self._local_in_position = False
                        # Propagate to siblings (epic is FLAT overall)
                        for sb in self._sibling_brokers:
                            sb._local_in_position = False
                    logger.info(f"Closed specific deal {deal_id} ({direction} {size})")
                    return True
            logger.warning(f"close_position_by_deal_id: {deal_id} not found on broker")
            return False
        except Exception as e:
            logger.error(f"close_position_by_deal_id failed: {e}", exc_info=True)
            return False

    async def close_position(self) -> bool:
        """Close all open positions for this epic."""
        if not await self.ensure_connected():
            return False
        try:
            positions = await self._shared.rest_call(self._shared.ig.fetch_open_positions)
            if positions is None:
                return True

            pos_list = (positions if isinstance(positions, list)
                        else positions.to_dict("records") if hasattr(positions, "to_dict")
                        else [])

            closed = []
            fill_levels = []
            for pos in pos_list:
                if pos.get("epic") == self.epic:
                    deal_id = pos.get("dealId", "")
                    direction = pos.get("direction", "")
                    size = pos.get("dealSize") or pos.get("size", 0)
                    close_dir = "SELL" if direction == "BUY" else "BUY"
                    try:
                        result = await self._shared.rest_call(
                            self._shared.ig.close_open_position,
                            deal_id=deal_id, direction=close_dir,
                            epic=None, expiry="DFB", level=None,
                            order_type="MARKET", quote_id=None, size=size,
                        )
                        closed.append(deal_id)
                        # Get actual fill level from confirmation
                        deal_ref = result.get("dealReference", "") if result else ""
                        if deal_ref:
                            confirm = await self._confirm_deal(deal_ref)
                            level = confirm.get("level")
                            if level:
                                fill_levels.append(float(level))
                    except Exception as e:
                        logger.error(f"Close failed for {deal_id}: {e}")

            for did in closed:
                self._position_deal_ids.pop(did, None)
            self._last_close_fills = fill_levels
            is_flat = len(self._position_deal_ids) == 0
            if is_flat:
                self._local_in_position = False
                # Propagate: siblings on same epic also FLAT. Without this,
                # stale sibling flags block future entries forever.
                for sb in self._sibling_brokers:
                    sb._local_in_position = False
            return is_flat

        except Exception as e:
            logger.error(f"close_position failed: {e}")
            return False

    async def get_position(self) -> dict:
        """Get current position for this epic."""
        if not await self.ensure_connected():
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}
        try:
            positions = await self._shared.rest_call(self._shared.ig.fetch_open_positions)
            if positions is None:
                return {"position": 0, "avg_cost": 0, "direction": "FLAT"}

            pos_list = (positions if isinstance(positions, list)
                        else positions.to_dict("records") if hasattr(positions, "to_dict")
                        else [])

            total_size = 0.0
            weighted_level = 0.0
            direction = ""
            stop_levels = {}
            for pos in pos_list:
                if pos.get("epic") == self.epic:
                    d = pos.get("direction", "")
                    size = float(pos.get("dealSize") or pos.get("size", 0))
                    level = float(pos.get("level") or pos.get("openLevel", 0))
                    stop = pos.get("stopLevel") or pos.get("stop_level")
                    deal_id = pos.get("dealId", "")
                    if deal_id:
                        self._position_deal_ids[deal_id] = {
                            "direction": d, "size": size, "level": level,
                        }
                        if stop is not None:
                            stop_levels[deal_id] = float(stop)
                    if not direction:
                        direction = d
                    total_size += size
                    weighted_level += level * size

            if total_size > 0:
                avg_cost = round(weighted_level / total_size, 2)
                return {
                    "position": total_size if direction == "BUY" else -total_size,
                    "avg_cost": avg_cost,
                    "direction": "LONG" if direction == "BUY" else "SHORT",
                    "stop_levels": stop_levels,
                }

            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}

        except Exception as e:
            logger.error(f"get_position failed: {e}")
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}

    # -- Internal helpers -----------------------------------------------------

    async def _confirm_deal(self, deal_reference: str, timeout_s: float = 10.0) -> dict:
        """Confirm deal with timeout. Falls back to position query if confirm hangs."""
        if not deal_reference:
            return {}
        try:
            await asyncio.sleep(0.5)
            confirm = await asyncio.wait_for(
                self._shared.rest_call(
                    self._shared.ig.fetch_deal_by_deal_reference, deal_reference,
                ),
                timeout=timeout_s,
            )
            return confirm if confirm else {}
        except asyncio.TimeoutError:
            # IG REST stalled — verify state via positions endpoint
            logger.error(
                f"Deal confirmation TIMEOUT after {timeout_s}s for {deal_reference} — "
                f"falling back to position query"
            )
            try:
                pos = await self.get_position()
                if pos.get("direction") != "FLAT":
                    logger.warning(
                        f"Position EXISTS after timeout: {pos['direction']} @ {pos.get('avg_cost', 0)}"
                    )
                    return {
                        "dealId": deal_reference,
                        "dealStatus": "ACCEPTED",
                        "level": pos.get("avg_cost", 0),
                    }
                else:
                    logger.warning(f"No position after timeout — order likely never filled")
                    return {"dealStatus": "REJECTED", "reason": "TIMEOUT_NO_POSITION"}
            except Exception as e2:
                logger.error(f"Position fallback also failed: {e2}")
                return {"dealId": deal_reference, "dealStatus": "UNKNOWN"}
        except Exception as e:
            logger.warning(f"Deal confirmation failed: {e}")
            return {"dealId": deal_reference}
