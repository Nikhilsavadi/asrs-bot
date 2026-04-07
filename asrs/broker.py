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

logger = logging.getLogger(__name__)


class IGBroker:
    """IG Markets broker -- one per epic. Shares session + stream manager."""

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

        # Tick-based stop monitoring (exit via market order)
        self._stop_monitor: dict | None = None
        self._stop_exit_active = False
        self._on_stop_callbacks: list = []

        # Track open position deal IDs (for multi-deal stop updates)
        self._position_deal_ids: dict[str, dict] = {}

        # Register tick callback for real-time entry detection
        self._stream.register_tick_callback(epic, self._on_tick)

    # -- Connection -----------------------------------------------------------

    async def connect(self) -> bool:
        """Connect using shared session. Subscribe to streaming."""
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

    def deactivate_bracket(self):
        """Deactivate the pending bracket (e.g. S2 cancels S1)."""
        if self._pending_bracket:
            self._pending_bracket["active"] = False

    # -- Tick-based stop monitor (exits via market order, not IG stop) ---------

    def activate_stop_monitor(self, direction: str, stop_level: float):
        """Activate tick-level stop monitoring. Exit via market order on hit."""
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

    async def _execute_stop_exit(self, direction: str, exit_price: float):
        """Close all positions via market order when stop is hit."""
        try:
            self._stop_monitor["active"] = False  # prevent re-trigger

            # Close all individual deals — retry up to 3 times
            import asyncio
            for attempt in range(1, 4):
                closed = await self.close_position()
                if closed:
                    break
                logger.error(f"Stop exit close attempt {attempt}/3 failed ({self.epic})")
                if attempt < 3:
                    await asyncio.sleep(2)

            # Verify position is flat and get actual fill
            pos = await self.get_position()
            fills = getattr(self, '_last_close_fills', [])
            actual_exit = sum(fills) / len(fills) if fills else exit_price
            if fills:
                logger.info(f"Actual close fills ({self.epic}): {fills} avg={actual_exit:.1f}")
            if pos["direction"] != "FLAT":
                logger.error(f"STOP EXIT FAILED — position still open ({self.epic})")
                # Re-enable monitor to try again on next tick
                if self._stop_monitor:
                    self._stop_monitor["active"] = True
                self._stop_exit_active = False
                return

            logger.info(f"Stop exit filled ({self.epic}): closed all deals @ ~{actual_exit}")

            # Fire callbacks to strategy
            for cb in self._on_stop_callbacks:
                try:
                    await cb({
                        "exit_price": actual_exit,
                        "exit_intended": exit_price,  # tick price at detection
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
        """
        # Stop monitor: check if price hit trailing stop
        if self._stop_monitor and self._stop_monitor.get("active") and not self._stop_exit_active:
            sm = self._stop_monitor
            stop = sm["stop_level"]
            hit = False
            if sm["direction"] == "LONG" and bid > 0 and bid <= stop:
                hit = True
            elif sm["direction"] == "SHORT" and ofr > 0 and ofr >= stop:
                hit = True
            if hit:
                self._stop_exit_active = True
                exit_price = bid if sm["direction"] == "LONG" else ofr
                logger.info(f"Stop hit ({self.epic}): {sm['direction']} stop={stop} "
                            f"exit_price={exit_price:.1f} (bid={bid:.1f} ofr={ofr:.1f})")
                loop = asyncio.get_event_loop()
                loop.create_task(self._execute_stop_exit(sm["direction"], exit_price))

        # Bracket trigger: entry signals
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return
        if self._tick_trigger_active:
            return

        spread = ofr - bid if (bid > 0 and ofr > 0) else 0
        if spread > self._max_spread_pts:
            return

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
        logger.info(f"Tick trigger ({self.epic}): {triggered_dir} @ {trigger_price:.1f} "
                     f"(bid={bid:.1f} ofr={ofr:.1f} spread={spread:.1f})")

        loop = asyncio.get_event_loop()
        loop.create_task(self._execute_tick_trigger(triggered_dir, trigger_price, bracket))

    async def _execute_tick_trigger(self, direction: str, price: float, bracket: dict):
        """Execute market order triggered by tick."""
        try:
            # Pre-entry safety: no existing position
            try:
                pos = await self.get_position()
                if pos["direction"] != "FLAT":
                    logger.error(f"BLOCKED: existing {pos['direction']} on {self.epic}")
                    self._tick_trigger_active = False
                    return
            except Exception:
                pass

            result = await self.place_market_order(action=direction, qty=bracket["qty"])
            if "error" in result:
                logger.error(f"Tick-triggered order failed: {result['error']}")
                self._tick_trigger_active = False
                return

            trigger_result = {
                "direction": "LONG" if direction == "BUY" else "SHORT",
                "fill_price": result.get("avg_price", price),
                "order_id": result.get("order_id", ""),
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
        return {"buy_order_id": f"pending_buy_{oca_group}",
                "sell_order_id": f"pending_sell_{oca_group}",
                "oca_group": oca_group}

    async def check_trigger_levels(self) -> dict | None:
        """Backup polling: check if price crossed bracket levels."""
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return None
        if self._tick_trigger_active:
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
                guaranteed_stop=False, level=None, limit_distance=None,
                limit_level=None, order_type="MARKET", quote_id=None,
                size=qty, stop_distance=self._disaster_stop_pts, stop_level=None,
                trailing_stop=False, trailing_stop_increment=None,
            )

            deal_ref = result.get("dealReference", "")
            confirm = await self._confirm_deal(deal_ref)

            if confirm.get("dealStatus") == "REJECTED":
                reason = confirm.get("reason", "Unknown")
                logger.error(f"Order REJECTED: {reason}")
                return {"error": f"Rejected: {reason}"}

            deal_id = confirm.get("dealId", deal_ref)
            fill_level = confirm.get("level", 0)

            self._position_deal_ids[deal_id] = {
                "direction": direction, "size": qty, "level": fill_level,
            }

            logger.info(f"Market order ({self.epic}): {direction} {qty}, fill={fill_level} ({deal_id})")
            return {"order_id": deal_id, "avg_price": fill_level}

        except Exception as e:
            logger.error(f"Market order failed ({self.epic}): {e}")
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
            return len(self._position_deal_ids) == 0

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

    async def _confirm_deal(self, deal_reference: str) -> dict:
        if not deal_reference:
            return {}
        try:
            await asyncio.sleep(0.5)
            confirm = await self._shared.rest_call(
                self._shared.ig.fetch_deal_by_deal_reference, deal_reference,
            )
            return confirm if confirm else {}
        except Exception as e:
            logger.warning(f"Deal confirmation failed: {e}")
            return {"dealId": deal_reference}
