"""
broker_ig.py — IG Markets Broker Adapter (thin wrapper over shared session)
═══════════════════════════════════════════════════════════════════════════════

Same public interface as the old monolithic broker. Delegates all IG
communication to the shared IGSharedSession (one REST session) and
IGStreamManager (Lightstreamer ticks + candles + trade events).

Per-bot state (OCA bracket simulation, order tracking) remains local.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from dax_bot import config
from shared.ig_session import IGSharedSession
from shared.ig_stream import IGStreamManager

logger = logging.getLogger(__name__)


class IGBroker:
    """IG Markets broker — delegates to shared session, keeps local order state."""

    def __init__(
        self,
        shared: IGSharedSession,
        stream: IGStreamManager,
        epic: str,
        currency: str,
    ):
        self._shared = shared
        self._stream = stream
        self.epic = epic
        self.currency = currency
        self.connected = False
        self.contract = None

        # Per-bot order tracking
        self._orders: dict[str, dict] = {}
        self._position_deal_ids: dict[str, dict] = {}
        self._pending_bracket: dict | None = None
        self._tick_trigger_active = False
        self._on_trigger_callbacks: list = []  # async callbacks for tick-triggered fills

        # Register for real-time tick-based entry detection
        self._stream.register_tick_callback(epic, self._on_tick)

    # ── Connection ─────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect using the shared session. Subscribe to streaming."""
        try:
            ok = await self._shared.ensure_connected()
            if not ok:
                self.connected = False
                return False

            # Fetch market info
            market = await self._shared.rest_call(
                self._shared.ig.fetch_market_by_epic, self.epic
            )
            self.contract = market
            self.connected = True

            instrument = market.get("instrument", {})
            name = instrument.get("name", self.epic)
            logger.info(f"IG connected — {name} ({self.epic})")

            # Subscribe to streaming (safe to call multiple times)
            await self._stream.subscribe_ticks(self.epic)
            await self._stream.subscribe_candles(self.epic)
            return True

        except Exception as e:
            logger.error(f"IG connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Mark this broker as disconnected (does NOT close shared session)."""
        self.connected = False
        logger.info(f"IG broker disconnected ({self.epic})")

    async def ensure_connected(self) -> bool:
        """Validate shared session."""
        ok = await self._shared.ensure_connected()
        self.connected = ok
        return ok

    # ── Market Data ────────────────────────────────────────────────────────

    async def get_5min_bars(self, duration: str = "1 D") -> pd.DataFrame:
        """
        Fetch historical 5-minute bars via REST.
        Streaming candles supplement intraday but morning routine needs history.
        """
        if not await self.ensure_connected():
            logger.error("get_5min_bars: not connected to IG")
            return pd.DataFrame()

        try:
            parts = duration.strip().split()
            num_days = int(parts[0]) if parts else 1

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=num_days + 1)
            start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
            end_str = end.strftime("%Y-%m-%dT%H:%M:%S")
            logger.info(f"IG fetching {self.epic} bars: {start_str} → {end_str}")

            result = await self._shared.rest_call(
                self._shared.ig.fetch_historical_prices_by_epic,
                epic=self.epic,
                resolution="MINUTE_5",
                start_date=start_str,
                end_date=end_str,
            )

            if result is None:
                logger.error(f"IG returned None for {self.epic} bars")
                return pd.DataFrame()

            if "prices" not in result:
                logger.error(f"IG result has no 'prices' key")
                return pd.DataFrame()

            df = result["prices"]
            if df.empty:
                logger.warning(f"IG returned empty DataFrame for {self.epic}")
                return pd.DataFrame()

            # IG returns multi-level columns: (bid/ask/last, Open/High/Low/Close)
            # Use mid prices: (bid + ask) / 2
            out = pd.DataFrame(index=df.index)
            for col in ["Open", "High", "Low", "Close"]:
                if ("bid", col) in df.columns and ("ask", col) in df.columns:
                    out[col] = (df[("bid", col)] + df[("ask", col)]) / 2
                elif ("last", col) in df.columns:
                    out[col] = df[("last", col)]
                else:
                    out[col] = df[col] if col in df.columns else 0

            if "Volume" in df.columns:
                out["Volume"] = df["Volume"]
            elif ("last", "Volume") in df.columns:
                out["Volume"] = df[("last", "Volume")]
            else:
                out["Volume"] = 0

            out.index = pd.to_datetime(out.index, utc=True)
            logger.info(f"IG fetched {len(out)} bars for {self.epic}")
            return out

        except Exception as e:
            logger.error(f"Failed to fetch IG bars: {e}", exc_info=True)
            return pd.DataFrame()

    async def get_overnight_bars(self) -> pd.DataFrame:
        """Fetch overnight bars (00:00-06:00 CET)."""
        df = await self.get_5min_bars("1 D")
        if df.empty:
            return df
        try:
            from zoneinfo import ZoneInfo
            cet = ZoneInfo("Europe/Berlin")
            df.index = df.index.tz_convert(cet)
            overnight = df.between_time("00:00", "06:00")
            logger.info(f"IG overnight bars: {len(overnight)}")
            return overnight
        except Exception as e:
            logger.error(f"Failed to filter overnight bars: {e}")
            return pd.DataFrame()

    def get_streaming_bars_df(self) -> pd.DataFrame:
        """Get today's 5-min bars from Lightstreamer stream (no REST call)."""
        return self._stream.get_today_bars_df(self.epic)

    def get_streaming_bar_count(self) -> int:
        """How many completed streaming bars today."""
        return self._stream.get_bar_count_today(self.epic)

    async def get_current_price(self) -> float | None:
        """Get current mid price — streaming first, REST fallback.
        Returns None if streaming price is stale (>30s old).
        """
        age = self._stream.get_tick_age(self.epic)
        if age < 30:
            return self._stream._prices.get(self.epic)
        if age != float("inf"):
            logger.warning("Price stale: %.1fs old — falling back to REST", age)
        return await self._stream.get_price(self.epic)

    # ── Tick-based entry trigger ────────────────────────────────────────

    def register_trigger_callback(self, callback):
        """Register async callback for tick-triggered fills."""
        self._on_trigger_callbacks.append(callback)

    def _on_tick(self, mid: float, bid: float = 0.0, ofr: float = 0.0):
        """Called on every tick from Lightstreamer (runs on event loop thread).
        Checks bracket levels using bid/offer (not mid) for accurate triggering.
        BUY triggers on offer (what you pay), SELL triggers on bid (what you get).
        """
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return
        if self._tick_trigger_active:
            return  # Already processing a trigger

        # Max spread check — skip if spread is too wide (> 3pts)
        spread = ofr - bid if (bid > 0 and ofr > 0) else 0
        if spread > config.MAX_SPREAD_PTS:
            return

        bracket = self._pending_bracket
        triggered_dir = None

        # Use offer for BUY (price you pay), bid for SELL (price you receive)
        if ofr > 0 and ofr >= bracket["buy_price"]:
            triggered_dir = "BUY"
        elif bid > 0 and bid <= bracket["sell_price"]:
            triggered_dir = "SELL"
        # Fallback to mid if bid/offer not available
        elif bid == 0 or ofr == 0:
            if mid >= bracket["buy_price"]:
                triggered_dir = "BUY"
            elif mid <= bracket["sell_price"]:
                triggered_dir = "SELL"

        if not triggered_dir:
            return

        # Mark active AND deactivate bracket immediately to prevent race
        # with check_trigger_levels() polling in monitor_cycle
        self._tick_trigger_active = True
        bracket["active"] = False
        trigger_price = ofr if triggered_dir == "BUY" else bid
        logger.info(f"Tick trigger: {triggered_dir} @ {trigger_price:.1f} (bid={bid:.1f} ofr={ofr:.1f} spread={spread:.1f})")

        # Schedule async order placement on event loop
        loop = asyncio.get_event_loop()
        loop.create_task(self._execute_tick_trigger(triggered_dir, price, bracket))

    async def _execute_tick_trigger(self, direction: str, price: float, bracket: dict):
        """Execute the market order triggered by a tick."""
        try:
            # Pre-entry safety: verify no existing position on this epic
            try:
                pos = await self.get_position()
                if pos["direction"] != "FLAT":
                    logger.error("BLOCKED: existing %s position on %s — cannot enter %s",
                                 pos["direction"], self.epic, direction)
                    self._tick_trigger_active = False
                    return
            except Exception as e:
                logger.warning("Pre-entry position check failed: %s — proceeding", e)

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

            # Fire registered callbacks (main.py handles fill + stop placement)
            for cb in self._on_trigger_callbacks:
                try:
                    await cb(trigger_result)
                except Exception as e:
                    logger.error(f"Trigger callback error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Tick trigger execution failed: {e}", exc_info=True)
        finally:
            self._tick_trigger_active = False

    # ── Order Management ───────────────────────────────────────────────────

    async def place_oca_bracket(
        self, buy_price: float, sell_price: float, qty: int, oca_group: str,
    ) -> dict:
        """
        Simulate OCA bracket. IG spread bet rejects working orders near market,
        so we store trigger levels locally and check via check_trigger_levels().
        """
        self._pending_bracket = {
            "buy_price": buy_price, "sell_price": sell_price,
            "qty": qty, "oca_group": oca_group, "active": True,
        }

        buy_id = f"pending_buy_{oca_group}"
        sell_id = f"pending_sell_{oca_group}"

        self._orders[buy_id] = {
            "type": "pending", "direction": "BUY", "oca_pair": sell_id,
            "oca_group": oca_group, "price": buy_price, "qty": qty,
        }
        self._orders[sell_id] = {
            "type": "pending", "direction": "SELL", "oca_pair": buy_id,
            "oca_group": oca_group, "price": sell_price, "qty": qty,
        }

        logger.info(f"OCA bracket (price-triggered): BUY@{buy_price} / SELL@{sell_price}")
        return {"buy_order_id": buy_id, "sell_order_id": sell_id, "oca_group": oca_group}

    async def check_trigger_levels(self) -> dict | None:
        """Check if current streaming price crossed any pending bracket level."""
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return None
        if self._tick_trigger_active:
            return None  # Tick callback already handling this

        # Use bid/offer for accurate trigger detection
        bid = self._stream._prices.get(f"{self.epic}_bid")
        ofr = self._stream._prices.get(f"{self.epic}_ofr")

        if bid is None or ofr is None:
            price = await self.get_current_price()
            if price is None:
                return None
            bid = ofr = price  # Fallback to mid

        # Spread check
        spread = ofr - bid
        if spread > config.MAX_SPREAD_PTS:
            logger.warning(f"Spread too wide ({spread:.1f}pts) — skipping trigger check")
            return None

        bracket = self._pending_bracket
        triggered_dir = None

        if ofr >= bracket["buy_price"]:
            triggered_dir = "BUY"
        elif bid <= bracket["sell_price"]:
            triggered_dir = "SELL"

        if not triggered_dir:
            return None

        # Pre-entry safety: verify no existing position
        try:
            pos = await self.get_position()
            if pos["direction"] != "FLAT":
                logger.error("BLOCKED: existing %s position — cannot enter %s",
                             pos["direction"], triggered_dir)
                return None
        except Exception as e:
            logger.warning("Pre-entry position check failed: %s — proceeding", e)

        trigger_price = ofr if triggered_dir == "BUY" else bid
        logger.info(f"Bracket triggered: {triggered_dir} @ {trigger_price:.1f} (bid={bid:.1f} ofr={ofr:.1f} spread={spread:.1f})")
        result = await self.place_market_order(action=triggered_dir, qty=bracket["qty"])

        if "error" in result:
            logger.error(f"Triggered market order failed: {result['error']}")
            return None

        self._pending_bracket["active"] = False
        return {
            "direction": "LONG" if triggered_dir == "BUY" else "SHORT",
            "fill_price": result.get("avg_price", price),
            "order_id": result.get("order_id", ""),
        }

    async def place_market_order(self, action: str, qty: int) -> dict:
        """Place a market order."""
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
                size=qty, stop_distance=None, stop_level=None,
                trailing_stop=False, trailing_stop_increment=None,
            )

            deal_ref = result.get("dealReference", "")
            confirm = await self._confirm_deal(deal_ref)

            # Verify deal was accepted by IG
            deal_status = confirm.get("dealStatus", "")
            if deal_status == "REJECTED":
                reason = confirm.get("reason", "Unknown")
                logger.error(f"Market order REJECTED by IG: {reason}")
                return {"error": f"Order rejected: {reason}"}

            deal_id = confirm.get("dealId", deal_ref)
            fill_level = confirm.get("level", 0)

            self._position_deal_ids[deal_id] = {
                "direction": direction, "size": qty, "level": fill_level,
            }

            logger.info(f"Market order: {direction} {qty}, fill={fill_level} ({deal_id})")
            return {"order_id": deal_id, "avg_price": fill_level}

        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return {"error": str(e)}

    async def place_stop_order(
        self, action: str, qty: int, stop_price: float, parent_id: int = 0,
    ) -> dict:
        """Place stop order or update position stop."""
        if not await self.ensure_connected():
            return {"error": "Not connected"}

        try:
            # Update existing position stop (all deals)
            if self._position_deal_ids:
                updated = []
                for deal_id in list(self._position_deal_ids.keys()):
                    try:
                        await self._shared.rest_call(
                            self._shared.ig.update_open_position,
                            limit_level=None, stop_level=stop_price, deal_id=deal_id,
                        )
                        updated.append(deal_id)
                        logger.info(f"Stop set on position {deal_id} @ {stop_price}")
                    except Exception as e:
                        logger.error(f"Failed to set stop on {deal_id}: {e}")
                if updated:
                    return {"order_id": f"stop_{updated[0]}"}
                return {"error": "Failed to set stop on any position"}

            # Create working order
            direction = "BUY" if action == "BUY" else "SELL"
            result = await self._shared.rest_call(
                self._shared.ig.create_working_order,
                currency_code=self.currency, direction=direction,
                epic=self.epic, expiry="DFB", guaranteed_stop=False,
                level=stop_price, size=qty,
                time_in_force="GOOD_TILL_CANCELLED", order_type="STOP",
                force_open=False,
            )
            deal_ref = result.get("dealReference", "")
            confirm = await self._confirm_deal(deal_ref)
            deal_id = confirm.get("dealId", deal_ref)

            self._orders[deal_id] = {
                "type": "working", "direction": direction,
                "price": stop_price, "qty": qty,
            }
            logger.info(f"Stop order: {direction} {qty} @ {stop_price} ({deal_id})")
            return {"order_id": deal_id}

        except Exception as e:
            logger.error(f"Stop order failed: {e}")
            return {"error": str(e)}

    async def place_limit_order(self, action: str, qty: int, limit_price: float) -> dict:
        """Place a limit order (for TP exits)."""
        if not await self.ensure_connected():
            return {"error": "Not connected"}

        try:
            direction = "BUY" if action == "BUY" else "SELL"
            result = await self._shared.rest_call(
                self._shared.ig.create_working_order,
                currency_code=self.currency, direction=direction,
                epic=self.epic, expiry="DFB", guaranteed_stop=False,
                level=limit_price, size=qty,
                time_in_force="GOOD_TILL_CANCELLED", order_type="LIMIT",
                force_open=False,
            )
            deal_ref = result.get("dealReference", "")
            confirm = await self._confirm_deal(deal_ref)
            deal_id = confirm.get("dealId", deal_ref)

            self._orders[deal_id] = {
                "type": "working", "direction": direction,
                "price": limit_price, "qty": qty,
            }
            logger.info(f"Limit order: {direction} {qty} @ {limit_price} ({deal_id})")
            return {"order_id": deal_id}

        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            return {"error": str(e)}

    async def modify_stop(self, order_id: str, new_stop: float) -> bool:
        """Modify an existing stop level."""
        if not await self.ensure_connected():
            return False

        try:
            deal_id = str(order_id).replace("stop_", "")

            # Try updating position stop (all deals)
            if deal_id in self._position_deal_ids or str(order_id).startswith("stop_"):
                if self._position_deal_ids:
                    updated_any = False
                    for pos_id in list(self._position_deal_ids.keys()):
                        try:
                            await self._shared.rest_call(
                                self._shared.ig.update_open_position,
                                limit_level=None, stop_level=new_stop, deal_id=pos_id,
                            )
                            logger.info(f"Stop modified on position {pos_id} → {new_stop}")
                            updated_any = True
                        except Exception as e:
                            logger.error(f"Failed to modify stop on {pos_id}: {e}")
                    return updated_any

            # Try updating working order
            if deal_id in self._orders:
                await self._shared.rest_call(
                    self._shared.ig.update_working_order,
                    good_till_date=None, level=new_stop,
                    limit_distance=None, limit_level=None,
                    stop_distance=None, stop_level=None,
                    guaranteed_stop=False,
                    time_in_force="GOOD_TILL_CANCELLED",
                    order_type="STOP", deal_id=deal_id,
                )
                self._orders[deal_id]["price"] = new_stop
                logger.info(f"Working order {deal_id} modified → {new_stop}")
                return True

            logger.warning(f"Order {order_id} not found for modification")
            return False

        except Exception as e:
            logger.error(f"Modify stop failed: {e}")
            return False

    async def modify_stop_qty(self, order_id: str, new_qty: int) -> bool:
        """IG doesn't support qty modification. Returns False to trigger re-place."""
        return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order."""
        if not await self.ensure_connected():
            return False

        try:
            deal_id = str(order_id).replace("stop_", "")
            await self._shared.rest_call(
                self._shared.ig.delete_working_order, deal_id=deal_id,
            )
            self._orders.pop(deal_id, None)
            logger.info(f"Order {deal_id} cancelled")
            return True

        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

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
                    except Exception as e:
                        logger.warning(f"Failed to cancel {deal_id}: {e}")

            self._orders.clear()
            # Deactivate local bracket simulation (prevents future price triggers)
            if self._pending_bracket:
                self._pending_bracket["active"] = False
            if count:
                logger.info(f"Cancelled {count} working orders")
            return count

        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
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

            closed_ids = []
            for pos in pos_list:
                if pos.get("epic") == self.epic:
                    deal_id = pos.get("dealId", "")
                    direction = pos.get("direction", "")
                    size = pos.get("dealSize") or pos.get("size", 0)
                    close_dir = "SELL" if direction == "BUY" else "BUY"
                    try:
                        await self._shared.rest_call(
                            self._shared.ig.close_open_position,
                            deal_id=deal_id, direction=close_dir,
                            epic=None, expiry="DFB", level=None,
                            order_type="MARKET", quote_id=None, size=size,
                        )
                        closed_ids.append(deal_id)
                        logger.info(f"Position closed: {close_dir} {size} ({deal_id})")
                    except Exception as e:
                        logger.error(f"Failed to close {deal_id}: {e}")

            for did in closed_ids:
                self._position_deal_ids.pop(did, None)
            remaining = len(self._position_deal_ids)
            if remaining:
                logger.error(f"{remaining} positions failed to close")
            return remaining == 0

        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return False

    # ── Position & P&L ─────────────────────────────────────────────────────

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
            for pos in pos_list:
                if pos.get("epic") == self.epic:
                    d = pos.get("direction", "")
                    size = float(pos.get("dealSize") or pos.get("size", 0))
                    level = float(pos.get("level") or pos.get("openLevel", 0))
                    deal_id = pos.get("dealId", "")
                    if deal_id:
                        self._position_deal_ids[deal_id] = {
                            "direction": d, "size": size, "level": level,
                        }
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
                }

            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}

        except Exception as e:
            logger.error(f"get_position failed: {e}")
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}

    async def get_open_orders(self) -> list:
        """Get all working orders."""
        if not await self.ensure_connected():
            return []

        try:
            orders = await self._shared.rest_call(self._shared.ig.fetch_working_orders)
            if orders is None:
                return []

            order_list = (orders if isinstance(orders, list)
                          else orders.to_dict("records") if hasattr(orders, "to_dict")
                          else [])

            return [
                {
                    "order_id": o.get("dealId", ""),
                    "action": o.get("direction", ""),
                    "type": o.get("orderType", ""),
                    "price": o.get("level") or o.get("orderLevel", 0),
                    "qty": o.get("orderSize") or o.get("size", 0),
                    "status": o.get("status", "WORKING"),
                }
                for o in order_list
            ]

        except Exception as e:
            logger.error(f"get_open_orders failed: {e}")
            return []

    async def get_order_status(self, order_id: str) -> str | None:
        """Check if a working order still exists."""
        if not await self.ensure_connected():
            return None

        try:
            orders = await self._shared.rest_call(self._shared.ig.fetch_working_orders)
            if orders is not None:
                order_list = (orders if isinstance(orders, list)
                              else orders.to_dict("records") if hasattr(orders, "to_dict")
                              else [])
                for o in order_list:
                    if o.get("dealId") == str(order_id):
                        return "Submitted"

            positions = await self._shared.rest_call(self._shared.ig.fetch_open_positions)
            if positions is not None:
                pos_list = (positions if isinstance(positions, list)
                            else positions.to_dict("records") if hasattr(positions, "to_dict")
                            else [])
                for p in pos_list:
                    if p.get("dealId") == str(order_id):
                        return "Filled"

            return "Cancelled"

        except Exception as e:
            logger.error(f"get_order_status failed: {e}")
            return None

    async def get_fill_price(self, order_id: str) -> float | None:
        """Get fill price for a filled order."""
        if not await self.ensure_connected():
            return None

        try:
            confirm = await self._shared.rest_call(
                self._shared.ig.fetch_deal_by_deal_reference, str(order_id),
            )
            return confirm.get("level") if confirm else None

        except Exception as e:
            logger.error(f"get_fill_price failed: {e}")
            return None

    def register_order_handler(self, callback):
        """Register for trade events via streaming."""
        self._stream.register_trade_callback(callback)
        logger.info(f"Order handler registered (streaming) for {self.epic}")

    def unregister_order_handler(self, callback):
        """No-op — trade callbacks persist for session lifetime."""
        pass

    # ── OCA simulation ─────────────────────────────────────────────────────

    async def cancel_oca_counterpart(self, filled_order_id: str) -> bool:
        """When one side of OCA fills, cancel the other."""
        if self._pending_bracket:
            self._pending_bracket["active"] = False

        order_info = self._orders.get(str(filled_order_id))
        if not order_info or "oca_pair" not in order_info:
            return True

        other_id = order_info["oca_pair"]
        if other_id.startswith("pending_"):
            self._orders.pop(other_id, None)
            logger.info(f"OCA: deactivated pending counterpart {other_id}")
            return True

        logger.info(f"OCA: cancelling counterpart {other_id}")
        return await self.cancel_order(other_id)

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _confirm_deal(self, deal_reference: str) -> dict:
        """Fetch deal confirmation to get the actual deal_id."""
        if not deal_reference:
            return {}
        try:
            await asyncio.sleep(0.5)
            confirm = await self._shared.rest_call(
                self._shared.ig.fetch_deal_by_deal_reference, deal_reference,
            )
            return confirm if confirm else {}
        except Exception as e:
            logger.warning(f"Deal confirmation failed for {deal_reference}: {e}")
            return {"dealId": deal_reference}
