"""
broker_ib.py — IBKR broker wrapper for ASRS strategy
═══════════════════════════════════════════════════════════════

One IBBroker per signal. Shares a single IBSharedSession and
IBStreamManager across all signals. Implements the same public
interface as asrs/broker.py:IGBroker so strategy.py is broker-agnostic.

Execution model (matches IGBroker's tick architecture):
  - Bracket entry: stored locally as pending, triggered by tick callback
  - Entry fill: market order via ib.placeOrder after tick breach
  - Stop exit: tick monitor fires market close on stop touch
  - Real stop order: placed on IBKR as backup (belt + braces)

Public surface (matches IGBroker):
    connect / ensure_connected
    get_5min_bars / get_streaming_bars_df / get_streaming_bar_count
    get_current_price
    register_trigger_callback / register_stop_callback
    place_oca_bracket / deactivate_bracket / check_trigger_levels
    activate_stop_monitor / update_stop_level / deactivate_stop_monitor
    place_market_order / place_stop_order
    modify_stop / modify_stop_all
    cancel_all_orders / close_position
    get_position
    _position_deal_ids (dict attr for reconciliation)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd
from ib_async import (
    Contract, Future, MarketOrder, StopOrder, LimitOrder, Trade,
)

from shared.ib_session import IBSharedSession
from shared.ib_stream import IBStreamManager, contract_key
from asrs.contract_resolver import (
    ContractSpec, get_spec, resolve_front_month, days_to_expiry,
)

logger = logging.getLogger(__name__)


class IBBroker:
    """IBKR broker — one per signal, shares session + stream manager."""

    def __init__(
        self,
        shared: IBSharedSession,
        stream: IBStreamManager,
        instrument: str,                # ASRS name: DAX / US30 / NIKKEI
        disaster_stop_pts: int = 200,
        max_spread_pts: float = 10.0,
        expiry: str | None = None,      # YYYYMMDD; default = front month
    ):
        self._shared = shared
        self._stream = stream
        self.instrument = instrument
        self.spec: ContractSpec = get_spec(instrument)
        self._expiry = expiry
        self._disaster_stop_pts = disaster_stop_pts
        self._max_spread_pts = max_spread_pts

        self.contract: Contract | None = None           # qualified at connect()
        self.contract_key: str = ""                      # set after qualify
        self.connected = False

        # Bracket simulation (same as IGBroker — tick callback drives entry)
        self._pending_bracket: dict | None = None
        self._tick_trigger_active = False
        self._on_trigger_callbacks: list[Callable] = []

        # Tick-based stop monitor (exit via market order)
        self._stop_monitor: dict | None = None
        self._stop_exit_active = False
        self._on_stop_callbacks: list[Callable] = []

        # Open position tracking — matches IGBroker's interface
        self._position_deal_ids: dict[str, dict] = {}

        # IBKR-specific: parallel real stop order (belt + braces)
        self._real_stop_trade: Trade | None = None

        # Currency passthrough for compatibility with main.py
        self.currency = self.spec.currency
        # "epic" alias so existing main.py code compiles unchanged
        self.epic = ""  # populated after qualify

    # ── Connection ───────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect via shared session, resolve front-month, subscribe streams."""
        if not await self._shared.ensure_connected():
            self.connected = False
            return False

        # Dynamically resolve front-month from IBKR's contract calendar
        # (handles CBOT/CME/EUREX expiry quirks without hardcoding rules)
        contract, expiry = await resolve_front_month(self.instrument, self._shared)
        if contract is None:
            logger.error(f"Failed to resolve {self.instrument} front-month")
            self.connected = False
            return False

        # qualify to populate conId + secIdList
        qualified = await self._shared.qualify(contract)
        if qualified is None:
            logger.error(f"Failed to qualify {self.instrument} contract")
            self.connected = False
            return False

        self.contract = qualified
        self.contract_key = contract_key(qualified)
        self.epic = self.contract_key  # alias for main.py compatibility

        # Log expiry / days-to-roll
        exp = qualified.lastTradeDateOrContractMonth
        days_left = days_to_expiry(exp)
        logger.info(
            f"IB contract qualified: {self.instrument} → "
            f"{qualified.localSymbol or qualified.symbol} "
            f"expiry {exp} ({days_left}d)"
        )
        if days_left <= 7:
            logger.warning(
                f"⚠️  {self.instrument} contract expires in {days_left} days — roll imminent"
            )

        # Subscribe to live data
        await self._stream.subscribe_ticks(qualified)
        await self._stream.subscribe_candles(qualified)

        # Register this broker's tick callback so bracket/stop monitor fires on every tick
        self._stream.register_tick_callback(self.contract_key, self._on_tick)

        self.connected = True
        return True

    async def ensure_connected(self) -> bool:
        ok = await self._shared.ensure_connected()
        self.connected = ok
        return ok

    # ── Market data ───────────────────────────────────────────────────

    async def get_5min_bars(self, duration: str = "1 D") -> pd.DataFrame:
        """REST-style historical fetch. duration like '1 D', '2 D', '5 D'."""
        if not await self.ensure_connected():
            return pd.DataFrame()
        if self.contract is None:
            return pd.DataFrame()
        try:
            bars = await self._shared.ib.reqHistoricalDataAsync(
                self.contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,        # UTC
                keepUpToDate=False,
            )
            if not bars:
                return pd.DataFrame()
            records = [{
                "Open": float(b.open), "High": float(b.high),
                "Low":  float(b.low),  "Close": float(b.close),
            } for b in bars]
            idx = pd.DatetimeIndex([
                b.date if isinstance(b.date, datetime)
                else pd.to_datetime(b.date, utc=True)
                for b in bars
            ])
            df = pd.DataFrame(records, index=idx)
            if df.index.tz is None:
                df.index = df.index.tz_localize(timezone.utc)
            return df
        except Exception as e:
            logger.error(f"get_5min_bars failed ({self.instrument}): {e}", exc_info=True)
            return pd.DataFrame()

    def get_streaming_bars_df(self) -> pd.DataFrame:
        return self._stream.get_today_bars_df(self.contract_key)

    def get_streaming_bar_count(self) -> int:
        return self._stream.get_bar_count_today(self.contract_key)

    async def get_current_price(self) -> float | None:
        age = self._stream.get_tick_age(self.contract_key)
        if age < 30:
            return self._stream.get_price_sync(self.contract_key)
        return await self._stream.get_price(self.contract_key)

    # ── Trigger / stop monitor registration ──────────────────────────

    def register_trigger_callback(self, callback: Callable) -> None:
        self._on_trigger_callbacks.append(callback)

    def register_stop_callback(self, callback: Callable) -> None:
        self._on_stop_callbacks.append(callback)

    def deactivate_bracket(self) -> None:
        if self._pending_bracket:
            self._pending_bracket["active"] = False

    def activate_stop_monitor(self, direction: str, stop_level: float) -> None:
        self._stop_monitor = {
            "active": True, "direction": direction, "stop_level": stop_level,
        }
        logger.info(
            f"Stop monitor active ({self.instrument}): {direction} stop={stop_level}"
        )

    def update_stop_level(self, new_stop: float) -> None:
        if self._stop_monitor and self._stop_monitor["active"]:
            old = self._stop_monitor["stop_level"]
            self._stop_monitor["stop_level"] = new_stop
            if abs(new_stop - old) > 0.1:
                logger.info(
                    f"Stop monitor updated ({self.instrument}): {old} → {new_stop}"
                )
        # Also move the real backup stop on IBKR
        if self._real_stop_trade:
            asyncio.get_event_loop().create_task(
                self._modify_real_stop(new_stop)
            )

    def deactivate_stop_monitor(self) -> None:
        if self._stop_monitor:
            self._stop_monitor["active"] = False
            self._stop_monitor = None
        # Cancel real backup stop
        if self._real_stop_trade:
            try:
                self._shared.ib.cancelOrder(self._real_stop_trade.order)
            except Exception:
                pass
            self._real_stop_trade = None

    # ── Tick callback: bracket + stop exit ───────────────────────────

    def _on_tick(self, mid: float, bid: float = 0.0, ofr: float = 0.0, last: float = 0.0) -> None:
        """
        Every tick — checks both stop monitor AND pending bracket.

        Stop checks use the LAST TRADE price (not bid/ask) so live execution
        mirrors the backtest semantics: stops only fire when an actual trade
        prints at or below the stop level. Bid/ask quote movements alone
        (e.g. market makers cancelling orders) do NOT fire stops — they're
        not real fills, just promises that can be withdrawn.

        This eliminates "phantom wick" exits like 2026-04-08 NIKKEI_S1 where
        the bid touched 55980 but no trade printed at that price, yet the
        old code fired the stop.
        """
        # 1. Stop monitor — uses LAST TRADE price for stop check
        if (
            self._stop_monitor
            and self._stop_monitor.get("active")
            and not self._stop_exit_active
        ):
            sm = self._stop_monitor
            stop = sm["stop_level"]
            # Prefer last trade; fall back to mid if last not available
            check_price = last if last and last > 0 else mid
            hit = False
            if sm["direction"] == "LONG" and check_price > 0 and check_price <= stop:
                hit = True
            elif sm["direction"] == "SHORT" and check_price > 0 and check_price >= stop:
                hit = True
            if hit:
                self._stop_exit_active = True
                # Exit price reference: last trade (what we'll record), the
                # market sell will fill at the prevailing bid for LONG, etc.
                exit_price = check_price
                logger.info(
                    f"Stop hit ({self.instrument}): {sm['direction']} "
                    f"stop={stop} last={check_price:.2f} "
                    f"(bid={bid:.2f} ofr={ofr:.2f})"
                )
                loop = asyncio.get_event_loop()
                loop.create_task(self._execute_stop_exit(sm["direction"], exit_price))

        # 2. Bracket trigger
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return
        if self._tick_trigger_active:
            return

        spread = ofr - bid if (bid > 0 and ofr > 0) else 0
        if spread > self._max_spread_pts:
            return

        bracket = self._pending_bracket
        triggered_dir = None
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
        logger.info(
            f"Tick trigger ({self.instrument}): {triggered_dir} @ {trigger_price:.2f} "
            f"(spread={spread:.1f})"
        )
        loop = asyncio.get_event_loop()
        loop.create_task(self._execute_tick_trigger(triggered_dir, trigger_price, bracket))

    async def _execute_tick_trigger(
        self, direction: str, price: float, bracket: dict,
    ) -> None:
        """Market order on tick breach of bracket level."""
        try:
            # Safety: no existing position
            try:
                pos = await self.get_position()
                if pos["direction"] != "FLAT":
                    logger.error(
                        f"BLOCKED: existing {pos['direction']} on {self.instrument}"
                    )
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

    async def _execute_stop_exit(self, direction: str, exit_price: float) -> None:
        """Market close all positions when tick monitor fires."""
        try:
            self._stop_monitor["active"] = False  # prevent re-trigger

            # Cancel real backup stop first
            if self._real_stop_trade:
                try:
                    self._shared.ib.cancelOrder(self._real_stop_trade.order)
                except Exception:
                    pass
                self._real_stop_trade = None

            for attempt in range(1, 4):
                closed = await self.close_position()
                if closed:
                    break
                logger.error(
                    f"Stop exit close attempt {attempt}/3 failed ({self.instrument})"
                )
                if attempt < 3:
                    await asyncio.sleep(2)

            pos = await self.get_position()
            fills = getattr(self, "_last_close_fills", [])
            actual_exit = sum(fills) / len(fills) if fills else exit_price
            if fills:
                logger.info(
                    f"Actual close fills ({self.instrument}): {fills} "
                    f"avg={actual_exit:.2f}"
                )
            if pos["direction"] != "FLAT":
                logger.error(
                    f"STOP EXIT FAILED — position still open ({self.instrument})"
                )
                if self._stop_monitor:
                    self._stop_monitor["active"] = True
                self._stop_exit_active = False
                return

            logger.info(
                f"Stop exit filled ({self.instrument}): closed @ ~{actual_exit:.2f}"
            )
            for cb in self._on_stop_callbacks:
                try:
                    await cb({
                        "exit_price": actual_exit,
                        "exit_intended": exit_price,
                        "direction": direction,
                    })
                except Exception as e:
                    logger.error(f"Stop exit callback error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Stop exit execution failed: {e}", exc_info=True)
        finally:
            self._stop_exit_active = False

    # ── Bracket placement ───────────────────────────────────────────

    async def place_oca_bracket(
        self, buy_price: float, sell_price: float, qty: int, oca_group: str,
    ) -> dict:
        """Store bracket locally; _on_tick handles the trigger."""
        self._pending_bracket = {
            "buy_price": buy_price, "sell_price": sell_price,
            "qty": qty, "oca_group": oca_group, "active": True,
        }
        logger.info(
            f"OCA bracket ({self.instrument}): BUY@{buy_price} / SELL@{sell_price}"
        )
        return {
            "buy_order_id":  f"pending_buy_{oca_group}",
            "sell_order_id": f"pending_sell_{oca_group}",
            "oca_group":     oca_group,
        }

    async def check_trigger_levels(self) -> dict | None:
        """Backup polling — check if a tick happened between _on_tick fires."""
        if not self._pending_bracket or not self._pending_bracket.get("active"):
            return None
        if self._tick_trigger_active:
            return None

        bid = self._stream._prices.get(f"{self.contract_key}_bid")
        ofr = self._stream._prices.get(f"{self.contract_key}_ofr")
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

        try:
            pos = await self.get_position()
            if pos["direction"] != "FLAT":
                logger.error(
                    f"BLOCKED: existing {pos['direction']} on {self.instrument}"
                )
                return None
        except Exception:
            pass

        trigger_price = ofr if triggered_dir == "BUY" else bid
        logger.info(
            f"Bracket triggered poll ({self.instrument}): {triggered_dir} @ {trigger_price:.2f}"
        )
        result = await self.place_market_order(
            action=triggered_dir, qty=bracket["qty"],
        )
        if "error" in result:
            return None

        self._pending_bracket["active"] = False
        return {
            "direction": "LONG" if triggered_dir == "BUY" else "SHORT",
            "fill_price": result.get("avg_price", trigger_price),
            "order_id":   result.get("order_id", ""),
        }

    # ── Order placement ─────────────────────────────────────────────

    async def place_market_order(self, action: str, qty: int) -> dict:
        """Market order entry. Also places a disaster stop on IBKR."""
        if not await self.ensure_connected():
            return {"error": "Not connected"}
        if self.contract is None:
            return {"error": "Contract not qualified"}
        try:
            order = MarketOrder(
                action=action, totalQuantity=qty, tif="DAY", transmit=True,
            )
            trade = self._shared.ib.placeOrder(self.contract, order)

            # Wait up to 5s for fill
            for _ in range(50):
                await asyncio.sleep(0.1)
                if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
                    break
                if trade.fills:
                    break

            status = trade.orderStatus.status
            # Some statuses like "Submitted" with fills present mean it filled
            # before the status field caught up. Treat fills as authoritative.
            if status not in ("Filled",) and not trade.fills:
                logger.error(
                    f"Market order not filled ({self.instrument}): status={status}"
                )
                return {"error": f"status={status}"}

            fill_price = float(trade.orderStatus.avgFillPrice or 0)
            if not fill_price and trade.fills:
                prices = [float(f.execution.price) for f in trade.fills if f.execution.price]
                if prices:
                    fill_price = sum(prices) / len(prices)
            order_id = str(trade.order.orderId)

            # Track deal — use orderId as the key (mirrors IG deal_id)
            self._position_deal_ids[order_id] = {
                "direction": action, "size": qty, "level": fill_price,
            }

            logger.info(
                f"Market order ({self.instrument}): {action} {qty} "
                f"fill={fill_price} id={order_id}"
            )
            return {"order_id": order_id, "avg_price": fill_price}

        except Exception as e:
            logger.error(f"Market order failed ({self.instrument}): {e}", exc_info=True)
            return {"error": str(e)}

    async def place_stop_order(
        self, action: str, qty: int, stop_price: float,
    ) -> dict:
        """
        Place a REAL stop order on IBKR as backup to the tick monitor.
        Stored in self._real_stop_trade so update_stop_level can move it.
        """
        if not await self.ensure_connected():
            return {"error": "Not connected"}
        if self.contract is None:
            return {"error": "Contract not qualified"}
        try:
            # Cancel old stop if exists (avoid orphans)
            if self._real_stop_trade:
                try:
                    self._shared.ib.cancelOrder(self._real_stop_trade.order)
                except Exception:
                    pass
                self._real_stop_trade = None

            order = StopOrder(
                action=action, totalQuantity=qty,
                stopPrice=stop_price, tif="DAY", transmit=True,
            )
            trade = self._shared.ib.placeOrder(self.contract, order)
            await asyncio.sleep(0.3)
            self._real_stop_trade = trade
            logger.info(
                f"Real stop placed ({self.instrument}): {action} {qty} @ {stop_price}"
            )
            return {"order_id": str(trade.order.orderId)}
        except Exception as e:
            logger.error(f"place_stop_order failed: {e}", exc_info=True)
            return {"error": str(e)}

    async def _modify_real_stop(self, new_stop: float) -> None:
        """Move the backup real stop (called by update_stop_level)."""
        if not self._real_stop_trade:
            return
        try:
            self._real_stop_trade.order.stopPrice = new_stop
            self._shared.ib.placeOrder(self.contract, self._real_stop_trade.order)
        except Exception as e:
            logger.error(f"modify_real_stop failed: {e}")

    async def modify_stop(self, order_id: str, new_stop: float) -> bool:
        """Modify a specific stop by order id."""
        if not await self.ensure_connected():
            return False
        try:
            for trade in self._shared.ib.trades():
                if str(trade.order.orderId) == str(order_id):
                    trade.order.stopPrice = new_stop
                    self._shared.ib.placeOrder(self.contract, trade.order)
                    logger.info(f"Stop modified: order {order_id} → {new_stop}")
                    return True
            logger.warning(f"modify_stop: order {order_id} not found")
            return False
        except Exception as e:
            logger.error(f"modify_stop failed: {e}")
            return False

    async def modify_stop_all(self, new_stop: float) -> bool:
        """Move real backup stop + update tick monitor."""
        self.update_stop_level(new_stop)
        return True

    async def cancel_all_orders(self) -> int:
        """Cancel all working orders for this contract."""
        if not await self.ensure_connected():
            return 0
        try:
            count = 0
            for trade in self._shared.ib.openTrades():
                if (
                    trade.contract.conId == self.contract.conId
                    and trade.orderStatus.status not in ("Filled", "Cancelled")
                ):
                    self._shared.ib.cancelOrder(trade.order)
                    count += 1
            if count:
                await asyncio.sleep(0.5)
                logger.info(f"Cancelled {count} orders on {self.instrument}")
            if self._pending_bracket:
                self._pending_bracket["active"] = False
            self._real_stop_trade = None
            return count
        except Exception as e:
            logger.error(f"cancel_all_orders failed: {e}")
            return 0

    async def close_position(self) -> bool:
        """Close all open positions for this contract at market."""
        if not await self.ensure_connected():
            return False
        if self.contract is None:
            return False
        try:
            closed_fills: list[float] = []
            total_closed = 0
            for pos in self._shared.ib.positions():
                if pos.contract.conId != self.contract.conId:
                    continue
                if pos.position == 0:
                    continue
                action = "SELL" if pos.position > 0 else "BUY"
                qty = abs(int(pos.position))
                order = MarketOrder(
                    action=action, totalQuantity=qty,
                    tif="DAY",  # explicit to avoid IBKR preset warning 10349
                )
                trade = self._shared.ib.placeOrder(self.contract, order)

                # Wait for fill (Filled OR fills list populated)
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
                        break
                    if trade.fills:  # paper sometimes fills before status updates
                        break

                # Capture fill price — try avgFillPrice first, then trade.fills
                fill = float(trade.orderStatus.avgFillPrice or 0)
                if not fill and trade.fills:
                    # Average the executions
                    prices = [float(f.execution.price) for f in trade.fills if f.execution.price]
                    if prices:
                        fill = sum(prices) / len(prices)
                if fill:
                    closed_fills.append(fill)
                total_closed += qty

            self._last_close_fills = closed_fills

            # Clear local deal tracking — verify via positions()
            await asyncio.sleep(0.3)
            flat = all(
                pos.contract.conId != self.contract.conId or pos.position == 0
                for pos in self._shared.ib.positions()
            )
            if flat:
                self._position_deal_ids.clear()
                if self._real_stop_trade:
                    try:
                        self._shared.ib.cancelOrder(self._real_stop_trade.order)
                    except Exception:
                        pass
                    self._real_stop_trade = None
            return flat
        except Exception as e:
            logger.error(f"close_position failed ({self.instrument}): {e}", exc_info=True)
            return False

    async def get_position(self) -> dict:
        """Return current position in a shape compatible with IGBroker."""
        if not await self.ensure_connected():
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}
        if self.contract is None:
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}
        try:
            total = 0.0
            weighted = 0.0
            direction = ""
            try:
                mult = float(self.contract.multiplier or 1) or 1.0
            except (ValueError, TypeError):
                mult = 1.0
            for pos in self._shared.ib.positions():
                if pos.contract.conId != self.contract.conId:
                    continue
                if pos.position == 0:
                    continue
                sign_dir = "BUY" if pos.position > 0 else "SELL"
                size = abs(float(pos.position))
                # IBKR's avgCost is fill_price * multiplier; divide back to get per-unit price
                avg = float(pos.avgCost or 0) / mult
                if not direction:
                    direction = sign_dir
                total += size
                weighted += avg * size

            if total > 0:
                avg_cost = round(weighted / total, 2)
                return {
                    "position": total if direction == "BUY" else -total,
                    "avg_cost": avg_cost,
                    "direction": "LONG" if direction == "BUY" else "SHORT",
                }
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}
        except Exception as e:
            logger.error(f"get_position failed: {e}")
            return {"position": 0, "avg_cost": 0, "direction": "FLAT"}
