"""
ig_stream.py — Streaming manager for tick prices, candles, and trade events
═══════════════════════════════════════════════════════════════════════════════

Bridges Lightstreamer's thread-based callbacks to the asyncio event loop.
Manages subscriptions for multiple epics (DAX + FTSE) through one connection.
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
from lightstreamer.client import Subscription, SubscriptionListener

from shared.ig_session import IGSharedSession

logger = logging.getLogger(__name__)

CET = ZoneInfo("Europe/Berlin")


# ── Subscription listeners (run on Lightstreamer's thread) ────────────────

class _TickListener(SubscriptionListener):
    """Receives CHART:{epic}:TICK updates and stores latest mid price."""

    def __init__(self, epic: str, prices: dict, events: dict,
                 tick_callbacks: dict,
                 loop: asyncio.AbstractEventLoop):
        self._epic = epic
        self._prices = prices
        self._events = events
        self._tick_callbacks = tick_callbacks
        self._loop = loop

    def onItemUpdate(self, update):
        try:
            bid = update.getValue("BID")
            ofr = update.getValue("OFR")
            if bid and ofr:
                mid = round((float(bid) + float(ofr)) / 2, 1)
                self._prices[self._epic] = mid
                self._prices[f"{self._epic}_bid"] = float(bid)
                self._prices[f"{self._epic}_ofr"] = float(ofr)
                self._prices[f"{self._epic}_time"] = time.time()
                # Log every 60th tick to avoid flooding
                count = self._prices.get(f"{self._epic}_count", 0) + 1
                self._prices[f"{self._epic}_count"] = count
                if count <= 3 or count % 60 == 0:
                    logger.info(f"Tick #{count} ({self._epic}): bid={bid} ofr={ofr} mid={mid}")
                # Wake any coroutine waiting for a fresh price
                event = self._events.get(self._epic)
                if event:
                    self._loop.call_soon_threadsafe(event.set)
                # Fire tick callbacks (for real-time entry triggers)
                # Pass bid, offer, mid so triggers can use the correct side
                for cb in self._tick_callbacks.get(self._epic, []):
                    self._loop.call_soon_threadsafe(cb, mid, float(bid), float(ofr))
        except Exception as e:
            logger.debug(f"Tick update error ({self._epic}): {e}")

    def onSubscription(self):
        logger.info(f"Tick subscription active: {self._epic}")

    def onSubscriptionError(self, code, message):
        logger.error(f"Tick subscription error ({self._epic}): {code} {message}")

    def onUnsubscription(self):
        logger.info(f"Tick subscription ended: {self._epic}")


class _CandleListener(SubscriptionListener):
    """Receives CHART:{epic}:MINUTE_5 updates and stores completed candles."""

    def __init__(self, epic: str, bars: dict, callbacks: dict,
                 loop: asyncio.AbstractEventLoop):
        self._epic = epic
        self._bars = bars
        self._callbacks = callbacks
        self._loop = loop

    def onItemUpdate(self, update):
        try:
            cons_end = update.getValue("CONS_END")
            # CONS_END = "1" means candle is complete
            if cons_end != "1":
                return

            bid_o = update.getValue("BID_OPEN")
            bid_h = update.getValue("BID_HIGH")
            bid_l = update.getValue("BID_LOW")
            bid_c = update.getValue("BID_CLOSE")
            ofr_o = update.getValue("OFR_OPEN")
            ofr_h = update.getValue("OFR_HIGH")
            ofr_l = update.getValue("OFR_LOW")
            ofr_c = update.getValue("OFR_CLOSE")

            if not all([bid_o, bid_h, bid_l, bid_c, ofr_o, ofr_h, ofr_l, ofr_c]):
                return

            # Compute candle START time in CET
            # CONS_END=1 fires at candle close (e.g., 09:05 for 09:00-09:05 bar)
            now_cet = datetime.now(CET)
            m, s = now_cet.minute, now_cet.second
            # Snap to nearest 5-minute boundary (the close time)
            if m % 5 == 4 and s >= 45:
                snap_min = ((m // 5) + 1) * 5
            else:
                snap_min = (m // 5) * 5
            snap_hour = now_cet.hour + snap_min // 60
            snap_min_mod = snap_min % 60
            end_time = now_cet.replace(hour=snap_hour, minute=snap_min_mod,
                                       second=0, microsecond=0)
            start_time = end_time - timedelta(minutes=5)

            bar = {
                "time": start_time,
                "Open": round((float(bid_o) + float(ofr_o)) / 2, 1),
                "High": round((float(bid_h) + float(ofr_h)) / 2, 1),
                "Low": round((float(bid_l) + float(ofr_l)) / 2, 1),
                "Close": round((float(bid_c) + float(ofr_c)) / 2, 1),
            }

            epic_bars = self._bars.setdefault(self._epic, deque(maxlen=300))
            epic_bars.append(bar)

            # Log bar construction with OHLC (Issue 5)
            logger.info(
                f"Bar built ({self._epic}): {start_time.strftime('%H:%M')}-"
                f"{end_time.strftime('%H:%M')} CET | "
                f"O={bar['Open']} H={bar['High']} L={bar['Low']} C={bar['Close']}"
            )

            # Fire registered callbacks
            for cb in self._callbacks.get(self._epic, []):
                self._loop.call_soon_threadsafe(
                    self._loop.create_task, cb(bar)
                )

        except Exception as e:
            logger.warning(f"Candle update error ({self._epic}): {e}")

    def onSubscription(self):
        logger.info(f"5-min candle subscription active: {self._epic}")

    def onSubscriptionError(self, code, message):
        logger.error(f"Candle subscription error ({self._epic}): {code} {message}")

    def onUnsubscription(self):
        logger.info(f"Candle subscription ended: {self._epic}")


class _TradeListener(SubscriptionListener):
    """Receives TRADE:{account_id} updates for fill/position events."""

    def __init__(self, callbacks: list, loop: asyncio.AbstractEventLoop):
        self._callbacks = callbacks
        self._loop = loop

    def onItemUpdate(self, update):
        try:
            # OPU = Open Position Update (contains all position fields)
            opu = update.getValue("OPU")
            if not opu:
                return

            # OPU is a JSON string with position details
            import json
            try:
                data = json.loads(opu)
            except (json.JSONDecodeError, TypeError):
                data = {"raw": opu}

            for cb in self._callbacks:
                self._loop.call_soon_threadsafe(
                    self._loop.create_task, cb(data)
                )

        except Exception as e:
            logger.debug(f"Trade update error: {e}")

    def onSubscription(self):
        logger.info("Trade subscription active")

    def onSubscriptionError(self, code, message):
        logger.error(f"Trade subscription error: {code} {message}")

    def onUnsubscription(self):
        logger.info("Trade subscription ended")


# ── Stream Manager ────────────────────────────────────────────────────────

class IGStreamManager:
    """Manages all Lightstreamer subscriptions for the shared IG session."""

    def __init__(self, session: IGSharedSession):
        self._session = session
        self._loop: asyncio.AbstractEventLoop | None = None

        # Latest tick prices per epic: epic -> mid price
        self._prices: dict[str, float] = {}
        self._price_events: dict[str, asyncio.Event] = {}
        self._tick_callbacks: dict[str, list[Callable]] = {}

        # 5-min candle accumulator per epic
        self._candle_bars: dict[str, deque] = {}
        self._candle_callbacks: dict[str, list[Callable]] = {}

        # Trade event callbacks
        self._trade_callbacks: list[Callable] = []

        # Active subscriptions for cleanup/resubscribe
        self._tick_subs: dict[str, Subscription] = {}
        self._candle_subs: dict[str, Subscription] = {}
        self._trade_sub: Subscription | None = None

    # ── Tick streaming ─────────────────────────────────────────────

    async def subscribe_ticks(self, epic: str):
        """Subscribe to real-time tick prices for an epic."""
        if not self._session.stream or not self._session.stream.ls_client:
            logger.warning(f"Cannot subscribe ticks for {epic} — no stream connection")
            return

        self._loop = asyncio.get_event_loop()
        self._price_events[epic] = asyncio.Event()

        sub = Subscription(
            mode="DISTINCT",
            items=[f"CHART:{epic}:TICK"],
            fields=["BID", "OFR", "LTP", "UTM"],
        )
        listener = _TickListener(
            epic, self._prices, self._price_events, self._tick_callbacks, self._loop
        )
        sub.addListener(listener)
        self._session.stream.subscribe(sub)
        self._tick_subs[epic] = sub

    async def get_price(self, epic: str) -> float | None:
        """
        Get latest price for an epic. Returns cached streaming price
        instantly. Falls back to REST if no tick received yet.
        """
        cached = self._prices.get(epic)
        if cached is not None:
            return cached

        # Fallback: one-off REST call
        try:
            market = await self._session.rest_call(
                self._session.ig.fetch_market_by_epic, epic
            )
            snapshot = market.get("snapshot", {})
            bid = snapshot.get("bid")
            offer = snapshot.get("offer")
            if bid is not None and offer is not None:
                price = round((bid + offer) / 2, 1)
                self._prices[epic] = price
                return price
        except Exception as e:
            logger.error(f"REST price fallback failed for {epic}: {e}")
        return None

    def get_price_sync(self, epic: str) -> float | None:
        """Non-async price lookup (for use in sync callbacks)."""
        return self._prices.get(epic)

    def get_tick_age(self, epic: str) -> float:
        """Seconds since last tick for this epic. Returns inf if no tick."""
        ts = self._prices.get(f"{epic}_time")
        if ts is None:
            return float("inf")
        return time.time() - ts

    # ── 5-min candle streaming ─────────────────────────────────────

    async def subscribe_candles(self, epic: str):
        """Subscribe to 5-minute candle completions for an epic."""
        if not self._session.stream or not self._session.stream.ls_client:
            logger.warning(f"Cannot subscribe candles for {epic} — no stream")
            return

        self._loop = asyncio.get_event_loop()

        sub = Subscription(
            mode="MERGE",
            items=[f"CHART:{epic}:5MINUTE"],
            fields=[
                "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
                "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE",
                "CONS_END", "UTM",
            ],
        )
        listener = _CandleListener(
            epic, self._candle_bars, self._candle_callbacks, self._loop
        )
        sub.addListener(listener)
        self._session.stream.subscribe(sub)
        self._candle_subs[epic] = sub

    def register_candle_callback(self, epic: str, callback: Callable):
        """Register async callback for completed 5-min candles."""
        self._candle_callbacks.setdefault(epic, []).append(callback)

    def register_tick_callback(self, epic: str, callback: Callable):
        """Register sync callback for every tick (runs on event loop thread via call_soon_threadsafe)."""
        self._tick_callbacks.setdefault(epic, []).append(callback)

    def get_today_bars_df(self, epic: str) -> pd.DataFrame:
        """Convert today's streaming candles to a DataFrame matching REST format.

        Returns DataFrame with CET DatetimeIndex and O/H/L/C columns,
        compatible with strategy.calculate_levels().
        """
        bars = self._candle_bars.get(epic, [])
        if not bars:
            return pd.DataFrame()

        today = datetime.now(CET).date()
        today_bars = [b for b in bars if b["time"].date() == today]

        if not today_bars:
            return pd.DataFrame()

        records = []
        times = []
        for b in today_bars:
            times.append(b["time"])
            records.append({
                "Open": b["Open"], "High": b["High"],
                "Low": b["Low"], "Close": b["Close"],
            })

        df = pd.DataFrame(records, index=pd.DatetimeIndex(times))
        # Ensure CET timezone on index
        if df.index.tz is None:
            df.index = df.index.tz_localize(CET)
        return df.sort_index()

    def get_bar_count_today(self, epic: str) -> int:
        """How many completed 5-min bars today."""
        bars = self._candle_bars.get(epic, [])
        today = datetime.now(CET).date()
        return sum(1 for b in bars if b["time"].date() == today)

    # ── Trade event streaming ──────────────────────────────────────

    async def subscribe_trades(self, account_id: str):
        """Subscribe to TRADE events (fills, position updates)."""
        if not self._session.stream or not self._session.stream.ls_client:
            logger.warning("Cannot subscribe trades — no stream connection")
            return

        self._loop = asyncio.get_event_loop()

        sub = Subscription(
            mode="DISTINCT",
            items=[f"TRADE:{account_id}"],
            fields=["OPU", "CONFIRMS", "WOU"],
        )
        listener = _TradeListener(self._trade_callbacks, self._loop)
        sub.addListener(listener)
        self._session.stream.subscribe(sub)
        self._trade_sub = sub

    def register_trade_callback(self, callback: Callable):
        """Register async callback for trade/fill events."""
        self._trade_callbacks.append(callback)

    # ── Resubscribe (after reconnect) ──────────────────────────────

    async def resubscribe_all(self):
        """Re-create all subscriptions after a session reconnect."""
        epics_tick = list(self._tick_subs.keys())
        epics_candle = list(self._candle_subs.keys())
        acc = self._session.acc_number

        self._tick_subs.clear()
        self._candle_subs.clear()
        self._trade_sub = None

        for epic in epics_tick:
            await self.subscribe_ticks(epic)
        for epic in epics_candle:
            await self.subscribe_candles(epic)
        if acc:
            await self.subscribe_trades(acc)

        logger.info(f"Resubscribed: {len(epics_tick)} tick, "
                     f"{len(epics_candle)} candle, trades={bool(acc)}")

    # ── Cleanup ────────────────────────────────────────────────────

    async def unsubscribe_all(self):
        """Unsubscribe everything before shutdown."""
        if self._session.stream:
            try:
                self._session.stream.unsubscribe_all()
            except Exception:
                pass
        self._tick_subs.clear()
        self._candle_subs.clear()
        self._trade_sub = None
