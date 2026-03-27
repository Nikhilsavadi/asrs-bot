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
                 loop: asyncio.AbstractEventLoop,
                 tick_bars: dict = None, candle_bars: dict = None,
                 candle_callbacks: dict = None):
        self._epic = epic
        self._prices = prices
        self._events = events
        self._tick_callbacks = tick_callbacks
        self._loop = loop
        self._tick_bars = tick_bars or {}      # shared tick-bar accumulator
        self._candle_bars = candle_bars or {}  # shared bar store (same as CONS_END)
        self._candle_callbacks = candle_callbacks or {}  # fire on tick-bar complete

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
                for cb in self._tick_callbacks.get(self._epic, []):
                    self._loop.call_soon_threadsafe(cb, mid, float(bid), float(ofr))

                # ── Tick-based bar builder ─────────────────────────────
                # Build 5-min OHLC bars in real-time from ticks
                # Bars emit on our clock, not IG's CONS_END delay
                self._update_tick_bar(mid)

        except Exception as e:
            logger.debug(f"Tick update error ({self._epic}): {e}")

    def _update_tick_bar(self, mid: float):
        """Accumulate tick into current 5-min bar. Emit when clock crosses boundary."""
        now = datetime.now(CET)
        bar_min = (now.minute // 5) * 5
        bar_start = now.replace(minute=bar_min, second=0, microsecond=0)

        current = self._tick_bars.get(self._epic)

        if current is None or current["time"] != bar_start:
            # New bar period — emit the old bar (if exists) and start fresh
            if current is not None and current["tick_count"] > 0:
                # Emit completed bar
                completed = {
                    "time": current["time"],
                    "Open": current["Open"],
                    "High": current["High"],
                    "Low": current["Low"],
                    "Close": current["Close"],
                }
                end_time = current["time"] + timedelta(minutes=5)

                # Store in same bar store as CONS_END bars (dedup by time)
                epic_bars = self._candle_bars.setdefault(self._epic, deque(maxlen=300))
                # Only add if not already there from CONS_END
                already = any(b["time"] == completed["time"] for b in epic_bars)
                if not already:
                    epic_bars.append(completed)
                    logger.info(
                        f"Tick-bar ({self._epic}): {completed['time'].strftime('%H:%M')}-"
                        f"{end_time.strftime('%H:%M')} CET | "
                        f"O={completed['Open']} H={completed['High']} "
                        f"L={completed['Low']} C={completed['Close']}"
                    )
                    # Fire candle callbacks so bots get tick-bar completions
                    for cb in self._candle_callbacks.get(self._epic, []):
                        self._loop.call_soon_threadsafe(
                            self._loop.create_task, cb(completed)
                        )

            # Start new bar
            self._tick_bars[self._epic] = {
                "time": bar_start,
                "Open": mid,
                "High": mid,
                "Low": mid,
                "Close": mid,
                "tick_count": 1,
            }
        else:
            # Update current bar
            current["High"] = max(current["High"], mid)
            current["Low"] = min(current["Low"], mid)
            current["Close"] = mid
            current["tick_count"] += 1

    def onSubscription(self):
        logger.info(f"Tick subscription active: {self._epic}")

    def onSubscriptionError(self, code, message):
        logger.error(f"Tick subscription error ({self._epic}): {code} {message}")

    def onUnsubscription(self):
        logger.info(f"Tick subscription ended: {self._epic}")


class _CandleListener(SubscriptionListener):
    """Receives CHART:{epic}:MINUTE_5 updates and stores completed candles."""

    def __init__(self, epic: str, bars: dict, callbacks: dict,
                 loop: asyncio.AbstractEventLoop, dedup_state: dict = None):
        self._epic = epic
        self._bars = bars
        self._callbacks = callbacks
        self._loop = loop
        # BUG #2 FIX: dedup state lives at manager level, survives reconnects
        self._dedup = dedup_state if dedup_state is not None else {}

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
            utm = update.getValue("UTM")

            if not all([bid_o, bid_h, bid_l, bid_c, ofr_o, ofr_h, ofr_l, ofr_c]):
                return

            # BUG #1 FIX: Use UTM (event timestamp) instead of datetime.now()
            # UTM is Unix time in milliseconds from Lightstreamer
            if utm:
                try:
                    event_utc = datetime.fromtimestamp(float(utm) / 1000, tz=CET)
                    m = event_utc.minute
                    # Snap to 5-min boundary: the close time is the next 5-min mark
                    snap_min = ((m // 5) + 1) * 5 if m % 5 != 0 else m
                    if snap_min == m and m % 5 == 0:
                        # Exactly on boundary — this IS the close time
                        snap_min = m
                    snap_hour = event_utc.hour + snap_min // 60
                    snap_min_mod = snap_min % 60
                    end_time = event_utc.replace(hour=snap_hour % 24, minute=snap_min_mod,
                                                  second=0, microsecond=0)
                    start_time = end_time - timedelta(minutes=5)
                except (ValueError, OSError):
                    # Fallback to system time if UTM parsing fails
                    now_cet = datetime.now(CET)
                    snap_min = (now_cet.minute // 5) * 5
                    end_time = now_cet.replace(minute=snap_min, second=0, microsecond=0)
                    start_time = end_time - timedelta(minutes=5)
            else:
                # No UTM available — fall back to system time
                now_cet = datetime.now(CET)
                snap_min = (now_cet.minute // 5) * 5
                end_time = now_cet.replace(minute=snap_min, second=0, microsecond=0)
                start_time = end_time - timedelta(minutes=5)

            # BUG #2 FIX: Dedup using manager-level state (survives reconnects)
            last_bar = self._dedup.get(self._epic)
            if last_bar == start_time:
                return
            self._dedup[self._epic] = start_time

            bar = {
                "time": start_time,
                "Open": round((float(bid_o) + float(ofr_o)) / 2, 1),
                "High": round((float(bid_h) + float(ofr_h)) / 2, 1),
                "Low": round((float(bid_l) + float(ofr_l)) / 2, 1),
                "Close": round((float(bid_c) + float(ofr_c)) / 2, 1),
            }

            # BUG #3 FIX: Bar range sanity check
            bar_range = bar["High"] - bar["Low"]
            if bar_range <= 0 or bar["High"] <= 0:
                logger.warning(f"Invalid bar data ({self._epic}): H={bar['High']} L={bar['Low']}")
                return

            epic_bars = self._bars.setdefault(self._epic, deque(maxlen=300))
            epic_bars.append(bar)

            # Log bar construction with OHLC
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

        # BUG #2 FIX: Dedup state persists across reconnects
        self._candle_dedup: dict[str, datetime] = {}

        # Tick-based bar builder: real-time OHLC from ticks (no CONS_END delay)
        self._tick_bars: dict[str, dict] = {}  # epic -> current accumulating bar
        self._tick_bar_lock: dict[str, bool] = {}  # prevent race conditions

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
            epic, self._prices, self._price_events, self._tick_callbacks, self._loop,
            tick_bars=self._tick_bars, candle_bars=self._candle_bars,
            candle_callbacks=self._candle_callbacks
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
            epic, self._candle_bars, self._candle_callbacks, self._loop,
            dedup_state=self._candle_dedup  # BUG #2 FIX: shared dedup state
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
