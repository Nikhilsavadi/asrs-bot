"""
ib_stream.py — Streaming layer for IBKR (ticks + 5-min bars)
═══════════════════════════════════════════════════════════════

Mirrors the public interface of shared/ig_stream.py:IGStreamManager
so the rest of the bot stays broker-agnostic.

Subscribes to:
  - reqMktData → ticks (bid / ask / last)            → tick callbacks
  - reqHistoricalDataAsync(keepUpToDate=True) → 5-min bars → candle callbacks

Per-instrument state is keyed by a string (e.g. "MYM_20260619") that
maps internally to a qualified Contract.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
from ib_async import Contract, BarDataList, Ticker, RealTimeBar

from shared.ib_session import IBSharedSession

logger = logging.getLogger(__name__)


def contract_key(contract: Contract) -> str:
    """Stable string key for a qualified IBKR contract."""
    if contract.conId:
        return f"{contract.localSymbol or contract.symbol}_{contract.conId}"
    return f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}"


class IBStreamManager:
    """
    Single streaming manager for all instruments. Holds dicts keyed by
    contract_key, mirroring IGStreamManager's interface so brokers
    can use either backend without code changes.
    """

    def __init__(self, session: IBSharedSession):
        self._session = session
        self.ib = session.ib

        # contract_key → Contract / Ticker / BarDataList
        self._contracts: dict[str, Contract] = {}
        self._tickers: dict[str, Ticker] = {}                 # reqMktData (quotes)
        self._rtbar_lists: dict[str, "RealTimeBarList"] = {}  # reqRealTimeBars (5sec TRADES)
        self._rtbar_received: set[str] = set()                # keys that have received >=1 rtbar
        self._use_mid_fallback: set[str] = set()              # keys with no rtbar permissions → mid-tick bars
        self._bar_lists: dict[str, BarDataList] = {}          # reqHistoricalData (warmup)

        # Last mid + bid/ofr per key. Also "{key}_bid", "{key}_ofr", "{key}_time"
        self._prices: dict[str, float] = {}

        # Completed 5-min bars per key (deque for fast appends, last 500 bars)
        self._candle_bars: dict[str, deque] = {}
        self._last_bar_emit: dict[str, datetime] = {}

        # In-progress 5-min bar built from REAL-TIME 5sec bars (trades-based)
        # Schema: {"start": datetime, "open": f, "high": f, "low": f, "close": f, "volume": int}
        self._building_bar: dict[str, dict] = {}

        # Callbacks: key → list of callables
        self._tick_callbacks: dict[str, list[Callable]] = {}
        self._candle_callbacks: dict[str, list[Callable]] = {}

        # Wire IB events to our dispatch
        self.ib.pendingTickersEvent += self._on_pending_tickers
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Subscribe ───────────────────────────────────────────────────

    async def subscribe_ticks(self, contract: Contract) -> str:
        """
        Subscribe to live data for a qualified contract.

        TWO subscriptions per contract:
          1. reqMktData → quote stream (bid/ask/last) for stop monitoring,
             entry triggers, and tick callbacks. Sampled at ~250ms.
          2. reqRealTimeBars(5sec, TRADES) → 5-second OHLC bars built from
             actual trades. Aggregated to 5-min in code via _on_5sec_bar.
             This replaces midpoint-based bar building which was inaccurate.
             reqRealTimeBars is free with standard L1 (no upgrade needed).

        Returns the contract_key used for callbacks / lookups.
        """
        key = contract_key(contract)
        if key in self._tickers:
            return key

        if not await self._session.ensure_connected():
            logger.error(f"Cannot subscribe ticks for {key} — IB not connected")
            return key

        self._contracts[key] = contract
        self._loop = asyncio.get_event_loop()

        # 1. reqMktData for quote stream (stop monitor, entry trigger, tick callbacks)
        #    genericTickList="233" = RT Volume → guaranteed last-trade price +
        #    last-size + last-time on every print. Without this, ticker.last
        #    can be sparse on thin contracts (NIY especially), which would
        #    silently degrade the last-trade stop-monitor fix.
        try:
            ticker = self.ib.reqMktData(contract, "233", False, False)
            self._tickers[key] = ticker
            logger.info(f"IB quotes subscribed: {key} ({contract.symbol})")
        except Exception as e:
            logger.error(f"reqMktData failed for {key}: {e}", exc_info=True)

        # 2. reqRealTimeBars(5 seconds, TRADES) — accurate trade-based bars.
        #    Each 5-second bar is built from actual trades during that window
        #    by IBKR's matching engine. We aggregate 60 of them into a 5-min bar.
        try:
            # useRTH=False is CRITICAL for futures: NIY (Nikkei Yen on CME)
            # trades through Asian session boundaries; useRTH=True would
            # silently drop bars outside the US RTH window. Same for FDXS
            # overnight Eurex extended hours.
            rtbars = self.ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=False)
            self._rtbar_lists[key] = rtbars
            rtbars.updateEvent += lambda bars, has_new, k=key: self._on_5sec_bar(k, bars, has_new)
            logger.info(f"IB real-time bars (5sec TRADES) subscribed: {key}")
        except Exception as e:
            logger.error(f"reqRealTimeBars failed for {key}: {e}", exc_info=True)

        # Permission probe: if no rtbar arrives within 20 s, fall back to
        # mid-tick bars for this key (CBOT data sub not yet active, etc.).
        async def _probe(k=key):
            await asyncio.sleep(20)
            if k not in self._rtbar_received:
                self._use_mid_fallback.add(k)
                logger.warning(
                    f"No real-time bars for {k} after 20s — falling back to "
                    f"mid-tick bar builder (less accurate; subscribe market "
                    f"data for this exchange to fix)"
                )
        if self._loop:
            self._loop.create_task(_probe())

        return key

    def _on_5sec_bar(self, key: str, bars, has_new_bar: bool) -> None:
        """
        Called when a new 5-second TRADES bar arrives. Aggregate into the
        in-progress 5-min bar. When the 5-min boundary crosses, finalise
        the bar and fire candle callbacks.
        """
        if not bars or not has_new_bar:
            return
        self._rtbar_received.add(key)
        latest = bars[-1]
        # latest.time is the bar END time (5 seconds after start)
        bar_time = latest.time
        if not isinstance(bar_time, datetime):
            bar_time = pd.to_datetime(bar_time, utc=True).to_pydatetime()
        elif bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        # Snap to 5-minute window the 5sec bar belongs to (floor of UTC minute)
        five_min_start = bar_time.replace(second=0, microsecond=0)
        five_min_start = five_min_start.replace(minute=(five_min_start.minute // 5) * 5)

        cur = self._building_bar.get(key)
        o = float(latest.open_); h = float(latest.high); l = float(latest.low); c = float(latest.close)
        v = float(getattr(latest, "volume", 0) or 0)

        if cur is None or cur["start"] != five_min_start:
            # Boundary crossed — finalise previous and start new
            if cur is not None:
                completed = {
                    "time": cur["start"],
                    "Open": cur["open"], "High": cur["high"],
                    "Low": cur["low"], "Close": cur["close"],
                    "Volume": cur["volume"],
                }
                deck = self._candle_bars.setdefault(key, deque(maxlen=500))
                if deck and deck[-1]["time"] == completed["time"]:
                    deck[-1] = completed
                else:
                    deck.append(completed)
                self._last_bar_emit[key] = datetime.now(timezone.utc)
                # Fire candle callbacks
                for cb in self._candle_callbacks.get(key, []):
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            if self._loop:
                                self._loop.create_task(cb(completed))
                        else:
                            cb(completed)
                    except Exception as e:
                        logger.error(f"Candle callback error ({key}): {e}", exc_info=True)
            # Start new in-progress 5-min bar from this 5sec bar
            self._building_bar[key] = {
                "start": five_min_start,
                "open": o, "high": h, "low": l, "close": c, "volume": v,
            }
        else:
            # Same 5-min window — extend OHLC
            if h > cur["high"]:
                cur["high"] = h
            if l < cur["low"]:
                cur["low"] = l
            cur["close"] = c
            cur["volume"] += v

    async def subscribe_candles(
        self, contract: Contract, what: str = "TRADES", use_rth: bool = True,
    ) -> str:
        """
        Pre-load today's 5-min bars from IBKR REST then rely on the
        in-process tick-bar builder for live updates.

        Why no keepUpToDate=True: in production we observed bar update
        events stalling 10+ minutes. The tick aggregator (fed by
        pendingTickersEvent) is more reliable because we control the
        timing — every tick → bar update.
        """
        key = contract_key(contract)
        if key in self._bar_lists:
            return key

        if not await self._session.ensure_connected():
            logger.error(f"Cannot subscribe candles for {key} — IB not connected")
            return key

        self._contracts.setdefault(key, contract)

        try:
            bars: BarDataList = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow=what,
                useRTH=use_rth,
                formatDate=2,
                keepUpToDate=False,        # one-shot warmup; live updates via ticks
            )
            if bars is None:
                logger.error(f"reqHistoricalDataAsync returned None for {key}")
                return key

            self._bar_lists[key] = bars
            self._candle_bars.setdefault(key, deque(maxlen=500))

            # Pre-populate with whatever IBKR has so far
            for b in bars:
                self._record_bar(key, b)

            logger.info(
                f"IB candles pre-loaded: {key} ({len(bars)} bars). "
                f"Live updates via tick aggregator."
            )
        except Exception as e:
            logger.error(f"reqHistoricalDataAsync failed for {key}: {e}", exc_info=True)
        return key

    # ── Tick handling ──────────────────────────────────────────────

    def _on_pending_tickers(self, tickers):
        """Called by IB when one or more subscribed tickers have new data.

        Two ticker types are dispatched here:
          1. Quote tickers (from reqMktData) → update bid/ask/last/mid cache,
             fire tick callbacks for stop monitoring and entry triggers.
          2. Trade-by-trade tickers (from reqTickByTickData) → walk the
             tickByTicks list of new trade prints and feed each one to the
             bar builder. THIS is the accurate source for OHLC bars.
        """
        now_ts = time.time()
        for ticker in tickers:
            # Identify ticker type by membership
            quote_key = None
            tbt_key = None
            for k, t in self._tickers.items():
                if t is ticker:
                    quote_key = k
                    break

            # === QUOTE UPDATES (stop monitor + entry trigger source) ===
            # Bars are built from reqRealTimeBars(5sec, TRADES) — see _on_5sec_bar.
            if quote_key is None:
                continue
            key = quote_key

            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ofr = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            last = ticker.last if ticker.last and ticker.last > 0 else 0.0

            if bid and ofr:
                mid = round((bid + ofr) / 2, 2)
            elif last:
                mid = round(last, 2)
            else:
                continue

            self._prices[key] = mid
            self._prices[f"{key}_bid"] = bid
            self._prices[f"{key}_ofr"] = ofr
            # Note: do NOT overwrite key_last here — tick-by-tick is authoritative
            if not self._prices.get(f"{key}_last"):
                self._prices[f"{key}_last"] = last
            self._prices[f"{key}_time"] = now_ts

            # Mid-tick bar fallback: only used when reqRealTimeBars permission
            # is denied for this key (e.g. CBOT without CBOT data sub).
            if key in self._use_mid_fallback:
                self._feed_tick_to_bar(key, mid)

            # Fire tick callbacks (stop monitor + entry triggers).
            for cb in self._tick_callbacks.get(key, []):
                try:
                    try:
                        cb(mid, bid, ofr, last=last)
                    except TypeError:
                        cb(mid, bid, ofr)
                except Exception as e:
                    logger.error(f"Tick callback error ({key}): {e}", exc_info=True)

    def _feed_tick_to_bar(self, key: str, price: float) -> None:
        """
        Aggregate ticks into 5-min OHLC bars in-process. Replaces the
        unreliable reqHistoricalData(keepUpToDate=True) update path.

        Bar boundary: floor of UTC minute / 5 * 5 (00, 05, 10, 15, ...).
        When a tick arrives in a new 5-min window, finalise the in-progress
        bar (write to deque, fire candle callbacks) and start a new one.
        """
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        bar_start = now.replace(minute=(now.minute // 5) * 5)
        cur = self._building_bar.get(key)

        if cur is None:
            self._building_bar[key] = {
                "start": bar_start,
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1,
            }
            return

        if cur["start"] == bar_start:
            # same window — update OHLC
            if price > cur["high"]:
                cur["high"] = price
            if price < cur["low"]:
                cur["low"] = price
            cur["close"] = price
            cur["volume"] += 1
            return

        # New window — finalise the previous bar and start fresh
        completed = {
            "time": cur["start"],
            "Open": cur["open"], "High": cur["high"],
            "Low": cur["low"], "Close": cur["close"],
            "Volume": cur["volume"],
        }
        deck = self._candle_bars.setdefault(key, deque(maxlen=500))
        # Replace if last bar has same timestamp (e.g. from REST pre-load)
        if deck and deck[-1]["time"] == completed["time"]:
            deck[-1] = completed
        else:
            deck.append(completed)
        self._last_bar_emit[key] = datetime.now(timezone.utc)

        # Fire candle callbacks for the completed bar
        for cb in self._candle_callbacks.get(key, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    if self._loop:
                        self._loop.create_task(cb(completed))
                else:
                    cb(completed)
            except Exception as e:
                logger.error(f"Candle callback error ({key}): {e}", exc_info=True)

        # Start new in-progress bar
        self._building_bar[key] = {
            "start": bar_start,
            "open": price, "high": price, "low": price, "close": price,
            "volume": 1,
        }

    # ── Candle handling ────────────────────────────────────────────

    def _record_bar(self, key: str, bar) -> None:
        """Append a single IB BarData into our deque (idempotent on time)."""
        bar_time = bar.date
        if not isinstance(bar_time, datetime):
            bar_time = pd.to_datetime(bar_time, utc=True).to_pydatetime()
        elif bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        deck = self._candle_bars.setdefault(key, deque(maxlen=500))
        # Dedup: if last bar has same time, replace it (live-updating)
        if deck and deck[-1]["time"] == bar_time:
            deck[-1] = {
                "time": bar_time,
                "Open": float(bar.open),
                "High": float(bar.high),
                "Low": float(bar.low),
                "Close": float(bar.close),
                "Volume": float(getattr(bar, "volume", 0) or 0),
            }
        else:
            deck.append({
                "time": bar_time,
                "Open": float(bar.open),
                "High": float(bar.high),
                "Low": float(bar.low),
                "Close": float(bar.close),
                "Volume": float(getattr(bar, "volume", 0) or 0),
            })
        self._last_bar_emit[key] = datetime.now(timezone.utc)

    def _on_bar_update(self, key: str, bars: BarDataList, has_new_bar: bool) -> None:
        """
        Called by IB whenever the live BarDataList updates.
        has_new_bar=True means a new 5-min bar just completed.
        """
        if not bars:
            return
        # Update / append the latest bar
        latest = bars[-1]
        self._record_bar(key, latest)

        # On new bar completion: fire candle callbacks
        if has_new_bar and len(bars) >= 2:
            completed = bars[-2]  # the newly closed bar (latest is now in-progress)
            self._record_bar(key, completed)
            payload = {
                "time":  completed.date,
                "Open":  float(completed.open),
                "High":  float(completed.high),
                "Low":   float(completed.low),
                "Close": float(completed.close),
            }
            for cb in self._candle_callbacks.get(key, []):
                try:
                    if asyncio.iscoroutinefunction(cb):
                        if self._loop:
                            self._loop.create_task(cb(payload))
                    else:
                        cb(payload)
                except Exception as e:
                    logger.error(f"Candle callback error ({key}): {e}", exc_info=True)

    # ── Public interface (mirrors IGStreamManager) ──────────────────

    def register_tick_callback(self, key: str, callback: Callable) -> None:
        self._tick_callbacks.setdefault(key, []).append(callback)

    def register_candle_callback(self, key: str, callback: Callable) -> None:
        self._candle_callbacks.setdefault(key, []).append(callback)
        logger.info(f"Candle callback registered for {key} (total: {len(self._candle_callbacks[key])})")

    async def get_price(self, key: str) -> float | None:
        cached = self._prices.get(key)
        if cached is not None:
            return cached
        # Fallback: snapshot via reqTickers
        contract = self._contracts.get(key)
        if not contract or not await self._session.ensure_connected():
            return None
        try:
            tickers = await self.ib.reqTickersAsync(contract)
            if tickers and tickers[0]:
                t = tickers[0]
                mid = t.midpoint() or t.last or t.close
                if mid:
                    return round(mid, 2)
        except Exception as e:
            logger.error(f"get_price snapshot failed for {key}: {e}")
        return None

    def get_price_sync(self, key: str) -> float | None:
        return self._prices.get(key)

    def get_tick_age(self, key: str) -> float:
        ts = self._prices.get(f"{key}_time")
        if ts is None:
            return float("inf")
        return time.time() - ts

    def get_today_bars_df(self, key: str) -> pd.DataFrame:
        """5-min bars for today (UTC date) as a DataFrame for the strategy."""
        raw = self._candle_bars.get(key)
        if not raw:
            return pd.DataFrame()
        bars = list(raw)
        today = datetime.now(timezone.utc).date()
        today_bars = [b for b in bars if b["time"].date() == today]
        if not today_bars:
            return pd.DataFrame()

        records = [{
            "Open": b["Open"], "High": b["High"],
            "Low":  b["Low"],  "Close": b["Close"],
        } for b in today_bars]
        idx = pd.DatetimeIndex([b["time"] for b in today_bars])
        df = pd.DataFrame(records, index=idx)
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        return df.sort_index()

    def get_bar_count_today(self, key: str) -> int:
        bars = list(self._candle_bars.get(key, []))
        today = datetime.now(timezone.utc).date()
        return sum(1 for b in bars if b["time"].date() == today)

    def get_last_bar_age(self, key: str) -> float:
        emit = self._last_bar_emit.get(key)
        if emit is None:
            return float("inf")
        return (datetime.now(timezone.utc) - emit).total_seconds()

    # ── Cleanup / resubscribe ───────────────────────────────────────

    async def resubscribe_all(self) -> None:
        """Re-subscribe everything after a reconnect."""
        contracts = list(self._contracts.values())
        self._tickers.clear()
        self._rtbar_lists.clear()
        self._bar_lists.clear()
        for c in contracts:
            await self.subscribe_ticks(c)
            await self.subscribe_candles(c)
        logger.info(f"IB stream resubscribed {len(contracts)} contracts")

    async def cancel(self, key: str) -> None:
        """Cancel both ticks and candles for a contract."""
        contract = self._contracts.get(key)
        if not contract:
            return
        if key in self._tickers:
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass
            self._tickers.pop(key, None)
        if key in self._rtbar_lists:
            try:
                self.ib.cancelRealTimeBars(self._rtbar_lists[key])
            except Exception:
                pass
            self._rtbar_lists.pop(key, None)
        if key in self._bar_lists:
            try:
                self.ib.cancelHistoricalData(self._bar_lists[key])
            except Exception:
                pass
            self._bar_lists.pop(key, None)
