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
        self._tickers: dict[str, Ticker] = {}
        self._bar_lists: dict[str, BarDataList] = {}

        # Last mid + bid/ofr per key. Also "{key}_bid", "{key}_ofr", "{key}_time"
        self._prices: dict[str, float] = {}

        # Completed 5-min bars per key (deque for fast appends, last 500 bars)
        self._candle_bars: dict[str, deque] = {}
        self._last_bar_emit: dict[str, datetime] = {}

        # Callbacks: key → list of callables
        self._tick_callbacks: dict[str, list[Callable]] = {}
        self._candle_callbacks: dict[str, list[Callable]] = {}

        # Wire IB events to our dispatch
        self.ib.pendingTickersEvent += self._on_pending_tickers
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Subscribe ───────────────────────────────────────────────────

    async def subscribe_ticks(self, contract: Contract) -> str:
        """
        Subscribe to live ticks for a qualified contract.
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

        try:
            ticker = self.ib.reqMktData(contract, "", False, False)
            self._tickers[key] = ticker
            logger.info(f"IB ticks subscribed: {key} ({contract.symbol})")
        except Exception as e:
            logger.error(f"reqMktData failed for {key}: {e}", exc_info=True)
        return key

    async def subscribe_candles(
        self, contract: Contract, what: str = "TRADES", use_rth: bool = True,
    ) -> str:
        """
        Subscribe to live-updating 5-min historical bars for a contract.
        Uses reqHistoricalDataAsync with keepUpToDate=True so IBKR streams
        bar updates as they form.
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
                formatDate=2,             # UTC
                keepUpToDate=True,
            )
            if bars is None:
                logger.error(f"reqHistoricalDataAsync returned None for {key}")
                return key

            self._bar_lists[key] = bars
            self._candle_bars.setdefault(key, deque(maxlen=500))

            # Pre-populate with whatever IBKR has so far
            for b in bars:
                self._record_bar(key, b)

            # Wire updateEvent so we get notified on every bar update
            bars.updateEvent += lambda bl, has_new_bar: self._on_bar_update(key, bl, has_new_bar)

            logger.info(f"IB candles subscribed: {key} ({len(bars)} initial bars)")
        except Exception as e:
            logger.error(f"reqHistoricalDataAsync failed for {key}: {e}", exc_info=True)
        return key

    # ── Tick handling ──────────────────────────────────────────────

    def _on_pending_tickers(self, tickers):
        """Called by IB when one or more subscribed tickers have new data."""
        now_ts = time.time()
        for ticker in tickers:
            # Find which key this ticker belongs to
            key = None
            for k, t in self._tickers.items():
                if t is ticker:
                    key = k
                    break
            if key is None:
                continue

            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ofr = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            last = ticker.last if ticker.last and ticker.last > 0 else 0.0

            # Mid: prefer bid/ask, fall back to last
            if bid and ofr:
                mid = round((bid + ofr) / 2, 2)
            elif last:
                mid = round(last, 2)
            else:
                continue  # nothing usable

            self._prices[key] = mid
            self._prices[f"{key}_bid"] = bid
            self._prices[f"{key}_ofr"] = ofr
            self._prices[f"{key}_time"] = now_ts

            # Fire tick callbacks
            for cb in self._tick_callbacks.get(key, []):
                try:
                    cb(mid, bid, ofr)
                except Exception as e:
                    logger.error(f"Tick callback error ({key}): {e}", exc_info=True)

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
        if key in self._bar_lists:
            try:
                self.ib.cancelHistoricalData(self._bar_lists[key])
            except Exception:
                pass
            self._bar_lists.pop(key, None)
