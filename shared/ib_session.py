"""
ib_session.py — Single shared IBKR (IB Gateway / TWS) session
═══════════════════════════════════════════════════════════════

One IB() instance per process. All brokers share it. Connection
managed via clientId + host + port from env. Async-native (ib_async
is built on asyncio, no executor needed).

Equivalent role to shared/ig_session.py but for Interactive Brokers.

Env vars:
    IB_HOST       default 127.0.0.1
    IB_PORT       default 7497  (paper) | 7496 (live)
    IB_CLIENT_ID  default 1
    IB_TIMEOUT    default 15  (seconds)
"""
import asyncio
import logging
import os
from typing import Awaitable, Callable

from ib_async import IB, Contract

logger = logging.getLogger(__name__)


class IBSharedSession:
    """Singleton wrapper around a single ib_async.IB() instance."""

    _instance: "IBSharedSession | None" = None

    def __init__(self):
        self.ib: IB = IB()
        self._host = os.getenv("IB_HOST", "127.0.0.1")
        # IB Gateway: 4002 paper / 4001 live
        # TWS desktop: 7497 paper / 7496 live
        self._port = int(os.getenv("IB_PORT", "4002"))
        self._client_id = int(os.getenv("IB_CLIENT_ID", "1"))
        self._timeout = int(os.getenv("IB_TIMEOUT", "15"))
        self._lock = asyncio.Lock()
        self._on_reconnect: list[Callable[[], Awaitable[None]]] = []
        self._on_disconnect: list[Callable[[], Awaitable[None]]] = []
        # Captured at first connect() — used by sync ib_async event handlers
        # to schedule async callbacks. asyncio.get_event_loop() is deprecated
        # on 3.10+ and raises on 3.12 when called outside a running loop.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Wire IB's connectivity events into our callback dispatchers
        self.ib.connectedEvent += self._fire_reconnect
        self.ib.disconnectedEvent += self._fire_disconnect

        # Track API errors for diagnostics (don't crash on benign warnings)
        self.ib.errorEvent += self._on_ib_error

    @classmethod
    def get_instance(cls) -> "IBSharedSession":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_paper(self) -> bool:
        return self._port in (4002, 7497)

    @property
    def mode(self) -> str:
        return "PAPER" if self.is_paper else "LIVE"

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    # ── Reconnect callbacks ──────────────────────────────────────────

    def on_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register an async callback to fire after every (re)connect."""
        self._on_reconnect.append(callback)

    def on_disconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register an async callback to fire on disconnect."""
        self._on_disconnect.append(callback)

    def _schedule(self, coro):
        """Schedule a coroutine on the captured loop from a sync context."""
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error("No event loop available to schedule callback")
                return
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            loop.create_task(coro)

    def _fire_reconnect(self) -> None:
        """Sync handler that schedules async reconnect callbacks."""
        for cb in self._on_reconnect:
            try:
                self._schedule(cb())
            except Exception as e:
                logger.error(f"on_reconnect callback failed: {e}", exc_info=True)

    def _fire_disconnect(self) -> None:
        for cb in self._on_disconnect:
            try:
                self._schedule(cb())
            except Exception as e:
                logger.error(f"on_disconnect callback failed: {e}", exc_info=True)

    def _on_ib_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract | None = None):
        """Filter and log IB API errors. Most low codes are informational."""
        # Codes 2100-2199 are informational (data farm status, etc)
        if 2100 <= errorCode < 2200:
            logger.debug(f"IB info {errorCode}: {errorString}")
            return
        # 1100/1101/1102 = connectivity events
        if errorCode in (1100, 1101, 1102):
            logger.warning(f"IB connectivity {errorCode}: {errorString}")
            return
        # Real errors
        logger.error(f"IB error {errorCode} (reqId={reqId}): {errorString}")

    # ── Connection ───────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to IB Gateway / TWS. Idempotent."""
        async with self._lock:
            return await self._connect_inner()

    async def _connect_inner(self) -> bool:
        if self.ib.isConnected():
            return True
        # Capture the running loop on first connect — sync IB event handlers
        # use this to schedule async callbacks safely.
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        try:
            await self.ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                timeout=self._timeout,
            )
            logger.info(
                f"IB connected ({self.mode}) "
                f"{self._host}:{self._port} clientId={self._client_id}"
            )
            # Market data type: 1=live, 2=frozen, 3=delayed, 4=delayed-frozen.
            # Default to 3 (delayed) so the bot works without paid subscriptions
            # in paper. Override via IB_MARKET_DATA_TYPE env when subscriptions
            # are active.
            mdt = int(os.getenv("IB_MARKET_DATA_TYPE", "3"))
            self.ib.reqMarketDataType(mdt)
            logger.info(f"Market data type: {mdt} (1=live, 3=delayed)")
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError) as e:
            logger.error(f"IB connect failed: {type(e).__name__}: {e}")
            return False
        except Exception as e:
            logger.error(f"IB connect failed: {e}", exc_info=True)
            return False

    async def disconnect(self) -> None:
        async with self._lock:
            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("IB disconnected")

    async def ensure_connected(self) -> bool:
        """Reconnect if needed. Used by every broker call."""
        if self.ib.isConnected():
            return True

        logger.warning("IB not connected — attempting reconnect")
        for attempt in range(1, 4):
            if await self.connect():
                logger.info(f"IB reconnected on attempt {attempt}")
                return True
            delay = attempt * 5
            logger.warning(f"IB reconnect attempt {attempt}/3 failed — retry in {delay}s")
            await asyncio.sleep(delay)

        logger.error("IB reconnect: all attempts failed")
        return False

    # ── Contract resolution ──────────────────────────────────────────

    async def qualify(self, contract: Contract) -> Contract | None:
        """
        Resolve a Contract spec to its exact specification.
        For Future contracts, this picks the front-month if expiry is unset.
        Returns None if qualification fails.
        """
        if not await self.ensure_connected():
            return None
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                return qualified[0]
            logger.error(f"Failed to qualify contract: {contract}")
            return None
        except Exception as e:
            logger.error(f"qualify failed for {contract}: {e}", exc_info=True)
            return None

    # ── Health / diagnostics ─────────────────────────────────────────

    async def server_time(self) -> float | None:
        """Round-trip ping to verify the connection is alive."""
        if not await self.ensure_connected():
            return None
        try:
            return await self.ib.reqCurrentTimeAsync()
        except Exception as e:
            logger.error(f"server_time failed: {e}")
            return None

    async def fetch_accounts(self) -> list[str]:
        if not await self.ensure_connected():
            return []
        try:
            return self.ib.managedAccounts()
        except Exception as e:
            logger.error(f"fetch_accounts failed: {e}")
            return []

    async def fetch_summary(self, account: str = "") -> dict:
        """Account summary: NetLiquidation, BuyingPower, etc."""
        if not await self.ensure_connected():
            return {}
        try:
            tags = "NetLiquidation,BuyingPower,AvailableFunds,MaintMarginReq,RealizedPnL,UnrealizedPnL"
            rows = await self.ib.accountSummaryAsync(account)
            return {r.tag: r.value for r in rows if r.tag in tags.split(",")}
        except Exception as e:
            logger.error(f"fetch_summary failed: {e}")
            return {}
