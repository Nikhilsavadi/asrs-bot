"""
ig_session.py — Single shared IG REST + Lightstreamer session
═══════════════════════════════════════════════════════════════

IG only allows ONE concurrent REST session per account. This module
holds a singleton IGService + IGStreamService used by both DAX and
FTSE bots. All REST calls are serialised through an asyncio.Lock
to prevent concurrent session conflicts.
"""

import asyncio
import logging
import os
from datetime import datetime

from trading_ig.rest import IGService
from trading_ig.stream import IGStreamService
import trading_ig.utils
import trading_ig.rest as _tir

logger = logging.getLogger(__name__)

# ── Monkey-patch conv_resol (trading-ig uses pandas offsets that break on >=2.2)
def _conv_resol(resolution):
    _map = {
        "SECOND": "SECOND", "MINUTE": "MINUTE", "MINUTE_2": "MINUTE_2",
        "MINUTE_3": "MINUTE_3", "MINUTE_5": "MINUTE_5", "MINUTE_10": "MINUTE_10",
        "MINUTE_15": "MINUTE_15", "MINUTE_30": "MINUTE_30",
        "HOUR": "HOUR", "HOUR_2": "HOUR_2", "HOUR_3": "HOUR_3", "HOUR_4": "HOUR_4",
        "DAY": "DAY", "WEEK": "WEEK", "MONTH": "MONTH",
    }
    return _map.get(resolution, resolution)

trading_ig.utils.conv_resol = _conv_resol
_tir.conv_resol = _conv_resol


class IGSharedSession:
    """Single IG REST + Lightstreamer session shared by all bots."""

    _instance: "IGSharedSession | None" = None

    def __init__(self):
        self.ig: IGService | None = None
        self.stream: IGStreamService | None = None
        self.connected = False
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_reconnect: list[callable] = []  # callbacks after reconnect

        # Credentials (read once)
        self._username = os.getenv("IG_USERNAME", "")
        self._password = os.getenv("IG_PASSWORD", "")
        self._api_key = os.getenv("IG_API_KEY", "")
        self._acc_number = os.getenv("IG_ACC_NUMBER", "")
        self._demo = os.getenv("IG_DEMO", "true").lower() == "true"

    @classmethod
    def get_instance(cls) -> "IGSharedSession":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def acc_number(self) -> str:
        return self._acc_number

    def on_reconnect(self, callback):
        """Register an async callback to run after session reconnection."""
        self._on_reconnect.append(callback)

    # ── Connection ────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Create ONE REST session + ONE Lightstreamer connection."""
        async with self._lock:
            return await self._connect_inner()

    async def _connect_inner(self) -> bool:
        try:
            self._loop = asyncio.get_event_loop()
            acc_type = "demo" if self._demo else "live"

            self.ig = IGService(
                username=self._username,
                password=self._password,
                api_key=self._api_key,
                acc_type=acc_type,
                acc_number=self._acc_number or None,
            )

            loop = self._loop
            # v3 with OAuth for accounts that support it (needed for Lightstreamer)
            # v2 doesn't work with IG demo Lightstreamer subscriptions
            session_version = "3" if self._acc_number else "2"

            # create_session returns a dict with 'lightstreamerEndpoint'
            ig_session = await loop.run_in_executor(
                None, lambda: self.ig.create_session(version=session_version)
            )

            # Read session tokens (needed for Lightstreamer auth with v3)
            if session_version == "3":
                await loop.run_in_executor(
                    None, lambda: self.ig.read_session(fetch_session_tokens="true")
                )
            logger.info(f"IG REST session created (v{session_version})")

            # Set up Lightstreamer streaming
            self.stream = IGStreamService(self.ig)
            self.stream.acc_number = self._acc_number

            # Build Lightstreamer password from security tokens
            cst = self.ig.session.headers["CST"]
            xst = self.ig.session.headers["X-SECURITY-TOKEN"]

            # LS endpoint comes from the create_session response dict
            ls_endpoint = ig_session.get("lightstreamerEndpoint", "") if ig_session else ""

            from lightstreamer.client import LightstreamerClient
            self.stream.lightstreamerEndpoint = ls_endpoint

            if ls_endpoint:
                self.stream.ls_client = LightstreamerClient(
                    self.stream.lightstreamerEndpoint, None
                )
                self.stream.ls_client.connectionDetails.setUser(self._acc_number)
                self.stream.ls_client.connectionDetails.setPassword(
                    f"CST-{cst}|XST-{xst}"
                )
                self.stream.ls_client.connect()
                logger.info(f"Lightstreamer connected → {self.stream.lightstreamerEndpoint}")
            else:
                logger.warning("No Lightstreamer endpoint — streaming unavailable")

            self.connected = True
            mode = "DEMO" if self._demo else "LIVE"
            logger.info(f"IG shared session connected ({mode})")
            return True

        except Exception as e:
            logger.error(f"IG shared session connect failed: {e}", exc_info=True)
            self.connected = False
            return False

    async def keepalive(self) -> bool:
        """Lightweight session keepalive — fetch accounts to keep token alive.
        Also checks Lightstreamer tick age and triggers resubscribe if stale.
        Run every 10 minutes via scheduler to prevent session expiry.
        """
        # 1. REST keepalive
        async with self._lock:
            if not self.connected or not self.ig:
                logger.warning("Keepalive: session not connected, skipping")
                # Fall through to ensure_connected
            else:
                try:
                    loop = asyncio.get_event_loop()
                    accounts = await loop.run_in_executor(None, self.ig.fetch_accounts)
                    if accounts is not None:
                        logger.info("Keepalive: REST session alive")
                        return True
                except Exception as e:
                    logger.warning(f"Keepalive REST failed: {e}")

        # REST dead — trigger full reconnect (includes stream resubscribe via callbacks)
        logger.warning("Keepalive: session expired, reconnecting...")
        return await self.ensure_connected()

    async def check_stream_health(self, stream_mgr, epic: str) -> bool:
        """Check if Lightstreamer is delivering ticks AND bars. Resubscribe if stale.
        Called by the enhanced keepalive job.
        """
        tick_age = stream_mgr.get_tick_age(epic)
        bar_age = stream_mgr.get_last_bar_age(epic) if hasattr(stream_mgr, 'get_last_bar_age') else 0

        # Ticks flowing AND bars building — healthy
        if tick_age < 120 and bar_age < 330:  # bars within ~1 bar interval + margin
            return True

        # Bars stale but ticks OK — tick-bar builder may be broken, resubscribe
        if tick_age < 120 and bar_age >= 330:
            logger.warning(f"Stream health: ticks OK but bars stale ({bar_age:.0f}s) — resubscribing")

        # Both stale
        elif tick_age >= 120:
            pass  # fall through to existing logic

        if tick_age == float("inf"):
            logger.warning("Stream health: no ticks ever received — resubscribing")
        else:
            logger.warning(f"Stream health: last tick {tick_age:.0f}s ago — resubscribing")

        # Check if Lightstreamer client is still connected
        if self.stream and self.stream.ls_client:
            try:
                status = self.stream.ls_client.getStatus()
                logger.info(f"Lightstreamer status: {status}")
                if "DISCONNECTED" in str(status).upper():
                    logger.warning("Lightstreamer disconnected — full reconnect needed")
                    return await self.ensure_connected()
            except Exception as e:
                logger.warning(f"Lightstreamer status check failed: {e}")
                return await self.ensure_connected()

        # LS client exists but ticks stale — try resubscribe only
        try:
            await stream_mgr.resubscribe_all()
            logger.info("Stream resubscribed after stale tick detection")
            return True
        except Exception as e:
            logger.error(f"Stream resubscribe failed: {e}")
            return await self.ensure_connected()

    async def ensure_connected(self) -> bool:
        """Validate session, reconnect if expired. Thread-safe via lock."""
        async with self._lock:
            if self.connected and self.ig:
                try:
                    loop = asyncio.get_event_loop()
                    accounts = await loop.run_in_executor(None, self.ig.fetch_accounts)
                    if accounts is not None:
                        return True
                except Exception:
                    pass

            logger.warning(f"IG shared session lost — reconnecting... "
                           f"(timestamp: {datetime.now().strftime('%H:%M:%S')})")
            self.connected = False

            # Disconnect Lightstreamer before re-creating session
            if self.stream and self.stream.ls_client:
                try:
                    self.stream.unsubscribe_all()
                    self.stream.ls_client.disconnect()
                except Exception:
                    pass

            for attempt in range(1, 4):
                if await self._connect_inner():
                    logger.info(f"IG shared session reconnected (attempt {attempt})")
                    # Fire reconnect callbacks (e.g. resubscribe streams)
                    for cb in self._on_reconnect:
                        try:
                            await cb()
                        except Exception as e:
                            logger.error(f"Reconnect callback failed: {e}")
                    return True
                delay = attempt * 5
                logger.warning(f"Reconnect attempt {attempt}/3 failed — retrying in {delay}s")
                await asyncio.sleep(delay)

            logger.error("All IG reconnection attempts failed")
            return False

    async def rest_call(self, func, *args, **kwargs):
        """
        Serialised REST call. Acquires lock, runs blocking func in executor,
        retries once on session expiry.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    None, lambda: func(*args, **kwargs)
                )
            except Exception as e:
                err_str = str(e).lower()
                if "session" in err_str or "token" in err_str or "401" in err_str:
                    logger.warning(f"REST call failed (session?): {e} — reconnecting")
                    self.connected = False
                    if await self._connect_inner():
                        return await loop.run_in_executor(
                            None, lambda: func(*args, **kwargs)
                        )
                raise

    async def disconnect(self):
        """Clean shutdown of both REST and streaming."""
        if self.stream:
            try:
                self.stream.disconnect()
            except Exception:
                pass
        if self.ig:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.ig.logout)
            except Exception:
                pass
        self.connected = False
        logger.info("IG shared session disconnected")
