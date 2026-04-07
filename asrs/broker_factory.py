"""
broker_factory.py — Pick IG or IBKR broker stack based on BROKER_TYPE env.

Usage in main.py:

    from asrs.broker_factory import get_broker_stack
    stack = get_broker_stack()
    await stack.connect_session()
    broker = stack.make_broker(inst_name, inst_cfg)
    await broker.connect()

Supported values for BROKER_TYPE env var:
    "ig"  — existing IG Markets (default, current production)
    "ib"  — Interactive Brokers (new, via IB Gateway)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class BrokerStack(Protocol):
    """Per-backend factory. Created once per process."""
    kind: str
    shared: Any
    stream: Any

    async def connect_session(self) -> bool: ...
    def make_broker(self, inst_name: str, inst_cfg: dict) -> Any: ...
    async def post_session_setup(self) -> None: ...


class _IGStack:
    kind = "ig"

    def __init__(self):
        from shared.ig_session import IGSharedSession
        from shared.ig_stream import IGStreamManager
        self.shared = IGSharedSession.get_instance()
        self.stream = IGStreamManager(self.shared)

    async def connect_session(self) -> bool:
        ok = await self.shared.connect()
        if ok:
            self.shared.on_reconnect(self.stream.resubscribe_all)
        return ok

    async def post_session_setup(self) -> None:
        """Subscribe to trades stream after session is up."""
        if getattr(self.shared, "acc_number", None):
            await self.stream.subscribe_trades(self.shared.acc_number)

    def make_broker(self, inst_name: str, inst_cfg: dict):
        from asrs.broker import IGBroker
        return IGBroker(
            self.shared,
            self.stream,
            epic=inst_cfg["epic"],
            currency=inst_cfg["currency"],
            disaster_stop_pts=inst_cfg["disaster_stop_pts"],
            max_spread_pts=inst_cfg["max_spread"],
        )


class _IBStack:
    kind = "ib"

    def __init__(self):
        from shared.ib_session import IBSharedSession
        from shared.ib_stream import IBStreamManager
        self.shared = IBSharedSession.get_instance()
        self.stream = IBStreamManager(self.shared)

    async def connect_session(self) -> bool:
        ok = await self.shared.connect()
        if ok:
            self.shared.on_reconnect(self.stream.resubscribe_all)
        return ok

    async def post_session_setup(self) -> None:
        """Nothing to do for IBKR — order/position events are wired per-broker."""
        return None

    def make_broker(self, inst_name: str, inst_cfg: dict):
        from asrs.broker_ib import IBBroker
        return IBBroker(
            self.shared,
            self.stream,
            instrument=inst_name,
            disaster_stop_pts=inst_cfg["disaster_stop_pts"],
            max_spread_pts=inst_cfg["max_spread"],
        )


def get_broker_stack() -> BrokerStack:
    """Return the configured broker stack. Honors BROKER_TYPE env var."""
    kind = os.getenv("BROKER_TYPE", "ig").lower().strip()
    if kind == "ib" or kind == "ibkr":
        logger.info("Broker stack: IBKR (IB Gateway)")
        return _IBStack()
    if kind == "ig":
        logger.info("Broker stack: IG Markets")
        return _IGStack()
    raise ValueError(f"Unknown BROKER_TYPE: {kind!r} (use 'ig' or 'ib')")
