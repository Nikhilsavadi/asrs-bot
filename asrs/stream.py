"""
stream.py -- Tick-bar builder + candle callbacks
================================================
Re-exports shared/ig_stream.py functionality with a clean interface.
The actual streaming implementation lives in shared/ig_stream.py
(Lightstreamer listeners, tick-bar builder, CONS_END dedup).

This module provides convenience wrappers used by main.py to wire up
candle callbacks per signal.
"""

import logging
from shared.ig_stream import IGStreamManager

logger = logging.getLogger(__name__)


def register_bar_callback(stream: IGStreamManager, epic: str, callback):
    """
    Register a candle-complete callback for an epic.
    Fires when either:
      - tick-bar builder emits a completed 5-min bar (primary, on system clock)
      - CONS_END from IG signals bar completion (backup, deduped)

    callback signature: async def on_bar(bar: dict)
        bar = {"time": datetime, "Open": float, "High": float, "Low": float, "Close": float}
    """
    stream.register_candle_callback(epic, callback)
    logger.info(f"Bar callback registered for {epic}")
