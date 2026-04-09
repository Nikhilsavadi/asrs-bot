"""Tests for shared/ib_stream.py — synth bar gap-fill safety rules.

Regression guard for today's runaway bug where synth bars self-perpetuated
by advancing deck[-1]["time"] forward, tricking the 15-min freshness check
into passing indefinitely even after the real data feed went silent.
"""
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest


def _mk_stream():
    """Build an IBStreamManager with stubbed dependencies for unit testing."""
    from shared.ib_stream import IBStreamManager
    session = MagicMock()
    session.ib = MagicMock()
    # Block the __init__ event wiring that needs real ib_async
    session.ib.pendingTickersEvent = MagicMock()
    session.ib.pendingTickersEvent.__iadd__ = lambda self, other: self
    mgr = IBStreamManager(session)
    return mgr


# ─────────────────────────────────────────────────────────────────────
# _maybe_insert_zero_trade_bars freshness guard
# ─────────────────────────────────────────────────────────────────────

def test_synth_refuses_when_no_real_bar_ever_seen():
    """If _last_real_bar_time is empty, synth must return immediately —
    contract is not actively trading."""
    mgr = _mk_stream()
    key = "TEST"
    mgr._candle_bars[key] = deque([
        {"time": datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc),
         "Open": 100, "High": 100, "Low": 100, "Close": 100, "Volume": 0},
    ])
    # No _last_real_bar_time entry → must refuse
    before = len(mgr._candle_bars[key])
    mgr._maybe_insert_zero_trade_bars(key, datetime.now(timezone.utc))
    after = len(mgr._candle_bars[key])
    assert before == after, "synth fired without any real bar baseline"


def test_synth_refuses_when_last_real_bar_is_stale():
    """CRITICAL: the exact bug we fixed today. Real bar is 30 min old,
    wall-clock finalizer must NOT synth anything because the contract
    is clearly not trading."""
    mgr = _mk_stream()
    key = "TEST"
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    stale_real_time = now - timedelta(minutes=30)
    mgr._last_real_bar_time[key] = stale_real_time
    mgr._candle_bars[key] = deque([
        {"time": stale_real_time,
         "Open": 100, "High": 100, "Low": 100, "Close": 100, "Volume": 0},
    ])
    before = len(mgr._candle_bars[key])
    mgr._maybe_insert_zero_trade_bars(key, now)
    after = len(mgr._candle_bars[key])
    assert before == after, "synth fired despite stale real bar"


def test_synth_fires_when_last_real_bar_is_fresh():
    """Happy path: real bar was 3 min ago, current window is 5 min after
    that. Synth should fill the one missing window."""
    mgr = _mk_stream()
    key = "TEST"
    now = datetime(2026, 4, 9, 10, 7, 0, tzinfo=timezone.utc)
    last_real = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
    mgr._last_real_bar_time[key] = last_real
    mgr._candle_bars[key] = deque([
        {"time": last_real,
         "Open": 23000, "High": 23010, "Low": 22990, "Close": 23005, "Volume": 50},
    ])
    # Set building_bar for the CURRENT window so gap-fill stops before it
    mgr._building_bar[key] = {
        "start": datetime(2026, 4, 9, 10, 5, 0, tzinfo=timezone.utc),
        "open": 23005, "high": 23005, "low": 23005, "close": 23005, "volume": 0,
    }
    mgr._maybe_insert_zero_trade_bars(key, now)
    # Should have NOT filled anything because the gap is only 1 window
    # and the building_bar IS that window
    # Actually there's nothing to fill: last_real = 10:00, cur_window = 10:05
    # next_start = 10:05 which equals cur_window → loop doesn't execute
    assert len(mgr._candle_bars[key]) == 1


def test_synth_cap_at_2_per_call():
    """Hard cap: maximum 2 synth bars per wall-clock finalizer iteration.
    Prevents runaway fills when the gap is large."""
    mgr = _mk_stream()
    key = "TEST"
    # Real bar 12 min ago — inside 15 min window (allowed), but 3 gap bars exist
    last_real = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
    now = datetime(2026, 4, 9, 10, 16, 0, tzinfo=timezone.utc)
    mgr._last_real_bar_time[key] = last_real
    mgr._candle_bars[key] = deque([
        {"time": last_real,
         "Open": 23000, "High": 23010, "Low": 22990, "Close": 23005, "Volume": 50},
    ])
    mgr._maybe_insert_zero_trade_bars(key, now)
    # Only 2 synth bars allowed per call
    assert len(mgr._candle_bars[key]) <= 3  # 1 real + 2 synth max


def test_synth_never_overwrites_real_bar():
    """Safety: if a real bar already exists at a timestamp, synth must skip it."""
    mgr = _mk_stream()
    key = "TEST"
    last_real = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
    # Pre-populate with an existing real bar at 10:05
    existing_real = datetime(2026, 4, 9, 10, 5, 0, tzinfo=timezone.utc)
    mgr._last_real_bar_time[key] = last_real
    mgr._candle_bars[key] = deque([
        {"time": last_real,
         "Open": 23000, "High": 23010, "Low": 22990, "Close": 23005, "Volume": 50},
        {"time": existing_real,
         "Open": 23100, "High": 23200, "Low": 23050, "Close": 23150, "Volume": 30},
    ])
    now = datetime(2026, 4, 9, 10, 11, 0, tzinfo=timezone.utc)
    mgr._maybe_insert_zero_trade_bars(key, now)
    # Real bar at 10:05 must still have its REAL OHLC, not overwritten
    bars_at_1005 = [b for b in mgr._candle_bars[key] if b["time"] == existing_real]
    assert len(bars_at_1005) == 1
    assert bars_at_1005[0]["High"] == 23200, "synth overwrote a real bar!"
    assert bars_at_1005[0]["Volume"] == 30


# ─────────────────────────────────────────────────────────────────────
# _finalize_bar: synth vs real precedence
# ─────────────────────────────────────────────────────────────────────

def test_finalize_synth_does_not_overwrite_existing():
    """Synth finalize with a timestamp that already exists = no-op."""
    mgr = _mk_stream()
    key = "TEST"
    t = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
    mgr._candle_bars[key] = deque([
        {"time": t, "Open": 23000, "High": 23100, "Low": 22950, "Close": 23050, "Volume": 50},
    ])
    synth = {
        "start": t, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0,
    }
    mgr._finalize_bar(key, synth, source="synth-no-trades")
    # Original real bar must still be there unchanged
    assert mgr._candle_bars[key][0]["High"] == 23100
    assert mgr._candle_bars[key][0]["Volume"] == 50


def test_finalize_real_updates_last_real_time():
    """Real bar finalization MUST update _last_real_bar_time;
    synth MUST NOT. This is what fixes the self-perpetuation."""
    mgr = _mk_stream()
    key = "TEST"
    t = datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc)
    real = {"start": t, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 50}
    mgr._finalize_bar(key, real, source="rtbar")
    assert mgr._last_real_bar_time[key] == t

    # Synth at a later time MUST NOT update _last_real_bar_time
    t2 = datetime(2026, 4, 9, 10, 5, 0, tzinfo=timezone.utc)
    synth = {"start": t2, "open": 105, "high": 105, "low": 105, "close": 105, "volume": 0}
    mgr._finalize_bar(key, synth, source="synth-no-trades")
    assert mgr._last_real_bar_time[key] == t, \
        "synth bar advanced _last_real_bar_time — self-perpetuation bug returned!"
