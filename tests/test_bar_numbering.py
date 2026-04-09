"""Tests for bar number logic in asrs/strategy.py.

Critical: these tests catch session-relative timing bugs that would
cause the strategy to fire on the wrong bar, which is how today's
NIKKEI bar4/bar5 race bug manifested.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock
import pandas as pd
import pytest

from asrs.strategy import Signal
import asrs.config as config


def _mk_signal(instrument: str, session: int):
    """Create a Signal instance without broker/stream for pure math tests."""
    mock_broker = MagicMock()
    mock_broker.register_stop_callback = MagicMock()
    mock_stream = MagicMock()
    mock_alert = MagicMock()
    return Signal(instrument, session, mock_broker, mock_stream, mock_alert)


# ─────────────────────────────────────────────────────────────────────
# _bar_number — session-relative bar index
# ─────────────────────────────────────────────────────────────────────

def test_bar_number_dax_s1_session_open():
    """DAX_S1 opens 09:00 CET. Bar at 09:00 CET = bar 1."""
    s = _mk_signal("DAX", 1)
    cet = ZoneInfo("Europe/Berlin")
    bar_time = datetime(2026, 4, 9, 9, 0, tzinfo=cet)
    assert s._bar_number(bar_time) == 1


def test_bar_number_dax_s1_bar_4():
    """Bar at 09:15 CET = bar 4 (09:00 + 15min)."""
    s = _mk_signal("DAX", 1)
    cet = ZoneInfo("Europe/Berlin")
    bar_time = datetime(2026, 4, 9, 9, 15, tzinfo=cet)
    assert s._bar_number(bar_time) == 4


def test_bar_number_dax_s1_before_open_returns_negative():
    """Bars before session open should return -1 (skip)."""
    s = _mk_signal("DAX", 1)
    cet = ZoneInfo("Europe/Berlin")
    bar_time = datetime(2026, 4, 9, 8, 55, tzinfo=cet)
    assert s._bar_number(bar_time) == -1


def test_bar_number_nikkei_s1_tz_conversion():
    """NIKKEI_S1 opens 10:00 JST. UTC bar at 01:00 converts → JST 10:00 → bar 1."""
    s = _mk_signal("NIKKEI", 1)
    utc = ZoneInfo("UTC")
    bar_time = datetime(2026, 4, 9, 1, 0, tzinfo=utc)
    assert s._bar_number(bar_time) == 1


def test_bar_number_us30_s2_bar_4():
    """US30_S2 opens 11:00 ET. Bar at 11:15 ET = bar 4."""
    s = _mk_signal("US30", 2)
    et = ZoneInfo("America/New_York")
    bar_time = datetime(2026, 4, 9, 11, 15, tzinfo=et)
    assert s._bar_number(bar_time) == 4


def test_bar_number_us30_s3_session_boundary():
    """US30_S3 opens 13:00 ET. Bar at 13:00 = bar 1, 13:05 = bar 2."""
    s = _mk_signal("US30", 3)
    et = ZoneInfo("America/New_York")
    assert s._bar_number(datetime(2026, 4, 9, 13, 0, tzinfo=et)) == 1
    assert s._bar_number(datetime(2026, 4, 9, 13, 5, tzinfo=et)) == 2
    assert s._bar_number(datetime(2026, 4, 9, 13, 15, tzinfo=et)) == 4


def test_bar_number_evening_bar_for_morning_session():
    """If an evening-session bar arrives (16:00+ JST) for NIKKEI_S1
    (which opened 10:00 JST), it should compute some bar number but
    it won't be bar 1/4/5 so strategy ignores it. Critical: doesn't
    crash and doesn't return anything that'd trigger morning_routine."""
    s = _mk_signal("NIKKEI", 1)
    jst = ZoneInfo("Asia/Tokyo")
    bar_time = datetime(2026, 4, 9, 16, 0, tzinfo=jst)
    bn = s._bar_number(bar_time)
    # 6h × 60min / 5min = 72 + 1 = 73
    assert bn == 73
    assert bn not in (1, 4, 5)  # strategy won't trigger on this


# ─────────────────────────────────────────────────────────────────────
# _find_bar — deque iteration + time matching
# ─────────────────────────────────────────────────────────────────────

def test_find_bar_matches_by_session_time_not_deque_position():
    """Critical: _find_bar MUST match by hour/minute, not by index.
    This is why today's bar4/bar5 fix works even with synth bars in the deque."""
    s = _mk_signal("DAX", 1)
    cet = ZoneInfo("Europe/Berlin")
    today = datetime.now(cet).date()

    # Build a DataFrame with many bars, one of which is at 09:15 CET
    # (the "bar 4" timestamp for DAX_S1 opening at 09:00). Put the bar 4
    # row in the MIDDLE of the deque to prove lookup is time-based, not
    # index-based.
    base_times = [
        datetime(today.year, today.month, today.day, 7, 0, tzinfo=cet),
        datetime(today.year, today.month, today.day, 7, 5, tzinfo=cet),
        datetime(today.year, today.month, today.day, 7, 10, tzinfo=cet),
        datetime(today.year, today.month, today.day, 9, 15, tzinfo=cet),  # bar 4 (middle)
        datetime(today.year, today.month, today.day, 10, 0, tzinfo=cet),
        datetime(today.year, today.month, today.day, 10, 5, tzinfo=cet),
    ]
    rows = [
        {"Open": 100, "High": 110, "Low": 90, "Close": 105},
        {"Open": 100, "High": 110, "Low": 90, "Close": 105},
        {"Open": 100, "High": 110, "Low": 90, "Close": 105},
        {"Open": 24000, "High": 24100, "Low": 23950, "Close": 24050},  # bar 4
        {"Open": 100, "High": 110, "Low": 90, "Close": 105},
        {"Open": 100, "High": 110, "Low": 90, "Close": 105},
    ]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(base_times))

    result = s._find_bar(df, 4)
    assert result is not None
    # Bar 4 high/low come from the 09:15 CET row, not index 3
    assert result["High"] == 24100
    assert result["Low"] == 23950
