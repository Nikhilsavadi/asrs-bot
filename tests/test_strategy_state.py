"""Tests for asrs/strategy.py state machine and atomic state writes."""
import json
import os
import pytest
from unittest.mock import MagicMock

from asrs.strategy import Signal, SignalState, Phase


def _mk_signal_with_state_dir(tmp_path, instrument="DAX", session=1):
    mock_broker = MagicMock()
    mock_broker.register_stop_callback = MagicMock()
    mock_stream = MagicMock()
    s = Signal(instrument, session, mock_broker, mock_stream, MagicMock())
    s._state_dir = str(tmp_path)
    return s


# ─────────────────────────────────────────────────────────────────────
# SignalState defaults
# ─────────────────────────────────────────────────────────────────────

def test_signal_state_defaults_are_idle():
    s = SignalState()
    assert s.phase == Phase.IDLE
    assert s.entries_used == 0
    assert s.bar_high == 0.0
    assert s.bar_low == 0.0
    assert s.trades == []


def test_signal_state_serializes_to_json_cleanly():
    s = SignalState()
    s.phase = Phase.LONG
    s.entries_used = 1
    s.entry_price = 24050.0
    d = s.to_dict()
    j = json.dumps(d)
    loaded = json.loads(j)
    assert loaded["phase"] == "LONG"
    assert loaded["entries_used"] == 1


# ─────────────────────────────────────────────────────────────────────
# Atomic state writes (today's pattern: tmp + fsync + rename)
# ─────────────────────────────────────────────────────────────────────

def test_save_state_atomic_no_partial_file(tmp_path):
    """State write must be atomic — no partial .tmp file left behind after save."""
    sig = _mk_signal_with_state_dir(tmp_path, "DAX", 1)
    sig.state.phase = Phase.LEVELS_SET
    sig.state.date = "2026-04-09"
    sig.state.bar_high = 24100.0
    sig.state.bar_low = 24050.0
    sig.save_state()

    expected = tmp_path / "DAX_S1.json"
    assert expected.exists()
    # .tmp file must NOT persist after successful save
    tmp_file = tmp_path / "DAX_S1.json.tmp"
    assert not tmp_file.exists()


def test_save_state_roundtrip(tmp_path):
    """Save then load → same values."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    sig = _mk_signal_with_state_dir(tmp_path, "US30", 2)
    sig.state.phase = Phase.LONG
    sig.state.date = today
    sig.state.entries_used = 2
    sig.state.entry_price = 48060.0
    sig.state.trailing_stop = 48020.0
    sig.save_state()

    # New signal instance loading the same file
    sig2 = _mk_signal_with_state_dir(tmp_path, "US30", 2)
    sig2.load_state()
    assert sig2.state.phase == Phase.LONG
    assert sig2.state.entries_used == 2
    assert sig2.state.entry_price == 48060.0
    assert sig2.state.trailing_stop == 48020.0


def test_load_state_resets_if_stale_date(tmp_path):
    """Loading a state file from yesterday should reset to IDLE."""
    sig = _mk_signal_with_state_dir(tmp_path, "DAX", 1)
    # Write a state file with a stale date
    state = SignalState()
    state.date = "2020-01-01"
    state.phase = Phase.LONG
    state.entries_used = 3
    state_file = tmp_path / "DAX_S1.json"
    with open(state_file, "w") as f:
        json.dump(state.to_dict(), f)

    sig.load_state()
    # Should have reset (today's date != 2020-01-01)
    assert sig.state.phase == Phase.IDLE
    assert sig.state.entries_used == 0
