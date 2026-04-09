"""Tests for asrs.main._validate_startup_env."""
import os
import pytest


def _clear_env(monkeypatch, *keys):
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def _set_valid_env(monkeypatch):
    """Set a minimal valid environment so tests can override individual vars."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("STARTING_EQUITY_GBP", "5000")
    monkeypatch.setenv("RISK_PCT_PER_TRADE", "0.5")
    monkeypatch.setenv("MAX_CONTRACTS", "10")
    monkeypatch.setenv("DAILY_LOSS_LIMIT_PCT", "3.0")
    monkeypatch.setenv("WEEKLY_LOSS_LIMIT_PCT", "6.0")
    monkeypatch.setenv("MAX_CONCURRENT_POSITIONS", "3")
    monkeypatch.setenv("CONSECUTIVE_LOSS_KILL", "6")
    monkeypatch.setenv("BROKER_TYPE", "ib")
    monkeypatch.setenv("IB_PORT", "4002")
    monkeypatch.setenv("IB_CLIENT_ID", "42")
    monkeypatch.setenv("IB_ADAPTIVE_PRIORITY", "Urgent")


def test_valid_env_passes(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    from asrs.main import _validate_startup_env
    _validate_startup_env()  # should not raise


def test_missing_telegram_token_fails(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()


def test_invalid_risk_pct_fails(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("RISK_PCT_PER_TRADE", "50")  # way too high
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()


def test_zero_max_contracts_fails(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_CONTRACTS", "0")
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()


def test_malformed_numeric_fails(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("STARTING_EQUITY_GBP", "not-a-number")
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()


def test_unknown_broker_type_fails(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("BROKER_TYPE", "robinhood")
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()


def test_invalid_adaptive_priority_fails(monkeypatch, tmp_path):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("IB_ADAPTIVE_PRIORITY", "VeryUrgent")
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()


def test_unwritable_runtime_dir_fails(monkeypatch):
    _set_valid_env(monkeypatch)
    monkeypatch.setenv("ASRS_RUNTIME_DIR", "/proc/impossible/to/write")
    from asrs.main import _validate_startup_env
    with pytest.raises(SystemExit):
        _validate_startup_env()
