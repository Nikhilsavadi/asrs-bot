"""Shared pytest fixtures for asrs-bot test suite."""
import os
import sys
import tempfile
import pytest

# Ensure repo root is on sys.path so `import asrs.strategy` works
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def tmp_runtime_dir(monkeypatch, tmp_path):
    """Isolate ASRS_RUNTIME_DIR for each test — no /tmp pollution."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("ASRS_RUNTIME_DIR", str(runtime))
    return runtime


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Temporary state dir for Signal state file tests."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def clean_pause_sentinel(tmp_runtime_dir):
    """Fresh pause sentinel path + ensure it's absent at start."""
    import telegram_cmd
    # Re-point PAUSE_SENTINEL to the isolated runtime dir
    sentinel = tmp_runtime_dir / "asrs-bot.paused"
    telegram_cmd.PAUSE_SENTINEL = str(sentinel)
    if sentinel.exists():
        sentinel.unlink()
    yield sentinel
    if sentinel.exists():
        sentinel.unlink()
