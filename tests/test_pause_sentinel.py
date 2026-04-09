"""Tests for pause sentinel — proves is_paused() reads the file on every call.

Regression guard for today's bug: stale in-memory `paused` global shadowing
the file state when the operator rm'd the sentinel externally.
"""
import os
import pytest


def test_is_paused_false_when_no_sentinel(tmp_runtime_dir):
    import telegram_cmd
    telegram_cmd.PAUSE_SENTINEL = str(tmp_runtime_dir / "asrs-bot.paused")
    assert telegram_cmd.is_paused() is False


def test_is_paused_true_when_sentinel_exists(tmp_runtime_dir):
    import telegram_cmd
    sentinel = tmp_runtime_dir / "asrs-bot.paused"
    telegram_cmd.PAUSE_SENTINEL = str(sentinel)
    sentinel.touch()
    assert telegram_cmd.is_paused() is True


def test_is_paused_responds_to_external_rm(tmp_runtime_dir):
    """CRITICAL: the bug we fixed today. Touch the file, verify paused,
    rm the file externally, verify NOT paused without any function call
    that would reset an in-memory cache."""
    import telegram_cmd
    sentinel = tmp_runtime_dir / "asrs-bot.paused"
    telegram_cmd.PAUSE_SENTINEL = str(sentinel)

    sentinel.touch()
    assert telegram_cmd.is_paused() is True

    sentinel.unlink()  # external rm, no Python function call
    assert telegram_cmd.is_paused() is False  # must read file, not cache


def test_is_paused_responds_to_external_touch(tmp_runtime_dir):
    """The reverse — external touch must pause immediately."""
    import telegram_cmd
    sentinel = tmp_runtime_dir / "asrs-bot.paused"
    telegram_cmd.PAUSE_SENTINEL = str(sentinel)

    assert telegram_cmd.is_paused() is False
    sentinel.touch()
    assert telegram_cmd.is_paused() is True


def test_set_paused_writes_file(tmp_runtime_dir):
    import telegram_cmd
    sentinel = tmp_runtime_dir / "asrs-bot.paused"
    telegram_cmd.PAUSE_SENTINEL = str(sentinel)
    telegram_cmd._set_paused(True)
    assert sentinel.exists()
    assert telegram_cmd.is_paused() is True


def test_set_paused_false_removes_file(tmp_runtime_dir):
    import telegram_cmd
    sentinel = tmp_runtime_dir / "asrs-bot.paused"
    telegram_cmd.PAUSE_SENTINEL = str(sentinel)
    sentinel.touch()
    telegram_cmd._set_paused(False)
    assert not sentinel.exists()
    assert telegram_cmd.is_paused() is False


def test_no_duplicate_is_paused_definition():
    """Regression: we accidentally had two is_paused() defs in the file.
    The second one overrode the first and broke the sentinel logic.
    This test ensures only one definition exists."""
    import ast
    with open(os.path.join(os.path.dirname(__file__), "..", "telegram_cmd.py")) as f:
        tree = ast.parse(f.read())
    is_paused_defs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "is_paused"
    ]
    assert len(is_paused_defs) == 1, \
        f"Expected 1 is_paused() definition, found {len(is_paused_defs)}"
