"""Tests for re-entry re-arm logic in monitor_cycle.

Regression guard for the US30_S2 -£250 bug discovered 2026-04-09:
- Live re-entry only fired when price was strictly BETWEEN buy/sell levels
- Backtest re-enters whenever price touches a level (no slippage modelling)
- On strong continuation moves, live missed all re-entries
- Fix: re-arm when price is within `max_slip` of either level, not just inside

Also guards against the slippage cascade alternative where immediate
re-arm on a runaway price burns all entries to slippage closes.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from asrs.strategy import Signal, Phase


def _mk_signal_with_levels(buy_level, sell_level, bar_range, max_slippage_pct=0.5,
                              buffer=5.0, current_price=None):
    """Build a signal pre-configured with bar 4 levels for re-entry testing."""
    mock_broker = MagicMock()
    mock_broker.register_stop_callback = MagicMock()
    mock_broker.ensure_connected = AsyncMock(return_value=True)
    mock_broker.get_current_price = AsyncMock(return_value=current_price)
    mock_broker.get_position = AsyncMock(return_value={"direction": "FLAT"})
    mock_stream = MagicMock()
    sig = Signal("US30", 2, mock_broker, mock_stream, AsyncMock())
    sig.state.phase = Phase.LEVELS_SET
    sig.state.buy_level = buy_level
    sig.state.sell_level = sell_level
    sig.state.bar_range = bar_range
    sig.state.entries_used = 1
    # Override config slippage params for predictable math
    sig.cfg = dict(sig.cfg)  # don't mutate the global
    sig.cfg["max_slippage_pct"] = max_slippage_pct
    sig.cfg["buffer"] = buffer
    sig.cfg["max_entries"] = 3
    sig._arm_bracket = AsyncMock()  # spy
    # monitor_cycle calls load_state() first which would reset our
    # in-memory state from a non-existent disk file. Stub it out.
    sig.load_state = MagicMock()
    return sig


# ─────────────────────────────────────────────────────────────────────
# Re-arm window math
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rearm_when_price_strictly_inside_levels():
    """Classic case: price between buy_level and sell_level → re-arm."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48050,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_awaited_once()


@pytest.mark.asyncio
async def test_rearm_when_price_just_above_buy_level():
    """Price 5pt above buy_level — within max_slip window → re-arm.
    Bar range 50, buffer 5, max_slip_pct 0.5 → max_slip = 30pt
    rearm_high = 48095 + 30 = 48125
    Price 48100 < 48125 → re-arm."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48100,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_rearm_when_price_far_above_buy_level():
    """Price 90pt above buy_level — outside max_slip window → wait.
    This was the US30_S2 case today: stop fired at 48185, price stayed
    above the re-arm window for 4+ hours, no re-entry possible."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48185,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_rearm_when_price_far_below_sell_level():
    """Symmetric case: price 90pt below sell_level → wait."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=47920,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_not_awaited()


@pytest.mark.asyncio
async def test_rearm_at_exactly_max_slip_above():
    """Edge case: price at exactly buy_level + max_slip - 1 → re-arm.
    rearm_high = 48095 + 30 = 48125, price 48124 < 48125 → re-arm."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48124,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_rearm_at_max_slip_boundary():
    """Edge case: price at exactly buy_level + max_slip → no re-arm
    (strict inequality). Prevents the boundary slippage cascade."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48125,  # exactly buy_level + max_slip
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_rearm_when_no_price_available():
    """Defensive: get_current_price returns None → don't re-arm."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=None,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_rearm_when_phase_is_done():
    """LEVELS_SET only — DONE phase should not re-arm."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48050,
    )
    sig.state.phase = Phase.DONE
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_not_awaited()
