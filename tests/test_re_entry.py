"""Tests for re-entry re-arm logic in monitor_cycle.

The re-arm gate requires price to be strictly BETWEEN buy_level and
sell_level. This does TWO jobs:

1. Slippage protection: if price is far above buy_level and we re-arm,
   the tick trigger fires immediately at the runaway price, the
   slippage check kills it, entries_used burns through all 3.

2. Direction-flip protection: after a LONG stop-out, requires price to
   pull back ALL THE WAY to inside the range before allowing a SHORT
   re-entry. Without this, the bot whipsaws on shallow wicks
   (LONG → stop → SHORT → stop → LONG → DONE with ~zero P&L).

Trade-off: on monotonic continuation moves where price never returns
inside the range (US30_S2 on 2026-04-09), live misses re-entries the
backtest captures. Backtest assumes free fills at level (no slippage
modelling), so it overstates this re-entry edge. Real headline PF is
lower than backtest 4.22 by the value of these missed re-entries.

We tried relaxing this to a max_slip window on 2026-04-09 to capture
the missed re-entries — reverted same day after realising it would
trigger direction-flip whipsaws on choppy days, which are MUCH more
common than monotonic continuations. The strict gate is the right
trade.
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
async def test_no_rearm_when_price_just_above_buy_level():
    """Price 5pt above buy_level — STRICT gate prevents re-arm.
    This is the direction-flip protection: requires meaningful pullback
    back inside the range, not just a tick wick. Today's US30_S2 case."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48100,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_not_awaited()


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
async def test_rearm_at_buy_level_minus_1():
    """Edge case: price 1pt below buy_level → re-arm (price IS inside)."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48094,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_awaited_once()


@pytest.mark.asyncio
async def test_rearm_at_exactly_buy_level():
    """Edge case: price exactly AT buy_level → re-arm (touches the level).
    Changed 2026-04-09 from strict < to <= so that pullback right to
    the breakout level (more common than back inside the range) fires
    re-entry. Slippage cost ~1pt, safe."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48095,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_awaited_once()


@pytest.mark.asyncio
async def test_rearm_at_exactly_sell_level():
    """Symmetric: price exactly AT sell_level → re-arm."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48016,
    )
    await sig._monitor_cycle_impl()
    sig._arm_bracket.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_rearm_just_above_buy_level():
    """Price 1pt above buy_level → no re-arm.
    Critical: this is the chop-protection case. If price is at 48096
    with buy_level=48095, we don't re-arm because re-arming would
    fire the tick trigger immediately at slippage and start the
    direction-flip cascade the user warned about."""
    sig = _mk_signal_with_levels(
        buy_level=48095, sell_level=48016, bar_range=50,
        current_price=48096,
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
