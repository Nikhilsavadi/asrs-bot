"""
US30 Bot main — S&P 500 ASRS strategy.
Reuses DAX bot logic with US30-specific config.
"""

import logging
import asyncio
from datetime import datetime

# Monkey-patch: make dax_bot code use spx_bot.config instead of dax_bot.config
# This is done by creating US30-specific instances with the right config.
from spx_bot import config
from dax_bot.strategy import (
    DailyState, Phase, calculate_levels, update_trail, process_stop_hit,
    check_add_to_winners, process_add_fill, classify_gap, get_bar,
    analyse_context, should_use_bar5,
)
from dax_bot.broker_ig import IGBroker
import httpx
from dax_bot.overnight import calculate_overnight_range, OvernightBias, OvernightResult

logger = logging.getLogger("US30_ASRS")

# Module-level state (initialized by run_all.py)
broker: IGBroker = None
_bar4_triggered = False
_tg_send = None  # Telegram send function, set by init()


async def _alert(text: str):
    """Send Telegram alert with [S3 US30] prefix."""
    if _tg_send:
        await _tg_send("[S3 US30] " + text)
    else:
        logger.info(f"ALERT (no TG): {text}")

# Override STATE_FILE for US30
import dax_bot.strategy as _strat
_US30_STATE_FILE = config.STATE_FILE


class US30DailyState(DailyState):
    """DailyState that uses US30 state file."""

    def save(self):
        import os, json
        from dataclasses import asdict
        os.makedirs(os.path.dirname(_US30_STATE_FILE), exist_ok=True)
        with open(_US30_STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "US30DailyState":
        import os, json
        today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
        try:
            if os.path.exists(_US30_STATE_FILE):
                with open(_US30_STATE_FILE) as f:
                    data = json.load(f)
                if data.get("date") == today:
                    state = cls()
                    for k, v in data.items():
                        if hasattr(state, k):
                            setattr(state, k, v)
                    return state
        except Exception as e:
            logger.warning(f"State load failed: {e}")
        state = cls()
        state.date = today
        return state


async def init(shared_session, stream_manager, tg_send=None):
    """Initialize US30 broker. Called by run_all.py."""
    global broker, _tg_send, _shared_session

    broker = IGBroker(shared_session, stream_manager, config.IG_EPIC, "GBP")
    _tg_send = tg_send
    _shared_session = shared_session
    logger.info(f"US30 bot initialized: {config.IG_EPIC}")


_shared_session = None


async def health_check():
    """14:00 UK — US30 health check before US open."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    logger.info("═══ US30 HEALTH CHECK ═══")
    mode = "DEMO" if config.IG_DEMO else "LIVE"

    price = None
    status = "Unknown"
    if broker:
        try:
            price = await broker.get_current_price()
            status = "Connected" if price else "No price"
        except Exception:
            status = "Unreachable"

    price_str = f"{price:.1f}" if price else "N/A"
    stream_bars = broker.get_streaming_bar_count() if broker else 0

    msg = (
        f"<b>US30 Health Check</b> [{mode}]\n"
        f"{now.strftime('%Y-%m-%d %H:%M')} ET\n"
        f"IG: {status}\n"
        f"Epic: {config.IG_EPIC}\n"
        f"US30: {price_str}\n"
        f"Streaming bars: {stream_bars}\n"
        f"Morning routine at {config.MORNING_HOUR}:{config.MORNING_MINUTE:02d} ET"
    )
    await _alert(msg)
    logger.info(f"Health check: IG={status}, US30={price_str}, bars={stream_bars}")


async def pre_trade_warmup():
    """14:20 UK (09:20 ET) — Verify connections before bar 4 window."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    logger.info("═══ US30 PRE-TRADE WARMUP ═══")
    issues = []

    # REST check
    if _shared_session:
        try:
            await _shared_session.keepalive()
            logger.info("Pre-warm: REST session alive")
        except Exception as e:
            issues.append(f"REST check failed: {e}")

    # Stream check
    if broker and _shared_session:
        try:
            stream_ok = await _shared_session.check_stream_health(
                broker._stream, config.IG_EPIC
            )
            if stream_ok:
                logger.info("Pre-warm: Lightstreamer alive")
            else:
                issues.append("Lightstreamer stale — resubscribed")
        except Exception as e:
            issues.append(f"Stream check failed: {e}")

    bar_count = broker.get_streaming_bar_count() if broker else 0
    logger.info(f"Pre-warm: {bar_count} streaming bars")

    if issues:
        await _alert(
            "<b>Pre-trade warmup</b>\n"
            + "\n".join(f"- {i}" for i in issues) +
            "\nRecovery attempted. ~30 min to morning routine."
        )
    else:
        logger.info("Pre-warm: all systems OK")


async def stream_alive_check():
    """14:40 UK (09:40 ET) — Check bars are flowing before bar 4 closes."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    bar_count = broker.get_streaming_bar_count() if broker else 0
    if bar_count == 0:
        logger.warning("Stream check: No US30 bars — attempting recovery")
        if broker and _shared_session:
            recovered = await _shared_session.check_stream_health(
                broker._stream, config.IG_EPIC
            )
            if not recovered:
                await _alert(
                    "<b>Stream check FAILED</b>\n"
                    "No US30 bars + recovery failed.\n"
                    "Morning routine will use REST fallback."
                )
    else:
        logger.info(f"Stream check: {bar_count} US30 bars — OK")


async def morning_routine():
    """09:51 ET — Calculate levels from bar 4/5 and place orders."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    logger.info("═══ US30 MORNING ROUTINE ═══")
    state = US30DailyState.load()
    if state.phase != Phase.IDLE:
        logger.info("Already processed today")
        return

    if not broker:
        logger.error("Broker not initialized")
        return

    # Get today's bars
    df = broker.get_streaming_bars_df()
    if df.empty:
        try:
            df = await broker.get_5min_bars("1 D")
        except Exception as e:
            logger.error(f"Failed to get bars: {e}")

    if df.empty:
        await _alert("ASRS ERROR\nNo bar data available for US30.\nCheck logs.")
        return

    today_date = now.date()

    # Gap direction
    try:
        prev_df = await broker.get_5min_bars("2 D")
        if not prev_df.empty:
            prev_day = prev_df[prev_df.index.date < today_date]
            today_bars = df[df.index.date == today_date]
            if not prev_day.empty and not today_bars.empty:
                gap_dir, gap_size = classify_gap(prev_day["Close"].iloc[-1], today_bars["Open"].iloc[0])
                state.gap_dir = gap_dir
                state.gap_size = gap_size
                logger.info(f"Gap: {gap_dir} ({gap_size:+.1f})")
    except Exception as e:
        logger.warning(f"Gap computation failed: {e}")

    # Overnight range
    overnight_result = OvernightResult()
    try:
        overnight_df = await broker.get_overnight_bars()
        today_bars = df[df.index.date == today_date]
        bar4_temp = get_bar(today_bars, 4) if not today_bars.empty else None
        if bar4_temp:
            overnight_result = calculate_overnight_range(
                overnight_df, bar4_temp["high"], bar4_temp["low"]
            )
    except Exception as e:
        logger.warning(f"Overnight range failed: {e}")

    state.overnight_high = overnight_result.range_high
    state.overnight_low = overnight_result.range_low
    state.overnight_range = overnight_result.range_size
    state.overnight_bias = overnight_result.bias.value
    state.save()

    # Calculate levels (uses config module — need to temporarily swap)
    # We patch the config reference in strategy module
    original_config = _strat.config
    _strat.config = config
    try:
        events = calculate_levels(state, df)
    finally:
        _strat.config = original_config

    logger.info(f"Level events: {events}")

    if "LEVELS_SET" not in events and "WAITING_FOR_BAR5" not in events:
        await _alert(f"US30 ASRS ERROR\nCannot calculate levels: {events}")
        return

    if "WAITING_FOR_BAR5" in events:
        logger.info("Waiting for bar 5...")
        await asyncio.sleep(300)
        df = broker.get_streaming_bars_df()
        if df.empty:
            df = await broker.get_5min_bars("1 D")
        _strat.config = config
        try:
            events = calculate_levels(state, df)
        finally:
            _strat.config = original_config
        if "LEVELS_SET" not in events:
            await _alert(f"US30 Bar 5 not available: {events}")
            return

    logger.info(f"Levels: Buy={state.buy_level} Sell={state.sell_level} "
                f"Bar={state.bar_number} Range={state.bar_range}")

    # Position sizing
    state.position_size = min(config.NUM_CONTRACTS, config.MAX_CONTRACTS)
    if state.range_flag == "NARROW" and config.NARROW_STD_MULTIPLIER > 1:
        state.position_size = min(config.NUM_CONTRACTS * config.NARROW_STD_MULTIPLIER, config.MAX_CONTRACTS)
    state.save()

    # Send signal
    await _alert(
        f"<b>US30 ASRS Signal</b>\n"
        f"Bar {state.bar_number}: H={state.bar_high:.1f} L={state.bar_low:.1f}\n"
        f"Range: {state.bar_range:.1f} ({state.range_flag})\n"
        f"Buy: {state.buy_level:.1f} | Sell: {state.sell_level:.1f}\n"
        f"Bias: {state.overnight_bias}\n"
        f"Contracts: {state.position_size}"
    )

    # Place orders
    bias = overnight_result.bias
    qty = state.position_size

    # One-sided based on bias
    if bias == OvernightBias.LONG_ONLY:
        buy_price = state.buy_level
        sell_price = 0.01  # Unreachable
    elif bias == OvernightBias.SHORT_ONLY:
        buy_price = 999999.0
        sell_price = state.sell_level
    else:
        buy_price = state.buy_level
        sell_price = state.sell_level

    result = await broker.place_oca_bracket(
        buy_price=buy_price, sell_price=sell_price,
        qty=qty, oca_group=f"US30_{state.date}_1",
    )

    if "error" not in result:
        state.buy_order_id = result.get("buy_order_id", "")
        state.sell_order_id = result.get("sell_order_id", "")
        state.phase = Phase.ORDERS_PLACED
        state.save()
        logger.info("Orders placed — US30 morning routine complete")
    else:
        await _alert(f"US30 ORDER FAILED: {result['error']}")


async def monitor_cycle():
    """Every minute during RTH — check triggers, manage positions."""
    state = US30DailyState.load()
    if state.phase == Phase.IDLE or state.phase == Phase.DONE:
        return

    if not broker:
        return

    # Swap config for strategy calls
    original_config = _strat.config
    _strat.config = config

    try:
        if state.phase == Phase.ORDERS_PLACED:
            # Check for trigger
            result = await broker.check_trigger_levels()
            if result:
                logger.info(f"US30 trigger: {result}")

        elif state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
            # Trail stop update
            df = broker.get_streaming_bars_df()
            if not df.empty:
                events = update_trail(state, df)
                if "TRAIL_UPDATED" in events or "TRAIL_TIGHT" in events:
                    try:
                        await broker.modify_stop(state.stop_order_id, state.trailing_stop)
                    except Exception:
                        pass

            # Check if stopped out
            pos = await broker.get_position()
            if pos and pos["direction"] == "FLAT" and state.contracts_active > 0:
                exit_price = await broker.get_fill_price(state.stop_order_id) or state.trailing_stop
                events = process_stop_hit(state, exit_price)
                logger.info(f"US30 stop hit: {events}")

                trade = state.trades[-1] if state.trades else {}
                await _alert(
                    f"<b>US30 EXIT</b>\n"
                    f"Direction: {trade.get('direction', '?')}\n"
                    f"Entry: {trade.get('entry', '?')} | Exit: {exit_price}\n"
                    f"PnL: {trade.get('pnl_pts', 0):.1f} pts\n"
                    f"MFE: {trade.get('mfe', 0):.1f} pts\n"
                    f"Reason: {trade.get('exit_reason', '?')}"
                )

                # Re-entry on profitable trail
                if "CAN_REENTER" in events:
                    if state.reentry_direction == "LONG":
                        state.buy_level = state.reentry_price
                        state.sell_level = 0.01
                    else:
                        state.buy_level = 999999.0
                        state.sell_level = state.reentry_price
                    state.save()
                    result = await broker.place_oca_bracket(
                        buy_price=state.buy_level, sell_price=state.sell_level,
                        qty=state.position_size or config.NUM_CONTRACTS,
                        oca_group=f"US30_{state.date}_{state.entries_used + 1}",
                    )
                    if "error" not in result:
                        state.buy_order_id = result.get("buy_order_id", "")
                        state.sell_order_id = result.get("sell_order_id", "")
                        state.phase = Phase.ORDERS_PLACED
                        state.save()
                        await _alert(
                            f"RE-ENTRY ARMED US30\n"
                            f"Direction: {state.reentry_direction}\n"
                            f"Trigger: {state.reentry_price}"
                        )

            # Add-to-winners
            if config.ADD_STRENGTH_ENABLED and state.adds_used < config.ADD_STRENGTH_MAX:
                price = await broker.get_current_price()
                if price:
                    add_events = check_add_to_winners(state, price)
                    if "ADD_TRIGGERED" in add_events:
                        add_action = "BUY" if state.direction == "LONG" else "SELL"
                        add_result = await broker.place_market_order(add_action, 1)
                        if "order_id" in add_result:
                            fill_price = add_result.get("avg_price", price)
                            process_add_fill(state, fill_price)
                            await _alert(
                                f"ADD TO WINNERS US30\n"
                                f"{state.direction} +1 @ {fill_price}\n"
                                f"Add #{state.adds_used} | Total: {state.contracts_active}"
                            )
    finally:
        _strat.config = original_config


async def end_of_day():
    """16:05 ET — Close everything."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    state = US30DailyState.load()
    if state.phase in (Phase.IDLE, Phase.DONE):
        return

    logger.info("US30 end of day — closing positions")
    if state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        close_action = "SELL" if state.direction == "LONG" else "BUY"
        await broker.place_market_order(close_action, state.contracts_active)
        await _alert("US30 END OF DAY — positions closed")

    state.phase = Phase.DONE
    state.save()
