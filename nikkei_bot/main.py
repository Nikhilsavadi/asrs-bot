"""
Nikkei 225 Bot main — Japan 225 ASRS strategy.
Reuses DAX bot logic with Nikkei-specific config.
"""

import logging
import asyncio
from datetime import datetime

# Monkey-patch: make dax_bot code use nikkei_bot.config instead of dax_bot.config
# This is done by creating Nikkei-specific instances with the right config.
from nikkei_bot import config
from dax_bot.strategy import (
    DailyState, Phase, calculate_levels, update_trail, process_stop_hit,
    check_add_to_winners, process_add_fill, classify_gap, get_bar,
    analyse_context, should_use_bar5,
)
from dax_bot.broker_ig import IGBroker
import httpx
from dax_bot.overnight import calculate_overnight_range, OvernightBias, OvernightResult

logger = logging.getLogger("NIKKEI_ASRS")

# Module-level state (initialized by run_all.py)
broker: IGBroker = None
_bar4_triggered = False
_tg_send = None  # Telegram send function, set by init()


async def _alert(text: str):
    """Send Telegram alert with [S4 NIKKEI] prefix."""
    if _tg_send:
        await _tg_send("[S4 NIKKEI] " + text)
    else:
        logger.info(f"ALERT (no TG): {text}")

# Override STATE_FILE for Nikkei
import dax_bot.strategy as _strat
_NIKKEI_STATE_FILE = config.STATE_FILE


class NikkeiDailyState(DailyState):
    """DailyState that uses Nikkei state file."""

    def save(self):
        import os, json
        from dataclasses import asdict
        os.makedirs(os.path.dirname(_NIKKEI_STATE_FILE), exist_ok=True)
        with open(_NIKKEI_STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "NikkeiDailyState":
        import os, json
        today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
        try:
            if os.path.exists(_NIKKEI_STATE_FILE):
                with open(_NIKKEI_STATE_FILE) as f:
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
    """Initialize Nikkei broker. Called by run_all.py."""
    global broker, _tg_send, _shared_session

    broker = IGBroker(shared_session, stream_manager, config.IG_EPIC, "GBP")
    broker.register_trigger_callback(on_tick_trigger)
    _tg_send = tg_send
    _shared_session = shared_session
    logger.info(f"Nikkei bot initialized: {config.IG_EPIC}")


_shared_session = None


async def on_tick_trigger(trigger: dict):
    """Called instantly when a tick crosses bracket levels (sub-second).
    Handles fill processing, stop placement, Telegram alert."""
    state = NikkeiDailyState.load()
    if state.phase != Phase.ORDERS_PLACED:
        return

    direction = trigger["direction"]
    fill_price = trigger["fill_price"]
    order_id = trigger.get("order_id", "")

    # Update state
    state.phase = Phase.LONG_ACTIVE if direction == "LONG" else Phase.SHORT_ACTIVE
    state.direction = direction
    state.entry_price = fill_price
    # Stop at the opposite side of the bar range (not the OCA bracket level which may be 999999)
    state.initial_stop = state.bar_high + config.BUFFER_PTS if direction == "SHORT" else state.bar_low - config.BUFFER_PTS
    state.trailing_stop = state.initial_stop
    state.contracts_active = state.position_size
    state.entries_used += 1
    state.stop_order_id = order_id
    state.last_add_price = fill_price
    state.max_favourable = fill_price  # Will be updated by trail
    intended = state.sell_level if direction == "SHORT" else state.buy_level
    state.trades.append({
        "entry": fill_price,
        "exit": 0,
        "exit_intended": 0,
        "entry_intended": intended,
        "direction": direction,
        "entry_time": datetime.now(config.TZ_JST).strftime("%H:%M:%S"),
        "entry_slippage": round(abs(fill_price - intended), 1),
        "exit_slippage": 0,
        "slippage_total": round(abs(fill_price - intended), 1),
        "pnl_pts": 0,
        "pnl_per_contract": 0,
        "mfe": 0,
        "exit_reason": "",
        "contracts_stopped": 0,
        "tp1_filled": False,
        "tp2_filled": False,
    })
    state.save()
    logger.info(f"NIKKEI tick trigger fill: {direction} @ {fill_price}")

    # Place stop on IG
    try:
        await broker.modify_stop(order_id, state.trailing_stop)
        logger.info(f"NIKKEI stop set @ {state.trailing_stop}")
    except Exception as e:
        logger.error(f"NIKKEI stop placement failed: {e}")

    # Telegram alert
    await _alert(
        f"📈 <b>NIKKEI {direction}</b>\n"
        f"Entry: {fill_price}\n"
        f"Stop: {state.trailing_stop}\n"
        f"Range: {state.bar_range:.1f}pts ({state.range_flag})\n"
    )


async def health_check():
    """00:00 JST — Nikkei health check before Tokyo open."""
    now = datetime.now(config.TZ_JST)
    if now.weekday() >= 5:
        return

    logger.info("═══ NIKKEI HEALTH CHECK ═══")
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
        f"<b>NIKKEI Health Check</b> [{mode}]\n"
        f"{now.strftime('%Y-%m-%d %H:%M')} JST\n"
        f"IG: {status}\n"
        f"Epic: {config.IG_EPIC}\n"
        f"NIKKEI: {price_str}\n"
        f"Streaming bars: {stream_bars}\n"
        f"Morning routine at 09:21 JST"
    )
    await _alert(msg)
    logger.info(f"Health check: IG={status}, NIKKEI={price_str}, bars={stream_bars}")


async def pre_trade_warmup():
    """08:50 JST — Verify connections before bar 4 window."""
    now = datetime.now(config.TZ_JST)
    if now.weekday() >= 5:
        return

    logger.info("═══ NIKKEI PRE-TRADE WARMUP ═══")
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
    """09:10 JST — Check bars are flowing before bar 4 closes."""
    now = datetime.now(config.TZ_JST)
    if now.weekday() >= 5:
        return

    bar_count = broker.get_streaming_bar_count() if broker else 0
    if bar_count == 0:
        logger.warning("Stream check: No NIKKEI bars — attempting recovery")
        if broker and _shared_session:
            recovered = await _shared_session.check_stream_health(
                broker._stream, config.IG_EPIC
            )
            if not recovered:
                await _alert(
                    "<b>Stream check FAILED</b>\n"
                    "No NIKKEI bars + recovery failed.\n"
                    "Morning routine will use REST fallback."
                )
    else:
        logger.info(f"Stream check: {bar_count} NIKKEI bars — OK")


async def morning_routine():
    """09:21 JST — Calculate levels from bar 4/5 and place orders."""
    now = datetime.now(config.TZ_JST)
    if now.weekday() >= 5:
        return

    from shared.holidays import is_holiday
    if is_holiday(now.date(), "NIKKEI"):
        logger.info(f"Nikkei market holiday ({now.date()}) — skipping")
        await _alert(f"📅 Nikkei market holiday today — no trading")
        return

    logger.info("═══ NIKKEI MORNING ROUTINE ═══")
    state = NikkeiDailyState.load()
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
        await _alert("ASRS ERROR\nNo bar data available for NIKKEI.\nCheck logs.")
        return

    today_date = now.date()

    # FAST PATH: Get bar 4 from streaming FIRST, arm bracket immediately
    # Gap and overnight are done AFTER bracket is armed
    # Calculate levels directly — streaming bars are in CET, Tokyo open is 01:00 CET (winter)
    # We can't use DAX's candle_number() which is hardcoded to 09:00 CET
    from zoneinfo import ZoneInfo
    cet = ZoneInfo("Europe/Berlin")
    jst = config.TZ_JST
    now_cet = datetime.now(cet)
    # Tokyo open at 09:00 JST — convert to CET for bar filtering
    now_jst = datetime.now(jst)
    tokyo_open_jst = now_jst.replace(hour=9, minute=0, second=0, microsecond=0)

    import pandas as pd
    tokyo_open_cet = pd.Timestamp(tokyo_open_jst).tz_convert(cet)

    # Filter bars from Tokyo open onwards
    nikkei_bars = df[df.index >= tokyo_open_cet].sort_index()
    logger.info(f"NIKKEI bars from 09:00 JST: {len(nikkei_bars)} bars")

    if len(nikkei_bars) < 4:
        await _alert(f"NIKKEI ASRS ERROR\nOnly {len(nikkei_bars)} bars since Tokyo open (need 4)")
        return

    # Bar 4 = 4th bar from Tokyo open
    bar4 = nikkei_bars.iloc[3]
    bar4_range = bar4["High"] - bar4["Low"]

    # Hybrid: bar 4 for narrow, bar 5 for normal/wide
    if bar4_range > config.NARROW_RANGE and len(nikkei_bars) >= 5:
        signal_bar = nikkei_bars.iloc[4]
        state.bar_number = 5
        logger.info(f"Using bar 5 (bar4 range {bar4_range:.1f} > NARROW {config.NARROW_RANGE})")
    else:
        signal_bar = bar4
        state.bar_number = 4

    state.bar_high = round(signal_bar["High"], 1)
    state.bar_low = round(signal_bar["Low"], 1)
    state.bar_range = round(signal_bar["High"] - signal_bar["Low"], 1)

    if state.bar_range < 3:
        await _alert(f"NIKKEI ASRS: Bar range too small ({state.bar_range}pts)")
        return

    # Max bar range check — skip absurdly wide bars
    if state.bar_range > config.MAX_BAR_RANGE:
        logger.warning(f"Bar range {state.bar_range:.1f} > MAX {config.MAX_BAR_RANGE} — skipping")
        await _alert(f"[S4 NIKKEI] SKIPPED: Bar range {state.bar_range:.1f}pts > max {config.MAX_BAR_RANGE}")
        return

    # Max risk check — at minimum stake, would this trade risk more than MAX_RISK_GBP?
    min_stake = 0.5  # IG minimum
    risk_gbp = state.bar_range * min_stake
    if risk_gbp > config.MAX_RISK_GBP:
        logger.warning(f"Risk £{risk_gbp:.0f} > MAX £{config.MAX_RISK_GBP:.0f} — skipping")
        await _alert(f"[S4 NIKKEI] SKIPPED: Risk £{risk_gbp:.0f} > cap £{config.MAX_RISK_GBP:.0f}")
        return

    # Range flag
    if state.bar_range <= config.NARROW_RANGE:
        state.range_flag = "NARROW"
    elif state.bar_range >= config.WIDE_RANGE:
        state.range_flag = "WIDE"
    else:
        state.range_flag = "NORMAL"

    # Set levels
    state.buy_level = round(state.bar_high + config.BUFFER_PTS, 1)
    state.sell_level = round(state.bar_low - config.BUFFER_PTS, 1)
    state.phase = Phase.LEVELS_SET
    state.oca_group = f"NIKKEI_{state.date}_1"
    state.save()

    events = ["LEVELS_SET"]
    logger.info(f"Levels: Buy={state.buy_level} Sell={state.sell_level} "
                f"Bar={state.bar_number} Range={state.bar_range} ({state.range_flag})")

    # Position sizing
    state.position_size = min(config.NUM_CONTRACTS, config.MAX_CONTRACTS)
    if state.range_flag == "NARROW" and config.NARROW_STD_MULTIPLIER > 1:
        state.position_size = min(config.NUM_CONTRACTS * config.NARROW_STD_MULTIPLIER, config.MAX_CONTRACTS)
    state.save()

    # ARM BRACKET IMMEDIATELY (both sides) — speed is critical at open
    qty = min(config.NUM_CONTRACTS, config.MAX_CONTRACTS)
    state.position_size = qty
    result = await broker.place_oca_bracket(
        buy_price=state.buy_level, sell_price=state.sell_level,
        qty=qty, oca_group=f"NIKKEI_{state.date}_1",
    )

    if "error" not in result:
        state.buy_order_id = result.get("buy_order_id", "")
        state.sell_order_id = result.get("sell_order_id", "")
        state.phase = Phase.ORDERS_PLACED
        state.oca_group = f"NIKKEI_{state.date}_1"
        state.save()
        logger.info(f"Bracket armed: Buy={state.buy_level} Sell={state.sell_level}")
    else:
        await _alert(f"NIKKEI ORDER FAILED: {result['error']}")
        return

    # NOW do slow REST calls for gap + overnight (bracket is already live)
    overnight_result = OvernightResult()
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

    # For Nikkei, "overnight" = pre-market bars before 09:00 JST (from streaming)
    try:
        pre_market = df[df.index < tokyo_open_cet]
        if not pre_market.empty:
            overnight_result = calculate_overnight_range(
                pre_market, state.bar_high, state.bar_low
            )
            logger.info(f"NIKKEI pre-market range: {overnight_result.range_high:.1f}-{overnight_result.range_low:.1f}, bias={overnight_result.bias.value}")
    except Exception as e:
        logger.warning(f"Pre-market range failed: {e}")

    state.overnight_high = overnight_result.range_high
    state.overnight_low = overnight_result.range_low
    state.overnight_range = overnight_result.range_size
    state.overnight_bias = overnight_result.bias.value

    # Apply bias filter: narrow to one side if bias is clear
    bias = overnight_result.bias
    if bias == OvernightBias.LONG_ONLY:
        logger.info("V58 bias: LONG ONLY — disabling sell side")
        broker._pending_bracket["sell_price"] = 0.01
    elif bias == OvernightBias.SHORT_ONLY:
        logger.info("V58 bias: SHORT ONLY — disabling buy side")
        broker._pending_bracket["buy_price"] = 999999.0

    state.save()

    # Send signal
    await _alert(
        f"<b>NIKKEI ASRS Signal</b>\n"
        f"Bar {state.bar_number}: H={state.bar_high:.1f} L={state.bar_low:.1f}\n"
        f"Range: {state.bar_range:.1f} ({state.range_flag})\n"
        f"Buy: {state.buy_level:.1f} | Sell: {state.sell_level:.1f}\n"
        f"Bias: {state.overnight_bias}\n"
        f"Contracts: {state.position_size}"
    )
    logger.info("Orders placed — NIKKEI morning routine complete")


async def monitor_cycle():
    """Every minute during TSE hours — check triggers, manage positions."""
    state = NikkeiDailyState.load()
    if state.phase == Phase.IDLE or state.phase == Phase.DONE:
        return

    if not broker:
        return

    # Swap config for strategy calls
    original_config = _strat.config
    _strat.config = config

    try:
        if state.phase == Phase.ORDERS_PLACED:
            # Re-arm bracket if lost (e.g. after restart)
            pb = broker._pending_bracket
            if not pb or not pb.get("active"):
                logger.warning(f"NIKKEI: bracket lost — re-arming from state (buy={state.buy_level}, sell={state.sell_level})")
                await broker.place_oca_bracket(
                    state.buy_level, state.sell_level,
                    qty=state.position_size, oca_group=state.oca_group
                )
            # Check for trigger
            result = await broker.check_trigger_levels()
            if result:
                logger.info(f"NIKKEI trigger: {result}")

        elif state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
            # Trail stop update
            df = broker.get_streaming_bars_df()
            if not df.empty:
                events = update_trail(state, df)
                if "TRAIL_UPDATED" in events or "TRAIL_TIGHT" in events:
                    # Update stop on ALL open deal IDs (original + adds)
                    all_deal_ids = [state.stop_order_id]
                    for add in state.add_positions:
                        if isinstance(add, dict) and add.get("deal_id"):
                            all_deal_ids.append(add["deal_id"])
                    for deal_id in all_deal_ids:
                        try:
                            await broker.modify_stop(deal_id, state.trailing_stop)
                        except Exception as e:
                            logger.warning(f"Stop update failed for {deal_id}: {e}")

            # Check if stopped out
            pos = await broker.get_position()
            if pos and pos["direction"] == "FLAT" and state.contracts_active > 0:
                exit_price = await broker.get_fill_price(state.stop_order_id) or state.trailing_stop

                # Detect manual close: if current price is far from trailing_stop,
                # someone closed manually (not a stop hit)
                current_price = await broker.get_current_price() or exit_price
                if state.direction == "LONG":
                    stop_distance = abs(current_price - state.trailing_stop)
                else:
                    stop_distance = abs(current_price - state.trailing_stop)
                manual_close = stop_distance > state.bar_range * 0.5
                if manual_close:
                    logger.warning(f"NIKKEI: position closed far from trail stop ({current_price} vs {state.trailing_stop}) — likely manual close")
                    exit_price = current_price

                # Ensure contracts_active reflects what was actually stopped
                if state.contracts_active == 0:
                    state.contracts_active = 1 + len(state.add_positions)
                    logger.warning(f"contracts_active was 0 at stop processing — restored to {state.contracts_active}")
                logger.info(f"NIKKEI stop processing: exit={exit_price}, trailing_stop={state.trailing_stop}, "
                            f"entry={state.entry_price}, contracts={state.contracts_active}, adds={len(state.add_positions)}")
                events = process_stop_hit(state, exit_price)
                if manual_close:
                    state.trades[-1]["exit_reason"] = "MANUAL_CLOSE"
                logger.info(f"NIKKEI stop hit: {events}{' (MANUAL)' if manual_close else ''}")

                trade = state.trades[-1] if state.trades else {}
                await _alert(
                    f"<b>NIKKEI EXIT</b>{' (MANUAL)' if manual_close else ''}\n"
                    f"Direction: {trade.get('direction', '?')}\n"
                    f"Entry: {trade.get('entry', '?')} | Exit: {exit_price}\n"
                    f"PnL: {trade.get('pnl_pts', 0):.1f} pts\n"
                    f"MFE: {trade.get('mfe', 0):.1f} pts\n"
                    f"Reason: {trade.get('exit_reason', '?')}"
                )

                # Re-entry: skip on manual close
                if manual_close:
                    state.phase = Phase.DONE
                    state.save()
                    logger.info("NIKKEI: skipping re-entry after manual close")
                elif "CAN_REENTER" in events:
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
                        oca_group=f"NIKKEI_{state.date}_{state.entries_used + 1}",
                    )
                    if "error" not in result:
                        state.buy_order_id = result.get("buy_order_id", "")
                        state.sell_order_id = result.get("sell_order_id", "")
                        state.phase = Phase.ORDERS_PLACED
                        state.save()
                        await _alert(
                            f"RE-ENTRY ARMED NIKKEI\n"
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
                            deal_id = add_result.get("order_id", "")
                            process_add_fill(state, fill_price)
                            # Store deal_id on the add position for stop management
                            if state.add_positions:
                                state.add_positions[-1]["deal_id"] = deal_id
                                state.save()
                            # Set stop on the new position immediately
                            try:
                                await broker.modify_stop(deal_id, state.trailing_stop)
                                logger.info(f"NIKKEI add stop set on {deal_id} @ {state.trailing_stop}")
                            except Exception as e:
                                logger.error(f"NIKKEI add stop FAILED on {deal_id}: {e}")
                            await _alert(
                                f"ADD TO WINNERS NIKKEI\n"
                                f"{state.direction} +1 @ {fill_price}\n"
                                f"Add #{state.adds_used} | Total: {state.contracts_active}\n"
                                f"Stop: {state.trailing_stop}"
                            )
    finally:
        _strat.config = original_config


async def end_of_day():
    """15:05 JST — Close everything."""
    now = datetime.now(config.TZ_JST)
    if now.weekday() >= 5:
        return

    state = NikkeiDailyState.load()
    if state.phase in (Phase.IDLE, Phase.DONE):
        return

    logger.info("NIKKEI end of day — closing positions")
    if state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        close_action = "SELL" if state.direction == "LONG" else "BUY"
        await broker.place_market_order(close_action, state.contracts_active)
        await _alert("NIKKEI END OF DAY — positions closed")

    state.phase = Phase.DONE
    state.save()
