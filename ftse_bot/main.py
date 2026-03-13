"""
main.py -- FTSE 1BN/1BP Trading Bot
==============================================================================

Tom Hougaard's First Bar Negative / First Bar Positive strategy on FTSE 100.
Exit: Candle trail (previous bar low/high) + Add-to-winners (S25_A2)

Usage:
    python -m ftse_bot.main              Run scheduled bot
    python -m ftse_bot.main --test       Test Telegram + broker
    python -m ftse_bot.main --status     Show current state
    python -m ftse_bot.main --cancel     Cancel all open orders
    python -m ftse_bot.main --close      Close position + cancel orders
"""

import asyncio
import os
import sys
import signal
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ftse_bot import config

# ── Broker ───────────────────────────────────────────────────────────────────
from ftse_bot.broker_ig import IGBroker as FTSEBroker

from ftse_bot.strategy import (
    DailyState, Phase, StopPhase,
    process_bar, get_order_directions, process_fill,
    update_candle_trail, update_stop, check_add_trigger, process_add,
    process_exit,
)
from ftse_bot import telegram_alerts as alerts
from ftse_bot import journal

try:
    import holidays
    UK_HOLIDAYS = holidays.UK(years=range(2024, 2028))
except ImportError:
    UK_HOLIDAYS = {}

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FTSE")

broker = None  # Set by run_all.py (shared session)
scheduler = AsyncIOScheduler(timezone=config.TZ_UK)


async def connect_with_retry(max_attempts: int = 5, delay: int = 30, alert: bool = True) -> bool:
    for attempt in range(1, max_attempts + 1):
        ok = await broker.connect()
        if ok:
            if attempt > 1 and alert:
                await alerts.send(f"🔌 <b>FTSE Connected</b> (attempt {attempt}/{max_attempts})")
            return True
        logger.warning(f"Connect attempt {attempt}/{max_attempts} failed")
        if attempt < max_attempts:
            if alert:
                await alerts.send(
                    f"⚠️ <b>FTSE Connection Failed</b>\n"
                    f"Attempt {attempt}/{max_attempts} — retrying in {delay}s"
                )
            await asyncio.sleep(delay)
    if alert:
        await alerts.send(f"❌ <b>FTSE Unreachable</b> — all {max_attempts} attempts failed")
    return False


# ==============================================================================
#  CORE ROUTINES
# ==============================================================================

def _is_trading_day() -> bool:
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        return False
    if UK_HOLIDAYS and now.date() in UK_HOLIDAYS:
        logger.info(f"UK bank holiday: {UK_HOLIDAYS.get(now.date())}")
        return False
    return True


async def health_check():
    """07:00 UK -- Confirm bot is alive."""
    if not _is_trading_day():
        return

    now = datetime.now(config.TZ_UK)
    logger.info("=== HEALTH CHECK ===")
    mode = "📄 DEMO" if config.IG_DEMO else "🔴 LIVE"
    broker_label = "IG"

    ok = await connect_with_retry(max_attempts=3, delay=20, alert=False)
    status = "✅ Connected" if ok else "❌ Unreachable"
    price = None
    if ok:
        price = await broker.get_current_price()
        await broker.disconnect()

    add_info = f"Add: S{int(config.ADD_STRENGTH_TRIGGER)}_A{config.ADD_STRENGTH_MAX}" if config.ADD_STRENGTH_ENABLED else "Add: OFF"
    msg = (
        f"🩺 <b>FTSE Health Check</b> [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UK\n"
        f"🔌 {broker_label}: {status}\n"
        f"💹 FTSE: {price or 'N/A'}\n"
        f"💰 {config.NUM_CONTRACTS}x £{config.STAKE_PER_POINT}/pt | {add_info}\n"
        f"📊 Trail: Candle (prev bar low/high)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ Bar read at 08:05 | Session close 16:30\n"
        f"<i>Bot is running ✅</i>"
    )
    await alerts.send(msg)


async def morning_bar():
    """08:05 UK -- Read the 08:00-08:05 bar, classify it, place orders."""
    if not _is_trading_day():
        logger.info("Not a trading day -- skipping")
        return

    # Check if trading is paused via Telegram command
    import telegram_cmd
    if telegram_cmd.is_paused():
        logger.info("Trading paused via /pause command -- skipping")
        await alerts.send("⏸ <b>FTSE PAUSED</b> — morning bar skipped. Use /resume to resume.")
        return

    logger.info("=== MORNING BAR ===")
    state = DailyState.load()
    if state.phase != Phase.IDLE:
        logger.info("Already processed today")
        return

    if not await connect_with_retry(max_attempts=5, delay=30):
        return

    await alerts.send(alerts.connection_status(True))
    await asyncio.sleep(5)

    df = await broker.get_5min_bars("1 D")
    if df.empty:
        await alerts.send(alerts.error_alert("No FTSE data -- market holiday?"))
        return

    # Find the 08:00 bar
    today = datetime.now(config.TZ_UK).date()
    today_bars = df[df.index.date == today]
    bar_0800 = None
    for idx, row in today_bars.iterrows():
        if idx.hour == 8 and idx.minute == 0:
            bar_0800 = row
            break
    if bar_0800 is None:
        for idx, row in today_bars.iterrows():
            if idx.hour == 8 and idx.minute == 5:
                bar_0800 = row
                break
    if bar_0800 is None:
        await alerts.send(alerts.error_alert(
            f"Cannot find 08:00 bar. Available: {[str(i.time()) for i in today_bars.index[:10]]}"
        ))
        return

    o, h, l, c = bar_0800["Open"], bar_0800["High"], bar_0800["Low"], bar_0800["Close"]
    events = process_bar(state, o, h, l, c)
    logger.info(f"Bar events: {events}")

    await alerts.send(alerts.bar_detected(state))

    if "DOJI_SKIP" in events:
        await alerts.send(alerts.no_trade(state, f"Doji bar + DOJI_ACTION={config.DOJI_ACTION}"))
        return

    if "LEVELS_SET" not in events:
        await alerts.send(alerts.error_alert(f"Cannot set levels: {events}"))
        return

    # Place orders based on bar type
    directions = get_order_directions(state.bar_type)

    # Order qty = NUM_CONTRACTS * stake_per_point (for IG, qty = stake in GBP/pt)
    order_qty = int(config.NUM_CONTRACTS * state.stake)

    if len(directions) == 2:
        # 1BN: OCA bracket (buy below + sell above)
        result = await broker.place_oca_bracket(
            buy_price=state.buy_level,
            sell_price=state.sell_level,
            qty=order_qty,
            oca_group=state.oca_group,
        )
        if "error" in result:
            await alerts.send(alerts.error_alert(f"Order failed: {result['error']}"))
            return
        state.buy_order_id = result["buy_order_id"]
        state.sell_order_id = result["sell_order_id"]

    elif directions == ["SELL"]:
        # 1BP: sell stop only — price-triggered
        broker._pending_bracket = {
            "buy_price": 999999,  # Will never trigger
            "sell_price": state.buy_level,
            "qty": order_qty,
            "oca_group": state.oca_group,
            "active": True,
        }
        sell_id = f"pending_sell_{state.oca_group}"
        broker._orders[sell_id] = {
            "type": "pending", "direction": "SELL",
            "price": state.buy_level, "qty": order_qty,
        }
        state.sell_order_id = sell_id

    state.phase = Phase.ORDERS_PLACED
    state.save()
    await alerts.send(alerts.orders_placed(state))
    logger.info(f"Orders placed: {directions}, qty={order_qty}")


async def _handle_fill_event(data):
    """
    Event-driven handler for IG streaming trade updates (OPU dict).
    Currently used for logging only — actual fill detection is via price-triggered
    polling in monitor_cycle.
    """
    if not isinstance(data, dict):
        return

    deal_status = data.get("dealStatus", data.get("status", ""))
    deal_id = data.get("dealId", "")
    level = data.get("level", 0)
    direction = data.get("direction", "")

    logger.info(f"[FTSE] Trade stream event: dealId={deal_id} status={deal_status} "
                f"direction={direction} level={level}")
    return


async def monitor_cycle():
    """Every 1 min -- check fills, update candle trail, check add-to-winners.
    When in ORDERS_PLACED phase on IG, polls price every 5s for fast entry detection.
    """
    try:
        state = DailyState.load()
        # Fast polling for IG price-triggered entries
        if (state.phase == Phase.ORDERS_PLACED
                and config.BROKER == "ig"
                and hasattr(broker, "check_trigger_levels")):
            if not await broker.ensure_connected():
                return
            for _ in range(12):  # 12 x 5s = 60s (one monitor cycle)
                state = DailyState.load()
                if state.phase != Phase.ORDERS_PLACED:
                    break
                trigger = await broker.check_trigger_levels()
                if trigger:
                    direction = trigger["direction"]
                    fill_price = trigger["fill_price"]
                    process_fill(state, direction, fill_price)
                    logger.info(f"Price trigger fill: {direction} @ {fill_price}")

                    stop_action = "SELL" if direction == "LONG" else "BUY"
                    stop_result = await broker.place_stop_order(
                        action=stop_action,
                        qty=int(state.contracts_active * state.stake),
                        stop_price=state.initial_stop,
                    )
                    if "order_id" in stop_result:
                        state.stop_order_id = stop_result["order_id"]
                        state.save()

                    await alerts.send(alerts.entry_filled(state))
                    return
                await asyncio.sleep(5)
            return

        await _monitor_inner()
    except Exception as e:
        logger.error(f"Monitor error: {e}", exc_info=True)
        await alerts.send(alerts.error_alert(f"Monitor error: {e}"))


async def _monitor_inner():
    state = DailyState.load()
    if state.phase in (Phase.IDLE, Phase.DONE):
        return
    if not await broker.ensure_connected():
        await alerts.send(alerts.error_alert("IG connection lost"))
        return

    # ── Check for entry fills ──────────────────────────────────────────
    if state.phase == Phase.ORDERS_PLACED:
        # IG price-triggered bracket: check if levels breached
        if hasattr(broker, "check_trigger_levels"):
            trigger = await broker.check_trigger_levels()
            if trigger:
                direction = trigger["direction"]
                fill_price = trigger["fill_price"]
                process_fill(state, direction, fill_price)
                logger.info(f"Price trigger fill: {direction} @ {fill_price}")

                stop_action = "SELL" if direction == "LONG" else "BUY"
                stop_result = await broker.place_stop_order(
                    action=stop_action,
                    qty=int(state.contracts_active * state.stake),
                    stop_price=state.initial_stop,
                )
                if "order_id" in stop_result:
                    state.stop_order_id = stop_result["order_id"]
                    state.save()

                await alerts.send(alerts.entry_filled(state))
                return

        # Fallback: check position directly
        pos = await broker.get_position()
        if pos["direction"] != "FLAT":
            direction = pos["direction"]
            fill_price = pos["avg_cost"]
            process_fill(state, direction, fill_price)
            logger.info(f"Fill detected: {direction} @ {fill_price}")

            if hasattr(broker, "cancel_oca_counterpart"):
                filled_id = state.buy_order_id if direction == "LONG" else state.sell_order_id
                await broker.cancel_oca_counterpart(str(filled_id))

            stop_action = "SELL" if direction == "LONG" else "BUY"
            stop_result = await broker.place_stop_order(
                action=stop_action,
                qty=int(state.contracts_active * state.stake),
                stop_price=state.initial_stop,
            )
            if "order_id" in stop_result:
                state.stop_order_id = stop_result["order_id"]
                state.save()

            await alerts.send(alerts.entry_filled(state))
            return

    # ── Position active -- manage trail + adds ────────────────────────
    if state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        pos = await broker.get_position()

        # Check if stopped out
        if pos["direction"] == "FLAT":
            if state.phase == Phase.DONE:
                return
            exit_price = await broker.get_fill_price(state.stop_order_id) or state.trailing_stop
            trade_result = process_exit(state, exit_price, "STOPPED")
            await alerts.send(alerts.exit_stopped(state, trade_result))
            journal.append_trade(trade_result)
            return

        # Get current price + recent bars for candle trail
        price = await broker.get_current_price()
        if price is None:
            return

        # Update MFE tracking
        update_stop(state, price)

        # ── Candle trail: get previous completed bar's low/high ───────
        df = await broker.get_5min_bars("1 D")
        if not df.empty:
            now = datetime.now(config.TZ_UK)
            today_bars = df[df.index.date == now.date()]

            # Get bars after entry (post 08:05)
            session_bars = today_bars[
                (today_bars.index.hour > 8) |
                ((today_bars.index.hour == 8) & (today_bars.index.minute >= 5))
            ]

            if len(session_bars) >= 2:
                # Previous completed bar (second-to-last)
                prev_bar = session_bars.iloc[-2]
                old_stop = state.trailing_stop
                events = update_candle_trail(
                    state,
                    prev_low=prev_bar["Low"],
                    prev_high=prev_bar["High"],
                )

                if "TRAIL_UPDATED" in events:
                    # Update stop order on broker
                    success = await broker.modify_stop(state.stop_order_id, state.trailing_stop)
                    if not success:
                        # Cancel old stop, place new one
                        await broker.cancel_order(state.stop_order_id)
                        stop_action = "SELL" if state.direction == "LONG" else "BUY"
                        result = await broker.place_stop_order(
                            action=stop_action,
                            qty=int(state.contracts_active * state.stake),
                            stop_price=state.trailing_stop,
                        )
                        if "order_id" in result:
                            state.stop_order_id = result["order_id"]
                            state.save()

                    # Alert only every TRAIL_MIN_ALERT_MOVE pts
                    if abs(state.trailing_stop - state.last_trail_alert_stop) >= config.TRAIL_MIN_ALERT_MOVE:
                        await alerts.send(alerts.trail_updated(state, old_stop))
                        state.last_trail_alert_stop = state.trailing_stop
                        state.save()

                    logger.info(f"Trail: {old_stop} -> {state.trailing_stop}")

        # ── Add-to-winners check ──────────────────────────────────────
        if check_add_trigger(state, price):
            add_price = round(state.last_add_price + config.ADD_STRENGTH_TRIGGER, 1) \
                if state.direction == "LONG" else \
                round(state.last_add_price - config.ADD_STRENGTH_TRIGGER, 1)

            logger.info(f"ADD TRIGGER: +{config.ADD_STRENGTH_TRIGGER}pts from last entry, adding at {add_price}")
            add_qty = int(state.stake)  # 1 contract per add

            result = await broker.place_market_order(
                action="BUY" if state.direction == "LONG" else "SELL",
                qty=add_qty,
            )
            if "order_id" in result:
                add = process_add(state, add_price, result.get("order_id", 0))
                logger.info(f"ADD #{state.adds_used}: {state.direction} +1 @ {add_price}")

                # Update stop order to cover new total size
                await broker.cancel_order(state.stop_order_id)
                stop_action = "SELL" if state.direction == "LONG" else "BUY"
                new_stop_result = await broker.place_stop_order(
                    action=stop_action,
                    qty=int(state.contracts_active * state.stake),
                    stop_price=state.trailing_stop,
                )
                if "order_id" in new_stop_result:
                    state.stop_order_id = new_stop_result["order_id"]
                    state.save()

                await alerts.send(
                    f"➕ <b>FTSE ADD #{state.adds_used}</b>\n"
                    f"{state.direction} +1 @ {add_price}\n"
                    f"Total: {state.contracts_active}x £{state.stake}/pt\n"
                    f"Stop: {state.trailing_stop}"
                )
            else:
                logger.warning(f"Add order failed: {result}")


async def session_close():
    """16:30 UK -- Cancel pending orders, close any open position."""
    if not _is_trading_day():
        return

    state = DailyState.load()
    if state.phase == Phase.IDLE:
        return

    logger.info("=== SESSION CLOSE 16:30 ===")

    if await broker.ensure_connected():
        await broker.cancel_all_orders()
        pos = await broker.get_position()

        if pos["direction"] != "FLAT":
            price = await broker.get_current_price()
            if price:
                trade_result = process_exit(state, price, "SESSION_CLOSE")
                await broker.close_position()
                await alerts.send(alerts.session_close(state, trade_result))
                journal.append_trade(trade_result)
            else:
                await broker.close_position()
                trade_result = process_exit(state, state.trailing_stop, "SESSION_CLOSE")
                await alerts.send(alerts.session_close(state, trade_result))
                journal.append_trade(trade_result)
        else:
            if state.phase == Phase.ORDERS_PLACED:
                state.phase = Phase.DONE
                state.save()
                await alerts.send(alerts.no_trade(state, "No trigger by 16:30"))
            elif state.phase == Phase.DONE:
                pass
            else:
                await alerts.send(alerts.session_close(state, None))

    state.phase = Phase.DONE
    state.save()
    logger.info("Session closed")


async def daily_summary():
    """17:00 UK -- Send daily summary with weekly P&L."""
    if not _is_trading_day():
        return

    state = DailyState.load()
    weekly = journal.get_weekly_pnl()
    await alerts.send(alerts.day_summary(state, weekly))
    await broker.disconnect()
    logger.info("Daily summary sent")


# ==============================================================================
#  SCHEDULER
# ==============================================================================

async def graceful_shutdown():
    logger.info("=== GRACEFUL SHUTDOWN ===")
    state = DailyState.load()
    try:
        if await broker.connect():
            cancelled = await broker.cancel_all_orders()
            if cancelled:
                logger.info(f"Shutdown: cancelled {cancelled} orders")
                await alerts.send(
                    f"🛑 <b>FTSE BOT SHUTDOWN</b>\n"
                    f"Cancelled {cancelled} orders\n"
                    f"Phase: {state.phase}\n"
                    f"<i>Bot will resume on restart</i>"
                )
            pos = await broker.get_position()
            if pos["direction"] != "FLAT":
                await alerts.send(
                    f"⚠️ <b>FTSE position still open!</b>\n"
                    f"{state.direction} {state.contracts_active}x @ {state.entry_price}\n"
                    f"Stop: {state.trailing_stop}\n"
                    f"<i>Monitor manually until restart</i>"
                )
            await broker.disconnect()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


async def startup_recovery():
    """On startup, check for existing positions."""
    state = DailyState.load()
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        return

    logger.info(f"=== RECOVERY: {state.direction} {state.contracts_active}x @ {state.entry_price} ===")
    if not await broker.connect():
        return

    pos = await broker.get_position()
    if pos["direction"] == "FLAT":
        logger.info("Position already closed")
        trade_result = process_exit(state, state.trailing_stop, "STOPPED")
        await alerts.send(alerts.exit_stopped(state, trade_result))
        journal.append_trade(trade_result)
        await broker.disconnect()
        return

    # Re-place trailing stop
    stop_action = "SELL" if state.direction == "LONG" else "BUY"
    result = await broker.place_stop_order(
        action=stop_action,
        qty=int(state.contracts_active * state.stake),
        stop_price=state.trailing_stop,
    )
    if "order_id" in result:
        state.stop_order_id = result["order_id"]
        state.save()

    await alerts.send(
        f"🔄 <b>FTSE BOT RECOVERED</b>\n"
        f"{state.direction} {state.contracts_active}x @ {state.entry_price}\n"
        f"Stop: {state.trailing_stop} ({state.stop_phase})\n"
        f"Adds: {state.adds_used}/{config.ADD_STRENGTH_MAX}\n"
        f"<i>Monitoring resumed</i>"
    )
    await broker.disconnect()


async def run_bot():
    with open("/tmp/ftse.pid", "w") as f:
        f.write(str(os.getpid()))

    # Telegram command handler — only start if DAX bot is available in the same
    # process (standalone mode). In Docker, the DAX container handles commands.
    import telegram_cmd
    try:
        from dax_bot.main import broker as _dax_broker
        # Both bots in same process — start command handler here
        asyncio.get_event_loop().create_task(
            telegram_cmd.poll_commands(dax_broker=_dax_broker, ftse_broker=broker)
        )
    except ImportError:
        # Running in separate container — DAX container handles Telegram commands
        logger.info("FTSE standalone: Telegram commands handled by DAX container")
        pass

    loop = asyncio.get_event_loop()
    shutdown_triggered = False

    def _handle_signal():
        nonlocal shutdown_triggered
        if shutdown_triggered:
            return
        shutdown_triggered = True
        logger.info("Shutdown signal received")
        loop.create_task(_shutdown_and_exit())

    async def _shutdown_and_exit():
        await graceful_shutdown()
        scheduler.shutdown(wait=False)
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    await startup_recovery()
    broker.register_order_handler(_handle_fill_event)

    # Schedule jobs
    scheduler.add_job(health_check, "cron",
        day_of_week="mon-fri", hour=7, minute=0,
        id="ftse_health", misfire_grace_time=120)

    scheduler.add_job(morning_bar, "cron",
        day_of_week="mon-fri", hour=config.BAR_HOUR, minute=config.BAR_MINUTE,
        id="ftse_morning", misfire_grace_time=120)

    scheduler.add_job(monitor_cycle, "cron",
        day_of_week="mon-fri", hour="8-16", minute="*",
        id="ftse_monitor", misfire_grace_time=30)

    scheduler.add_job(session_close, "cron",
        day_of_week="mon-fri", hour=config.SESSION_END_H, minute=config.SESSION_END_M,
        id="ftse_close", misfire_grace_time=120)

    scheduler.add_job(daily_summary, "cron",
        day_of_week="mon-fri", hour=config.SUMMARY_HOUR, minute=config.SUMMARY_MINUTE,
        id="ftse_summary", misfire_grace_time=120)

    scheduler.start()

    mode = "DEMO" if config.IG_DEMO else "LIVE"
    broker_info = f"IG:         {config.IG_EPIC}"

    add_label = f"S{int(config.ADD_STRENGTH_TRIGGER)}_A{config.ADD_STRENGTH_MAX}" if config.ADD_STRENGTH_ENABLED else "OFF"
    print(f"""
╔══════════════════════════════════════════════╗
║     FTSE 1BN/1BP Trading Bot                 ║
║     Candle Trail + Add-to-Winners            ║
╠══════════════════════════════════════════════╣
║  Mode:       {mode:<32}║
║  {broker_info:<43}║
║  Contracts:  {config.NUM_CONTRACTS}x £{config.STAKE_PER_POINT}/pt{' ' * (25 - len(str(config.NUM_CONTRACTS)) - len(str(config.STAKE_PER_POINT)))}║
║  Trail:      Candle (prev bar low/high)      ║
║  Add:        {add_label:<32}║
║  Doji:       {config.DOJI_ACTION:<32}║
║                                              ║
║  07:00  Health check                         ║
║  08:05  Read 1st bar + place orders          ║
║  08:05-16:30  Monitor every 1 min            ║
║  16:30  Session close                        ║
║  17:00  Daily summary                        ║
╚══════════════════════════════════════════════╝
""")

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if not shutdown_triggered:
            await graceful_shutdown()
            scheduler.shutdown()
            await broker.disconnect()


# ==============================================================================
#  CLI
# ==============================================================================

async def cmd_test():
    tg_ok = await alerts.send(
        f"🇬🇧 <b>FTSE Bot Test</b>\n"
        f"Broker: {config.BROKER}\n"
        f"Epic: {config.IG_EPIC}\n"
        f"Contracts: {config.NUM_CONTRACTS}x £{config.STAKE_PER_POINT}/pt\n"
        f"Add: S{int(config.ADD_STRENGTH_TRIGGER)}_A{config.ADD_STRENGTH_MAX}"
    )
    print(f"Telegram: {'OK' if tg_ok else 'FAILED'}")
    ib_ok = await broker.connect()
    if ib_ok:
        price = await broker.get_current_price()
        print(f"Broker: OK | FTSE: {price}")
        await broker.disconnect()
    else:
        print("Broker: FAILED")


async def cmd_status():
    state = DailyState.load()
    print(f"Date:       {state.date}")
    print(f"Phase:      {state.phase}")
    print(f"Bar type:   {state.bar_type}")
    print(f"Bar:        O={state.bar_open} H={state.bar_high} L={state.bar_low} C={state.bar_close}")
    print(f"Width:      {state.bar_width}")
    print(f"Stake:      £{state.stake}/pt x {state.contracts_active} {'(halved)' if state.stake_halved else ''}")
    if state.direction:
        print(f"Position:   {state.direction} @ {state.entry_price}")
        print(f"Stop:       {state.trailing_stop} ({state.stop_phase})")
        print(f"Adds:       {state.adds_used}/{config.ADD_STRENGTH_MAX}")
    if state.exit_price:
        print(f"Exit:       {state.exit_price} ({state.exit_reason})")
        print(f"P&L:        {state.pnl_pts} pts (£{state.pnl_gbp})")
    weekly = journal.get_weekly_pnl()
    print(f"Weekly:     {weekly} pts")


async def cmd_cancel():
    if await broker.connect():
        print(f"Cancelled {await broker.cancel_all_orders()} orders")
        await broker.disconnect()


async def cmd_close():
    if await broker.connect():
        await broker.cancel_all_orders()
        await broker.close_position()
        print("Done")
        await broker.disconnect()


def main():
    cmd = sys.argv[1].lower().strip("-") if len(sys.argv) > 1 else "run"
    cmds = {
        "run": run_bot, "test": cmd_test, "status": cmd_status,
        "cancel": cmd_cancel, "close": cmd_close,
    }
    if cmd == "help":
        print("Commands: --test | --status | --cancel | --close | --help")
        return
    if cmd in cmds:
        asyncio.run(cmds[cmd]())
    else:
        print(f"Unknown: {cmd}. Try --help")


if __name__ == "__main__":
    main()
