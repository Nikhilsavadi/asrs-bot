"""
run_all.py — Run DAX ASRS + Gold ORB bots in a single process.
═══════════════════════════════════════════════════════════════
Single container, single event loop. Shared IG REST + Lightstreamer
session across all strategies. Each strategy is fully independent:
  - DAX: scheduler-driven (08:21-17:35 UK time)
  - Gold ORB: event-driven (Lightstreamer candle callbacks)

No shared state between strategies. Shutdown is sequential.

Usage:
    python run_all.py
"""

import asyncio
import os
import sys
import signal
import logging
from logging.handlers import RotatingFileHandler

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ── Logging: console + rotating file ────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "/data")
LOG_FILE = os.path.join(LOG_DIR, "asrs.log")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)

# Add file handler (5MB per file, keep 3 backups = 20MB max)
os.makedirs(LOG_DIR, exist_ok=True)
_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("ASRS")


async def main():
    # Write PID for healthcheck
    with open("/tmp/asrs.pid", "w") as f:
        f.write(str(os.getpid()))

    loop = asyncio.get_event_loop()
    shutdown_triggered = False

    # ── Import DAX bot ─────────────────────────────────────────
    from dax_bot import config as dax_config

    # ── Initialize trade journal DB ───────────────────────────────
    from shared.journal_db import init_db, migrate_csv, seed_scaling_ladder
    init_db()
    migrate_csv()
    seed_scaling_ladder(os.getenv("SCALING_LADDER", ""), "ALL")

    # ── Shared IG session (ONE REST + ONE Lightstreamer connection) ──
    from shared.ig_session import IGSharedSession
    from shared.ig_stream import IGStreamManager

    shared = IGSharedSession.get_instance()
    await shared.connect()

    stream_mgr = IGStreamManager(shared)

    # Subscribe to streaming (DAX)
    await stream_mgr.subscribe_ticks(dax_config.IG_EPIC)
    await stream_mgr.subscribe_candles(dax_config.IG_EPIC)
    if shared.acc_number:
        await stream_mgr.subscribe_trades(shared.acc_number)

    # Re-subscribe all streams after any session reconnect
    shared.on_reconnect(stream_mgr.resubscribe_all)

    # ── Thin broker adapter (shares the single session) ─────────
    from dax_bot.broker_ig import IGBroker
    dax_broker = IGBroker(shared, stream_mgr, dax_config.IG_EPIC, dax_config.CURRENCY)

    # ── Inject broker into DAX bot module ───────────────────────
    import dax_bot.main as dax_main
    dax_main.broker = dax_broker

    # ── Register candle callback for early bar monitoring ─────
    stream_mgr.register_candle_callback(dax_config.IG_EPIC, dax_main.on_candle_complete)

    # ── Register tick-based entry trigger (sub-second vs 5s polling) ─────
    dax_broker.register_trigger_callback(dax_main.on_tick_trigger)

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 3: Gold ORB (fully event-driven, no scheduler)
    # Runs independently from DAX — triggered by Lightstreamer
    # candle callbacks on Gold epic.
    # Sessions: Gold Asian 00-07, London 07-12, US 13-18 UTC
    # ══════════════════════════════════════════════════════════════
    import telegram_cmd

    gold_main = None  # Will be set if init succeeds

    async def tg_send_gold(text):
        """Send Telegram message with [S2] prefix for Gold ORB."""
        await telegram_cmd._send("[S2] " + text)

    try:
        import gold_bot.main as _gold_main
        await _gold_main.setup(shared, stream_mgr, tg_send=tg_send_gold)
        gold_main = _gold_main
        logger.info("Strategy 2 (Gold ORB) initialized")
    except Exception as e:
        logger.error("Strategy 2 init failed: %s", e, exc_info=True)
        try:
            await tg_send_gold("Strategy 2 INIT FAILED: " + str(e))
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 3: US30 ASRS (scheduler-driven, US market hours)
    # Same ASRS rules as DAX, adapted for S&P 500.
    # RTH: 09:30-16:00 ET, bar 4 at 09:50 ET
    # ══════════════════════════════════════════════════════════════
    spx_main = None
    enable_spx = os.getenv("ENABLE_SPX", "true").lower() == "true"

    if enable_spx:
        try:
            import spx_bot.config as spx_config
            import spx_bot.main as _spx_main

            # Subscribe to US30 streaming
            await stream_mgr.subscribe_ticks(spx_config.IG_EPIC)
            await stream_mgr.subscribe_candles(spx_config.IG_EPIC)

            # Initialize US30 bot
            await _spx_main.init(shared, stream_mgr, tg_send=telegram_cmd._send)
            spx_main = _spx_main
            logger.info("Strategy 3 (US30 ASRS) initialized")
        except Exception as e:
            logger.error("Strategy 3 (US30) init failed: %s", e, exc_info=True)

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 4: Nikkei 225 ASRS (scheduler-driven, Tokyo hours)
    # Same ASRS rules as DAX, adapted for Nikkei price scale.
    # TSE: 09:00-15:00 JST = 00:00-06:00 UTC
    # ══════════════════════════════════════════════════════════════
    nikkei_main = None
    enable_nikkei = os.getenv("ENABLE_NIKKEI", "true").lower() == "true"

    if enable_nikkei:
        try:
            import nikkei_bot.config as nikkei_config
            import nikkei_bot.main as _nikkei_main

            # Subscribe to Nikkei streaming
            await stream_mgr.subscribe_ticks(nikkei_config.IG_EPIC)
            await stream_mgr.subscribe_candles(nikkei_config.IG_EPIC)

            # Initialize Nikkei bot
            await _nikkei_main.init(shared, stream_mgr, tg_send=telegram_cmd._send)
            nikkei_main = _nikkei_main
            logger.info("Strategy 4 (Nikkei ASRS) initialized")
        except Exception as e:
            logger.error("Strategy 4 (Nikkei) init failed: %s", e, exc_info=True)

    # ── Graceful shutdown ─────────────────────────────────────────
    dax_scheduler = AsyncIOScheduler(timezone=dax_config.TZ_UK)
    dax_main.scheduler = dax_scheduler

    async def _shutdown():
        nonlocal shutdown_triggered
        if shutdown_triggered:
            return
        shutdown_triggered = True
        logger.info("=== SHUTDOWN ===")
        # Shutdown DAX first (scheduler-driven, needs clean stop)
        try:
            await dax_main.graceful_shutdown()
        except Exception as e:
            logger.error("DAX shutdown error: %s", e)
        # Shutdown Gold ORB (close any open positions)
        if gold_main is not None:
            try:
                await gold_main.graceful_shutdown()
            except Exception as e:
                logger.error("Gold ORB shutdown error: %s", e)
        dax_scheduler.shutdown(wait=False)
        await stream_mgr.unsubscribe_all()
        await shared.disconnect()
        loop.stop()

    def _handle_signal():
        if not shutdown_triggered:
            loop.create_task(_shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    # ── DAX Startup recovery ──────────────────────────────────────
    await dax_main.startup_recovery()

    # Register event handlers
    dax_broker.register_order_handler(dax_main._handle_fill_event)

    # ── Dashboard ─────────────────────────────────────────────────
    from dax_bot.dashboard import start_dashboard
    start_dashboard(port=8080)

    # ── Telegram command handler (DAX only) ───────────────────────
    loop.create_task(
        telegram_cmd.poll_commands(dax_broker=dax_broker)
    )

    # ── DAX schedule (all times UK) ──────────────────────────────
    dax_scheduler.add_job(dax_main.health_check, "cron",
        day_of_week="mon-fri", hour=7, minute=0,
        id="dax_health", misfire_grace_time=120)

    dax_scheduler.add_job(dax_main.pre_trade_warmup, "cron",
        day_of_week="mon-fri", hour=8, minute=0,
        id="dax_prewarm", misfire_grace_time=120)

    dax_scheduler.add_job(dax_main.stream_alive_check, "cron",
        day_of_week="mon-fri", hour=8, minute=10,
        id="dax_stream_check", misfire_grace_time=120)

    dax_scheduler.add_job(dax_main.morning_routine, "cron",
        day_of_week="mon-fri",
        hour=dax_config.MORNING_HOUR, minute=dax_config.MORNING_MINUTE,
        id="dax_morning", misfire_grace_time=120)

    dax_scheduler.add_job(dax_main.failsafe_check, "cron",
        day_of_week="mon-fri", hour=8, minute=25,
        id="dax_failsafe", misfire_grace_time=120)

    # Session 2: 10:21 UK = 11:21 CET (after bar 4 of session 2 closes at 11:20)
    dax_scheduler.add_job(dax_main.session2_routine, "cron",
        day_of_week="mon-fri", hour=10, minute=21,
        id="dax_session2", misfire_grace_time=120)

    dax_scheduler.add_job(dax_main.monitor_cycle, "cron",
        day_of_week="mon-fri", hour="8-17", minute="*",
        id="dax_monitor", misfire_grace_time=30)

    dax_scheduler.add_job(dax_main.end_of_day, "cron",
        day_of_week="mon-fri",
        hour=dax_config.SUMMARY_HOUR, minute=dax_config.SUMMARY_MINUTE,
        id="dax_eod", misfire_grace_time=120)

    # Overnight bar collection — hourly 23:00-07:00 UTC (00:00-08:00 CET)
    dax_scheduler.add_job(dax_main.collect_overnight_bars, "cron",
        day_of_week="mon-fri", hour="23,0-7", minute=0,
        id="overnight_bars", misfire_grace_time=300)

    # ── US30 schedule (US/Eastern times, converted to UTC for scheduler) ──
    if spx_main is not None:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")

        dax_scheduler.add_job(spx_main.health_check, "cron",
            day_of_week="mon-fri", hour=9, minute=0,
            id="spx_health", misfire_grace_time=120,
            timezone=tz_et)

        dax_scheduler.add_job(spx_main.pre_trade_warmup, "cron",
            day_of_week="mon-fri", hour=9, minute=20,
            id="spx_prewarm", misfire_grace_time=120,
            timezone=tz_et)

        dax_scheduler.add_job(spx_main.stream_alive_check, "cron",
            day_of_week="mon-fri", hour=9, minute=40,
            id="spx_stream_check", misfire_grace_time=120,
            timezone=tz_et)

        dax_scheduler.add_job(spx_main.morning_routine, "cron",
            day_of_week="mon-fri", hour=9, minute=56,
            id="spx_morning", misfire_grace_time=120,
            timezone=tz_et)

        dax_scheduler.add_job(spx_main.monitor_cycle, "cron",
            day_of_week="mon-fri", hour="9-16", minute="*",
            id="spx_monitor", misfire_grace_time=30,
            timezone=tz_et)

        # US30 Session 2: 11:21 ET (after bar 4 of session 2 closes at 11:20)
        dax_scheduler.add_job(spx_main.session2_routine, "cron",
            day_of_week="mon-fri", hour=11, minute=21,
            id="spx_session2", misfire_grace_time=120,
            timezone=tz_et)

        dax_scheduler.add_job(spx_main.end_of_day, "cron",
            day_of_week="mon-fri", hour=16, minute=5,
            id="spx_eod", misfire_grace_time=120,
            timezone=tz_et)

    # ── Nikkei schedule (all times UTC; TSE 09:00-15:00 JST = 00:00-06:00 UTC) ──
    if nikkei_main is not None:
        dax_scheduler.add_job(nikkei_main.health_check, "cron",
            day_of_week="mon-fri", hour=0, minute=0,
            id="nikkei_health", misfire_grace_time=120)

        dax_scheduler.add_job(nikkei_main.pre_trade_warmup, "cron",
            day_of_week="mon-fri", hour=0, minute=10,
            id="nikkei_prewarm", misfire_grace_time=120)

        dax_scheduler.add_job(nikkei_main.morning_routine, "cron",
            day_of_week="mon-fri", hour=0, minute=26,
            id="nikkei_morning", misfire_grace_time=120)

        dax_scheduler.add_job(nikkei_main.monitor_cycle, "cron",
            day_of_week="mon-fri", hour="0-6", minute="*",
            id="nikkei_monitor", misfire_grace_time=30)

        dax_scheduler.add_job(nikkei_main.end_of_day, "cron",
            day_of_week="mon-fri", hour=6, minute=5,
            id="nikkei_eod", misfire_grace_time=120)

    # ── Session keepalive every 10 minutes ────────────────────────
    async def keepalive_with_stream_check():
        await shared.keepalive()
        await shared.check_stream_health(stream_mgr, dax_config.IG_EPIC)

    dax_scheduler.add_job(keepalive_with_stream_check, "interval", minutes=10,
        id="ig_keepalive", misfire_grace_time=60)

    # ── Monitoring: missed job alerts ──────────────────────────────
    from apscheduler.events import EVENT_JOB_MISSED, EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    from shared.monitoring import create_job_listener, heartbeat_ping, send_startup_alert, position_safety_audit

    job_listener = create_job_listener(telegram_cmd._send, loop)
    dax_scheduler.add_listener(job_listener, EVENT_JOB_MISSED | EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    dax_scheduler.add_job(heartbeat_ping, "interval", minutes=5,
        id="heartbeat", misfire_grace_time=60)

    # Position safety audit — verify all open positions have stops
    async def _safety_audit():
        from shared.ig_session import IGSharedSession
        shared = IGSharedSession.get_instance()
        if shared and shared.ig:
            await position_safety_audit(shared, telegram_cmd._send)

    dax_scheduler.add_job(_safety_audit, "interval", minutes=5,
        id="safety_audit", misfire_grace_time=60)

    dax_scheduler.start()

    # ── Build gold status string ────────────────────────────────
    ig_mode = "DEMO" if dax_config.IG_DEMO else "LIVE"

    gold_str = "disabled"
    try:
        from gold_bot import config as gold_config
        if gold_config.ENABLE_GOLD:
            sessions = list(gold_config.INSTRUMENTS["GOLD"]["sessions"].keys())
            weekly = " + Weekly" if gold_config.WEEKLY_ORB_ENABLED else ""
            gold_str = "15m ORB (" + ", ".join(sessions) + ")" + weekly
    except Exception:
        pass

    us30_str = "disabled"
    if spx_main is not None:
        us30_str = f"ASRS bar4/5 ({spx_main.config.IG_EPIC})"

    nikkei_str = "disabled"
    if nikkei_main is not None:
        nikkei_str = f"ASRS bar4/5 ({nikkei_main.config.IG_EPIC})"

    # ── Startup alert via Telegram (single combined message) ──
    await send_startup_alert(telegram_cmd._send, gold_str=gold_str, spx_str=us30_str, nikkei_str=nikkei_str)

    logger.info("Bot started: DAX=%s, S3 ORB=%s, US30=%s, Nikkei=%s, mode=%s",
                dax_config.IG_EPIC, gold_str, us30_str, nikkei_str, ig_mode)

    print("")
    print("=" * 56)
    print("  ASRS Trading Bot (DAX + US30 + Nikkei)")
    print("  IG Markets - Streaming Edition")
    print("=" * 56)
    print("  Broker:      IG %s" % ig_mode)
    print("  S1 DAX:      %s" % dax_config.IG_EPIC)
    print("  S3 US30:     %s" % us30_str)
    print("  S4 Nikkei:   %s" % nikkei_str)
    print("  Telegram:    Commands active")
    print("-" * 56)
    print("  DAX:    08:21 Morning -> 17:35 EOD (UK time)")
    print("  Gold:   07:00-09:00 London, 13:30-15:30 US (UTC)")
    print("  EUR:    07:00-12:00 London, 12:00-15:00 US (UTC)")
    print("  Nikkei: 00:21 Morning -> 06:05 EOD (UTC)")
    print("  Every 10m: Keepalive (REST + stream)")
    print("=" * 56)
    print("")

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if not shutdown_triggered:
            await _shutdown()


if __name__ == "__main__":
    asyncio.run(main())
