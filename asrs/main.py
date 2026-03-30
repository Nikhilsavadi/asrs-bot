"""
main.py -- Single entry point for all 6 ASRS signals
=====================================================
Replaces run_all.py + dax_bot/main.py + spx_bot/main.py + nikkei_bot/main.py.

Creates 6 Signal instances (DAX_S1, DAX_S2, US30_S1, US30_S2, NIKKEI_S1, NIKKEI_S2),
wires them to one shared IG session, and schedules everything.

Usage:
    python -m asrs.main
"""

import asyncio
import os
import sys
import signal as sig
import logging
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.events import EVENT_JOB_MISSED, EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from asrs import config
from asrs.strategy import Signal
from asrs.broker import IGBroker
from asrs.stream import register_bar_callback
from asrs.alerts import send as tg_send

# -- Logging ------------------------------------------------------------------
LOG_DIR = os.getenv("LOG_DIR", "/data")
LOG_FILE = os.path.join(LOG_DIR, "asrs.log")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
os.makedirs(LOG_DIR, exist_ok=True)
_fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logging.getLogger().addHandler(_fh)

logger = logging.getLogger("ASRS")

# Global registry of all signals (for Telegram commands)
ALL_SIGNALS: list[Signal] = []


async def main():
    """Boot all 6 signals with one shared IG session."""

    # Write PID for healthcheck
    with open("/tmp/asrs.pid", "w") as f:
        f.write(str(os.getpid()))

    loop = asyncio.get_event_loop()
    shutdown_triggered = False

    # -- Journal DB -----------------------------------------------------------
    from asrs.journal import init_db, migrate_csv, seed_scaling_ladder
    init_db()
    migrate_csv()
    seed_scaling_ladder(os.getenv("SCALING_LADDER", ""), "ALL")

    # -- Shared IG session (ONE REST + ONE Lightstreamer) ---------------------
    from shared.ig_session import IGSharedSession
    from shared.ig_stream import IGStreamManager

    shared = IGSharedSession.get_instance()
    await shared.connect()
    stream_mgr = IGStreamManager(shared)

    # Subscribe to trades stream
    if shared.acc_number:
        await stream_mgr.subscribe_trades(shared.acc_number)

    # Re-subscribe all streams after reconnect
    shared.on_reconnect(stream_mgr.resubscribe_all)

    # -- Create broker + signal instances per instrument ----------------------
    signals: dict[str, Signal] = {}  # keyed by "DAX_S1", "DAX_S2", etc.

    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        epic = inst_cfg["epic"]
        currency = inst_cfg["currency"]

        # Subscribe to streaming for this epic (safe to call multiple times)
        await stream_mgr.subscribe_ticks(epic)
        await stream_mgr.subscribe_candles(epic)

        # One broker PER SIGNAL (not per epic) so each signal tracks its
        # own deal IDs independently. Brokers share the same IG session
        # and stream manager -- only the local order state differs.
        # Note: S2's broker registers a second tick callback on the same
        # epic, which is fine -- IGStreamManager supports multiple callbacks.
        for sn in (1, 2):
            broker = IGBroker(
                shared, stream_mgr, epic, currency,
                disaster_stop_pts=inst_cfg["disaster_stop_pts"],
                max_spread_pts=inst_cfg["max_spread"],
            )
            signal = Signal(inst_name, sn, broker, stream_mgr, tg_send)
            key = f"{inst_name}_S{sn}"
            signals[key] = signal
            ALL_SIGNALS.append(signal)

            # Register tick trigger callback on this signal's broker
            broker.register_trigger_callback(signal.on_tick_trigger)
            logger.info(f"Signal created: {key} (epic={epic})")

        # Link S1 <-> S2 siblings (R24: S2 cancels S1 bracket)
        s1 = signals[f"{inst_name}_S1"]
        s2 = signals[f"{inst_name}_S2"]
        s1.set_sibling(s2)
        s2.set_sibling(s1)

        # Register candle callback -- fires on_bar_complete for both S1 and S2
        async def _on_bar(bar, _s1=s1, _s2=s2):
            await _s1.on_bar_complete(bar)
            await _s2.on_bar_complete(bar)
        register_bar_callback(stream_mgr, epic, _on_bar)

    # -- Scheduler ------------------------------------------------------------
    scheduler = AsyncIOScheduler(timezone=config.TZ_UK)

    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        sched_tz = ZoneInfo(inst_cfg["scheduler_timezone"])
        s1 = signals[f"{inst_name}_S1"]
        s2 = signals[f"{inst_name}_S2"]
        prefix = inst_name.lower()

        # -- S1 morning routine (1 min after bar 4 close) --------------------
        s1_open_h = inst_cfg["s1_open_hour"]
        s1_open_m = inst_cfg["s1_open_minute"]
        # Bar 4 closes at open + 20min. Schedule at open + 21min.
        s1_routine_m = s1_open_m + 21
        s1_routine_h = s1_open_h + s1_routine_m // 60
        s1_routine_m = s1_routine_m % 60

        scheduler.add_job(s1.morning_routine, "cron",
            day_of_week="mon-fri", hour=s1_routine_h, minute=s1_routine_m,
            id=f"{prefix}_s1_morning", misfire_grace_time=120,
            timezone=sched_tz)

        # Failsafe: retry 4 minutes later if still IDLE
        fs_m = s1_routine_m + 4
        fs_h = s1_routine_h + fs_m // 60
        fs_m = fs_m % 60

        async def _failsafe(sig=s1):
            sig.load_state()
            if sig.state.phase == "IDLE":
                logger.warning(f"[{sig.name}] Failsafe: still IDLE -- retrying")
                await sig.morning_routine()

        scheduler.add_job(_failsafe, "cron",
            day_of_week="mon-fri", hour=fs_h, minute=fs_m,
            id=f"{prefix}_s1_failsafe", misfire_grace_time=120,
            timezone=sched_tz)

        # -- S2 morning routine (1 min after S2 bar 4 close) -----------------
        s2_open_h = inst_cfg["s2_open_hour"]
        s2_open_m = inst_cfg["s2_open_minute"]
        s2_routine_m = s2_open_m + 21
        s2_routine_h = s2_open_h + s2_routine_m // 60
        s2_routine_m = s2_routine_m % 60

        scheduler.add_job(s2.morning_routine, "cron",
            day_of_week="mon-fri", hour=s2_routine_h, minute=s2_routine_m,
            id=f"{prefix}_s2_morning", misfire_grace_time=120,
            timezone=sched_tz)

        # -- Monitor cycle: every minute during trading hours -----------------
        # From 1 hour before S1 open to session end
        monitor_start_h = max(0, s1_open_h - 1)
        monitor_end_h = inst_cfg["session_end_hour"]

        hour_range = f"{monitor_start_h}-{monitor_end_h}"

        async def _monitor(s1=s1, s2=s2):
            await s1.monitor_cycle()
            await s2.monitor_cycle()

        scheduler.add_job(_monitor, "cron",
            day_of_week="mon-fri", hour=hour_range, minute="*",
            id=f"{prefix}_monitor", misfire_grace_time=30,
            timezone=sched_tz)

        # -- End of day: force close ------------------------------------------
        eod_h = inst_cfg["session_end_hour"]
        eod_m = inst_cfg["session_end_minute"] + 5  # 5 min after session end
        eod_h = eod_h + eod_m // 60
        eod_m = eod_m % 60

        async def _eod(s1=s1, s2=s2):
            await s1.end_of_day()
            await s2.end_of_day()

        scheduler.add_job(_eod, "cron",
            day_of_week="mon-fri", hour=eod_h, minute=eod_m,
            id=f"{prefix}_eod", misfire_grace_time=120,
            timezone=sched_tz)

        # -- Pre-trade warmup (10 min before S1 routine) ----------------------
        warmup_m = s1_routine_m - 10
        warmup_h = s1_routine_h
        if warmup_m < 0:
            warmup_m += 60
            warmup_h -= 1

        async def _warmup(b=s1.broker, epic=inst_cfg["epic"], name=inst_name, s=s1):
            ok = await b.ensure_connected()
            bar_count = b.get_streaming_bar_count()
            tick_age = stream_mgr.get_tick_age(epic)
            bar_age = stream_mgr.get_last_bar_age(epic) if hasattr(stream_mgr, 'get_last_bar_age') else 0

            issues = []
            if not ok: issues.append("IG connection FAILED")
            if bar_count == 0: issues.append("No streaming bars")
            if tick_age > 120: issues.append(f"Ticks stale ({tick_age:.0f}s)")
            if bar_age > 600: issues.append(f"Bars stale ({bar_age:.0f}s)")

            if issues:
                await tg_send(f"⚠️ [{name}] PRE-TRADE CHECK\n" + "\n".join(f"  ❌ {i}" for i in issues))
            else:
                await tg_send(
                    f"✅ [{name}] PRE-TRADE CHECK\n"
                    f"  IG: Connected\n"
                    f"  Bars: {bar_count}\n"
                    f"  Morning routine in 10 mins"
                )

        scheduler.add_job(_warmup, "cron",
            day_of_week="mon-fri", hour=warmup_h, minute=warmup_m,
            id=f"{prefix}_warmup", misfire_grace_time=120,
            timezone=sched_tz)

    # -- Session keepalive every 10 minutes -----------------------------------
    async def _keepalive():
        await shared.keepalive()

    scheduler.add_job(_keepalive, "interval", minutes=10,
        id="ig_keepalive", misfire_grace_time=60)

    # -- Monitoring: missed job alerts ----------------------------------------
    try:
        from shared.monitoring import (
            create_job_listener, heartbeat_ping, send_startup_alert,
            position_safety_audit,
        )
        job_listener = create_job_listener(tg_send, loop)
        scheduler.add_listener(job_listener, EVENT_JOB_MISSED | EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

        scheduler.add_job(heartbeat_ping, "interval", minutes=5,
            id="heartbeat", misfire_grace_time=60)

        async def _safety_audit():
            if shared and shared.ig:
                await position_safety_audit(shared, tg_send)

        scheduler.add_job(_safety_audit, "interval", minutes=5,
            id="safety_audit", misfire_grace_time=60)
    except ImportError:
        logger.warning("shared.monitoring not available -- skipping")

    # -- Monthly report (R7 in spec: 1st of month, 08:00 UK) -----------------
    async def _monthly_report():
        try:
            from asrs.journal import get_monthly_report
            report = get_monthly_report()
            await tg_send(report)
        except Exception as e:
            logger.warning(f"Monthly report failed: {e}")

    scheduler.add_job(_monthly_report, "cron",
        day=1, hour=8, minute=0,
        id="monthly_report", misfire_grace_time=3600)

    # -- Telegram command handler ---------------------------------------------
    try:
        import telegram_cmd
        loop.create_task(
            telegram_cmd.poll_commands(
                # Pass first DAX broker for backward compat with /kill etc
                dax_broker=signals["DAX_S1"].broker
            )
        )
    except ImportError:
        logger.warning("telegram_cmd not available")

    # -- Graceful shutdown ----------------------------------------------------
    async def _shutdown():
        nonlocal shutdown_triggered
        if shutdown_triggered:
            return
        shutdown_triggered = True
        logger.info("=== SHUTDOWN ===")
        scheduler.shutdown(wait=False)
        await stream_mgr.unsubscribe_all()
        await shared.disconnect()
        loop.stop()

    def _handle_signal():
        if not shutdown_triggered:
            loop.create_task(_shutdown())

    for s in (sig.SIGTERM, sig.SIGINT):
        loop.add_signal_handler(s, _handle_signal)

    # -- Start ----------------------------------------------------------------
    scheduler.start()

    ig_mode = "DEMO" if config.IG_DEMO else "LIVE"
    signal_list = ", ".join(s.name for s in ALL_SIGNALS)

    try:
        await tg_send(
            f"<b>ASRS Bot Started</b> [{ig_mode}]\n"
            f"Signals: {signal_list}\n"
            f"Instruments: {', '.join(config.INSTRUMENTS.keys())}"
        )
    except Exception:
        pass

    logger.info(f"Bot started: {signal_list}, mode={ig_mode}")
    print("")
    print("=" * 56)
    print("  ASRS Trading Bot (Unified)")
    print(f"  Mode: IG {ig_mode}")
    print("=" * 56)
    for s in ALL_SIGNALS:
        tz_name = s.cfg["scheduler_timezone"]
        print(f"  {s.name}: {s.cfg['epic']} ({tz_name})")
    print("-" * 56)
    print("  Every 10m: Keepalive | Every 5m: Heartbeat + Safety")
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
