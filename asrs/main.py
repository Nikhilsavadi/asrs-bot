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
import time
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

# Heartbeat file path -- must be inside /app/data (volume-mounted to host ./data)
HEARTBEAT_FILE = "/app/data/heartbeat"

# Stream health: track consecutive resubscribe failures per epic
_resub_fail_count: dict[str, int] = {}


async def reconcile_positions(signals, shared_session, stream_mgr, tg_send):
    """
    Reconcile open IG positions with signal state on startup.

    After a VPS restart mid-trade, the bot has no in-memory state.
    This fetches all open positions from IG and matches each to the
    correct signal by epic + direction, restoring LONG/SHORT state
    and activating the disaster stop monitor.
    """
    try:
        positions = await shared_session.rest_call(
            shared_session.ig.fetch_open_positions
        )
        if positions is None:
            logger.info("Reconciliation: No open positions -- clean start")
            return

        pos_list = (positions if isinstance(positions, list)
                    else positions.to_dict("records") if hasattr(positions, "to_dict")
                    else [])

        if not pos_list:
            logger.info("Reconciliation: No open positions -- clean start")
            return

        # Build epic -> list of positions
        epic_positions: dict[str, list] = {}
        for pos in pos_list:
            epic = pos.get("epic", "")
            epic_positions.setdefault(epic, []).append(pos)

        matched = 0
        for key, signal in signals.items():
            epic = signal.cfg["epic"]
            if epic not in epic_positions:
                continue

            for pos in epic_positions[epic]:
                direction_raw = pos.get("direction", "")
                direction = "LONG" if direction_raw == "BUY" else "SHORT"
                entry_level = float(pos.get("level") or pos.get("openLevel", 0))
                size = float(pos.get("dealSize") or pos.get("size", 0))
                deal_id = pos.get("dealId", "")

                if entry_level <= 0:
                    continue

                # Only reconcile signals that are currently IDLE or have no position
                from asrs.strategy import Phase
                if signal.state.phase in (Phase.LONG, Phase.SHORT):
                    continue  # already tracking a position

                # Set signal state
                signal.state.phase = Phase.LONG if direction == "LONG" else Phase.SHORT
                signal.state.direction = direction
                signal.state.entry_price = entry_level
                if deal_id:
                    signal.state.deal_ids = [deal_id]

                # Track deal in broker
                signal.broker._position_deal_ids[deal_id] = {
                    "direction": direction_raw, "size": size, "level": entry_level,
                }

                # Activate disaster stop monitor
                disaster_pts = signal.cfg["disaster_stop_pts"]
                if direction == "LONG":
                    stop_level = entry_level - disaster_pts
                else:
                    stop_level = entry_level + disaster_pts

                signal.state.initial_stop = stop_level
                signal.state.trailing_stop = stop_level
                signal.broker.activate_stop_monitor(direction, stop_level)

                signal.save_state()
                matched += 1

                msg = (
                    f"[{signal.name}] RECONCILED: {direction} @ {entry_level}, "
                    f"stop monitor active @ {stop_level}"
                )
                logger.info(msg)
                await tg_send(msg)

                # Remove this position so it doesn't match another signal
                epic_positions[epic].remove(pos)
                break  # one position per signal

        if matched == 0:
            logger.info("Reconciliation: Positions found but none matched signals")

    except Exception as e:
        logger.error(f"Reconciliation failed: {e}", exc_info=True)
        try:
            await tg_send(f"RECONCILIATION FAILED: {e}")
        except Exception:
            pass


def write_heartbeat():
    """Write current timestamp to heartbeat file for external watchdog."""
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        logger.warning(f"Heartbeat write failed: {e}")


async def check_stream_health_all(shared_session, stream_mgr, tg_send):
    """
    Check stream health for all epics. Send Telegram alerts on staleness.
    Track consecutive resubscribe failures -- critical alert after 3.
    """
    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        epic = inst_cfg["epic"]

        tick_age = stream_mgr.get_tick_age(epic)
        bar_age = stream_mgr.get_last_bar_age(epic)

        # Healthy: ticks within 120s AND bars within 330s (just over one 5-min bar)
        if tick_age < 120 and bar_age < 330:
            _resub_fail_count[epic] = 0
            continue

        # Build alert message
        issues = []
        if tick_age >= 120:
            issues.append(f"ticks stale ({tick_age:.0f}s)")
        if bar_age >= 330:
            issues.append(f"bars stale ({bar_age:.0f}s)")

        alert_msg = f"[{inst_name}] Stream: {', '.join(issues)} -- resubscribing..."
        logger.warning(alert_msg)
        await tg_send(alert_msg)

        # Attempt resubscribe via existing check_stream_health
        try:
            ok = await shared_session.check_stream_health(stream_mgr, epic)
            if ok:
                _resub_fail_count[epic] = 0
                logger.info(f"[{inst_name}] Stream resubscribe successful")
            else:
                count = _resub_fail_count.get(epic, 0) + 1
                _resub_fail_count[epic] = count
                logger.error(f"[{inst_name}] Stream resubscribe failed ({count}/3)")
                if count >= 3:
                    await tg_send(
                        f"CRITICAL: [{inst_name}] Stream resubscribe failed {count} times! "
                        f"Manual intervention may be needed."
                    )
        except Exception as e:
            count = _resub_fail_count.get(epic, 0) + 1
            _resub_fail_count[epic] = count
            logger.error(f"[{inst_name}] Stream health check exception: {e}")
            if count >= 3:
                await tg_send(
                    f"CRITICAL: [{inst_name}] Stream resubscribe failed {count} times! "
                    f"Error: {e}"
                )


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
        # Determine how many sessions (2 or 3) based on config
        max_session = 3 if f"s3_open_hour" in inst_cfg else 2
        inst_signals = []

        for sn in range(1, max_session + 1):
            broker = IGBroker(
                shared, stream_mgr, epic, currency,
                disaster_stop_pts=inst_cfg["disaster_stop_pts"],
                max_spread_pts=inst_cfg["max_spread"],
            )
            signal = Signal(inst_name, sn, broker, stream_mgr, tg_send)
            key = f"{inst_name}_S{sn}"
            signals[key] = signal
            ALL_SIGNALS.append(signal)
            inst_signals.append(signal)

            # Register tick trigger callback on this signal's broker
            broker.register_trigger_callback(signal.on_tick_trigger)
            logger.info(f"Signal created: {key} (epic={epic})")

        # Link siblings: each session cancels previous session's bracket
        for i in range(1, len(inst_signals)):
            inst_signals[i - 1].set_sibling(inst_signals[i])
            inst_signals[i].set_sibling(inst_signals[i - 1])

        # Register candle callback -- fires on_bar_complete for all sessions
        async def _on_bar(bar, _sigs=inst_signals):
            for sig in _sigs:
                await sig.on_bar_complete(bar)
        register_bar_callback(stream_mgr, epic, _on_bar)

    # -- Position Reconciliation (before scheduler starts) --------------------
    await reconcile_positions(signals, shared, stream_mgr, tg_send)

    # -- Scheduler ------------------------------------------------------------
    scheduler = AsyncIOScheduler(timezone=config.TZ_UK)

    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        sched_tz = ZoneInfo(inst_cfg["scheduler_timezone"])
        prefix = inst_name.lower()
        max_session = 3 if "s3_open_hour" in inst_cfg else 2
        inst_sigs = [signals[f"{inst_name}_S{sn}"] for sn in range(1, max_session + 1)]

        # -- Morning routines + failsafes for each session --------------------
        for sn in range(1, max_session + 1):
            signal_obj = signals[f"{inst_name}_S{sn}"]
            open_h = inst_cfg[f"s{sn}_open_hour"]
            open_m = inst_cfg[f"s{sn}_open_minute"]
            routine_m = open_m + 21
            routine_h = open_h + routine_m // 60
            routine_m = routine_m % 60

            scheduler.add_job(signal_obj.morning_routine, "cron",
                day_of_week="mon-fri", hour=routine_h, minute=routine_m,
                id=f"{prefix}_s{sn}_morning", misfire_grace_time=120,
                timezone=sched_tz)

            # Failsafe: retry 4 minutes later if still IDLE
            fs_m = routine_m + 4
            fs_h = routine_h + fs_m // 60
            fs_m = fs_m % 60

            async def _failsafe(_s=signal_obj):
                _s.load_state()
                if _s.state.phase == "IDLE":
                    logger.warning(f"[{_s.name}] Failsafe: still IDLE -- retrying")
                    await _s.morning_routine()

            scheduler.add_job(_failsafe, "cron",
                day_of_week="mon-fri", hour=fs_h, minute=fs_m,
                id=f"{prefix}_s{sn}_failsafe", misfire_grace_time=120,
                timezone=sched_tz)

        # -- Monitor cycle: every minute during trading hours -----------------
        s1_open_h = inst_cfg["s1_open_hour"]
        monitor_start_h = max(0, s1_open_h - 1)
        monitor_end_h = inst_cfg["session_end_hour"]
        hour_range = f"{monitor_start_h}-{monitor_end_h}"

        async def _monitor(_sigs=inst_sigs):
            for sig in _sigs:
                await sig.monitor_cycle()

        scheduler.add_job(_monitor, "cron",
            day_of_week="mon-fri", hour=hour_range, minute="*",
            id=f"{prefix}_monitor", misfire_grace_time=30,
            timezone=sched_tz)

        # -- End of day: force close ------------------------------------------
        eod_h = inst_cfg["session_end_hour"]
        eod_m = inst_cfg["session_end_minute"] + 5
        eod_h = eod_h + eod_m // 60
        eod_m = eod_m % 60

        async def _eod(_sigs=inst_sigs):
            for sig in _sigs:
                await sig.end_of_day()

        scheduler.add_job(_eod, "cron",
            day_of_week="mon-fri", hour=eod_h, minute=eod_m,
            id=f"{prefix}_eod", misfire_grace_time=120,
            timezone=sched_tz)

        # -- Pre-trade warmup (30 min before S1 routine) ----------------------
        s1_open_h = inst_cfg["s1_open_hour"]
        s1_open_m = inst_cfg["s1_open_minute"]
        s1_routine_m = s1_open_m + 21
        s1_routine_h = s1_open_h + s1_routine_m // 60
        s1_routine_m = s1_routine_m % 60
        warmup_m = s1_routine_m - 30
        warmup_h = s1_routine_h
        if warmup_m < 0:
            warmup_m += 60
            warmup_h -= 1

        async def _warmup(b=inst_sigs[0].broker, epic=inst_cfg["epic"], name=inst_name):
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
                    f"  Morning routine in 30 mins"
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

    # -- Heartbeat writer (every 60s for external watchdog) ------------------
    def _write_heartbeat():
        write_heartbeat()

    scheduler.add_job(_write_heartbeat, "interval", seconds=60,
        id="heartbeat_writer", misfire_grace_time=30)
    write_heartbeat()  # Write immediately on startup

    # -- Stream health check (every 5 minutes, with Telegram alerts) ---------
    async def _stream_health():
        await check_stream_health_all(shared, stream_mgr, tg_send)

    scheduler.add_job(_stream_health, "interval", minutes=5,
        id="stream_health", misfire_grace_time=60)

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

    # -- Visual performance report (PNG) ----------------------------------------
    async def _send_visual_report():
        try:
            from reports import generate_full_report
            from telegram_cmd import _send_photo
            image_bytes = generate_full_report()
            await _send_photo(image_bytes, caption="ASRS Performance Report")
        except Exception as e:
            logger.warning(f"Visual report failed: {e}")

    # Sunday 08:00 UK
    scheduler.add_job(_send_visual_report, "cron",
        day_of_week="sun", hour=8, minute=0,
        id="weekly_visual_report", misfire_grace_time=3600,
        timezone=config.TZ_UK)

    # 1st of month 08:00 UK
    scheduler.add_job(_send_visual_report, "cron",
        day=1, hour=8, minute=5,
        id="monthly_visual_report", misfire_grace_time=3600,
        timezone=config.TZ_UK)

    # -- Governance alerts (every 5 minutes) ------------------------------------
    async def _governance_check():
        try:
            from reports import check_governance_alerts
            alerts = check_governance_alerts()
            for alert in alerts:
                await tg_send(f"🚨 <b>GOVERNANCE</b>\n{alert}")
        except Exception as e:
            logger.warning(f"Governance check failed: {e}")

    scheduler.add_job(_governance_check, "interval", minutes=5,
        id="governance_check", misfire_grace_time=60)

    # -- Weekly report (Friday 17:00 UK) ----------------------------------------
    async def _weekly_report():
        try:
            from shared.journal_db import get_weekly_pnl, get_cumulative_pnl_by_instrument
            from datetime import datetime as dt
            from zoneinfo import ZoneInfo
            now = dt.now(ZoneInfo("Europe/London"))

            week = get_weekly_pnl()
            ytd_by_inst = get_cumulative_pnl_by_instrument()

            # This month
            first_of_month = now.replace(day=1).date().isoformat()
            today_str = now.date().isoformat()
            from shared.journal_db import _get_conn
            conn = _get_conn()
            month_row = conn.execute(
                "SELECT COALESCE(SUM(pnl_pts), 0) as pnl, COUNT(*) as trades "
                "FROM trades WHERE date >= ? AND date <= ?",
                (first_of_month, today_str),
            ).fetchone()
            month_pnl = month_row["pnl"] if month_row else 0
            month_trades = month_row["trades"] if month_row else 0

            # YTD
            year_start = f"{now.year}-01-01"
            ytd_row = conn.execute(
                "SELECT COALESCE(SUM(pnl_pts), 0) as pnl, COUNT(*) as trades "
                "FROM trades WHERE date >= ? AND date <= ?",
                (year_start, today_str),
            ).fetchone()
            ytd_pnl = ytd_row["pnl"] if ytd_row else 0
            ytd_trades = ytd_row["trades"] if ytd_row else 0

            # Per instrument this week
            inst_lines = []
            for inst in ["DAX", "US30", "NIKKEI"]:
                iw = get_weekly_pnl(inst)
                if iw["trades"] > 0:
                    p = iw["pnl_gbp"]
                    inst_lines.append(f"  {inst}: {'+' if p >= 0 else ''}{p:.1f}pts ({iw['trades']} trades)")

            msg = (
                f"<b>WEEKLY REPORT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Week ending {now.strftime('%d %b %Y')}\n\n"
                f"<b>This Week:</b> {'+' if week['pnl_gbp'] >= 0 else ''}{week['pnl_gbp']:.1f}pts ({week['trades']} trades)\n"
            )
            if inst_lines:
                msg += "\n".join(inst_lines) + "\n"
            msg += (
                f"\n<b>This Month:</b> {'+' if month_pnl >= 0 else ''}{month_pnl:.1f}pts ({month_trades} trades)\n"
                f"<b>YTD:</b> {'+' if ytd_pnl >= 0 else ''}{ytd_pnl:.1f}pts ({ytd_trades} trades)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━"
            )
            await tg_send(msg)
        except Exception as e:
            logger.warning(f"Weekly report failed: {e}")

    scheduler.add_job(_weekly_report, "cron",
        day_of_week="fri", hour=20, minute=0,
        id="weekly_report", misfire_grace_time=3600,
        timezone=config.TZ_UK)

    # -- Dashboard sync (every 30 mins) ----------------------------------------
    async def _dashboard_sync():
        try:
            from sync_to_railway import sync_all
            sync_all()
        except Exception as e:
            logger.warning(f"Dashboard sync failed: {e}")

    scheduler.add_job(_dashboard_sync, "interval", minutes=30,
        id="dashboard_sync", misfire_grace_time=120)

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
    # Ensure 'from asrs.main import ALL_SIGNALS' works even when run as __main__
    import sys
    sys.modules["asrs.main"] = sys.modules[__name__]
    asyncio.run(main())
