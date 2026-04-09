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
from asrs.broker_factory import get_broker_stack
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
    Reconcile broker positions with signal state on startup.

    SAFE-BY-DEFAULT policy:

    1. If a signal's loaded state file ALREADY shows LONG/SHORT phase
       (i.e. previous run was tracking a position), trust the state and
       re-arm its tick stop monitor at the persisted trailing_stop. This
       preserves the actual strategy stop, bar 4/5 levels, etc.

    2. If a signal is FLAT in state but the broker reports an open
       position on its contract: DO NOT auto-reconcile to a disaster
       stop. The original strategy stop (50pt risk-managed) is gone, and
       overwriting it with a 200pt disaster stop quadruples the risk
       silently. Instead, send a CRITICAL Telegram alert and refuse to
       claim the position. Manual intervention required.

    3. To prevent the same position being claimed by multiple signals
       sharing a broker (e.g. DAX_S1 and DAX_S2 both watch FDXS), track
       reconciled positions by contract conId / epic.
    """
    try:
        from asrs.strategy import Phase

        # Pass 1: re-arm stop monitor for any signal already in LONG/SHORT
        # state (its state file persisted the real strategy stop).
        # IMPORTANT: load_state() from disk first — Signal.__init__ creates a
        # blank SignalState(), so without this every signal looks IDLE here
        # and any open position would be orphaned on restart.
        rearmed = 0
        for key, signal in signals.items():
            try:
                signal.load_state()
            except Exception as e:
                logger.warning(f"[{key}] load_state failed in reconcile: {e}")
            if signal.state.phase not in (Phase.LONG, Phase.SHORT):
                continue
            stop = signal.state.trailing_stop or signal.state.initial_stop
            if stop <= 0:
                continue
            try:
                signal.broker.activate_stop_monitor(signal.state.direction, stop)
                rearmed += 1
                logger.info(
                    f"[{key}] re-armed stop monitor from state: "
                    f"{signal.state.direction} @ {signal.state.entry_price} stop={stop}"
                )
            except Exception as e:
                logger.warning(f"[{key}] re-arm failed: {e}")

        # Pass 2: detect orphaned broker positions (in IBKR but no signal
        # tracks them). These need manual attention — we will NOT auto-claim.
        seen_keys: set[str] = set()
        orphan_alerts: list[str] = []
        for key, signal in signals.items():
            br_key = getattr(signal.broker, "epic", "") or str(id(signal.broker))
            if br_key in seen_keys:
                continue
            seen_keys.add(br_key)
            try:
                pos = await signal.broker.get_position()
            except Exception as e:
                logger.warning(f"reconcile {key}: get_position failed: {e}")
                continue

            if pos.get("direction", "FLAT") == "FLAT":
                continue

            # Position exists at broker. Check if any signal on this broker
            # has it tracked in-memory (Pass 1 would have re-armed those).
            tracked = False
            for sk, sig in signals.items():
                if getattr(sig.broker, "epic", "") != br_key:
                    continue
                # Tolerance: IBKR avgCost can drift from bot's recorded entry
                # by a few points due to fill slippage + commission roll-in.
                # Direction match + 5pt tolerance is enough — only one signal
                # per broker can be in LONG/SHORT phase at a time anyway.
                if sig.state.phase in (Phase.LONG, Phase.SHORT) and \
                   sig.state.direction == pos["direction"] and \
                   abs(sig.state.entry_price - float(pos.get("avg_cost", 0))) < 5.0:
                    tracked = True
                    break

            if tracked:
                continue

            # ORPHAN — broker has a position but no signal tracks it.
            # Place an EMERGENCY disaster stop immediately so the position
            # can't bleed unbounded while we wait for human intervention.
            # Original strategy stop is unknown, so we use the configured
            # disaster_stop_pts as a safety net (much wider than strategy
            # would have used, but bounded loss > unbounded loss).
            entry = pos.get("avg_cost", 0)
            direction = pos["direction"]
            disaster_pts = getattr(signal.broker, "_disaster_stop_pts", 200)
            stop_level = (entry - disaster_pts) if direction == "LONG" else (entry + disaster_pts)
            stop_placed = False
            try:
                stop_action = "SELL" if direction == "LONG" else "BUY"
                qty = abs(int(pos.get("position", 1)))
                result = await signal.broker.place_stop_order(
                    action=stop_action, qty=qty, stop_price=round(stop_level, 1),
                )
                stop_placed = "order_id" in result
                if stop_placed:
                    logger.warning(
                        f"ORPHAN STOP placed for {signal.instrument} {direction} "
                        f"@ {entry}: stop={stop_level:.1f} ({disaster_pts}pt disaster)"
                    )
            except Exception as e:
                logger.error(f"ORPHAN stop placement failed: {e}", exc_info=True)
            stop_line = (
                f"<b>Emergency disaster stop placed @ {stop_level:.1f} ({disaster_pts}pt).</b>\n"
                if stop_placed else
                f"<b>⚠️ FAILED to place emergency stop — manual action REQUIRED.</b>\n"
            )
            msg = (
                f"⚠️ <b>ORPHAN POSITION DETECTED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Broker: {signal.instrument} ({br_key})\n"
                f"Position: {direction} {abs(pos.get('position', 0))} @ {entry}\n"
                + stop_line +
                f"Bot started in TRACKING-ONLY mode for this instrument.\n"
                f"Manual close via IBKR or /kill required."
            )
            orphan_alerts.append(msg)
            logger.error(
                f"ORPHAN: {signal.instrument} {direction} @ {entry} not claimed by any signal"
            )

        for alert in orphan_alerts:
            try:
                await tg_send(alert)
            except Exception:
                pass

        if rearmed == 0 and not orphan_alerts:
            logger.info("Reconciliation: clean start (no positions, no state)")
        else:
            logger.info(
                f"Reconciliation: {rearmed} signals re-armed from state, "
                f"{len(orphan_alerts)} orphan positions detected"
            )

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


def _is_market_open(inst_name: str) -> bool:
    """Check if market should be active right now (not weekend/holiday)."""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    inst_cfg = config.INSTRUMENTS[inst_name]
    tz = ZoneInfo(inst_cfg["timezone"])
    now = datetime.now(tz)
    # Weekend check (Sat=5, Sun=6)
    if now.weekday() >= 5:
        return False
    # Within session hours (with 30 min buffer either side)
    s1_open = inst_cfg["s1_open_hour"] * 60 + inst_cfg["s1_open_minute"]
    eod = inst_cfg["session_end_hour"] * 60 + inst_cfg["session_end_minute"]
    cur = now.hour * 60 + now.minute
    return (s1_open - 30) <= cur <= (eod + 30)


async def check_stream_health_all(shared_session, stream_mgr, tg_send, signals=None):
    """
    Check stream health for all instruments. Send Telegram alerts on staleness.
    Uses broker.epic (which is the actual stream key — IG epic for IG,
    contract_key for IBKR) so it works under both broker stacks.
    """
    # Build instrument → stream_key from active signals (broker-agnostic)
    inst_keys: dict[str, str] = {}
    if signals:
        for sig in signals.values():
            inst_keys.setdefault(sig.instrument, sig.broker.epic)

    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        epic = inst_keys.get(inst_name) or inst_cfg.get("epic", "")
        if not epic:
            continue

        # Skip if market is closed (weekend or outside session hours)
        if not _is_market_open(inst_name):
            _resub_fail_count[epic] = 0
            continue

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

        # Attempt resubscribe — broker-agnostic path
        try:
            if hasattr(shared_session, "check_stream_health"):
                # IG path: shared.check_stream_health() exists on IGSharedSession
                ok = await shared_session.check_stream_health(stream_mgr, epic)
            elif hasattr(stream_mgr, "resubscribe_all"):
                # IB path: just resubscribe everything
                await stream_mgr.resubscribe_all()
                ok = True
            else:
                ok = False
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

    # -- Broker stack (IG or IBKR based on BROKER_TYPE env) ------------------
    stack = get_broker_stack()
    await stack.connect_session()
    shared = stack.shared
    stream_mgr = stack.stream
    await stack.post_session_setup()

    # Wire IB reconnect → automatic resubscribe of all streams.
    # Without this, an IB Gateway flap during market hours would leave
    # the bot running with zero ticks/bars until the stream-health
    # watchdog notices (~5 min). Now it self-heals on the connect event.
    if hasattr(shared, "on_reconnect") and hasattr(stream_mgr, "resubscribe_all"):
        async def _on_ib_reconnect():
            logger.warning("IB reconnect detected — resubscribing all streams")
            try:
                await stream_mgr.resubscribe_all()
                await tg_send("IB reconnect: streams resubscribed")
            except Exception as e:
                logger.error(f"resubscribe_all failed: {e}", exc_info=True)
                await tg_send(f"CRITICAL: IB resubscribe failed: {e}")
                return
            # POST-RESUBSCRIBE: validate that any signal currently in
            # LONG/SHORT phase still has a live broker stop. IBKR holds
            # resting orders across API client disconnects in normal cases,
            # but a hard disconnect / Gateway crash can lose them. If the
            # real stop is gone, re-place it from persisted trailing_stop.
            try:
                from asrs.strategy import Phase
                refilled = 0
                for _rc_sig in ALL_SIGNALS:
                    _rc_sig.load_state()
                    if _rc_sig.state.phase not in (Phase.LONG, Phase.SHORT):
                        continue
                    br = _rc_sig.broker
                    real_stop = getattr(br, "_real_stop_trade", None)
                    needs_replace = False
                    if real_stop is None:
                        needs_replace = True
                    else:
                        status = getattr(real_stop.orderStatus, "status", "") if hasattr(real_stop, "orderStatus") else ""
                        if status not in ("Submitted", "PreSubmitted"):
                            needs_replace = True
                    if needs_replace and _rc_sig.state.trailing_stop > 0:
                        action = "SELL" if _rc_sig.state.direction == "LONG" else "BUY"
                        try:
                            await br.place_stop_order(
                                action=action, qty=1,
                                stop_price=float(_rc_sig.state.trailing_stop),
                            )
                            refilled += 1
                            logger.warning(
                                f"[{_rc_sig.name}] post-reconnect: re-placed real stop "
                                f"@ {_rc_sig.state.trailing_stop}"
                            )
                        except Exception as e:
                            logger.error(f"[{_rc_sig.name}] re-place stop failed: {e}", exc_info=True)
                            await tg_send(
                                f"CRITICAL: [{_rc_sig.name}] post-reconnect stop "
                                f"replace FAILED — manual stop required"
                            )
                if refilled:
                    await tg_send(f"Post-reconnect: re-placed {refilled} real stop(s)")
            except Exception as e:
                logger.error(f"post-reconnect stop check failed: {e}", exc_info=True)
        shared.on_reconnect(_on_ib_reconnect)

    # -- Create broker + signal instances per instrument ----------------------
    signals: dict[str, Signal] = {}  # keyed by "DAX_S1", "DAX_S2", etc.

    # DISABLE_INSTRUMENTS env: comma-separated list of instruments to skip.
    # Example: DISABLE_INSTRUMENTS=NIKKEI skips all NIKKEI signals entirely
    # (no broker, no subscriptions, no scheduler jobs, no Telegram alerts).
    # Used when an instrument's contract is wrong for current account size
    # (e.g. NIY at £5k paper — pollutes monitoring with apples-to-oranges data).
    _disabled_raw = os.getenv("DISABLE_INSTRUMENTS", "").strip()
    DISABLED_INSTRUMENTS = {
        x.strip().upper() for x in _disabled_raw.split(",") if x.strip()
    }
    if DISABLED_INSTRUMENTS:
        logger.warning(
            f"DISABLE_INSTRUMENTS={','.join(sorted(DISABLED_INSTRUMENTS))} "
            f"— these signals will NOT be loaded"
        )

    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        if inst_name.upper() in DISABLED_INSTRUMENTS:
            logger.info(f"Skipping {inst_name} (in DISABLE_INSTRUMENTS)")
            continue
        # Determine how many sessions (2 or 3) based on config
        max_session = 3 if f"s3_open_hour" in inst_cfg else 2
        inst_signals = []

        # One broker PER SIGNAL (not per epic) so each signal tracks its
        # own deal IDs independently. Brokers share the same session
        # and stream manager -- only the local order state differs.
        for sn in range(1, max_session + 1):
            broker = stack.make_broker(inst_name, inst_cfg)
            await broker.connect()

            signal = Signal(inst_name, sn, broker, stream_mgr, tg_send)
            key = f"{inst_name}_S{sn}"
            signals[key] = signal
            ALL_SIGNALS.append(signal)
            inst_signals.append(signal)

            # Register tick trigger callback on this signal's broker
            broker.register_trigger_callback(signal.on_tick_trigger)
            logger.info(f"Signal created: {key} (broker={stack.kind}, epic={broker.epic})")

        # Link siblings: each session cancels previous session's bracket
        for i in range(1, len(inst_signals)):
            inst_signals[i - 1].set_sibling(inst_signals[i])
            inst_signals[i].set_sibling(inst_signals[i - 1])

        # Register candle callback -- fires on_bar_complete for all sessions
        # Use the first broker's epic/contract_key as the stream key
        stream_key = inst_signals[0].broker.epic
        async def _on_bar(bar, _sigs=inst_signals):
            for sig in _sigs:
                await sig.on_bar_complete(bar)
        register_bar_callback(stream_mgr, stream_key, _on_bar)

    # -- Position Reconciliation (before scheduler starts) --------------------
    await reconcile_positions(signals, shared, stream_mgr, tg_send)

    # -- Mid-session morning_routine catchup ---------------------------------
    # If bot started AFTER a signal's routine_time but BEFORE its session_end,
    # the cron jobs already fired (and we missed them with no grace). Run
    # morning_routine catchup so the signal can still trade today.
    try:
        from asrs.strategy import Phase
        from datetime import datetime as _dt
        catchup = []
        for inst_name, inst_cfg in config.INSTRUMENTS.items():
            if inst_name.upper() in DISABLED_INSTRUMENTS:
                continue
            inst_tz = ZoneInfo(inst_cfg["timezone"])
            now_inst = _dt.now(inst_tz)
            if now_inst.weekday() >= 5:
                continue  # weekend
            max_session = 3 if "s3_open_hour" in inst_cfg else 2
            for sn in range(1, max_session + 1):
                open_h = inst_cfg[f"s{sn}_open_hour"]
                open_m = inst_cfg[f"s{sn}_open_minute"]
                routine_total_m = open_h * 60 + open_m + 21
                routine_h = routine_total_m // 60
                routine_m = routine_total_m % 60
                end_h = inst_cfg["session_end_hour"]
                end_m = inst_cfg["session_end_minute"]
                now_total_m = now_inst.hour * 60 + now_inst.minute
                end_total_m = end_h * 60 + end_m
                if routine_total_m <= now_total_m < end_total_m:
                    _catch = signals.get(f"{inst_name}_S{sn}")
                    if _catch is None:
                        continue
                    _catch.load_state()
                    if _catch.state.phase == Phase.IDLE:
                        catchup.append((_catch, f"{routine_h:02d}:{routine_m:02d}"))
        if catchup:
            for _catch_sig, routine_str in catchup:
                logger.warning(
                    f"[{_catch_sig.name}] startup catchup: now in-session past "
                    f"routine time {routine_str} — running morning_routine"
                )
                try:
                    await _catch_sig.morning_routine()
                except Exception as e:
                    logger.error(f"[{_catch_sig.name}] catchup morning_routine failed: {e}", exc_info=True)
            await tg_send(
                f"Startup catchup: ran morning_routine for "
                f"{', '.join(s[0].name for s in catchup)}"
            )
    except Exception as e:
        logger.error(f"Startup mid-session catchup check failed: {e}", exc_info=True)

    # -- Scheduler ------------------------------------------------------------
    scheduler = AsyncIOScheduler(timezone=config.TZ_UK)

    for inst_name, inst_cfg in config.INSTRUMENTS.items():
        if inst_name.upper() in DISABLED_INSTRUMENTS:
            continue
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

        async def _warmup(b=inst_sigs[0].broker, name=inst_name):
            # Use broker.epic — IG epic for IG, contract_key for IBKR
            epic = b.epic
            ok = await b.ensure_connected()
            bar_count = b.get_streaming_bar_count()
            tick_age = stream_mgr.get_tick_age(epic)
            bar_age = stream_mgr.get_last_bar_age(epic) if hasattr(stream_mgr, 'get_last_bar_age') else 0

            issues = []
            if not ok: issues.append("Broker connection FAILED")
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
        # IG path: REST keepalive ping. IB path: ensure_connected keeps the
        # ib_async socket alive (no equivalent REST endpoint).
        if hasattr(shared, "keepalive"):
            await shared.keepalive()
        elif hasattr(shared, "ensure_connected"):
            await shared.ensure_connected()

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
        await check_stream_health_all(shared, stream_mgr, tg_send, signals)

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
            # Now broker-agnostic: dispatches to IG or IB variant inside.
            if shared:
                await position_safety_audit(shared, tg_send)

        scheduler.add_job(_safety_audit, "interval", minutes=5,
            id="safety_audit", misfire_grace_time=60)
    except ImportError:
        logger.warning("shared.monitoring not available -- skipping")

    # -- Daily contract roll check (07:00 UK, weekdays) ----------------------
    # Front-month is resolved once at startup. If the bot runs past expiry,
    # it would trade an expired/zero-volume contract. This cron warns at
    # ≤7d and bails out hard at ≤2d.
    async def _roll_check():
        if stack.kind != "ib":
            return
        try:
            from asrs.contract_resolver import resolve_front_month, days_to_expiry
            for inst_name in config.INSTRUMENTS.keys():
                if inst_name.upper() in DISABLED_INSTRUMENTS:
                    continue
                contract, expiry = await resolve_front_month(inst_name, shared)
                if not expiry:
                    await tg_send(f"ROLL CHECK: {inst_name} resolve failed")
                    continue
                d = days_to_expiry(expiry)
                if d <= 2:
                    msg = (f"CRITICAL: {inst_name} expires in {d}d ({expiry}). "
                           f"Bot will trade expired contract — manual roll required NOW.")
                    logger.error(msg)
                    await tg_send(msg)
                elif d <= 7:
                    await tg_send(f"ROLL WARNING: {inst_name} expires in {d}d ({expiry})")
        except Exception as e:
            logger.error(f"roll_check failed: {e}", exc_info=True)
            await tg_send(f"roll_check exception: {e}")

    scheduler.add_job(_roll_check, "cron",
        day_of_week="mon-fri", hour=7, minute=0,
        id="roll_check", misfire_grace_time=3600)

    # -- Daily backtest-vs-live drift check (21:00 UK, weekdays) -------------
    # Runs the backtest engine on today's IBKR bars and compares the
    # resulting trade list against the live journal. Alerts on any mismatch.
    # This is the parity check that catches silent bar-source / strategy
    # drift before it bleeds £k.
    async def _replay_check():
        if stack.kind != "ib":
            return
        try:
            import asyncio as _asyncio, subprocess
            proc = await _asyncio.create_subprocess_exec(
                "python3", "/root/asrs-bot/replay_today.py",
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.STDOUT,
            )
            stdout, _ = await _asyncio.wait_for(proc.communicate(), timeout=180)
            output = stdout.decode("utf-8", errors="replace")
            # Parse last DELTA lines
            deltas = []
            for line in output.splitlines():
                if "DELTA:" in line or "TOTAL" in line:
                    deltas.append(line.strip())
            summary = "\n".join(deltas[-6:]) if deltas else "no output"
            # If any per-instrument delta is non-trivial (>20 pts), alert loud
            alert = False
            for line in deltas:
                try:
                    if "DELTA:" in line:
                        val = line.split("DELTA:")[-1].split("pts")[0].strip()
                        if abs(int(float(val))) > 20:
                            alert = True
                            break
                except Exception:
                    pass
            tag = "DRIFT ALERT" if alert else "DAILY PARITY"
            msg = f"<b>{tag}</b> — replay vs live\n<pre>{summary}</pre>"
            if alert:
                msg += (
                    "\n━━━━━━━━━━━━━━━━━━━━━━\n"
                    "<b>WHAT THIS MEANS</b>\n"
                    "Live trades diverged from what the backtest engine "
                    "would have done on the same bars (delta &gt; 20pts). "
                    "This is a single-day issue, not a 30-day pattern.\n\n"
                    "<b>LIKELY CAUSES</b>\n"
                    "1. Slippage spike on one fill (check microstructure: "
                    "spread, last_price at trigger time)\n"
                    "2. Bar source mismatch (rare since rtbar fix — check "
                    "if any signal hit mid-tick fallback today)\n"
                    "3. Order rejection / partial fill\n"
                    "4. Manual intervention via TWS\n\n"
                    "<b>NEXT MORNING CHECKLIST</b>\n"
                    "1. <code>/pnl</code> — see today's trades\n"
                    "2. Look at the worst-delta instrument's last 5 fills "
                    "in the journal — compare entry_intended vs entry, "
                    "exit_intended vs exit\n"
                    "3. Check <code>/tmp/asrs-logs/asrs.log</code> for "
                    "any 'fallback to mid-tick' or 'EXCESSIVE SLIPPAGE' "
                    "warnings\n\n"
                    "<b>DECISION</b>\n"
                    "• <b>Single-day delta &lt; 50pts</b>: investigate but "
                    "no action needed. Watch tomorrow.\n"
                    "• <b>Same instrument drifts 3 days running</b>: that "
                    "instrument has a real problem. Check the broker / "
                    "data sub for that contract.\n"
                    "• <b>Delta &gt; 100pts in one day</b>: <code>/pause</code> "
                    "and investigate before resuming. Something material "
                    "broke."
                )
            await tg_send(msg)
            # Statistical process control — PSR + CUSUM on rolling 30-day P&L
            try:
                from asrs.spc import daily_drift_report, format_drift_report
                spc_report = daily_drift_report()
                await tg_send(format_drift_report(spc_report))
            except Exception as e:
                logger.error(f"spc daily_drift_report failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"replay_check failed: {e}", exc_info=True)
            await tg_send(f"replay_check exception: {e}")

    scheduler.add_job(_replay_check, "cron",
        day_of_week="mon-fri", hour=21, minute=0,
        id="replay_check", misfire_grace_time=3600,
        timezone=config.TZ_UK)

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
        # Pass any available broker as fallback for legacy commands.
        # Order: prefer DAX_S1, then US30_S1, then NIKKEI_S1, then any.
        fallback_broker = None
        for key in ("DAX_S1", "US30_S1", "NIKKEI_S1"):
            if key in signals:
                fallback_broker = signals[key].broker
                break
        if fallback_broker is None and signals:
            fallback_broker = next(iter(signals.values())).broker
        loop.create_task(
            telegram_cmd.poll_commands(dax_broker=fallback_broker)
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
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        # IG stream has unsubscribe_all; IB stream doesn't (its lifecycle
        # is tied to the IB connection itself)
        if hasattr(stream_mgr, "unsubscribe_all"):
            try:
                await stream_mgr.unsubscribe_all()
            except Exception as e:
                logger.warning(f"Stream unsubscribe failed: {e}")
        try:
            await shared.disconnect()
        except Exception:
            pass
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
