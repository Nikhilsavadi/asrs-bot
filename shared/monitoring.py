"""
monitoring.py — Alerting & Monitoring for ASRS Bots
═══════════════════════════════════════════════════════

1. Missed job alerts (APScheduler EVENT_JOB_MISSED / EVENT_JOB_ERROR)
2. Heartbeat ping to external uptime monitor (Healthchecks.io etc.)
3. Container startup/restart alert via Telegram
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)

TZ_UK = ZoneInfo("Europe/London")

# Ping URL for push-based uptime monitoring (e.g. https://hc-ping.com/xxx)
HEALTHCHECK_PING_URL = os.getenv("HEALTHCHECK_PING_URL", "")

# Track last heartbeat for /health endpoint
_last_heartbeat: float = 0
_start_time: float = time.time()
_last_dax_job: str = ""


# ── Missed Job Alerts ─────────────────────────────────────────────────────

def create_job_listener(send_func, loop: asyncio.AbstractEventLoop):
    """
    Create an APScheduler event listener for missed/errored jobs.

    Args:
        send_func: async function to send Telegram message (telegram_cmd._send)
        loop: the asyncio event loop to dispatch from scheduler thread
    """
    from apscheduler.events import JobEvent

    def _on_job_event(event: JobEvent):
        global _last_dax_job

        job_id = getattr(event, "job_id", "unknown")
        now = datetime.now(TZ_UK).strftime("%H:%M:%S")

        # Track last job execution
        if "dax" in job_id or job_id in ("heartbeat", "ig_keepalive"):
            _last_dax_job = f"{job_id} @ {now}"

        # Only alert on missed/error, not normal execution
        exception = getattr(event, "exception", None)
        scheduled_run_time = getattr(event, "scheduled_run_time", None)

        if exception:
            msg = (
                f"🚨 <b>JOB ERROR</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Job: <code>{job_id}</code>\n"
                f"Error: <code>{exception}</code>\n"
                f"Time: {now}"
            )
            loop.call_soon_threadsafe(asyncio.ensure_future, send_func(msg))
        elif hasattr(event, "code"):
            # EVENT_JOB_MISSED
            from apscheduler.events import EVENT_JOB_MISSED
            if event.code == EVENT_JOB_MISSED:
                msg = (
                    f"⚠️ <b>JOB MISSED</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Job: <code>{job_id}</code>\n"
                    f"Scheduled: {scheduled_run_time}\n"
                    f"Time: {now}"
                )
                loop.call_soon_threadsafe(asyncio.ensure_future, send_func(msg))

    return _on_job_event


# ── Heartbeat Ping ────────────────────────────────────────────────────────

async def heartbeat_ping():
    """
    Ping external monitoring service (runs every 5 minutes via APScheduler).
    Also updates internal heartbeat timestamp for /health endpoint.
    """
    global _last_heartbeat
    _last_heartbeat = time.time()

    if not HEALTHCHECK_PING_URL:
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.get(HEALTHCHECK_PING_URL, timeout=10)
    except Exception as e:
        logger.warning(f"Heartbeat ping failed: {e}")


# ── Startup Alert ─────────────────────────────────────────────────────────

async def send_startup_alert(send_func, gold_str: str = "disabled", spx_str: str = "disabled", nikkei_str: str = "disabled"):
    """Send single combined Telegram alert on container start."""
    now = datetime.now(TZ_UK)
    msg = (
        f"🔄 <b>BOT STARTED</b>  {now.strftime('%Y-%m-%d %H:%M:%S')} UK\n"
        f"S1 DAX: ASRS active\n"
        f"S3 US30: {spx_str}\n"
        f"S4 Nikkei: {nikkei_str}\n"
        f"<i>All schedules active.</i>"
    )
    await send_func(msg)


# ── Position Safety Audit ─────────────────────────────────────────────────

async def position_safety_audit(shared_session, send_func):
    """
    Every 5 mins: check ALL open positions have stops set.
    If any position has no stop, set a disaster stop and alert.

    Broker-agnostic: dispatches to IG or IB implementation based on
    which kind of session was passed.
    """
    if hasattr(shared_session, "ib"):
        return await _position_safety_audit_ib(shared_session, send_func)
    return await _position_safety_audit_ig(shared_session, send_func)


async def _position_safety_audit_ib(shared_session, send_func):
    """
    IBKR variant: check that every open position is either:
      (a) protected by a real IBKR StopOrder, OR
      (b) protected by the bot's active tick-stop monitor

    The bot deliberately uses tick monitors instead of real stops because
    they handle trail/breakeven updates without the IBKR API spam, so
    "no real stop" is normal — only alert if BOTH real stop AND tick
    monitor are missing.
    """
    try:
        ib = shared_session.ib
        positions = ib.positions()
        if not positions:
            return

        # Build conId → has_real_stop_order map
        stop_conids: set[int] = set()
        for trade in ib.openTrades():
            order_type = getattr(trade.order, "orderType", "")
            if order_type in ("STP", "STOP"):
                stop_conids.add(trade.contract.conId)

        # Build conId → has_active_tick_monitor map by inspecting brokers
        tick_monitor_conids: set[int] = set()
        try:
            from asrs.main import ALL_SIGNALS
            for sig in ALL_SIGNALS:
                br = sig.broker
                if (getattr(br, "_stop_monitor", None) is not None
                    and br._stop_monitor.get("active")
                    and getattr(br, "contract", None) is not None):
                    tick_monitor_conids.add(br.contract.conId)
        except Exception:
            pass

        for pos in positions:
            if pos.position == 0:
                continue
            conid = pos.contract.conId
            if conid in stop_conids:
                continue
            if conid in tick_monitor_conids:
                continue
            # Truly unprotected — no real stop AND no tick monitor
            sym = pos.contract.localSymbol or pos.contract.symbol
            direction = "LONG" if pos.position > 0 else "SHORT"
            await send_func(
                f"🚨 <b>IB SAFETY AUDIT</b>\n"
                f"Position {sym} ({direction} {abs(pos.position)}) has "
                f"NO real stop AND NO tick monitor active!\n"
                f"This is a real exposure — check /status immediately."
            )
            logger.warning(
                f"IB SAFETY AUDIT: {sym} {direction} unprotected (no stop, no monitor)"
            )
    except Exception as e:
        logger.warning(f"IB safety audit failed: {e}")


async def _position_safety_audit_ig(ig_session, send_func):
    """IG variant: check open positions via REST, set emergency stop if missing."""
    try:
        positions = ig_session.ig.fetch_open_positions()
        if hasattr(positions, 'to_dict'):
            positions = positions.to_dict('records')
        if not positions:
            return

        for pos in positions:
            deal_id = pos.get('dealId', '')
            epic = pos.get('epic', '')
            direction = pos.get('direction', '')
            stop_level = pos.get('stopLevel') or pos.get('stop_level')
            level = pos.get('level', 0)
            size = pos.get('size', 0)

            if stop_level is None or stop_level == 0:
                # NO STOP — CRITICAL
                # Set disaster stop immediately
                disaster_dist = 500  # default
                if 'DAX' in epic or 'GER' in epic:
                    disaster_dist = 200
                elif 'DOW' in epic:
                    disaster_dist = 500
                elif 'NIKKEI' in epic:
                    disaster_dist = 1000

                if direction == 'BUY':
                    emergency_stop = float(level) - disaster_dist
                else:
                    emergency_stop = float(level) + disaster_dist

                try:
                    ig_session.ig.update_open_position(
                        limit_level=None, stop_level=emergency_stop, deal_id=deal_id,
                    )
                    logger.warning(f"SAFETY AUDIT: Set emergency stop on {deal_id} ({epic} {direction}) @ {emergency_stop}")
                except Exception as e:
                    logger.error(f"SAFETY AUDIT: Failed to set stop on {deal_id}: {e}")

                await send_func(
                    f"🚨 <b>SAFETY AUDIT</b>\n"
                    f"Position {deal_id} had NO STOP!\n"
                    f"Epic: {epic} | Dir: {direction} | Entry: {level}\n"
                    f"Emergency stop set @ {emergency_stop}\n"
                    f"<b>Check immediately!</b>"
                )
    except Exception as e:
        logger.warning(f"Position safety audit failed: {e}")


# ── Health Status (for /health endpoint) ──────────────────────────────────

def get_health() -> dict:
    """Return health status dict for the /health Flask endpoint."""
    uptime = time.time() - _start_time
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)

    heartbeat_age = time.time() - _last_heartbeat if _last_heartbeat else -1

    return {
        "status": "ok",
        "uptime": f"{hours}h {minutes}m",
        "uptime_seconds": int(uptime),
        "last_heartbeat_age_seconds": int(heartbeat_age) if heartbeat_age >= 0 else None,
        "last_dax_job": _last_dax_job or "none yet",
        "pid": os.getpid(),
        "time_uk": datetime.now(TZ_UK).strftime("%Y-%m-%d %H:%M:%S"),
    }
