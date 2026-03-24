"""
telegram_cmd.py — Telegram Command Handler for DAX Bot
═══════════════════════════════════════════════════════════════════════════════

Polls for /commands from authorized chat ID and executes them against
the DAX bot instance. Runs as an async background task in the event loop.

Commands:
  /status      — Bot health, IG connection, positions, today's P&L
  /positions   — Open trades with entry, current price, stop, unrealised P&L
  /close       — Close all positions
  /pause       — Stop taking new trades, keep managing existing
  /resume      — Resume normal trading
  /logs        — Last 20 lines from bot log
  /pnl         — Today's realised P&L, trades, wins/losses
  /restart     — Disconnect and reconnect to IG API
"""

import asyncio
import logging
import logging.handlers
import collections
import httpx
from datetime import datetime

import os


# ── In-memory log buffer for /logs command ──────────────────────────────────
class MemoryLogHandler(logging.Handler):
    """Stores last N log lines in a deque for /logs retrieval."""
    def __init__(self, capacity: int = 100):
        super().__init__()
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def emit(self, record):
        try:
            self.buffer.append(self.format(record))
        except Exception:
            pass

    def get_lines(self, n: int = 20) -> list[str]:
        return list(self.buffer)[-n:]


# Install the memory handler on the root logger so all bot logs are captured
_memory_handler = MemoryLogHandler(capacity=200)
_memory_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S"
))
logging.getLogger().addHandler(_memory_handler)

# Import config
try:
    from dax_bot import config as dax_config
except ImportError:
    dax_config = None

logger = logging.getLogger("TelegramCmd")

_cfg = dax_config
TG_TOKEN = _cfg.TG_TOKEN if _cfg else os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = _cfg.TG_CHAT_ID if _cfg else os.getenv("TELEGRAM_CHAT_ID", "")
API_BASE = f"https://api.telegram.org/bot{TG_TOKEN}"

# Global pause flag — checked by bot main loops before placing new orders
paused = False


async def _send(text: str):
    """Send HTML message to Telegram."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
    except Exception as e:
        logger.error(f"TG send failed: {e}")


async def _get_offset() -> int | None:
    """Get latest update offset to skip old messages."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{API_BASE}/getUpdates",
                params={"timeout": 0, "limit": 1, "offset": -1},
                timeout=5,
            )
            if r.status_code == 200:
                results = r.json().get("result", [])
                if results:
                    return results[-1]["update_id"] + 1
    except Exception:
        pass
    return None


async def handle_status(dax_broker, **kwargs):
    """Show bot health, IG connection, position, today's P&L."""
    global paused
    _cfg = dax_config
    now = datetime.now(_cfg.TZ_UK)
    mode = "DEMO" if getattr(_cfg, "IG_DEMO", True) else "LIVE"

    # DAX state
    dax_state = None
    try:
        from dax_bot.strategy import DailyState as DaxState
        dax_state = DaxState.load()
    except ImportError:
        pass

    # IG connection check
    dax_ok = dax_broker.connected if dax_broker else False

    # DAX position
    dax_pos = "FLAT"
    dax_phase = "N/A"
    dax_pnl = ""
    stream_bars = 0
    if dax_state:
        dax_phase = str(dax_state.phase)
        if dax_state.direction:
            dax_pos = f"{dax_state.direction} @ {dax_state.entry_price}"
        if hasattr(dax_state, 'trades') and dax_state.trades:
            total = sum(t.get("pnl_pts", 0) for t in dax_state.trades if isinstance(t.get("pnl_pts"), (int, float)))
            dax_pnl = f"{'+' if total >= 0 else ''}{round(total, 1)} pts"
    if dax_broker:
        stream_bars = dax_broker.get_streaming_bar_count()

    pause_str = "PAUSED" if paused else "ACTIVE"

    # US30 state
    us30_phase = "N/A"
    us30_pos = "FLAT"
    us30_pnl = ""
    us30_bars = 0
    try:
        from spx_bot.main import broker as us30_broker
        from spx_bot.strategy import US30DailyState
        us30_state = US30DailyState.load()
        us30_phase = str(us30_state.phase)
        if us30_state.direction:
            us30_pos = f"{us30_state.direction} @ {us30_state.entry_price}"
        if hasattr(us30_state, 'trades') and us30_state.trades:
            total_us = sum(t.get("pnl_pts", 0) for t in us30_state.trades if isinstance(t.get("pnl_pts"), (int, float)))
            us30_pnl = f"{'+' if total_us >= 0 else ''}{round(total_us, 1)} pts"
        if us30_broker:
            us30_bars = us30_broker.get_streaming_bar_count()
    except Exception:
        pass

    # S2 phases
    dax_s2_phase = "N/A"
    dax_s2_pos = ""
    us30_s2_phase = "N/A"
    us30_s2_pos = ""
    if dax_state:
        dax_s2_phase = getattr(dax_state, 's2_phase', 'IDLE')
        if getattr(dax_state, 's2_direction', ''):
            dax_s2_pos = f" | {dax_state.s2_direction} @ {dax_state.s2_entry_price}"
    try:
        us30_state_s2 = us30_state if 'us30_state' in dir() else None
        if us30_state_s2:
            us30_s2_phase = getattr(us30_state_s2, 's2_phase', 'IDLE')
            if getattr(us30_state_s2, 's2_direction', ''):
                us30_s2_pos = f" | {us30_state_s2.s2_direction} @ {us30_state_s2.s2_entry_price}"
    except Exception:
        pass

    msg = (
        f"📊 <b>BOT STATUS</b> [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UK\n"
        f"🔄 Trading: <b>{pause_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>DAX S1</b>\n"
        f"  IG: {'✅' if dax_ok else '❌'} | Phase: {dax_phase}\n"
        f"  Position: {dax_pos}\n"
        f"  Today P&L: {dax_pnl or 'N/A'}\n"
        f"  Streaming bars: {stream_bars}\n"
        f"<b>DAX S2</b> (14:00 CET)\n"
        f"  Phase: {dax_s2_phase}{dax_s2_pos}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>US30 S1</b>\n"
        f"  Phase: {us30_phase}\n"
        f"  Position: {us30_pos}\n"
        f"  Today P&L: {us30_pnl or 'N/A'}\n"
        f"  Streaming bars: {us30_bars}\n"
        f"<b>US30 S2</b> (11:00 ET)\n"
        f"  Phase: {us30_s2_phase}{us30_s2_pos}\n"
    )

    # Nikkei
    nk_phase = "N/A"
    nk_pos = "FLAT"
    nk_pnl = ""
    nk_bars = 0
    nk_s2_phase = "N/A"
    try:
        from nikkei_bot.main import broker as nk_broker
        from nikkei_bot.main import NikkeiDailyState
        nk_state = NikkeiDailyState.load()
        nk_phase = str(nk_state.phase)
        if nk_state.direction:
            nk_pos = f"{nk_state.direction} @ {nk_state.entry_price}"
        if hasattr(nk_state, 'trades') and nk_state.trades:
            total_nk = sum(t.get("pnl_pts", 0) for t in nk_state.trades if isinstance(t.get("pnl_pts"), (int, float)))
            nk_pnl = f"{'+' if total_nk >= 0 else ''}{round(total_nk, 1)} pts"
        if nk_broker:
            nk_bars = nk_broker.get_streaming_bar_count()
        nk_s2_phase = getattr(nk_state, 's2_phase', 'IDLE')
    except Exception:
        pass

    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>NIKKEI S1</b> (10:00 JST)\n"
        f"  Phase: {nk_phase}\n"
        f"  Position: {nk_pos}\n"
        f"  Today P&L: {nk_pnl or 'N/A'}\n"
        f"  Streaming bars: {nk_bars}\n"
        f"<b>NIKKEI S2</b> (12:00 JST)\n"
        f"  Phase: {nk_s2_phase}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )
    await _send(msg)


async def handle_positions(dax_broker, **kwargs):
    """Show all open trades with entry, current price, stop, unrealised P&L."""
    lines = ["📈 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━━━"]
    found = False

    # DAX
    try:
        from dax_bot.strategy import DailyState as DaxState, Phase as DaxPhase
        dax_state = DaxState.load()
        if dax_state.phase in (DaxPhase.LONG_ACTIVE, DaxPhase.SHORT_ACTIVE):
            price = None
            if dax_broker and dax_broker.connected:
                price = await dax_broker.get_current_price()
            unrealised = ""
            if price:
                if dax_state.direction == "LONG":
                    ur = round(price - dax_state.entry_price, 1)
                else:
                    ur = round(dax_state.entry_price - price, 1)
                unrealised = f"{'+' if ur >= 0 else ''}{ur} pts"

            add_max = dax_config.ADD_STRENGTH_MAX if dax_config else 2
            lines.append(
                f"\n<b>DAX {dax_state.direction}</b>\n"
                f"  Entry: {dax_state.entry_price}\n"
                f"  Current: {price or 'N/A'}\n"
                f"  Stop: {dax_state.trailing_stop}\n"
                f"  Contracts: {dax_state.contracts_active}\n"
                f"  Adds: {dax_state.adds_used}/{add_max}\n"
                f"  Unrealised: {unrealised or 'N/A'}"
            )
            found = True
    except ImportError:
        pass

    if not found:
        lines.append("\nNo open positions.")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
    await _send("\n".join(lines))


async def handle_close(dax_broker, target: str = "all", **kwargs):
    """Close positions."""
    results = []

    if target in ("all", "dax") and dax_broker:
        try:
            if await dax_broker.ensure_connected():
                await dax_broker.cancel_all_orders()
                pos = await dax_broker.get_position()
                if pos["direction"] != "FLAT":
                    await dax_broker.close_position()
                    results.append("DAX: Position closed ✅")

                    # Update state
                    try:
                        from dax_bot.strategy import DailyState as DaxState, process_stop_hit
                        dax_state = DaxState.load()
                        if dax_state.direction:
                            price = await dax_broker.get_current_price()
                            if price:
                                process_stop_hit(dax_state, price)
                    except ImportError:
                        pass
                else:
                    cancelled = await dax_broker.cancel_all_orders()
                    results.append(f"DAX: No position (cancelled {cancelled} orders)")
            else:
                results.append("DAX: ❌ Cannot connect to IG")
        except Exception as e:
            results.append(f"DAX: ❌ {e}")

    msg = (
        f"🛑 <b>CLOSE ALL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(results)
    )
    await _send(msg)


async def handle_kill(dax_broker, **kwargs):
    """EMERGENCY: Close ALL positions on IG, pause trading, reset all state."""
    global paused
    paused = True
    results = []

    try:
        from shared.ig_session import IGSharedSession
        shared = IGSharedSession.get_instance()
        if not shared or not shared.ig:
            shared = IGSharedSession()
            await shared.connect()

        # Fetch ALL open positions on the account
        positions = shared.ig.fetch_open_positions()
        if hasattr(positions, 'to_dict'):
            positions = positions.to_dict('records')

        if not positions:
            results.append("No open positions found.")
        else:
            for pos in positions:
                deal_id = pos.get('dealId', '')
                epic = pos.get('epic', '')
                direction = pos.get('direction', '')
                size = pos.get('size', 1)
                try:
                    close_dir = 'SELL' if direction == 'BUY' else 'BUY'
                    shared.ig.close_open_position(
                        deal_id=deal_id, direction=close_dir,
                        epic=epic, expiry='DFB', level=None,
                        order_type='MARKET', quote_id=None, size=size,
                    )
                    results.append(f"  {epic} {direction} x{size}: CLOSED")
                except Exception as e:
                    results.append(f"  {epic} {direction}: FAILED ({e})")

        # Reset DAX state
        try:
            from dax_bot.strategy import DailyState as DaxState, Phase as DaxPhase
            ds = DaxState.load()
            ds.phase = DaxPhase.DONE
            ds.s2_phase = "DONE"
            ds.direction = ""
            ds.contracts_active = 0
            ds.save()
            results.append("DAX state: RESET")
        except Exception:
            pass

        # Reset US30 state
        try:
            from spx_bot.strategy import US30DailyState, Phase as US30Phase
            us = US30DailyState.load()
            us.phase = US30Phase.DONE
            us.s2_phase = "DONE"
            us.direction = ""
            us.contracts_active = 0
            us.save()
            results.append("US30 state: RESET")
        except Exception:
            pass

    except Exception as e:
        results.append(f"ERROR: {e}")

    msg = (
        f"🚨 <b>EMERGENCY KILL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"ALL positions closed. Trading PAUSED.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(results)
        + "\n━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Use /resume to resume trading.</i>"
    )
    await _send(msg)


async def handle_pause():
    """Pause new trades."""
    global paused
    paused = True
    await _send(
        "⏸ <b>TRADING PAUSED</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "No new trades will be placed.\n"
        "Existing positions will continue to be managed.\n"
        "<i>Use /resume to resume trading.</i>"
    )


async def handle_resume():
    """Resume trading."""
    global paused
    paused = False
    await _send(
        "▶️ <b>TRADING RESUMED</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "New trades will be placed normally.\n"
        "<i>Bot is fully active.</i>"
    )


async def handle_logs():
    """Show last 20 lines from in-memory log buffer."""
    try:
        lines = _memory_handler.get_lines(20)
        if not lines:
            output = "No logs captured yet."
        else:
            output = "\n".join(lines)

        # Truncate if too long for Telegram (4096 char limit)
        if len(output) > 3800:
            output = output[-3800:]

        await _send(f"📋 <b>LAST 20 LOG LINES</b>\n━━━━━━━━━━━━━━━━━━━━━━\n<pre>{output}</pre>")
    except Exception as e:
        await _send(f"❌ <b>LOGS ERROR</b>\n{e}")


async def handle_pnl():
    """Today's realised P&L, trades, wins/losses."""
    _cfg = dax_config
    now = datetime.now(_cfg.TZ_UK)
    today = now.strftime("%Y-%m-%d")

    lines = [
        f"💰 <b>TODAY'S P&L</b> ({today})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    ]

    # DAX journal
    try:
        from dax_bot import journal as dax_journal
        dax_trades = dax_journal.load_all_trades()
        dax_today = [t for t in dax_trades if t.get("date") == today]
        if dax_today:
            dax_pnl = sum(t.get("pnl_pts", 0) for t in dax_today)
            dax_wins = sum(1 for t in dax_today if t.get("pnl_pts", 0) > 0)
            dax_losses = sum(1 for t in dax_today if t.get("pnl_pts", 0) < 0)
            ps = "+" if dax_pnl >= 0 else ""
            lines.append(
                f"\n<b>DAX</b>: {ps}{round(dax_pnl, 1)} pts\n"
                f"  Trades: {len(dax_today)} | W: {dax_wins} L: {dax_losses}"
            )
            for t in dax_today:
                p = t.get("pnl_pts", 0)
                lines.append(f"  {t.get('direction', '?')} {'+' if p >= 0 else ''}{round(p, 1)} pts")
        else:
            lines.append("\n<b>DAX</b>: No trades today")
    except ImportError:
        lines.append("\n<b>DAX</b>: N/A")

    # US30 + Nikkei from shared journal DB
    try:
        from shared import journal_db
        for inst in ["US30", "NIKKEI"]:
            inst_trades = journal_db.get_trades_for_date(today, instrument=inst)
            if inst_trades:
                inst_pnl = sum(t.get("pnl_pts", 0) for t in inst_trades)
                inst_wins = sum(1 for t in inst_trades if t.get("pnl_pts", 0) > 0)
                inst_losses = sum(1 for t in inst_trades if t.get("pnl_pts", 0) < 0)
                ps = "+" if inst_pnl >= 0 else ""
                lines.append(
                    f"\n<b>{inst}</b>: {ps}{round(inst_pnl, 1)} pts\n"
                    f"  Trades: {len(inst_trades)} | W: {inst_wins} L: {inst_losses}"
                )
                for t in inst_trades:
                    p = t.get("pnl_pts", 0)
                    lines.append(f"  {t.get('direction', '?')} {'+' if p >= 0 else ''}{round(p, 1)} pts")
            else:
                lines.append(f"\n<b>{inst}</b>: No trades today")
    except Exception as e:
        lines.append(f"\n<b>US30/NIKKEI</b>: {e}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    await _send("\n".join(lines))


async def handle_journal(text: str):
    """Show trade journal — recent trades, cumulative P&L, scaling tier."""
    from shared import journal_db

    parts = text.strip().split()
    instrument = None
    n = 5

    for p in parts[1:]:
        p_lower = p.lower()
        if p_lower == "dax":
            instrument = "DAX"
        elif p_lower == "week":
            n = -1
        else:
            try:
                n = int(p)
            except ValueError:
                pass

    if n == -1:
        weekly = journal_db.get_weekly_pnl(instrument)
        by_inst = journal_db.get_cumulative_pnl_by_instrument()
        tier = journal_db.get_current_scaling_tier()

        lines = [
            "📓 <b>WEEKLY JOURNAL</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"This week: <b>£{weekly['pnl_gbp']:.2f}</b> ({weekly['trades']} trades)",
            "",
        ]
        for inst, data in by_inst.items():
            ps = "+" if data["pnl_gbp"] >= 0 else ""
            lines.append(f"  {inst}: {ps}£{data['pnl_gbp']:.2f} ({data['trades']} trades)")

        lines.append("")
        lines.append(f"📊 Cumulative: <b>£{tier['cumulative_pnl']:.2f}</b>")
        lines.append(f"📈 Current stake: £{tier['current_stake']}/pt")
        if tier["next_threshold"]:
            remaining = tier["next_threshold"] - tier["cumulative_pnl"]
            lines.append(f"🎯 Next: £{tier['next_stake']}/pt at £{tier['next_threshold']:.0f} "
                        f"(£{remaining:.0f} to go)")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        await _send("\n".join(lines))
        return

    trades = journal_db.get_recent_trades(n, instrument)
    tier = journal_db.get_current_scaling_tier()

    inst_label = instrument or "ALL"
    lines = [
        f"📓 <b>TRADE JOURNAL</b> ({inst_label}, last {len(trades)})",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not trades:
        lines.append("No trades recorded yet.")
    else:
        for t in trades:
            pnl = t.get("pnl_gbp", 0) or t.get("pnl_pts", 0)
            ps = "+" if pnl >= 0 else ""
            icon = "✅" if pnl >= 0 else "❌"
            lines.append(
                f"{icon} {t.get('date', '?')} [{t.get('instrument', '?')}] "
                f"{t.get('direction', '?')} {ps}£{pnl:.1f}"
            )

    lines.append("")
    lines.append(f"📊 Cumulative: <b>£{tier['cumulative_pnl']:.2f}</b>")
    lines.append(f"📈 Stake: £{tier['current_stake']}/pt")
    if tier["next_threshold"]:
        remaining = tier["next_threshold"] - tier["cumulative_pnl"]
        lines.append(f"🎯 Next tier: £{tier['next_stake']}/pt at £{tier['next_threshold']:.0f} "
                    f"(£{remaining:.0f} to go)")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    await _send("\n".join(lines))


async def handle_set(text: str):
    """Handle /set key value — hot-reload config."""
    from shared.config_reload import apply_set

    parts = text.strip().split(maxsplit=2)
    if len(parts) < 3:
        from shared.config_reload import ALLOWED_KEYS
        keys = ", ".join(sorted(ALLOWED_KEYS.keys()))
        await _send(
            f"⚙️ <b>USAGE</b>: <code>/set key value</code>\n\n"
            f"Available keys:\n<code>{keys}</code>"
        )
        return

    result = apply_set(parts[1], parts[2])
    await _send(result)


async def handle_config():
    """Show all configurable keys and current values."""
    from shared.config_reload import get_current_config
    await _send(get_current_config())


async def handle_restart(dax_broker, **kwargs):
    """Disconnect and reconnect to IG API."""
    results = []

    if dax_broker:
        try:
            await dax_broker.disconnect()
            ok = await dax_broker.connect()
            results.append(f"DAX IG: {'✅ Reconnected' if ok else '❌ Failed'}")
        except Exception as e:
            results.append(f"DAX IG: ❌ {e}")

    msg = (
        f"🔄 <b>IG RESTART</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(results)
    )
    await _send(msg)


async def poll_commands(dax_broker=None, **kwargs):
    """
    Main polling loop. Runs forever as a background asyncio task.
    Polls Telegram getUpdates every 3 seconds for new /commands.
    """
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.warning("Telegram not configured — command handler disabled")
        return

    logger.info("Telegram command handler started")

    # Register command menu with Telegram
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/setMyCommands",
                json={"commands": [
                    {"command": "status", "description": "Bot health & overview"},
                    {"command": "positions", "description": "Open trades with live P&L"},
                    {"command": "pnl", "description": "Today's realised P&L"},
                    {"command": "close", "description": "Close all positions"},
                    {"command": "pause", "description": "Stop new trades"},
                    {"command": "resume", "description": "Resume trading"},
                    {"command": "restart", "description": "Reconnect IG API"},
                    {"command": "logs", "description": "Last 20 log lines"},
                    {"command": "journal", "description": "Trade journal & scaling"},
                    {"command": "config", "description": "View current config"},
                    {"command": "set", "description": "Change config: /set key value"},
                ]},
                timeout=10,
            )
        logger.info("Telegram command menu registered")
    except Exception as e:
        logger.warning(f"Failed to set command menu: {e}")

    offset = await _get_offset()

    while True:
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "timeout": 3,
                    "allowed_updates": ["message"],
                }
                if offset:
                    params["offset"] = offset

                r = await client.get(
                    f"{API_BASE}/getUpdates",
                    params=params,
                    timeout=8,
                )

                if r.status_code != 200:
                    await asyncio.sleep(5)
                    continue

                updates = r.json().get("result", [])
                for update in updates:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = (msg.get("text") or "").strip()

                    # Auth check: only respond to our chat ID
                    if chat_id != str(TG_CHAT_ID):
                        continue

                    if not text.startswith("/"):
                        continue

                    cmd = text.lower()
                    logger.info(f"Command received: {cmd}")

                    try:
                        if cmd == "/status":
                            await handle_status(dax_broker)
                        elif cmd == "/positions":
                            await handle_positions(dax_broker)
                        elif cmd in ("/close", "/close dax", "/close_dax"):
                            await handle_close(dax_broker, "dax")
                        elif cmd == "/kill":
                            await handle_kill(dax_broker)
                        elif cmd == "/pause":
                            await handle_pause()
                        elif cmd == "/resume":
                            await handle_resume()
                        elif cmd == "/logs":
                            await handle_logs()
                        elif cmd == "/pnl":
                            await handle_pnl()
                        elif cmd == "/restart":
                            await handle_restart(dax_broker)
                        elif cmd.startswith("/journal"):
                            await handle_journal(text)
                        elif cmd.startswith("/set "):
                            await handle_set(text)
                        elif cmd == "/set":
                            await handle_set(text)
                        elif cmd == "/config":
                            await handle_config()
                        elif cmd == "/help":
                            await _send(
                                "🤖 <b>COMMANDS</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━━━\n"
                                "/status — Bot health & overview\n"
                                "/positions — Open trades detail\n"
                                "/close — Close all positions\n"
                                "/pause — Stop new trades\n"
                                "/resume — Resume trading\n"
                                "/logs — Last 20 log lines\n"
                                "/pnl — Today's P&L\n"
                                "/journal — Trade journal & scaling\n"
                                "/config — View current config\n"
                                "/set key val — Change config live\n"
                                "/restart — Reconnect IG API\n"
                                "/help — This message"
                            )
                        else:
                            await _send(f"❓ Unknown command: <code>{text}</code>\nUse /help for available commands.")

                    except Exception as e:
                        logger.error(f"Command handler error: {e}", exc_info=True)
                        await _send(f"❌ <b>COMMAND ERROR</b>\n<code>{cmd}</code>\n\n<pre>{e}</pre>")

        except httpx.TimeoutException:
            pass  # Normal — long poll timeout
        except Exception as e:
            logger.error(f"Command poll error: {e}")
            await asyncio.sleep(10)

        await asyncio.sleep(1)


def is_paused() -> bool:
    """Check if trading is paused. Called by bot main loops."""
    return paused
