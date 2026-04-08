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


async def _send_photo(image_bytes: bytes, caption: str = ""):
    """Send a photo (PNG bytes) to Telegram."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/sendPhoto",
                data={"chat_id": TG_CHAT_ID, "caption": caption},
                files={"photo": ("report.png", image_bytes, "image/png")},
                timeout=30,
            )
    except Exception as e:
        logger.error(f"TG photo send failed: {e}")


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
    """Show bot health — reads from unified asrs.main.ALL_SIGNALS."""
    global paused
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Europe/London"))
    mode = "DEMO"  # TODO: read from config
    pause_str = "PAUSED" if paused else "ACTIVE"

    # Try new unified signals first
    try:
        from asrs.main import ALL_SIGNALS
        if not ALL_SIGNALS:
            await _send(f"📊 <b>BOT STATUS</b> [{mode}]\n━━━━━━━━━━━━━━━━━━━━━━\nNo signals loaded yet — bot may still be starting.")
            return
        if ALL_SIGNALS:
            # Broker-agnostic connection check
            broker0 = ALL_SIGNALS[0].broker
            shared0 = getattr(broker0, "_shared", None)
            if shared0 is None:
                broker_label = "Broker"
                broker_ok = False
            elif hasattr(shared0, "ig") and shared0.ig is not None:
                broker_label = "IG"
                broker_ok = True
            elif hasattr(shared0, "ib") and shared0.connected:
                broker_label = "IBKR"
                broker_ok = True
            else:
                broker_label = "Broker"
                broker_ok = False
            msg = (
                f"📊 <b>BOT STATUS</b> [{mode}]\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UK\n"
                f"🔄 Trading: <b>{pause_str}</b>\n"
                f"  {broker_label}: {'✅' if broker_ok else '❌'}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
            )
            from datetime import datetime as dt
            uk = ZoneInfo("Europe/London")

            for signal in ALL_SIGNALS:
                state = signal.state
                phase = state.phase.name if hasattr(state.phase, 'name') else str(state.phase)
                pos = "FLAT"
                if state.direction:
                    pos = f"{state.direction} @ {state.entry_price}"
                # P&L: in-memory trades first, fall back to journal DB
                pnl_str = "N/A"
                if state.trades:
                    total = sum(t.get("pnl_pts", 0) for t in state.trades)
                    pnl_str = f"{'+' if total >= 0 else ''}{round(total, 1)} pts"
                else:
                    try:
                        from shared import journal_db
                        today_str = now.strftime("%Y-%m-%d")
                        db_trades = journal_db.get_trades_for_date(today_str, instrument=signal.instrument)
                        if db_trades:
                            total = sum(t.get("pnl_pts", 0) for t in db_trades)
                            pnl_str = f"{'+' if total >= 0 else ''}{round(total, 1)} pts (journal)"
                    except Exception:
                        pass
                bars = signal.broker.get_streaming_bar_count() if hasattr(signal.broker, 'get_streaming_bar_count') else 0

                # Schedule times in local + UK
                cfg = signal.cfg
                sched_tz = ZoneInfo(cfg["scheduler_timezone"])
                tz_label = cfg["timezone"].split("/")[-1]
                s_key = f"s{signal.session}"
                open_h = cfg[f"{s_key}_open_hour"]
                open_m = cfg[f"{s_key}_open_minute"]
                routine_m = open_m + 21
                routine_h = open_h + routine_m // 60
                routine_m = routine_m % 60
                today = now.date()
                local_time = dt(today.year, today.month, today.day, routine_h, routine_m, tzinfo=sched_tz)
                uk_time = local_time.astimezone(uk)
                time_str = f"{routine_h:02d}:{routine_m:02d} {tz_label} / {uk_time.strftime('%H:%M')} UK"

                msg += (
                    f"<b>{signal.name}</b> ({time_str})\n"
                    f"  Phase: {phase}\n"
                    f"  Position: {pos}\n"
                    f"  Today P&L: {pnl_str}\n"
                    f"  Streaming bars: {bars}\n"
                )

                # Add separator between instruments (after S2)
                if signal.session == 2:
                    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"

            await _send(msg)
            return
    except Exception as e:
        await _send(f"❌ <b>STATUS ERROR</b>\n{e}")


async def handle_positions(dax_broker, **kwargs):
    """Show all open trades with entry, current price, stop, unrealised P&L."""
    lines = ["📈 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━━━"]
    found = False

    try:
        from asrs.main import ALL_SIGNALS
        from asrs.strategy import Phase
        for signal in ALL_SIGNALS:
            s = signal.state
            if s.phase not in (Phase.LONG, Phase.SHORT):
                continue
            price = await signal.broker.get_current_price()
            unrealised = ""
            if price:
                if s.direction == "LONG":
                    ur = round(price - s.entry_price, 1)
                else:
                    ur = round(s.entry_price - price, 1)
                unrealised = f"{'+' if ur >= 0 else ''}{ur} pts"

            add_max = signal.cfg.get("add_max", 2)
            lines.append(
                f"\n<b>{signal.name} {s.direction}</b>\n"
                f"  Entry: {s.entry_price}\n"
                f"  Current: {price or 'N/A'}\n"
                f"  Stop: {s.trailing_stop}\n"
                f"  Adds: {s.adds_used}/{add_max}\n"
                f"  BE: {'Yes' if s.breakeven_hit else 'No'}\n"
                f"  Unrealised: {unrealised or 'N/A'}"
            )
            found = True
    except Exception as e:
        lines.append(f"\nError: {e}")

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
    """EMERGENCY: Close ALL positions across all brokers, pause trading, reset state."""
    global paused
    paused = True
    results = []

    try:
        from asrs.main import ALL_SIGNALS
        from asrs.strategy import Phase

        # Close positions via each broker (broker-agnostic — works for IG and IB)
        seen_brokers: set[int] = set()
        for signal in ALL_SIGNALS:
            broker_id = id(signal.broker)
            if broker_id in seen_brokers:
                continue
            seen_brokers.add(broker_id)
            try:
                pos = await signal.broker.get_position()
                if pos.get("direction", "FLAT") != "FLAT":
                    closed = await signal.broker.close_position()
                    label = getattr(signal.broker, "epic", "?")
                    results.append(
                        f"  {signal.instrument} ({label}): {pos['direction']} → "
                        f"{'CLOSED' if closed else 'FAILED'}"
                    )
                else:
                    label = getattr(signal.broker, "epic", "?")
                    results.append(f"  {signal.instrument} ({label}): flat (skip)")
            except Exception as e:
                results.append(f"  {signal.instrument}: ERROR {e}")
            try:
                await signal.broker.cancel_all_orders()
            except Exception:
                pass

        # Reset all signal states
        for signal in ALL_SIGNALS:
            signal.state.phase = Phase.DONE
            signal.state.direction = ""
            signal.state.contracts_active = 0
            signal.state.deal_ids = []
            if hasattr(signal.broker, '_pending_bracket'):
                signal.broker._pending_bracket = None
            if hasattr(signal.broker, 'deactivate_stop_monitor'):
                signal.broker.deactivate_stop_monitor()
            results.append(f"{signal.name}: RESET")

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


async def handle_morning(text: str):
    """Manually trigger a signal's morning routine.
    Usage: /morning DAX_S1
    Used when the scheduled morning routine was missed (e.g. after a
    bot restart during market hours). Resets the signal to IDLE first
    so the morning routine can fully re-run.

    Safety: after morning_routine arms a bracket, if current price is
    already outside the bracket levels, deactivate the bracket and put
    the signal in re-entry-waiting mode. Prevents stale-trigger fills
    when the call is made well after the actual bar 4/5 window.
    """
    parts = text.strip().split()
    if len(parts) < 2:
        await _send("Usage: /morning <SIGNAL_NAME>")
        return
    target = parts[1].upper()
    try:
        from asrs.main import ALL_SIGNALS
        from asrs.strategy import Phase
        for signal in ALL_SIGNALS:
            if signal.name != target:
                continue

            signal.state.phase = Phase.IDLE
            signal.state.entries_used = 0
            signal.state.direction = ""
            signal.state.deal_ids = []
            signal._bar4_triggered = False
            signal.save_state()
            signal.broker.deactivate_bracket()
            await _send(f"🔄 [{signal.name}] Reset to IDLE, running morning routine now...")
            await signal.morning_routine()

            # Safety: if price is already outside the bracket, defuse it
            try:
                buy_lvl = signal.state.buy_level
                sell_lvl = signal.state.sell_level
                price = await signal.broker.get_current_price()
                if buy_lvl > 0 and sell_lvl > 0 and price:
                    if price >= buy_lvl or price <= sell_lvl:
                        signal.broker.deactivate_bracket()
                        signal.state.phase = Phase.LEVELS_SET  # waiting state
                        signal.save_state()
                        side = "ABOVE BUY" if price >= buy_lvl else "BELOW SELL"
                        await _send(
                            f"⚠️ [{signal.name}] Price {price} already {side} "
                            f"({sell_lvl} / {buy_lvl}).\n"
                            f"Bracket DEFUSED — will arm once price returns between levels."
                        )
                        return
            except Exception as e:
                logger.warning(f"morning safety check failed: {e}")

            await _send(f"✅ [{signal.name}] Morning routine complete. Check /status.")
            return
        names = [s.name for s in ALL_SIGNALS]
        await _send(f"❌ Signal '{target}' not found.\nAvailable: {', '.join(names)}")
    except Exception as e:
        import traceback
        await _send(f"❌ Morning trigger error: {e}\n<pre>{traceback.format_exc()[-500:]}</pre>")


async def handle_reset(text: str):
    """Reset a signal to IDLE so it can re-run morning routine.
    Usage: /reset US30_S1 or /reset all
    """
    parts = text.strip().split()
    target = parts[1].upper() if len(parts) > 1 else ""

    try:
        from asrs.main import ALL_SIGNALS
        from asrs.strategy import Phase
        reset = []
        for signal in ALL_SIGNALS:
            if target == "ALL" or signal.name == target:
                signal.state.phase = Phase.IDLE
                signal.state.entries_used = 0
                signal.state.direction = ""
                signal.state.deal_ids = []
                signal._bar4_triggered = False
                signal.save_state()
                signal.broker.deactivate_bracket()
                reset.append(signal.name)
        if reset:
            await _send(f"🔄 Reset to IDLE: {', '.join(reset)}\nMorning routine will re-run on next bar 4.")
        else:
            names = [s.name for s in ALL_SIGNALS]
            await _send(f"❌ Signal '{target}' not found.\nAvailable: {', '.join(names)}")
    except Exception as e:
        await _send(f"❌ Reset error: {e}")


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


_HIST_DAILY_CACHE: list | None = None


def _load_hist_daily() -> list:
    """Load and cache firstrate daily P&L series (cross-instrument totals)."""
    global _HIST_DAILY_CACHE
    if _HIST_DAILY_CACHE is not None:
        return _HIST_DAILY_CACHE
    try:
        import pandas as pd
        df = pd.read_csv("data/backtest_firstrate_results.csv")
        # Sum P&L per calendar day across all instruments
        daily = df.groupby("date")["pnl_pts"].sum().sort_index()
        _HIST_DAILY_CACHE = [(str(d), float(p)) for d, p in daily.items()]
    except Exception:
        _HIST_DAILY_CACHE = []
    return _HIST_DAILY_CACHE


def historical_day_match(today_pts: float) -> str:
    """
    Return a one-paragraph comparison: how many backtest days had similar
    P&L to today, and what happened on the day AFTER each of those.
    """
    series = _load_hist_daily()
    if not series:
        return ""
    pts_list = [p for _d, p in series]
    n_total = len(pts_list)
    if n_total < 100:
        return ""

    # Bucket: within ±20% of today's value (or ±10pts if today is small)
    tolerance = max(abs(today_pts) * 0.20, 10.0)
    similar_idx = [
        i for i, p in enumerate(pts_list)
        if abs(p - today_pts) <= tolerance
    ]
    n_match = len(similar_idx)
    pct = n_match / n_total * 100
    sign_today = "+" if today_pts >= 0 else ""

    # Special case: today's P&L was never seen in the 18-year backtest
    if n_match == 0:
        worst_day = min(pts_list)
        best_day = max(pts_list)
        is_worse = today_pts < worst_day
        is_better = today_pts > best_day
        if is_worse:
            verdict = (
                f"⚠️ Today ({sign_today}{today_pts:.0f}pts) is WORSE than the "
                f"worst day in 18 years ({worst_day:+.0f}pts).\n"
                f"This is unprecedented in backtest data. Verify execution."
            )
        elif is_better:
            verdict = (
                f"🎉 Today ({sign_today}{today_pts:.0f}pts) is BETTER than the "
                f"best day in 18 years ({best_day:+.0f}pts).\n"
                f"This is unprecedented in backtest data."
            )
        else:
            verdict = (
                f"Today ({sign_today}{today_pts:.0f}pts) is rare — no exact "
                f"matches in 18yr data within ±{tolerance:.0f}pts.\n"
                f"Backtest range: {worst_day:+.0f} to {best_day:+.0f}pts."
            )
        return (
            f"\n📚 <b>HISTORICAL CONTEXT</b> (18yr backtest)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n{verdict}"
        )

    # Average next-day P&L for those matches
    next_day_pnls = [pts_list[i + 1] for i in similar_idx if i + 1 < n_total]
    next_day_avg = sum(next_day_pnls) / len(next_day_pnls) if next_day_pnls else 0
    next_day_pos_pct = (
        sum(1 for p in next_day_pnls if p > 0) / len(next_day_pnls) * 100
        if next_day_pnls else 0
    )

    # Next 5 days
    next_5_pnls = []
    for i in similar_idx:
        win = pts_list[i + 1: i + 6]
        if win:
            next_5_pnls.append(sum(win))
    next_5_avg = sum(next_5_pnls) / len(next_5_pnls) if next_5_pnls else 0

    sign_next = "+" if next_day_avg >= 0 else ""
    sign_5 = "+" if next_5_avg >= 0 else ""

    return (
        f"\n📚 <b>HISTORICAL CONTEXT</b> (18yr backtest)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Days like today ({sign_today}{today_pts:.0f}±{tolerance:.0f}pts): "
        f"<b>{n_match}</b> / {n_total} ({pct:.1f}%)\n"
        f"Next day avg: <b>{sign_next}{next_day_avg:.0f}pts</b> "
        f"({next_day_pos_pct:.0f}% positive)\n"
        f"Next 5 days avg: <b>{sign_5}{next_5_avg:.0f}pts</b>"
    )


async def handle_pnl():
    """Today's realised P&L, trades, wins/losses — all from unified journal DB."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Europe/London"))
    today = now.strftime("%Y-%m-%d")

    lines = [
        f"💰 <b>TODAY'S P&L</b> ({today})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    ]

    try:
        from shared import journal_db
        day_total = 0
        day_trades = 0
        day_wins = 0

        for inst in ["DAX", "US30", "NIKKEI"]:
            inst_trades = journal_db.get_trades_for_date(today, instrument=inst)
            if inst_trades:
                inst_pnl = sum(t.get("pnl_pts", 0) for t in inst_trades)
                inst_wins = sum(1 for t in inst_trades if t.get("pnl_pts", 0) > 0)
                inst_losses = sum(1 for t in inst_trades if t.get("pnl_pts", 0) < 0)
                day_total += inst_pnl
                day_trades += len(inst_trades)
                day_wins += inst_wins
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

        # Total at top
        icon = "🟢" if day_total >= 0 else "🔴"
        lines.insert(1,
            f"\n{icon} <b>{'+'if day_total>=0 else ''}{round(day_total,1)} pts</b> "
            f"({day_trades} trades, {day_wins} wins)\n"
        )
    except Exception as e:
        lines.append(f"\nError: {e}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    # Historical context — find similar days in 18-year backtest
    try:
        ctx = historical_day_match(day_total)
        if ctx:
            lines.append(ctx)
    except Exception as e:
        logger.warning(f"historical context failed: {e}")

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


async def handle_report():
    """Generate and send full performance report as image."""
    try:
        await _send("Generating full report...")
        from reports import generate_full_report
        image_bytes = generate_full_report()
        await _send_photo(image_bytes, caption="ASRS Performance Report")
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        await _send(f"Report generation failed: {e}")


async def handle_chart():
    """Generate and send equity curve chart as image."""
    try:
        await _send("Generating chart...")
        from reports import generate_chart_only
        image_bytes = generate_chart_only()
        await _send_photo(image_bytes, caption="ASRS Equity Curves")
    except Exception as e:
        logger.error(f"Chart generation failed: {e}", exc_info=True)
        await _send(f"Chart generation failed: {e}")


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
                    {"command": "report", "description": "Full performance report (image)"},
                    {"command": "chart", "description": "Equity curves only (image)"},
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
                        elif cmd == "/report":
                            await handle_report()
                        elif cmd == "/chart":
                            await handle_chart()
                        elif cmd.startswith("/reset"):
                            await handle_reset(text)
                        elif cmd.startswith("/morning"):
                            await handle_morning(text)
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
                                "/report — Full performance report (image)\n"
                                "/chart — Equity curves only (image)\n"
                                "/config — View current config\n"
                                "/set key val — Change config live\n"
                                "/restart — Reconnect IG API\n"
                                "/reset US30_S1 — Reset signal to IDLE\n"
                                "/reset all — Reset all signals\n"
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
