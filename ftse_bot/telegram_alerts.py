"""
telegram_alerts.py -- Telegram Alert Formatting & Delivery for FTSE Bot
"""

import logging
import re
import httpx

from ftse_bot import config

logger = logging.getLogger(__name__)


async def send(text: str) -> bool:
    """Send HTML message via Telegram Bot API."""
    if not config.TG_TOKEN or not config.TG_CHAT_ID:
        logger.warning("Telegram not configured -- console output only")
        print("\n" + "=" * 50)
        print(re.sub(r'<[^>]+>', '', text))
        print("=" * 50 + "\n")
        return False

    url = f"https://api.telegram.org/bot{config.TG_TOKEN}/sendMessage"
    payload = {"chat_id": config.TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                logger.info("Telegram sent")
                return True
            logger.error(f"Telegram {r.status_code}: {r.text}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
    return False


# -- Formatters ----------------------------------------------------------------

def bar_detected(state) -> str:
    """Alert when first bar is classified."""
    bar_type = state.bar_type
    icon = {"1BN": "🔴", "1BP": "🟢", "DOJI": "⚪"}.get(bar_type, "?")
    bar_dir = {"1BN": "Bearish", "1BP": "Bullish", "DOJI": "Doji"}.get(bar_type, "?")
    stake_note = f" (halved: width > {config.BAR_WIDTH_THRESHOLD})" if state.stake_halved else ""

    mode = "DEMO" if config.IG_DEMO else "LIVE"

    return (
        f"{icon} <b>FTSE 1st Bar: {bar_type} ({bar_dir})</b> [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Open:  {state.bar_open}\n"
        f"High:  {state.bar_high}\n"
        f"Low:   {state.bar_low}\n"
        f"Close: {state.bar_close}\n"
        f"Width: {state.bar_width} pts\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Stake: {config.CURRENCY}{state.stake}/pt{stake_note}\n"
    )


def orders_placed(state) -> str:
    """Alert when orders are placed."""
    lines = [f"📥 <b>FTSE ORDERS PLACED</b>\n━━━━━━━━━━━━━━━━━━━━━━"]

    if state.buy_order_id:
        lines.append(
            f"🟩 Buy stop: {state.buy_level} (ID: {state.buy_order_id})\n"
            f"   Stop if filled: {round(state.buy_level - state.bar_width, 1)}"
        )
    if state.sell_order_id:
        lines.append(
            f"🟥 Sell stop: {state.sell_level} (ID: {state.sell_order_id})\n"
            f"   Stop if filled: {round(state.sell_level + state.bar_width, 1)}"
        )

    lines.append(
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Stake: {config.CURRENCY}{state.stake}/pt\n"
        f"<i>Session ends 16:30 UK</i>"
    )
    return "\n".join(lines)


def entry_filled(state) -> str:
    """Alert when entry order fills."""
    icon = "🟩" if state.direction == "LONG" else "🟥"
    return (
        f"{icon} <b>ENTRY FILLED -- {state.direction} FTSE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: <b>{state.entry_price}</b>\n"
        f"Stop:  {state.initial_stop} ({state.bar_width} pts)\n"
        f"Stake: {config.CURRENCY}{state.stake}/pt\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Trail: Candle (prev bar low/high)</i>"
    )


def stop_to_breakeven(state) -> str:
    return (
        f"🔄 <b>STOP -> BREAKEVEN -- FTSE {state.direction}</b>\n"
        f"Entry: {state.entry_price}\n"
        f"Stop:  {state.initial_stop} -> <b>{state.trailing_stop}</b>\n"
        f"Profit locked via candle trail"
    )


def trail_updated(state, old_stop: float) -> str:
    direction = "↑" if state.direction == "LONG" else "↓"
    if state.direction == "LONG":
        locked = round(state.trailing_stop - state.entry_price, 1)
    else:
        locked = round(state.entry_price - state.trailing_stop, 1)

    return (
        f"📈 <b>TRAIL {direction} -- FTSE {state.direction}</b>\n"
        f"Stop: {old_stop} -> <b>{state.trailing_stop}</b>\n"
        f"Locked: {locked} pts profit"
    )


def exit_stopped(state, trade: dict) -> str:
    pnl = trade.get("pnl_pts", 0)
    pnl_gbp = trade.get("pnl_gbp", 0)
    icon = "✅" if pnl >= 0 else "❌"
    ps = "+" if pnl >= 0 else ""

    return (
        f"🛑 {icon} <b>STOPPED OUT -- FTSE {trade.get('direction', '')}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: {trade.get('entry', '?')}\n"
        f"Exit:  {trade.get('exit', '?')}\n"
        f"P&L:   <b>{ps}{pnl} pts</b> ({ps}{config.CURRENCY}{abs(pnl_gbp)})\n"
        f"MFE:   {trade.get('mfe', 0)} pts\n"
        f"Phase: {trade.get('stop_phase', '?')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )


def session_close(state, trade: dict | None) -> str:
    if trade:
        pnl = trade.get("pnl_pts", 0)
        pnl_gbp = trade.get("pnl_gbp", 0)
        ps = "+" if pnl >= 0 else ""
        icon = "✅" if pnl >= 0 else "❌"
        return (
            f"⏰ {icon} <b>SESSION CLOSE -- 16:30 FTSE {trade.get('direction', '')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Entry: {trade.get('entry', '?')}\n"
            f"Exit:  {trade.get('exit', '?')} (market close)\n"
            f"P&L:   <b>{ps}{pnl} pts</b> ({ps}{config.CURRENCY}{abs(pnl_gbp)})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
    return (
        f"⏰ <b>SESSION CLOSE -- 16:30</b>\n"
        f"No position to close."
    )


def no_trade(state, reason: str) -> str:
    return (
        f"⚠️ <b>NO TRADE TODAY -- FTSE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {state.date}\n"
        f"Reason: {reason}"
    )


def day_summary(state, weekly_pnl: float) -> str:
    """End-of-day summary at 17:00."""
    if state.phase == Phase.IDLE:
        return (
            f"📊 <b>FTSE DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 {state.date}\n"
            f"No trade taken today.\n"
            f"Weekly P&L: {'+' if weekly_pnl >= 0 else ''}{weekly_pnl} pts"
        )

    pnl = state.pnl_pts
    pnl_gbp = state.pnl_gbp
    ps = "+" if pnl >= 0 else ""
    icon = "🟢" if pnl >= 0 else "🔴"

    return (
        f"📊 <b>FTSE DAILY SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {state.date} | Bar: {state.bar_type}\n"
        f"{icon} P&L: <b>{ps}{pnl} pts</b> ({ps}{config.CURRENCY}{abs(pnl_gbp)})\n"
        f"Direction: {state.direction or 'N/A'}\n"
        f"Entry: {state.entry_price or 'N/A'} -> Exit: {state.exit_price or 'N/A'}\n"
        f"Reason: {state.exit_reason or 'N/A'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Weekly P&L: {'+' if weekly_pnl >= 0 else ''}{weekly_pnl} pts\n"
        f"Stake: {config.CURRENCY}{state.stake}/pt"
    )


def error_alert(msg: str) -> str:
    return f"❌ <b>FTSE BOT ERROR</b>\n━━━━━━━━━━━━━━━━━━━━━━\n{msg}"


def connection_status(connected: bool) -> str:
    if connected:
        mode = "Demo" if config.IG_DEMO else "Live"
        return f"🔌 <b>FTSE IG Connected</b> ({mode} | {config.IG_EPIC})"
    return "🔌 <b>FTSE IG Disconnected</b> -- will retry"


# Import Phase here to avoid circular import at module level
from ftse_bot.strategy import Phase
