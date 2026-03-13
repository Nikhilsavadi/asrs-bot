"""
alerts.py — Telegram Alert Formatting & Delivery
"""

import logging
import httpx
from dax_bot import config

logger = logging.getLogger(__name__)


async def send(text: str) -> bool:
    """Send HTML message via Telegram Bot API."""
    if not config.TG_TOKEN or not config.TG_CHAT_ID:
        logger.warning("Telegram not configured — console output only")
        import re
        print("\n" + "═" * 50)
        print(re.sub(r'<[^>]+>', '', text))
        print("═" * 50 + "\n")
        return False

    url = f"https://api.telegram.org/bot{config.TG_TOKEN}/sendMessage"
    payload = {"chat_id": config.TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                logger.info("Telegram ✅")
                return True
            logger.error(f"Telegram {r.status_code}: {r.text}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
    return False


# ── Formatters ─────────────────────────────────────────────────────────────────

def morning_levels(state) -> str:
    risk = round(state.buy_level - state.sell_level, 1)
    max_per_pt = round(config.RISK_GBP / risk, 2) if risk > 0 else 0
    flag = {"NARROW": "⚠️", "WIDE": "🔶", "NORMAL": "✅"}.get(state.range_flag, "")
    bar_label = f"Bar #{state.bar_number}"
    if state.use_5th_bar and state.bar5_rule_matched:
        bar_label += f" (rule: {state.bar5_rule_matched})"
    elif not state.use_5th_bar:
        bar_label += " (default)"

    ctx = ""
    if state.context_directional:
        ctx = "\n📈 Opening: Directional"
    elif state.context_choppy:
        ctx = "\n🔀 Opening: Choppy"
    elif state.context_overlap:
        ctx = "\n📊 Opening: Overlapping"

    mode = "📄 DEMO" if config.IG_DEMO else "🔴 LIVE"

    return (
        f"📐 <b>ASRS — DAX 40</b> [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {state.date}\n"
        f"🕐 {bar_label}: {state.bar_high} / {state.bar_low}\n"
        f"   Range: {state.bar_range} pts {flag}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟩 <b>BUY stop: {state.buy_level}</b>\n"
        f"🟥 <b>SELL stop: {state.sell_level}</b>\n"
        f"   Risk: {risk} pts | OCA: auto-cancel\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 £{config.RISK_GBP} risk → max £{max_per_pt}/pt\n"
        f"📦 Contracts: {config.NUM_CONTRACTS}x Micro DAX (€1/pt)\n"
        f"   Risk = €{round(risk * config.NUM_CONTRACTS, 1)}"
        f"{ctx}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Orders placed on IG automatically ✅</i>\n"
        f"<i>Max {config.MAX_ENTRIES} entries | Trail: 5-min low/high</i>"
    )


def orders_placed(state) -> str:
    return (
        f"✅ <b>ORDERS LIVE — IG</b>\n"
        f"Buy stop: {state.buy_level} (ID: {state.buy_order_id})\n"
        f"Sell stop: {state.sell_level} (ID: {state.sell_order_id})\n"
        f"OCA group: {state.oca_group}\n"
        f"<i>When one fills, the other auto-cancels</i>"
    )


def entry_triggered(state) -> str:
    risk = round(abs(state.entry_price - state.initial_stop), 1)
    icon = "🟩" if state.direction == "LONG" else "🟥"

    # Entry slippage from trade log
    slip_str = ""
    if state.trades:
        slip = state.trades[-1].get("entry_slippage", 0)
        if slip != 0:
            slip_str = f"\n📊 Slippage: {'+' if slip > 0 else ''}{slip} pts"

    msg = (
        f"{icon} <b>ENTRY #{state.entries_used} — {state.direction} DAX</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: <b>{state.entry_price}</b>\n"
        f"Stop:  {state.trailing_stop}\n"
        f"Risk:  {risk} pts (€{round(risk * config.NUM_CONTRACTS, 1)})\n"
        f"Qty:   {config.NUM_CONTRACTS} contracts"
        f"{slip_str}\n"
    )

    if config.PARTIAL_EXIT:
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"C1 TP: {state.tp1_price} (+{config.TP1_PTS} pts)\n"
            f"C2 TP: {state.tp2_price} (+{config.TP2_PTS} pts)\n"
            f"C3: candle trail\n"
        )

    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"

    if state.entries_used >= config.MAX_ENTRIES:
        msg += f"<i>Final entry — no more flips</i>\n"
    else:
        msg += f"<i>Entries: {state.entries_used}/{config.MAX_ENTRIES} | Flip available</i>\n"

    msg += f"<i>Trail: prev candle low/high</i>"
    return msg


def tp_filled(state, tp_num: int) -> str:
    tp_pts = config.TP1_PTS if tp_num == 1 else config.TP2_PTS
    tp_price = state.tp1_price if tp_num == 1 else state.tp2_price
    return (
        f"🎯 <b>TP{tp_num} FILLED — C{tp_num} closed</b>\n"
        f"Exit: {tp_price} (+{tp_pts} pts)\n"
        f"Contracts remaining: {state.contracts_active}\n"
        f"<i>{'C3 riding candle trail' if state.contracts_active == 1 else 'C2 TP + C3 trail still active'}</i>"
    )


def trail_updated(state, old_stop: float) -> str:
    direction = "↑" if state.direction == "LONG" else "↓"
    locked = ""
    if state.direction == "LONG" and state.trailing_stop > state.entry_price:
        locked = f"\n🔒 Profit locked: {round(state.trailing_stop - state.entry_price, 1)} pts"
    elif state.direction == "SHORT" and state.trailing_stop < state.entry_price:
        locked = f"\n🔒 Profit locked: {round(state.entry_price - state.trailing_stop, 1)} pts"

    return (
        f"📏 <b>TRAIL {direction} — DAX {state.direction}</b>\n"
        f"Stop: {old_stop} → <b>{state.trailing_stop}</b>\n"
        f"Entry: {state.entry_price}{locked}"
    )


def exit_stopped(state, trade: dict) -> str:
    pnl = trade.get("pnl_pts", 0)
    icon = "✅" if pnl >= 0 else "❌"
    ps = "+" if pnl >= 0 else ""
    pnl_eur = round(pnl, 1)  # Total P&L already accounts for all contracts

    msg = (
        f"{icon} <b>EXIT — DAX {trade.get('direction', '')}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: {trade.get('entry', '?')}\n"
        f"Exit:  {trade.get('exit', '?')} (stop)\n"
        f"P&L:   <b>{ps}{pnl} pts total</b> (€{abs(pnl_eur)})\n"
    )

    if config.PARTIAL_EXIT:
        tp1 = "filled" if trade.get("tp1_filled") else "not hit"
        tp2 = "filled" if trade.get("tp2_filled") else "not hit"
        stopped = trade.get("contracts_stopped", config.NUM_CONTRACTS)
        msg += (
            f"C1 TP1: {tp1} | C2 TP2: {tp2}\n"
            f"C3 stopped: {stopped} contract(s)\n"
        )

    msg += f"MFE:   {trade.get('mfe', 0)} pts\n"

    # Slippage summary
    total_slip = trade.get("slippage_total", 0)
    if total_slip != 0:
        msg += f"📊 Slippage: {'+' if total_slip > 0 else ''}{total_slip} pts total\n"

    if state.entries_used < config.MAX_ENTRIES:
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔄 <i>Flip — placing new OCA bracket</i>"
        )
    else:
        msg += (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏁 <i>Max entries — done for today</i>"
        )
    return msg


def day_summary(state) -> str:
    trades = state.trades
    if not trades:
        return (
            f"📊 <b>ASRS SUMMARY</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 {state.date}\nNo trades triggered."
        )

    total_pnl = sum(t.get("pnl_pts", 0) for t in trades)
    ps = "+" if total_pnl >= 0 else ""
    icon = "🟢" if total_pnl >= 0 else "🔴"

    bar_info = f"Bar {state.bar_number}"
    if state.bar5_rule_matched:
        bar_info += f" ({state.bar5_rule_matched})"

    msg = (
        f"📊 <b>ASRS SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {state.date} | {bar_info}\n"
        f"{icon} Net: <b>{ps}{round(total_pnl, 1)} pts</b>\n"
        f"Trades: {len(trades)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    for t in trades:
        p = t.get("pnl_pts", 0)
        s = "+" if p >= 0 else ""
        msg += (
            f"#{t.get('num', '?')} {t.get('direction', '?')}: "
            f"{t.get('entry', '?')} → {t.get('exit', '?')} "
            f"= {s}{round(p, 1)} pts\n"
        )

    # Slippage summary across all trades
    total_slip = sum(t.get("slippage_total", 0) for t in trades)
    slip_str = ""
    if total_slip != 0:
        slip_str = f"\n📊 Slippage: {'+' if total_slip > 0 else ''}{round(total_slip, 1)} pts"

    # total_pnl already includes all 3 contracts (TP1 + TP2 + trail)
    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"€ Total: {ps}€{abs(round(total_pnl, 1))} ({config.NUM_CONTRACTS}x micro)"
        f"{slip_str}\n"
    )
    return msg


def error_alert(msg: str) -> str:
    return f"🚨 <b>ASRS ERROR</b>\n━━━━━━━━━━━━━━━━━━━━━━\n{msg}"


def connection_status(connected: bool) -> str:
    if connected:
        mode = "Demo" if config.IG_DEMO else "Live"
        return f"🔌 <b>IG Connected</b> ({mode} | {config.IG_EPIC})"
    return "🔌 <b>IG Disconnected</b> — will retry"
