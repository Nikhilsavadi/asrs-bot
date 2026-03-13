"""
telegram_bot.py — Interactive Telegram (Inline Keyboards + Callbacks)
═══════════════════════════════════════════════════════════════════════════════

Handles:
  - Sending messages with inline keyboard buttons
  - Polling for button presses (callback queries)
  - Timeout handling (default action if no response)

Separate from alerts.py to keep alert formatting clean.
"""

import asyncio
import logging
import httpx

from dax_bot import config

logger = logging.getLogger(__name__)

API_BASE = f"https://api.telegram.org/bot{config.TG_TOKEN}"


async def send_with_buttons(
    text: str,
    buttons: list[list[dict]],
) -> int | None:
    """
    Send message with inline keyboard buttons.

    buttons format:
    [
        [{"text": "TRADE", "callback_data": "trade"},
         {"text": "SKIP", "callback_data": "skip"}],
        [{"text": "LONG ONLY", "callback_data": "long_only"},
         {"text": "SHORT ONLY", "callback_data": "short_only"}],
    ]

    Returns message_id for later reference, or None on failure.
    """
    if not config.TG_TOKEN or not config.TG_CHAT_ID:
        logger.warning("Telegram not configured")
        print(f"\n[BUTTONS] {text}")
        for row in buttons:
            print(f"  Options: {[b['text'] for b in row]}")
        return None

    payload = {
        "chat_id": config.TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "reply_markup": {
            "inline_keyboard": buttons,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{API_BASE}/sendMessage", json=payload, timeout=10)
            if r.status_code == 200:
                data = r.json()
                msg_id = data["result"]["message_id"]
                logger.info(f"Sent interactive message (id: {msg_id})")
                return msg_id
            logger.error(f"Telegram error: {r.status_code} {r.text}")
    except Exception as e:
        logger.error(f"Send failed: {e}")
    return None


async def wait_for_callback(
    timeout_sec: int = 300,
    poll_interval: int = 3,
) -> str | None:
    """
    Poll for a callback query (button press) from the user.

    Returns the callback_data string, or None if timeout.
    Uses getUpdates with offset to avoid processing old callbacks.
    """
    if not config.TG_TOKEN:
        return None

    # Get current update offset
    offset = await _get_latest_offset()

    elapsed = 0
    while elapsed < timeout_sec:
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "timeout": poll_interval,
                    "allowed_updates": ["callback_query"],
                }
                if offset:
                    params["offset"] = offset

                r = await client.get(
                    f"{API_BASE}/getUpdates",
                    params=params,
                    timeout=poll_interval + 5,
                )

                if r.status_code == 200:
                    data = r.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1

                        if "callback_query" in update:
                            cb = update["callback_query"]
                            cb_data = cb.get("data", "")
                            cb_id = cb.get("id", "")
                            user = cb.get("from", {}).get("first_name", "User")

                            # Acknowledge the button press
                            await _answer_callback(cb_id, f"✅ {cb_data.upper()}")

                            logger.info(f"Callback received: {cb_data} from {user}")
                            return cb_data

        except httpx.TimeoutException:
            pass  # Normal — poll timeout
        except Exception as e:
            logger.error(f"Callback poll error: {e}")

        elapsed += poll_interval

    logger.info(f"No response after {timeout_sec}s")
    return None


async def _get_latest_offset() -> int | None:
    """Get the latest update offset to avoid old callbacks."""
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


async def _answer_callback(callback_id: str, text: str):
    """Acknowledge a callback query (removes loading state on button)."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text},
                timeout=5,
            )
    except Exception:
        pass


async def edit_message(message_id: int, new_text: str):
    """Edit a previously sent message (e.g., to show the decision)."""
    if not config.TG_TOKEN or not config.TG_CHAT_ID:
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/editMessageText",
                json={
                    "chat_id": config.TG_CHAT_ID,
                    "message_id": message_id,
                    "text": new_text,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
    except Exception as e:
        logger.error(f"Edit message failed: {e}")


# ── Pre-built button layouts ──────────────────────────────────────────────────

DECISION_BUTTONS = [
    [
        {"text": "✅ TRADE", "callback_data": "trade"},
        {"text": "❌ SKIP", "callback_data": "skip"},
    ],
    [
        {"text": "🟩 LONG ONLY", "callback_data": "long_only"},
        {"text": "🟥 SHORT ONLY", "callback_data": "short_only"},
    ],
]


def format_decision_message(state, conditions, rule_match) -> str:
    """Format the human decision request message."""
    risk = round(state.buy_level - state.sell_level, 1)
    mode = "📄 PAPER" if config.IB_PORT in (4002, 4004, 7497) else "🔴 LIVE"

    msg = (
        f"⚠️ <b>ASRS — DECISION NEEDED</b> [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {state.date}\n"
        f"🟩 Buy: <b>{state.buy_level}</b>\n"
        f"🟥 Sell: <b>{state.sell_level}</b>\n"
        f"   Risk: {risk} pts\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Today's Conditions:</b>\n"
        f"{conditions.summary_str()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Backtest Says:</b>\n"
        f"{rule_match.confidence}\n"
        f"{rule_match.stats}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Tap a button below (5 min timeout → "
        f"{'SKIP' if _default_is_skip() else 'TRADE'})</i>"
    )
    return msg


def format_decision_result(decision: str, source: str) -> str:
    """Format confirmation of the decision made."""
    icons = {
        "trade": "✅ TRADING",
        "skip": "⛔ SKIPPING",
        "long_only": "🟩 LONG ONLY",
        "short_only": "🟥 SHORT ONLY",
    }
    label = icons.get(decision, decision.upper())
    return f"{label} — decided by {source}"


def _default_is_skip() -> bool:
    """Check what happens on no response."""
    from rules import load_rules
    rules = load_rules()
    return rules.get("no_response_action", "SKIP") == "SKIP"
