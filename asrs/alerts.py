"""
alerts.py -- Telegram alert formatting & delivery
==================================================
Single send() function used by all signals. Messages are plain HTML
with signal name prefix for identification.
"""

import logging
import httpx
from asrs import config

logger = logging.getLogger(__name__)


async def send(text: str) -> bool:
    """Send HTML message via Telegram Bot API."""
    if not config.TG_TOKEN or not config.TG_CHAT_ID:
        logger.warning("Telegram not configured -- console output only")
        import re
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
                return True
            logger.error(f"Telegram {r.status_code}: {r.text}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
    return False
