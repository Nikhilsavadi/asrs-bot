#!/usr/bin/env python3
"""
watchdog.py -- Runs on HOST, monitors bot heartbeat, restarts if stale.
========================================================================

The ASRS bot writes a timestamp to /app/data/heartbeat every 60 seconds.
Since ./data is mounted as /app/data inside the container, the host can
read it at ./data/heartbeat (or the absolute path below).

This script:
  1. Reads the heartbeat file every 30 seconds
  2. If the timestamp is older than 180 seconds, sends a Telegram alert
     and restarts the Docker container via docker compose
  3. Runs as a simple loop -- deploy as a systemd service on the host

Usage:
    python3 watchdog.py

Environment variables:
    TELEGRAM_BOT_TOKEN  -- Telegram bot token (required for alerts)
    TELEGRAM_CHAT_ID    -- Telegram chat ID (required for alerts)
    HEARTBEAT_FILE      -- Path to heartbeat file (default: /root/asrs-bot/data/heartbeat)
    COMPOSE_DIR         -- Path to docker-compose.yml directory (default: /root/asrs-bot)
    STALE_THRESHOLD     -- Seconds before heartbeat is considered stale (default: 180)
"""

import os
import subprocess
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("watchdog")

# Configuration
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
HEARTBEAT_FILE = os.getenv("HEARTBEAT_FILE", "/root/asrs-bot/data/heartbeat")
COMPOSE_DIR = os.getenv("COMPOSE_DIR", "/root/asrs-bot")
STALE_THRESHOLD = int(os.getenv("STALE_THRESHOLD", "180"))
CHECK_INTERVAL = 30  # seconds between checks

# Cooldown: don't restart more than once per 5 minutes
_last_restart: float = 0
RESTART_COOLDOWN = 300


def send_telegram(text: str):
    """Send alert via Telegram HTTP API (no bot framework needed)."""
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.warning("Telegram not configured -- alert skipped")
        return

    try:
        import httpx
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        with httpx.Client(timeout=10) as client:
            client.post(url, json={
                "chat_id": TG_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            })
    except ImportError:
        # Fall back to urllib if httpx not installed on host
        import urllib.request
        import urllib.parse
        import json
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": TG_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.error(f"Telegram fallback send failed: {e}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


def read_heartbeat() -> float | None:
    """Read timestamp from heartbeat file. Returns None if file missing/invalid."""
    try:
        with open(HEARTBEAT_FILE, "r") as f:
            return float(f.read().strip())
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as e:
        logger.warning(f"Heartbeat file read error: {e}")
        return None


def restart_container():
    """Restart the ASRS bot Docker container."""
    global _last_restart

    now = time.time()
    if now - _last_restart < RESTART_COOLDOWN:
        logger.warning(
            f"Restart cooldown active ({RESTART_COOLDOWN}s) -- skipping"
        )
        return

    _last_restart = now
    logger.info("Restarting ASRS bot container...")

    try:
        result = subprocess.run(
            ["docker", "compose", "restart"],
            cwd=COMPOSE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Container restart successful")
            send_telegram("WATCHDOG: Container restarted successfully")
        else:
            logger.error(f"Container restart failed: {result.stderr}")
            send_telegram(f"WATCHDOG: Container restart FAILED\n<pre>{result.stderr[:500]}</pre>")
    except subprocess.TimeoutExpired:
        logger.error("Container restart timed out (120s)")
        send_telegram("WATCHDOG: Container restart TIMED OUT")
    except Exception as e:
        logger.error(f"Container restart error: {e}")
        send_telegram(f"WATCHDOG: Container restart error: {e}")


def main():
    """Main watchdog loop."""
    logger.info(
        f"Watchdog started | heartbeat={HEARTBEAT_FILE} | "
        f"threshold={STALE_THRESHOLD}s | check_interval={CHECK_INTERVAL}s"
    )

    while True:
        try:
            ts = read_heartbeat()

            if ts is None:
                # No heartbeat file yet -- bot may not have started
                logger.info("No heartbeat file found -- waiting for bot to start")
            else:
                age = time.time() - ts
                if age > STALE_THRESHOLD:
                    logger.warning(
                        f"Heartbeat STALE: {age:.0f}s old (threshold={STALE_THRESHOLD}s)"
                    )
                    send_telegram(
                        f"WATCHDOG: Bot heartbeat stale ({age:.0f}s), restarting..."
                    )
                    restart_container()
                else:
                    logger.info(f"Heartbeat OK: {age:.0f}s old")

        except Exception as e:
            logger.error(f"Watchdog check error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
