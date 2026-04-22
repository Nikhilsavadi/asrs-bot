"""Persistent append-only audit log for bracket arm/cancel/trigger events.

Writes to /app/data/bracket_audit.log so events survive container restarts.
One line per event, JSONL for easy grep/jq."""
import json, os, time
from datetime import datetime, timezone

_LOG_PATH = os.environ.get("BRACKET_AUDIT_LOG", "/app/data/bracket_audit.log")


def _write(event: dict):
    try:
        event["ts"] = datetime.now(timezone.utc).isoformat()
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass  # audit log must never crash trading


def arm_success(signal: str, epic: str, buy: float, sell: float, qty: float):
    _write({"event": "ARM_SUCCESS", "signal": signal, "epic": epic,
            "buy": buy, "sell": sell, "qty": qty})


def arm_fail(signal: str, epic: str, reason: str):
    _write({"event": "ARM_FAIL", "signal": signal, "epic": epic, "reason": reason})


def cancel(signal: str, epic: str, reason: str):
    _write({"event": "CANCEL", "signal": signal, "epic": epic, "reason": reason})


def trigger_fired(signal: str, epic: str, direction: str, price: float, slippage: float):
    _write({"event": "TRIGGER_FIRED", "signal": signal, "epic": epic,
            "direction": direction, "price": price, "slippage": slippage})


def trigger_blocked(signal: str, epic: str, reason: str, direction: str, price: float):
    _write({"event": "TRIGGER_BLOCKED", "signal": signal, "epic": epic,
            "reason": reason, "direction": direction, "price": price})
