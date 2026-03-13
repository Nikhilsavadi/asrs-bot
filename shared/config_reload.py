"""
config_reload.py — Hot-reload config via Telegram /set commands
═══════════════════════════════════════════════════════════════

Only whitelisted keys can be changed. Validates type and range.
Writes to .env for persistence, patches module attributes in-place.
"""

import os
import logging

logger = logging.getLogger(__name__)

ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")

# ── Whitelisted config keys ───────────────────────────────────────────────
# Maps friendly name → { env var, module path, attribute, type, min, max }

ALLOWED_KEYS = {
    # DAX
    "contracts":       {"env": "NUM_CONTRACTS",        "mod": "dax_bot.config",  "attr": "NUM_CONTRACTS",        "type": int,   "min": 1,   "max": 10},
    "buffer":          {"env": "BUFFER_PTS",           "mod": "dax_bot.config",  "attr": "BUFFER_PTS",           "type": int,   "min": 0,   "max": 10},
    "max_entries":     {"env": "MAX_ENTRIES",          "mod": "dax_bot.config",  "attr": "MAX_ENTRIES",          "type": int,   "min": 1,   "max": 5},
    "risk":            {"env": "RISK_PER_TRADE_GBP",   "mod": "dax_bot.config",  "attr": "RISK_GBP",            "type": float, "min": 10,  "max": 500},
    "tp1":             {"env": "TP1_PTS",              "mod": "dax_bot.config",  "attr": "TP1_PTS",              "type": float, "min": 5,   "max": 100},
    "tp2":             {"env": "TP2_PTS",              "mod": "dax_bot.config",  "attr": "TP2_PTS",              "type": float, "min": 10,  "max": 200},
    "add_trigger":     {"env": "ADD_STRENGTH_TRIGGER", "mod": "dax_bot.config",  "attr": "ADD_STRENGTH_TRIGGER", "type": float, "min": 10,  "max": 100},
    "add_max":         {"env": "ADD_STRENGTH_MAX",     "mod": "dax_bot.config",  "attr": "ADD_STRENGTH_MAX",     "type": int,   "min": 0,   "max": 5},
    "trail_ema":       {"env": "TRAIL_EMA_PERIOD",     "mod": "dax_bot.config",  "attr": "TRAIL_EMA_PERIOD",     "type": int,   "min": 3,   "max": 50},
    "partial_exit":    {"env": "PARTIAL_EXIT",         "mod": "dax_bot.config",  "attr": "PARTIAL_EXIT",         "type": "bool"},
    # FTSE (requires ENABLE_FTSE=true + restart)
    "enable_ftse":     {"env": "ENABLE_FTSE",           "mod": "run_all",         "attr": "ENABLE_FTSE",          "type": "bool"},
    "ftse_contracts":  {"env": "FTSE_NUM_CONTRACTS",   "mod": "ftse_bot.config", "attr": "NUM_CONTRACTS",        "type": int,   "min": 1,   "max": 10},
    "ftse_stake":      {"env": "STAKE_PER_POINT",      "mod": "ftse_bot.config", "attr": "STAKE_PER_POINT",      "type": float, "min": 0.5, "max": 20},
    "ftse_add_trigger": {"env": "FTSE_ADD_TRIGGER",    "mod": "ftse_bot.config", "attr": "ADD_STRENGTH_TRIGGER", "type": float, "min": 10,  "max": 100},
    "ftse_add_max":    {"env": "FTSE_ADD_MAX",         "mod": "ftse_bot.config", "attr": "ADD_STRENGTH_MAX",     "type": int,   "min": 0,   "max": 5},
}


def apply_set(key: str, value_str: str) -> str:
    """
    Validate, update .env, and patch config module in-place.
    Returns a confirmation or error message string.
    """
    key = key.lower().strip()
    if key not in ALLOWED_KEYS:
        available = ", ".join(sorted(ALLOWED_KEYS.keys()))
        return f"❌ Unknown key: <code>{key}</code>\n\nAvailable: {available}"

    spec = ALLOWED_KEYS[key]

    # Type validation
    try:
        if spec["type"] == "bool":
            val_str = value_str.strip().lower()
            if val_str in ("true", "1", "yes", "on"):
                typed_val = True
                env_val = "true"
            elif val_str in ("false", "0", "no", "off"):
                typed_val = False
                env_val = "false"
            else:
                return f"❌ Invalid bool: <code>{value_str}</code> (use true/false)"
        else:
            typed_val = spec["type"](value_str)
            env_val = str(typed_val)

            # Range validation
            if "min" in spec and typed_val < spec["min"]:
                return f"❌ {key} must be >= {spec['min']} (got {typed_val})"
            if "max" in spec and typed_val > spec["max"]:
                return f"❌ {key} must be <= {spec['max']} (got {typed_val})"
    except (ValueError, TypeError):
        return f"❌ Invalid {spec['type'].__name__}: <code>{value_str}</code>"

    # Get old value
    import importlib
    try:
        mod = importlib.import_module(spec["mod"])
        old_val = getattr(mod, spec["attr"], "?")
    except ImportError:
        return f"❌ Module {spec['mod']} not available"

    # Update os.environ
    os.environ[spec["env"]] = env_val

    # Patch module attribute in-place
    setattr(mod, spec["attr"], typed_val)

    # Write to .env file for persistence
    _update_env_file(spec["env"], env_val)

    logger.info(f"Config updated: {key} = {typed_val} (was {old_val})")

    return (
        f"✅ <b>CONFIG UPDATED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<code>{key}</code>: {old_val} → <b>{typed_val}</b>\n"
        f"ENV: {spec['env']}={env_val}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Active immediately. Persisted to .env.</i>"
    )


def get_current_config() -> str:
    """Format all configurable keys and their current values."""
    import importlib

    lines = [
        "⚙️ <b>CURRENT CONFIG</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "<b>DAX</b>",
    ]

    for key, spec in ALLOWED_KEYS.items():
        try:
            mod = importlib.import_module(spec["mod"])
            val = getattr(mod, spec["attr"], "?")
        except ImportError:
            val = "N/A"

        # Group headers
        if key == "enable_ftse":
            lines.append("")
            lines.append("<b>FTSE</b>")
        elif key == "ftse_contracts" and "<b>FTSE</b>" not in lines:
            lines.append("")
            lines.append("<b>FTSE</b>")

        range_str = ""
        if "min" in spec and "max" in spec:
            range_str = f" ({spec['min']}–{spec['max']})"

        lines.append(f"  <code>{key:<18}</code> = <b>{val}</b>{range_str}")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("<i>Use /set key value to change</i>")

    return "\n".join(lines)


def _update_env_file(env_key: str, env_val: str):
    """Update or append a key=value in the .env file."""
    if not os.path.exists(ENV_FILE):
        logger.warning(f".env file not found at {ENV_FILE}")
        return

    try:
        with open(ENV_FILE, "r") as f:
            lines = f.readlines()

        found = False
        new_lines = []
        for line in lines:
            stripped = line.strip()
            # Match KEY=... (ignoring comments)
            if stripped.startswith(f"{env_key}="):
                new_lines.append(f"{env_key}={env_val}\n")
                found = True
            elif stripped.startswith(f"# {env_key}="):
                # Uncomment and update
                new_lines.append(f"{env_key}={env_val}\n")
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f"{env_key}={env_val}\n")

        with open(ENV_FILE, "w") as f:
            f.writelines(new_lines)

        logger.info(f".env updated: {env_key}={env_val}")

    except Exception as e:
        logger.error(f"Failed to update .env: {e}")
