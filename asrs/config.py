"""
config.py -- Per-instrument configuration + global settings
============================================================
All instrument-specific parameters live in INSTRUMENTS dict.
Signal class reads from its config dict -- no per-instrument imports.
"""

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# -- Timezones (used by scheduler and bar numbering) -------------------------
TZ_UK  = ZoneInfo("Europe/London")
TZ_CET = ZoneInfo("Europe/Berlin")
TZ_ET  = ZoneInfo("America/New_York")
TZ_JST = ZoneInfo("Asia/Tokyo")

# -- IG Markets credentials (shared across all instruments) -------------------
IG_USERNAME   = os.getenv("IG_USERNAME", "")
IG_PASSWORD   = os.getenv("IG_PASSWORD", "")
IG_API_KEY    = os.getenv("IG_API_KEY", "")
IG_ACC_NUMBER = os.getenv("IG_ACC_NUMBER", "")
IG_DEMO       = os.getenv("IG_DEMO", "true").lower() == "true"

# -- Telegram -----------------------------------------------------------------
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# -- Global defaults ----------------------------------------------------------
NUM_CONTRACTS = int(os.getenv("NUM_CONTRACTS", "1"))  # Starting stake
MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "5"))   # Hard ceiling

# -- Bar 5 rules: range tokens that trigger bar 5 instead of bar 4 -----------
# "NORMAL,WIDE" means: if bar 4 range is NORMAL or WIDE, wait for bar 5
BAR5_RULES = [r.strip() for r in os.getenv("BAR5_RULES", "NORMAL,WIDE").split(",") if r.strip()]

# -- Per-instrument configuration ---------------------------------------------
INSTRUMENTS = {
    "DAX": {
        "epic": "IX.D.DAX.DAILY.IP",
        "currency": "GBP",
        "label": "DAX 40",
        # Entry levels
        "buffer": 2.0,
        "narrow_range": 15,
        "wide_range": 40,
        # Risk management
        "max_risk_gbp": 50.0,
        "max_entries": 2,
        "max_bar_range": 120,
        "max_spread": 10.0,
        "max_slippage_pct": 0.5,
        "disaster_stop_pts": 200,
        # Trail
        "breakeven_pts": 15.0,
        "tight_threshold": 100.0,
        "trail_min_move": 3.0,
        # Adds
        "add_trigger": 25.0,
        "add_max": 2,
        # Session 1: times in instrument timezone
        "s1_open_hour": 9, "s1_open_minute": 0,
        # Session 2
        "s2_open_hour": 14, "s2_open_minute": 0,
        # Session end (force close)
        "session_end_hour": 17, "session_end_minute": 30,
        # Timezones
        "timezone": "Europe/Berlin",
        "scheduler_timezone": "Europe/Berlin",  # Config hours are CET — scheduler must use CET
    },
    "US30": {
        "epic": "IX.D.DOW.DAILY.IP",
        "currency": "GBP",
        "label": "US30 (Dow)",
        "buffer": 5.0,
        "narrow_range": 30,
        "wide_range": 100,
        "max_risk_gbp": 50.0,
        "max_entries": 2,
        "max_bar_range": 300,
        "max_spread": 10.0,
        "max_slippage_pct": 0.5,
        "disaster_stop_pts": 1000,
        "breakeven_pts": 20.0,
        "tight_threshold": 80.0,
        "trail_min_move": 5.0,
        "add_trigger": 30.0,
        "add_max": 2,
        "s1_open_hour": 9, "s1_open_minute": 30,
        "s2_open_hour": 11, "s2_open_minute": 0,
        "session_end_hour": 16, "session_end_minute": 0,
        "timezone": "America/New_York",
        "scheduler_timezone": "America/New_York",
    },
    "NIKKEI": {
        "epic": "IX.D.NIKKEI.DAILY.IP",
        "currency": "GBP",
        "label": "Nikkei 225",
        "buffer": 2.0,
        "narrow_range": 50,
        "wide_range": 150,
        "max_risk_gbp": 75.0,
        "max_entries": 2,
        "max_bar_range": 250,
        "max_spread": 50.0,
        "max_slippage_pct": 0.5,
        "disaster_stop_pts": 1000,
        "breakeven_pts": 50.0,
        "tight_threshold": 300.0,
        "trail_min_move": 5.0,
        "add_trigger": 80.0,
        "add_max": 2,
        "s1_open_hour": 10, "s1_open_minute": 0,
        "s2_open_hour": 12, "s2_open_minute": 0,
        "session_end_hour": 15, "session_end_minute": 0,
        "timezone": "Asia/Tokyo",
        "scheduler_timezone": "Asia/Tokyo",
    },
}


def get_tz(instrument: str) -> ZoneInfo:
    """Return the instrument's primary timezone."""
    return ZoneInfo(INSTRUMENTS[instrument]["timezone"])


def get_scheduler_tz(instrument: str) -> ZoneInfo:
    """Return the timezone used by APScheduler for this instrument."""
    return ZoneInfo(INSTRUMENTS[instrument]["scheduler_timezone"])
