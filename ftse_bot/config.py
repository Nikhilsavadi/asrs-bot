"""
config.py -- FTSE 1BN/1BP Bot Configuration
"""

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# -- Timezone ------------------------------------------------------------------
TZ_UK = ZoneInfo("Europe/London")

# -- Broker --------------------------------------------------------------------
BROKER = "ig"

# -- IG Markets Connection ----------------------------------------------------
IG_USERNAME   = os.getenv("IG_USERNAME", "")
IG_PASSWORD   = os.getenv("IG_PASSWORD", "")
IG_API_KEY    = os.getenv("IG_API_KEY", "")
IG_ACC_NUMBER = os.getenv("IG_ACC_NUMBER", "")
IG_DEMO       = os.getenv("IG_DEMO", "true").lower() == "true"
IG_EPIC       = os.getenv("FTSE_IG_EPIC", "IX.D.FTSE.DAILY.IP")

# -- Instrument ----------------------------------------------------------------
INSTRUMENT = "FTSE 100 DFB"
SEC_TYPE   = "SPREAD_BET"
EXCHANGE   = "IG"
CURRENCY   = "GBP"

# -- Strategy ------------------------------------------------------------------
BUFFER_PTS              = float(os.getenv("FTSE_BUFFER_PTS", "1"))        # 1pt above/below bar
BAR_WIDTH_THRESHOLD     = float(os.getenv("FTSE_BAR_WIDTH_THRESHOLD", "30"))
TRAIL_MIN_ALERT_MOVE    = float(os.getenv("FTSE_TRAIL_MIN_ALERT", "5"))   # Only alert every 5pt trail move
DOJI_ACTION             = os.getenv("DOJI_ACTION", "SKIP").upper()        # SKIP, TREAT_AS_1BN, TREAT_AS_1BP
MAX_ENTRIES             = int(os.getenv("FTSE_MAX_ENTRIES", "2"))         # Re-entries per day after stop

# -- Exit Strategy: Candle Trail -----------------------------------------------
# Previous candle low/high as trailing stop (same as DAX bot)
TRAIL_MODE              = "CANDLE"  # CANDLE = previous bar low/high

# -- Add-to-Winners (Strength Mode) -------------------------------------------
ADD_STRENGTH_ENABLED    = os.getenv("FTSE_ADD_ENABLED", "true").lower() == "true"
ADD_STRENGTH_TRIGGER    = float(os.getenv("FTSE_ADD_TRIGGER", "25"))     # +25pts from last entry
ADD_STRENGTH_MAX        = int(os.getenv("FTSE_ADD_MAX", "2"))            # Max 2 extra positions

# -- Position Sizing -----------------------------------------------------------
NUM_CONTRACTS           = int(os.getenv("FTSE_NUM_CONTRACTS", "3"))      # 3x GBP/pt
STAKE_PER_POINT         = float(os.getenv("STAKE_PER_POINT", "1"))       # GBP per point per contract

# -- Telegram ------------------------------------------------------------------
TG_TOKEN   = os.getenv("FTSE_TELEGRAM_TOKEN", "")
TG_CHAT_ID = os.getenv("FTSE_TELEGRAM_CHAT_ID", "")

# -- Schedule (UK time) -------------------------------------------------------
BAR_HOUR        = 8
BAR_MINUTE      = 5     # Read the 08:00-08:05 bar at 08:05
SESSION_END_H   = 16
SESSION_END_M   = 30    # Hard close at 16:30
SUMMARY_HOUR    = 17
SUMMARY_MINUTE  = 0     # Daily summary at 17:00

# -- Logging -------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
