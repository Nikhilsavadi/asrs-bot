"""
config.py -- Strategy 3: Gold 15-min ORB (Asian/London/US) + Weekly ORB
"""

import os
from zoneinfo import ZoneInfo

TZ_UTC = ZoneInfo("UTC")
TZ_UK  = ZoneInfo("Europe/London")

# -- Instruments -------------------------------------------------------
GOLD_EPIC = os.getenv("GOLD_IG_EPIC", "CS.D.USCGC.TODAY.IP")

INSTRUMENTS = {
    "GOLD": {
        "epic": GOLD_EPIC,
        "spread": 0.4,
        "min_range": 2.0,       # 15-min bars have wider ranges than 5-min
        "max_range": 50.0,
        "currency": "USD",
        "sessions": {
            "ASIAN":  {"start": (0, 0),  "end": (7, 0)},
            "LONDON": {"start": (7, 0),  "end": (12, 0)},
            "US":     {"start": (13, 0), "end": (18, 0)},
        },
        "session_opens": [(0, 0), (7, 0), (13, 0)],
    },
}

# -- 15-min ORB Strategy -----------------------------------------------
CANDLE_TF_MINUTES       = 15        # Aggregate 5-min into 15-min bars
RANGE_BARS              = 4         # 4 x 15-min = 1 hour opening range
CONFIRMS_REQUIRED       = 2
CONFIRM_BODY_RATIO      = 0.6
CONFIRM_RANGE_MULT      = 1.5
TARGET_R                = 2.0
MAX_TRADES_PER_SESSION  = 2
EXCLUSION_BARS          = 1         # Skip first bar after range (15 mins)
MAX_DAILY_LOSS_R        = 6.0
MAX_DAILY_LOSS_GBP      = float(os.getenv("GOLD_MAX_DAILY_LOSS_GBP", "150"))  # Hard GBP cap

# -- Weekly ORB ---------------------------------------------------------
WEEKLY_ORB_ENABLED      = os.getenv("WEEKLY_ORB_ENABLED", "true").lower() == "true"
WEEKLY_MIN_RANGE        = 15.0
WEEKLY_MAX_RANGE        = 200.0
WEEKLY_TARGET_R         = 2.0
WEEKLY_CONFIRMS         = 2

# -- Risk --------------------------------------------------------------
BASE_RISK_GBP = float(os.getenv("GOLD_RISK_GBP", "25"))
MIN_STAKE     = 0.50
MAX_STAKE     = 50.0

# -- Telegram (reuse DAX token) ----------------------------------------
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# -- Enable/disable ----------------------------------------------------
ENABLE_GOLD = os.getenv("ENABLE_GOLD", "true").lower() == "true"
