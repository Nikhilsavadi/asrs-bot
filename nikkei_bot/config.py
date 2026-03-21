"""
Nikkei 225 Bot config — Japan 225 ASRS strategy.
Same rules as DAX but adapted for Tokyo market hours and Nikkei price scale.
"""

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# -- Timezones ----------------------------------------------------------------
TZ_JST    = ZoneInfo("Asia/Tokyo")
TZ_UK     = ZoneInfo("Europe/London")
TZ_CET    = TZ_JST  # Nikkei uses JST as primary timezone (aliased for shared code compat)

# -- Broker --------------------------------------------------------------------
IG_EPIC       = os.getenv("NIKKEI_IG_EPIC", "IX.D.NIKKEI.DAILY.IP")
INSTRUMENT    = "Japan 225 DFB"
IG_DEMO       = os.getenv("IG_DEMO", "true").lower() == "true"

# -- Strategy ------------------------------------------------------------------
# Nikkei is ~35000-50000, moves in whole numbers
BUFFER_PTS    = float(os.getenv("NIKKEI_BUFFER_PTS", "2.0"))
NARROW_RANGE  = int(os.getenv("NIKKEI_NARROW_RANGE", "50"))
WIDE_RANGE    = int(os.getenv("NIKKEI_WIDE_RANGE", "150"))
MAX_ENTRIES   = int(os.getenv("NIKKEI_MAX_ENTRIES", "2"))    # entry + re-entry
ENABLE_FLIPS  = os.getenv("NIKKEI_ENABLE_FLIPS", "false").lower() == "true"
NUM_CONTRACTS = int(os.getenv("NIKKEI_NUM_CONTRACTS", "1"))
MAX_CONTRACTS = int(os.getenv("NIKKEI_MAX_CONTRACTS", "5"))
NARROW_STD_MULTIPLIER = int(os.getenv("NIKKEI_NARROW_STD_MULTIPLIER", "2"))

# Bar 5 rules: use bar 5 for NORMAL+WIDE ranges (same as DAX hybrid)
BAR5_RULES_STR = os.getenv("NIKKEI_BAR5_RULES", "NORMAL,WIDE")
BAR5_RULES = [r.strip() for r in BAR5_RULES_STR.split(",") if r.strip()]

# -- Trailing Stop (scaled for Nikkei) ----------------------------------------
TRAIL_MIN_MOVE          = float(os.getenv("NIKKEI_TRAIL_MIN_MOVE", "5"))
TRAIL_BREAKEVEN_PTS     = float(os.getenv("NIKKEI_TRAIL_BREAKEVEN_PTS", "50"))
TRAIL_TIGHT_THRESHOLD   = float(os.getenv("NIKKEI_TRAIL_TIGHT_THRESHOLD", "300"))
TRAIL_BREAKEVEN_TRIGGER = float(os.getenv("NIKKEI_TRAIL_BREAKEVEN_TRIGGER", "5"))

# -- EMA Trail -----------------------------------------------------------------
TRAIL_EMA_PERIOD       = int(os.getenv("NIKKEI_TRAIL_EMA_PERIOD", "10"))
TRAIL_EMA_TRIGGER      = float(os.getenv("NIKKEI_TRAIL_EMA_TRIGGER", "10"))
TRAIL_EMA_BUFFER       = float(os.getenv("NIKKEI_TRAIL_EMA_BUFFER", "0.005"))
ADD_EMA_TOUCH_ZONE     = float(os.getenv("NIKKEI_ADD_EMA_TOUCH_ZONE", "0.003"))

# -- Add-to-Winners (scaled for Nikkei) ----------------------------------------
ADD_STRENGTH_ENABLED   = os.getenv("NIKKEI_ADD_STRENGTH_ENABLED", "false").lower() == "true"  # DISABLED: IG creates separate positions per add, stops not managed
ADD_STRENGTH_TRIGGER   = float(os.getenv("NIKKEI_ADD_STRENGTH_TRIGGER", "80"))
ADD_STRENGTH_MAX       = int(os.getenv("NIKKEI_ADD_STRENGTH_MAX", "2"))

# -- Partial Exit --------------------------------------------------------------
PARTIAL_EXIT   = os.getenv("NIKKEI_PARTIAL_EXIT", "false").lower() == "true"
TP1_PTS        = float(os.getenv("NIKKEI_TP1_PTS", "25"))
TP2_PTS        = float(os.getenv("NIKKEI_TP2_PTS", "60"))

# -- Risk Management -----------------------------------------------------------
MAX_SLIPPAGE_PTS   = int(os.getenv("NIKKEI_MAX_SLIPPAGE_PTS", "30"))
MAX_SPREAD_PTS     = float(os.getenv("NIKKEI_MAX_SPREAD_PTS", "50.0"))   # Nikkei has wider IG spread ~30pts
DISASTER_STOP_PTS  = int(os.getenv("NIKKEI_DISASTER_STOP_PTS", "1000"))
MAX_DAILY_LOSS_GBP = float(os.getenv("NIKKEI_MAX_DAILY_LOSS_GBP", "200"))
STALE_PRICE_SECS   = int(os.getenv("NIKKEI_STALE_PRICE_SECS", "30"))

# -- SPX Regime Filter (not applicable for Nikkei -- disabled) -----------------
SPX_REGIME_ENABLED = False
SPX_EPIC           = IG_EPIC

# -- Telegram ------------------------------------------------------------------
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_GBP   = float(os.getenv("NIKKEI_RISK_PER_TRADE_GBP", "25"))

# -- Schedule (JST -- Asia/Tokyo) ----------------------------------------------
# TSE opens 09:00 JST. Bar 4 closes at 09:20 JST. Schedule at 09:21 JST.
# 09:21 JST = 00:21 UTC (winter), 00:21 UTC (no DST in Japan)
MORNING_HOUR   = 0    # UTC hour for 09:21 JST (00:21 UTC)
MORNING_MINUTE = 21

# -- Session 2 (disabled -- Nikkei S2 not worth it based on backtest) ----------
SESSION2_ENABLED   = os.getenv("NIKKEI_SESSION2_ENABLED", "false").lower() == "true"
SESSION2_HOUR_JST  = 12   # placeholder
SESSION2_BAR_COUNT = 4

# -- State ---------------------------------------------------------------------
STATE_FILE = os.getenv("NIKKEI_STATE_FILE", "data/nikkei_daily_state.json")
