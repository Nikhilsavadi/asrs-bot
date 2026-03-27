"""
US30 Bot config — Dow Jones ASRS strategy.
Same rules as DAX but adapted for US market hours and US30 price scale.
"""

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# ── Timezones ──────────────────────────────────────────────────────────────────
TZ_ET     = ZoneInfo("America/New_York")
TZ_UK     = ZoneInfo("Europe/London")
TZ_CET    = TZ_ET  # SPX uses ET as primary timezone (aliased for shared code compat)

# ── Broker ───────────────────────────────────────────────────────────────────
IG_EPIC       = os.getenv("US30_IG_EPIC", "IX.D.DOW.DAILY.IP")  # Wall Street DFB (Dow Jones)
INSTRUMENT    = "US30 DFB"
IG_DEMO       = os.getenv("IG_DEMO", "true").lower() == "true"

# ── Strategy ─────────────────────────────────────────────────────────────────
# US30 is ~35000-45000, similar scale to DAX but slightly larger moves
BUFFER_PTS    = float(os.getenv("US30_BUFFER_PTS", "5.0"))
NARROW_RANGE  = int(os.getenv("US30_NARROW_RANGE", "30"))
WIDE_RANGE    = int(os.getenv("US30_WIDE_RANGE", "100"))
MAX_ENTRIES   = int(os.getenv("US30_MAX_ENTRIES", "2"))    # entry + re-entry
ENABLE_FLIPS  = os.getenv("US30_ENABLE_FLIPS", "false").lower() == "true"
NUM_CONTRACTS = int(os.getenv("US30_NUM_CONTRACTS", "1"))
MAX_CONTRACTS = int(os.getenv("US30_MAX_CONTRACTS", "5"))
NARROW_STD_MULTIPLIER = int(os.getenv("US30_NARROW_STD_MULTIPLIER", "2"))

# Bar 5 rules: use bar 5 for NORMAL+WIDE ranges (same as DAX hybrid)
BAR5_RULES_STR = os.getenv("US30_BAR5_RULES", "NORMAL,WIDE")
BAR5_RULES = [r.strip() for r in BAR5_RULES_STR.split(",") if r.strip()]

# ── Trailing Stop (scaled for US30) ─────────────────────────────────────────
TRAIL_MIN_MOVE          = float(os.getenv("US30_TRAIL_MIN_MOVE", "5"))
TRAIL_BREAKEVEN_PTS     = float(os.getenv("US30_TRAIL_BREAKEVEN_PTS", "20"))     # ~15 DAX pts
TRAIL_TIGHT_THRESHOLD   = float(os.getenv("US30_TRAIL_TIGHT_THRESHOLD", "80"))   # ~100 DAX pts
TRAIL_BREAKEVEN_TRIGGER = float(os.getenv("US30_TRAIL_BREAKEVEN_TRIGGER", "5"))

# ── EMA Trail ────────────────────────────────────────────────────────────────
TRAIL_EMA_PERIOD       = int(os.getenv("US30_TRAIL_EMA_PERIOD", "10"))
TRAIL_EMA_TRIGGER      = float(os.getenv("US30_TRAIL_EMA_TRIGGER", "10"))
TRAIL_EMA_BUFFER       = float(os.getenv("US30_TRAIL_EMA_BUFFER", "0.005"))
ADD_EMA_TOUCH_ZONE     = float(os.getenv("US30_ADD_EMA_TOUCH_ZONE", "0.003"))

# ── Add-to-Winners (scaled for US30) ────────────────────────────────────────
ADD_STRENGTH_ENABLED   = os.getenv("US30_ADD_STRENGTH_ENABLED", "true").lower() == "true"  # Multi-position stops managed: set stop on each deal_id
ADD_STRENGTH_TRIGGER   = float(os.getenv("US30_ADD_STRENGTH_TRIGGER", "30"))     # ~25 DAX pts
ADD_STRENGTH_MAX       = int(os.getenv("US30_ADD_STRENGTH_MAX", "2"))

# ── Partial Exit ─────────────────────────────────────────────────────────────
PARTIAL_EXIT   = os.getenv("US30_PARTIAL_EXIT", "false").lower() == "true"
TP1_PTS        = float(os.getenv("US30_TP1_PTS", "25"))    # ~20 DAX pts
TP2_PTS        = float(os.getenv("US30_TP2_PTS", "60"))    # ~50 DAX pts

# ── Risk Management ──────────────────────────────────────────────────────────
MAX_SLIPPAGE_PTS   = int(os.getenv("US30_MAX_SLIPPAGE_PTS", "15"))
MAX_SLIPPAGE_PCT   = float(os.getenv("US30_MAX_SLIPPAGE_PCT", "0.5"))  # Max slippage as % of initial risk (50%)
MAX_SPREAD_PTS     = float(os.getenv("US30_MAX_SPREAD_PTS", "15.0"))
DISASTER_STOP_PTS  = int(os.getenv("US30_DISASTER_STOP_PTS", "500"))
MAX_DAILY_LOSS_GBP = float(os.getenv("US30_MAX_DAILY_LOSS_GBP", "200"))
MAX_RISK_GBP       = float(os.getenv("US30_MAX_RISK_GBP", "50"))       # Cap risk per trade
MAX_BAR_RANGE      = int(os.getenv("US30_MAX_BAR_RANGE", "300"))        # Skip if bar range > this
STALE_PRICE_SECS   = int(os.getenv("US30_STALE_PRICE_SECS", "30"))

# ── Telegram ─────────────────────────────────────────────────────────────────
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_GBP   = float(os.getenv("US30_RISK_PER_TRADE_GBP", "25"))

# ── Schedule (US/Eastern time) ───────────────────────────────────────────────
# RTH opens 9:30 ET. Bar 4 closes at 9:50 ET. Schedule at 9:51.
MORNING_HOUR   = 9
MORNING_MINUTE = 51

# ── Session 2 (11:00 ET continuation) ──────────────────────────────────────
SESSION2_ENABLED   = os.getenv("US30_SESSION2_ENABLED", "true").lower() == "true"
SESSION2_HOUR_ET   = 11   # 11:00 ET
SESSION2_BAR_COUNT = 4    # Use bar 4 of session 2 (11:00-11:20 ET)

# ── State ────────────────────────────────────────────────────────────────────
STATE_FILE = os.getenv("US30_STATE_FILE", "data/us30_daily_state.json")
