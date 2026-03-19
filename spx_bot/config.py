"""
SPX Bot config — S&P 500 ASRS strategy.
Same rules as DAX but adapted for US market hours and SPX price scale.
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
IG_EPIC       = os.getenv("SPX_IG_EPIC", "IX.D.SPTRD.DAILY.IP")  # S&P 500 DFB
INSTRUMENT    = "SPX DFB"
IG_DEMO       = os.getenv("IG_DEMO", "true").lower() == "true"

# ── Strategy ─────────────────────────────────────────────────────────────────
# SPX moves ~1/3 of DAX in absolute points, so thresholds scaled accordingly
BUFFER_PTS    = float(os.getenv("SPX_BUFFER_PTS", "0.5"))
NARROW_RANGE  = int(os.getenv("SPX_NARROW_RANGE", "5"))
WIDE_RANGE    = int(os.getenv("SPX_WIDE_RANGE", "15"))
MAX_ENTRIES   = int(os.getenv("SPX_MAX_ENTRIES", "2"))    # entry + re-entry
ENABLE_FLIPS  = os.getenv("SPX_ENABLE_FLIPS", "false").lower() == "true"
NUM_CONTRACTS = int(os.getenv("SPX_NUM_CONTRACTS", "1"))
MAX_CONTRACTS = int(os.getenv("SPX_MAX_CONTRACTS", "5"))
NARROW_STD_MULTIPLIER = int(os.getenv("SPX_NARROW_STD_MULTIPLIER", "2"))

# Bar 5 rules: use bar 5 for NORMAL+WIDE ranges (same as DAX hybrid)
BAR5_RULES_STR = os.getenv("SPX_BAR5_RULES", "NORMAL,WIDE")
BAR5_RULES = [r.strip() for r in BAR5_RULES_STR.split(",") if r.strip()]

# ── Trailing Stop (scaled for SPX) ──────────────────────────────────────────
TRAIL_MIN_MOVE          = float(os.getenv("SPX_TRAIL_MIN_MOVE", "1"))
TRAIL_BREAKEVEN_PTS     = float(os.getenv("SPX_TRAIL_BREAKEVEN_PTS", "5"))      # ~15 DAX pts
TRAIL_TIGHT_THRESHOLD   = float(os.getenv("SPX_TRAIL_TIGHT_THRESHOLD", "30"))   # ~100 DAX pts
TRAIL_BREAKEVEN_TRIGGER = float(os.getenv("SPX_TRAIL_BREAKEVEN_TRIGGER", "2"))

# ── EMA Trail ────────────────────────────────────────────────────────────────
TRAIL_EMA_PERIOD       = int(os.getenv("SPX_TRAIL_EMA_PERIOD", "10"))
TRAIL_EMA_TRIGGER      = float(os.getenv("SPX_TRAIL_EMA_TRIGGER", "3"))
TRAIL_EMA_BUFFER       = float(os.getenv("SPX_TRAIL_EMA_BUFFER", "0.005"))
ADD_EMA_TOUCH_ZONE     = float(os.getenv("SPX_ADD_EMA_TOUCH_ZONE", "0.003"))

# ── Add-to-Winners (scaled) ─────────────────────────────────────────────────
ADD_STRENGTH_ENABLED   = os.getenv("SPX_ADD_STRENGTH_ENABLED", "true").lower() == "true"
ADD_STRENGTH_TRIGGER   = float(os.getenv("SPX_ADD_STRENGTH_TRIGGER", "8"))      # ~25 DAX pts
ADD_STRENGTH_MAX       = int(os.getenv("SPX_ADD_STRENGTH_MAX", "2"))

# ── Partial Exit ─────────────────────────────────────────────────────────────
PARTIAL_EXIT   = os.getenv("SPX_PARTIAL_EXIT", "false").lower() == "true"
TP1_PTS        = float(os.getenv("SPX_TP1_PTS", "7"))     # ~20 DAX pts
TP2_PTS        = float(os.getenv("SPX_TP2_PTS", "17"))    # ~50 DAX pts

# ── Risk Management ──────────────────────────────────────────────────────────
MAX_SLIPPAGE_PTS   = int(os.getenv("SPX_MAX_SLIPPAGE_PTS", "5"))
MAX_SPREAD_PTS     = float(os.getenv("SPX_MAX_SPREAD_PTS", "5.0"))
MAX_DAILY_LOSS_GBP = float(os.getenv("SPX_MAX_DAILY_LOSS_GBP", "200"))
STALE_PRICE_SECS   = int(os.getenv("SPX_STALE_PRICE_SECS", "30"))

# ── SPX Regime Filter (not applicable for SPX itself — disabled) ────────────
SPX_REGIME_ENABLED = False
SPX_EPIC           = IG_EPIC

# ── Telegram ─────────────────────────────────────────────────────────────────
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_GBP   = float(os.getenv("SPX_RISK_PER_TRADE_GBP", "25"))

# ── Schedule (US/Eastern time) ───────────────────────────────────────────────
# RTH opens 9:30 ET. Bar 4 closes at 9:50 ET. Schedule at 9:51.
MORNING_HOUR   = 9
MORNING_MINUTE = 51

# ── State ────────────────────────────────────────────────────────────────────
STATE_FILE = os.getenv("SPX_STATE_FILE", "data/spx_daily_state.json")
