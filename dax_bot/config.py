"""
config.py — Configuration & Constants
"""

import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# ── Timezones ──────────────────────────────────────────────────────────────────
TZ_CET    = ZoneInfo("Europe/Berlin")
TZ_UK     = ZoneInfo("Europe/London")

# ── Broker ───────────────────────────────────────────────────────────────────
BROKER       = "ig"

# ── IG Markets Connection ────────────────────────────────────────────────────
IG_USERNAME   = os.getenv("IG_USERNAME", "")
IG_PASSWORD   = os.getenv("IG_PASSWORD", "")
IG_API_KEY    = os.getenv("IG_API_KEY", "")
IG_ACC_NUMBER = os.getenv("IG_ACC_NUMBER", "")
IG_DEMO       = os.getenv("IG_DEMO", "true").lower() == "true"
IG_EPIC       = os.getenv("IG_EPIC", "IX.D.DAX.DAILY.IP")

# ── Instrument ─────────────────────────────────────────────────────────────────
INSTRUMENT   = "DAX DFB"
SEC_TYPE     = "SPREAD_BET"
EXCHANGE     = "IG"
CURRENCY     = os.getenv("CURRENCY", "EUR")

# ── Strategy ───────────────────────────────────────────────────────────────────
BUFFER_PTS    = int(os.getenv("BUFFER_PTS", "2"))
NARROW_RANGE  = int(os.getenv("NARROW_RANGE", "15"))
WIDE_RANGE    = int(os.getenv("WIDE_RANGE", "40"))
MAX_ENTRIES   = int(os.getenv("MAX_ENTRIES", "3"))   # entry + flip + re-entry
NUM_CONTRACTS = int(os.getenv("NUM_CONTRACTS", "1"))   # £1/pt to start — scale up later
MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "5"))   # Hard ceiling regardless of scaling ladder
NARROW_STD_MULTIPLIER = int(os.getenv("NARROW_STD_MULTIPLIER", "2"))  # 2x size on STANDARD+NARROW days
MAX_DAILY_LOSS_GBP = float(os.getenv("MAX_DAILY_LOSS_GBP", "200"))  # Stop trading if day P&L < -£200
MAX_SLIPPAGE_PTS   = int(os.getenv("MAX_SLIPPAGE_PTS", "10"))       # Close entry if fill slips >10pts
MAX_SPREAD_PTS     = float(os.getenv("MAX_SPREAD_PTS", "10.0"))   # Skip entry if spread > this

# ── Signal Bar Selection ─────────────────────────────────────────────────────
# Always use bar 4. Bar 5 disabled — backtest showed bar 5 rules reduced P&L
# by £6.5k over 624 days (overfit to earlier data).
DEFAULT_SIGNAL_BAR = int(os.getenv("DEFAULT_SIGNAL_BAR", "4"))
BAR5_RULES_STR = os.getenv("BAR5_RULES", "")  # Empty = always bar 4
BAR5_RULES = [r.strip() for r in BAR5_RULES_STR.split(",") if r.strip()]

# ── Partial Exit TPs (3-contract mode) ───────────────────────────────────────
# C1 exits at entry +/- TP1_PTS, C2 at TP2_PTS, C3 rides full EMA trail
PARTIAL_EXIT   = os.getenv("PARTIAL_EXIT", "false").lower() == "true"
TP1_PTS        = float(os.getenv("TP1_PTS", "20"))     # Contract 1 fixed TP
TP2_PTS        = float(os.getenv("TP2_PTS", "50"))     # Contract 2 fixed TP

# ── Trailing Stop ──────────────────────────────────────────────────────────────
TRAIL_MIN_MOVE = float(os.getenv("TRAIL_MIN_MOVE", "3"))  # Min pts to bother alerting
TRAIL_BREAKEVEN_PTS = float(os.getenv("TRAIL_BREAKEVEN_PTS", "15"))  # Move stop to entry after +Xpts
TRAIL_TIGHT_THRESHOLD = float(os.getenv("TRAIL_TIGHT_THRESHOLD", "100"))  # Switch to tight trail after +Xpts

# ── EMA Trail & Add to Winners ───────────────────────────────────────────────
TRAIL_EMA_PERIOD       = int(os.getenv("TRAIL_EMA_PERIOD", "10"))
TRAIL_BREAKEVEN_TRIGGER = float(os.getenv("TRAIL_BREAKEVEN_TRIGGER", "5"))
TRAIL_EMA_TRIGGER      = float(os.getenv("TRAIL_EMA_TRIGGER", "10"))
TRAIL_EMA_BUFFER       = float(os.getenv("TRAIL_EMA_BUFFER", "0.005"))
ADD_EMA_TOUCH_ZONE     = float(os.getenv("ADD_EMA_TOUCH_ZONE", "0.003"))

# ── Add-to-Winners (Strength Mode) ──────────────────────────────────────────
# When trade moves +TRIGGER pts from last entry, add 1 more contract (up to MAX adds)
ADD_STRENGTH_ENABLED   = os.getenv("ADD_STRENGTH_ENABLED", "true").lower() == "true"
ADD_STRENGTH_TRIGGER   = float(os.getenv("ADD_STRENGTH_TRIGGER", "25"))   # pts profit before add
ADD_STRENGTH_MAX       = int(os.getenv("ADD_STRENGTH_MAX", "2"))          # max extra positions

# ── Telegram ───────────────────────────────────────────────────────────────────
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_GBP   = float(os.getenv("RISK_PER_TRADE_GBP", "100"))

# ── Schedule (UK time) ────────────────────────────────────────────────────────
# Bar 4 closes at 08:20 UK (09:20 CET). We schedule at 08:21 to give
# the streaming candle 1 minute to arrive. Failsafe retry at 08:25.
MORNING_HOUR   = 8
MORNING_MINUTE = 21
MONITOR_START  = "08:21"
MONITOR_END    = "17:30"
SUMMARY_HOUR   = 17
SUMMARY_MINUTE = 35

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
