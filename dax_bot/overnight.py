"""
overnight.py — Overnight Range Analysis (Tom Hougaard V58 Theory)
═══════════════════════════════════════════════════════════════════════════════

Core concept from Tom's "V58 out of hours SRS Review":

    The overnight range (12am–6am CET) tells you WHERE the market has
    been before European traders arrive. If the SRS bar forms ABOVE or
    BELOW that range, it means the market has already extended in one
    direction — and the "new theory" says to FADE that move.

Three scenarios:

    SRS ABOVE overnight range → SHORT bias (fade the overnight up move)
      - Normal: buy above SRS → New theory: SELL above SRS
      - Buy below SRS still valid (fading back into range)

    SRS BELOW overnight range → LONG bias (fade the overnight down move)
      - Normal: sell below SRS → New theory: BUY below SRS
      - Sell above SRS may still apply

    SRS INSIDE overnight range → STANDARD (no bias, OCA bracket both sides)

Implementation:
    - Fetch bars from 00:00–06:00 CET (requires useRTH=False on IBKR)
    - Calculate high/low of that range
    - Compare bar 4 position to the range
    - Return bias: SHORT_ONLY, LONG_ONLY, or STANDARD
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


# ── Overnight Bias ─────────────────────────────────────────────────────────────

class OvernightBias(str, Enum):
    SHORT_ONLY = "SHORT_ONLY"    # SRS above overnight range → fade the up move
    LONG_ONLY  = "LONG_ONLY"     # SRS below overnight range → fade the down move
    STANDARD   = "STANDARD"      # SRS inside overnight range → normal OCA bracket
    NO_DATA    = "NO_DATA"       # Couldn't determine (no overnight bars)


@dataclass
class OvernightResult:
    """Result of overnight range analysis."""
    bias:           OvernightBias = OvernightBias.NO_DATA
    range_high:     float = 0.0
    range_low:      float = 0.0
    range_size:     float = 0.0
    bar4_high:      float = 0.0
    bar4_low:       float = 0.0
    bar4_vs_range:  str = ""         # "ABOVE", "BELOW", "INSIDE", "PARTIAL_ABOVE", "PARTIAL_BELOW"
    overlap_pct:    float = 0.0      # How much of bar4 overlaps with overnight range

    def summary(self) -> str:
        """Human-readable summary for Telegram."""
        if self.bias == OvernightBias.NO_DATA:
            return "Overnight: No data"
        return (
            f"Overnight range: {self.range_low:.0f}–{self.range_high:.0f} "
            f"({self.range_size:.0f} pts)\n"
            f"Bar 4 vs range: {self.bar4_vs_range}\n"
            f"Bias: {self.bias.value}"
        )

    def emoji(self) -> str:
        if self.bias == OvernightBias.SHORT_ONLY:
            return "🔴"
        elif self.bias == OvernightBias.LONG_ONLY:
            return "🟢"
        elif self.bias == OvernightBias.STANDARD:
            return "⚪"
        return "❓"


# ── Core Calculation ───────────────────────────────────────────────────────────

def calculate_overnight_range(
    overnight_bars: pd.DataFrame,
    bar4_high: float,
    bar4_low: float,
    tolerance_pct: float = 0.25,
) -> OvernightResult:
    """
    Calculate overnight range and determine SRS bias.

    Args:
        overnight_bars: DataFrame of bars from 00:00–06:00 CET
        bar4_high: High of the 4th 5-min candle
        bar4_low: Low of the 4th 5-min candle
        tolerance_pct: What % of bar4 must be outside range to count as
                       "above" or "below" (0.25 = 75% outside)

    Returns:
        OvernightResult with bias and range data
    """
    result = OvernightResult(bar4_high=bar4_high, bar4_low=bar4_low)

    if overnight_bars.empty:
        logger.warning("No overnight bars available")
        return result

    # Calculate overnight high/low
    result.range_high = round(overnight_bars["High"].max(), 1)
    result.range_low = round(overnight_bars["Low"].min(), 1)
    result.range_size = round(result.range_high - result.range_low, 1)

    if result.range_size <= 0:
        logger.warning("Overnight range is zero")
        result.bias = OvernightBias.NO_DATA
        return result

    # Determine bar4 position relative to overnight range
    bar4_range = bar4_high - bar4_low
    if bar4_range <= 0:
        result.bias = OvernightBias.STANDARD
        return result

    # Calculate overlap between bar4 and overnight range
    overlap_high = min(bar4_high, result.range_high)
    overlap_low = max(bar4_low, result.range_low)
    overlap = max(0, overlap_high - overlap_low)
    result.overlap_pct = round(overlap / bar4_range * 100, 1) if bar4_range > 0 else 0

    # Classify position
    if bar4_low >= result.range_high:
        # Bar4 entirely above overnight range
        result.bar4_vs_range = "ABOVE"
        result.bias = OvernightBias.SHORT_ONLY

    elif bar4_high <= result.range_low:
        # Bar4 entirely below overnight range
        result.bar4_vs_range = "BELOW"
        result.bias = OvernightBias.LONG_ONLY

    elif bar4_low > result.range_low and bar4_high > result.range_high:
        # Bar4 mostly above (partial overlap at bottom)
        above_pct = (bar4_high - result.range_high) / bar4_range * 100
        if above_pct > (1 - tolerance_pct) * 100:
            result.bar4_vs_range = "PARTIAL_ABOVE"
            result.bias = OvernightBias.SHORT_ONLY
        else:
            result.bar4_vs_range = "PARTIAL_ABOVE"
            result.bias = OvernightBias.STANDARD  # Too much overlap

    elif bar4_high < result.range_high and bar4_low < result.range_low:
        # Bar4 mostly below (partial overlap at top)
        below_pct = (result.range_low - bar4_low) / bar4_range * 100
        if below_pct > (1 - tolerance_pct) * 100:
            result.bar4_vs_range = "PARTIAL_BELOW"
            result.bias = OvernightBias.LONG_ONLY
        else:
            result.bar4_vs_range = "PARTIAL_BELOW"
            result.bias = OvernightBias.STANDARD  # Too much overlap

    else:
        # Bar4 inside overnight range
        result.bar4_vs_range = "INSIDE"
        result.bias = OvernightBias.STANDARD

    logger.info(
        f"Overnight: {result.range_low}–{result.range_high} ({result.range_size}pts) | "
        f"Bar4: {result.bar4_vs_range} | Bias: {result.bias.value}"
    )

    return result


# ── Backtest helper ────────────────────────────────────────────────────────────

def calculate_overnight_range_from_df(
    full_day_df: pd.DataFrame,
    bar4_high: float,
    bar4_low: float,
) -> OvernightResult:
    """
    For backtesting: extract overnight bars from a full-day DataFrame
    that includes pre-market data (useRTH=False).

    Overnight = 00:00 to 06:00 CET (or local equivalent).
    """
    if full_day_df.empty:
        return OvernightResult()

    # Filter to 00:00–06:00 CET
    overnight = full_day_df.between_time("00:00", "06:00")

    if overnight.empty:
        return OvernightResult()

    return calculate_overnight_range(overnight, bar4_high, bar4_low)
