"""
contract_resolver.py — Map ASRS instrument names to IBKR Future contract specs.

Handles:
  - Per-instrument symbol / exchange / multiplier metadata
  - Quarterly front-month resolution (3rd Friday of Mar/Jun/Sep/Dec)
  - Days-to-expiry calculation
  - Roll-warning threshold

The actual qualification (resolving to a real contract) happens via
IBSharedSession.qualify() — this module just builds the spec.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ib_async import Future

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractSpec:
    instrument: str          # ASRS name: "DAX" / "US30" / "NIKKEI"
    symbol: str              # IBKR symbol: "DAX" / "MYM" / "NIY"
    trading_class: str       # IBKR tradingClass (often = symbol but not always)
    exchange: str            # "EUREX" / "CBOT" / "OSE.JPN"
    currency: str            # "EUR" / "USD" / "JPY"
    multiplier: int          # contract multiplier in points (€5/pt, $0.50/pt → 0.5, ¥500/pt)
    multiplier_str: str      # how IBKR labels it (e.g. "5", "0.5")
    min_tick: float          # minimum price increment
    description: str         # human-readable label


# Default sizing strategy: smallest practical contract per instrument
SPECS: dict[str, ContractSpec] = {
    "DAX": ContractSpec(
        instrument="DAX",
        symbol="DAX",
        trading_class="FDXM",         # Mini DAX (€5/pt)
        exchange="EUREX",
        currency="EUR",
        multiplier=5,
        multiplier_str="5",
        min_tick=1.0,
        description="Mini DAX Future (€5/pt)",
    ),
    "US30": ContractSpec(
        instrument="US30",
        symbol="MYM",                  # Micro Dow ($0.50/pt)
        trading_class="MYM",
        exchange="CBOT",
        currency="USD",
        multiplier=0.5,                # 0.5 USD per index point
        multiplier_str="0.5",
        min_tick=1.0,
        description="Micro Dow Future ($0.50/pt)",
    ),
    "NIKKEI": ContractSpec(
        instrument="NIKKEI",
        symbol="NIY",                  # Nikkei 225 Yen (¥500/pt)
        trading_class="NIY",
        exchange="OSE.JPN",
        currency="JPY",
        multiplier=500,                # 500 JPY per index point
        multiplier_str="500",
        min_tick=5.0,
        description="Nikkei 225 Yen Future (¥500/pt)",
    ),
}


# ── Quarterly expiry helpers ─────────────────────────────────────────

QUARTERLY_MONTHS = (3, 6, 9, 12)


def _third_friday(year: int, month: int) -> date:
    """3rd Friday of the given month."""
    first = date(year, month, 1)
    # weekday(): Mon=0..Sun=6 ; Friday=4
    days_to_first_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_first_friday + 14)


def front_month_expiry(today: date | None = None, lookahead_days: int = 0) -> str:
    """
    Return the YYYYMMDD string for the next quarterly expiry from `today`.

    If today is within `lookahead_days` of the current quarterly expiry,
    return the NEXT quarter (so we don't qualify a contract about to expire).
    """
    today = today or date.today()
    cutoff = today + timedelta(days=lookahead_days)

    for offset in range(0, 13):
        m = today.month + offset
        y = today.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        if m not in QUARTERLY_MONTHS:
            continue
        expiry = _third_friday(y, m)
        if expiry > cutoff:
            return expiry.strftime("%Y%m%d")
    raise RuntimeError("front_month_expiry: no expiry found in 12 months")


def days_to_expiry(expiry_str: str, today: date | None = None) -> int:
    """Days from `today` until the expiry date (YYYYMMDD)."""
    today = today or date.today()
    expiry = datetime.strptime(expiry_str, "%Y%m%d").date()
    return (expiry - today).days


def should_roll(expiry_str: str, today: date | None = None, threshold_days: int = 7) -> bool:
    """True if expiry is within `threshold_days` of today."""
    return days_to_expiry(expiry_str, today) <= threshold_days


# ── Contract construction ────────────────────────────────────────────

def make_future(instrument: str, expiry: str | None = None) -> Future:
    """
    Build an unqualified ib_async.Future for the given ASRS instrument.

    If expiry is None, the front-month is selected (with a 3-day buffer
    so we never grab a contract about to expire).
    """
    spec = SPECS.get(instrument)
    if spec is None:
        raise ValueError(f"Unknown instrument: {instrument}")

    if expiry is None:
        expiry = front_month_expiry(lookahead_days=3)

    return Future(
        symbol=spec.symbol,
        lastTradeDateOrContractMonth=expiry,
        exchange=spec.exchange,
        currency=spec.currency,
        tradingClass=spec.trading_class,
        multiplier=spec.multiplier_str,
    )


def get_spec(instrument: str) -> ContractSpec:
    spec = SPECS.get(instrument)
    if spec is None:
        raise ValueError(f"Unknown instrument: {instrument}")
    return spec


def all_instruments() -> list[str]:
    return list(SPECS.keys())
