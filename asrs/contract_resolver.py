"""
contract_resolver.py — Map ASRS instrument names to IBKR Future specs.

Strategy: store the "search template" for each instrument (symbol +
exchange + currency + tradingClass), then query IBKR's reqContractDetails
at resolve time to pick the actual front-month contract. This works
across all the quirks of different exchanges (CBOT 3rd Thursday,
EUREX 3rd Friday, CME 2nd Thursday for NKD, etc.) without hardcoding
calendar rules.

Note: requires an active IBSharedSession to resolve (it's a network
call). Specs alone are static.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ib_async import Future, Contract

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractSpec:
    instrument: str
    symbol: str               # IBKR symbol
    exchange: str
    currency: str
    trading_class: str        # often empty; FDXM uses it
    description: str
    expected_multiplier: float  # for sanity checking after resolve


# Search templates — DO NOT include expiry or multiplier
# (we let IBKR fill those in via reqContractDetails)
SPECS: dict[str, ContractSpec] = {
    "DAX": ContractSpec(
        instrument="DAX",
        symbol="DAX",
        exchange="EUREX",
        currency="EUR",
        trading_class="FDXS",          # Sub-mini DAX €1/pt (~£0.85/pt)
        description="Sub-Mini DAX Future (€1/pt, EUREX)",
        expected_multiplier=1.0,
    ),
    "US30": ContractSpec(
        instrument="US30",
        symbol="MYM",                  # Micro Dow $0.50/pt (~£0.40/pt)
        exchange="CBOT",
        currency="USD",
        trading_class="",
        description="Micro Dow Future ($0.50/pt, CBOT)",
        expected_multiplier=0.5,
    ),
    "NIKKEI": ContractSpec(
        instrument="NIKKEI",
        symbol="NIY",                  # Nikkei Yen ¥500/pt (~£2.65/pt)
        exchange="CME",
        currency="JPY",
        trading_class="",
        description="Nikkei 225 Yen Future (¥500/pt, CME)",
        expected_multiplier=500.0,
    ),
}


def _make_search_contract(spec: ContractSpec) -> Future:
    """Build a Future spec for reqContractDetails (no expiry, no multiplier)."""
    kwargs = dict(
        symbol=spec.symbol,
        exchange=spec.exchange,
        currency=spec.currency,
    )
    if spec.trading_class:
        kwargs["tradingClass"] = spec.trading_class
    return Future(**kwargs)


QUARTERLY_MONTHS = {3, 6, 9, 12}


async def resolve_front_month(
    instrument: str, session, lookahead_days: int = 3,
    quarterly_only: bool = True,
) -> tuple[Contract | None, str | None]:
    """
    Query IBKR for the actual contract calendar of `instrument` and pick
    the nearest expiry that is > today + lookahead_days. By default only
    quarterly (Mar/Jun/Sep/Dec) expiries are considered — skips monthly
    serials that may exist (e.g. NIY May/Jul/Aug) which are usually low
    liquidity.

    Returns (contract, expiry_str) or (None, None) on failure.
    """
    spec = get_spec(instrument)
    if not await session.ensure_connected():
        return None, None

    search = _make_search_contract(spec)
    try:
        details = await session.ib.reqContractDetailsAsync(search)
    except Exception as e:
        logger.error(f"reqContractDetails failed for {instrument}: {e}")
        return None, None

    if not details:
        logger.error(f"No contracts returned for {instrument}")
        return None, None

    cutoff = date.today() + timedelta(days=lookahead_days)
    candidates: list[tuple[date, Contract]] = []
    for d in details:
        c = d.contract
        try:
            exp_date = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d").date()
        except Exception:
            continue
        if exp_date <= cutoff:
            continue
        if quarterly_only and exp_date.month not in QUARTERLY_MONTHS:
            continue
        candidates.append((exp_date, c))

    if not candidates:
        # Fallback: relax quarterly filter if nothing found
        if quarterly_only:
            logger.warning(
                f"{instrument}: no quarterly expiry found, falling back to any month"
            )
            return await resolve_front_month(
                instrument, session, lookahead_days, quarterly_only=False
            )
        logger.error(f"No future expiries found for {instrument}")
        return None, None

    # Sort by date ascending → first one is front-month
    candidates.sort(key=lambda t: t[0])
    exp_date, contract = candidates[0]

    # Sanity check multiplier
    try:
        mult = float(contract.multiplier)
        if abs(mult - spec.expected_multiplier) > 0.01:
            logger.warning(
                f"{instrument}: multiplier mismatch — got {mult}, expected {spec.expected_multiplier}"
            )
    except (ValueError, TypeError):
        pass

    expiry_str = contract.lastTradeDateOrContractMonth
    days_left = (exp_date - date.today()).days
    logger.info(
        f"Resolved {instrument} front-month: {contract.localSymbol or contract.symbol} "
        f"expiry={expiry_str} ({days_left}d) mult={contract.multiplier}"
    )
    return contract, expiry_str


def days_to_expiry(expiry_str: str, today: date | None = None) -> int:
    today = today or date.today()
    expiry = datetime.strptime(expiry_str, "%Y%m%d").date()
    return (expiry - today).days


def should_roll(expiry_str: str, today: date | None = None, threshold_days: int = 7) -> bool:
    return days_to_expiry(expiry_str, today) <= threshold_days


def get_spec(instrument: str) -> ContractSpec:
    spec = SPECS.get(instrument)
    if spec is None:
        raise ValueError(f"Unknown instrument: {instrument}")
    return spec


def all_instruments() -> list[str]:
    return list(SPECS.keys())


# ── Compatibility shim for code that imports the old static API ─────
# (kept so existing tests / scripts don't break)

def make_future(instrument: str, expiry: str | None = None) -> Future:
    """
    Build an unqualified Future spec for an instrument.
    If expiry is given, use it directly; otherwise return a search-style
    spec that resolve_front_month() can use to look up the actual contract.

    Note: when expiry is None this returns a contract WITHOUT a specific
    expiry, which is only useful for reqContractDetails search — not for
    qualifyContractsAsync. Use resolve_front_month() for actual qualification.
    """
    spec = get_spec(instrument)
    kwargs = dict(symbol=spec.symbol, exchange=spec.exchange, currency=spec.currency)
    if spec.trading_class:
        kwargs["tradingClass"] = spec.trading_class
    if expiry:
        kwargs["lastTradeDateOrContractMonth"] = expiry
    return Future(**kwargs)
