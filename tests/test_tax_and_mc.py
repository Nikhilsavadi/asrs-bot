"""Tests for monte_carlo_4yr.py — CGT math + contract sizing."""
import numpy as np
import pytest

from monte_carlo_4yr import (
    apply_tax_post, contracts_for, INSTRUMENTS, MAX_CONTRACTS,
    WIFE_GIFT_AMOUNT, TRADING_DAYS_PER_YEAR,
)


# ─────────────────────────────────────────────────────────────────────
# apply_tax_post — UK CGT at year-end boundaries
# ─────────────────────────────────────────────────────────────────────

def test_tax_zero_when_no_gain():
    """No gain → no tax (even above allowance)."""
    n = TRADING_DAYS_PER_YEAR * 2 + 1
    primary = np.full(n, 5000.0)  # flat, no gain
    wife = np.zeros(n)
    p_post, w_post, events = apply_tax_post(
        primary, wife, start_equity=5000,
        tax_rate=0.20, allowance=3000,
    )
    assert np.allclose(p_post, primary)
    assert len(events) == 0


def test_tax_respects_allowance():
    """£2.5k gain in year 1 is under £3k allowance → zero tax."""
    n = TRADING_DAYS_PER_YEAR + 1
    primary = np.linspace(5000, 7500, n)  # £2.5k gain
    wife = np.zeros(n)
    p_post, _, events = apply_tax_post(primary, wife, 5000, 0.20, 3000)
    # Total tax should be zero
    assert len(events) == 0
    assert p_post[-1] == pytest.approx(7500, abs=1)


def test_tax_20_percent_above_allowance():
    """£10k gain - £3k allowance = £7k taxable × 20% = £1.4k tax."""
    n = TRADING_DAYS_PER_YEAR + 1
    primary = np.linspace(5000, 15000, n)
    wife = np.zeros(n)
    p_post, _, events = apply_tax_post(primary, wife, 5000, 0.20, 3000)
    expected_tax = (10000 - 3000) * 0.20
    assert len(events) == 1
    assert events[0]["p_tax"] == pytest.approx(expected_tax, rel=0.01)
    # Post-tax equity should be 15000 - 1400 = 13600
    assert p_post[-1] == pytest.approx(13600, abs=1)


def test_tax_wife_gift_excluded_from_gain():
    """Spousal gift MUST NOT be taxed as gain on the wife account."""
    n = TRADING_DAYS_PER_YEAR + 1
    primary = np.linspace(5000, 5000, n)  # flat
    wife = np.zeros(n)
    wife[100:] = WIFE_GIFT_AMOUNT  # gift received mid-year, no subsequent gain
    p_post, w_post, events = apply_tax_post(primary, wife, 5000, 0.20, 3000)
    # Wife gain = (post_ye_wife - 0 anchor) - gift_amount = 0, NO tax
    assert len(events) == 0, f"wife gift incorrectly taxed: {events}"
    assert w_post[-1] == pytest.approx(WIFE_GIFT_AMOUNT, abs=1)


def test_tax_multiple_years_compound_correctly():
    """4 year-ends each with £10k gain → 4 tax events, each ~£1.4k."""
    years = 4
    n = TRADING_DAYS_PER_YEAR * years + 1
    # Ramp £10k per year
    primary = np.zeros(n)
    for i in range(n):
        primary[i] = 5000 + (i / TRADING_DAYS_PER_YEAR) * 10000
    wife = np.zeros(n)
    p_post, _, events = apply_tax_post(primary, wife, 5000, 0.20, 3000)
    assert len(events) == years
    for ev in events:
        assert ev["p_tax"] == pytest.approx((10000 - 3000) * 0.20, rel=0.05)


# ─────────────────────────────────────────────────────────────────────
# contracts_for — per-instrument sizing used by MC
# ─────────────────────────────────────────────────────────────────────

def test_contracts_for_nikkei_gated_below_30k():
    """NIKKEI must return 0 contracts if equity < £30k (margin gate)."""
    assert contracts_for("NIKKEI", 5000) == 0
    assert contracts_for("NIKKEI", 15000) == 0
    assert contracts_for("NIKKEI", 29999) == 0


def test_contracts_for_nikkei_enabled_at_30k():
    """At exactly £30k, NIKKEI should be sizable."""
    assert contracts_for("NIKKEI", 30000) >= 1


def test_contracts_for_dax_scales_with_equity():
    """DAX should scale linearly with equity until max_contracts cap."""
    c1 = contracts_for("DAX", 5000)
    c2 = contracts_for("DAX", 20000)
    assert c2 > c1 or c2 == MAX_CONTRACTS


def test_contracts_for_cap_at_max():
    """No instrument should exceed MAX_CONTRACTS regardless of equity."""
    for inst in INSTRUMENTS:
        c = contracts_for(inst, 10_000_000)
        assert c <= MAX_CONTRACTS
