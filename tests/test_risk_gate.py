"""Tests for asrs/risk_gate.py — position sizing + equity + gate checks."""
import pytest
from asrs.risk_gate import (
    RiskGateConfig, position_size_contracts, current_equity_gbp,
    today_pnl_gbp, week_pnl_gbp,
)


# ─────────────────────────────────────────────────────────────────────
# position_size_contracts — vol-targeted contract sizing
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def at_equity(monkeypatch):
    """Helper: set the risk_gate's equity to a specific amount for a test."""
    import asrs.risk_gate as rg
    from unittest.mock import patch

    def _set(equity: float, risk_pct: float = 0.5):
        # Patch the module-level functions that position_size_contracts calls
        monkeypatch.setattr(rg, "current_equity_gbp", lambda: equity)
        # Also update CFG's risk_pct to keep them in sync
        rg.CFG.risk_pct_per_trade = risk_pct
    return _set


def test_sizing_dax_at_5k_forces_minimum_1_lot(at_equity):
    """£5k × 0.5% = £25 budget. DAX 50pt stop × £0.86/pt = £43/lot.
    25/43 = 0.58 → floor 0 → clamped to min 1 lot."""
    at_equity(5000)
    qty = position_size_contracts("DAX_S1", "DAX",
                                    stop_distance_pts=50, gbp_per_pt=0.86,
                                    max_contracts=10)
    assert qty >= 1


def test_sizing_scales_linearly_with_equity(at_equity):
    """£40k × 0.5% = £200 budget. DAX 50pt × £0.86 = £43/lot → 4 lots."""
    at_equity(40000)
    qty = position_size_contracts("DAX_S1", "DAX", 50, 0.86, 10)
    assert qty == 4


def test_sizing_capped_by_max_contracts(at_equity):
    """At high equity, sizing caps at max_contracts regardless."""
    at_equity(1_000_000)
    qty = position_size_contracts("US30_S1", "US30",
                                    stop_distance_pts=30, gbp_per_pt=0.40,
                                    max_contracts=10)
    assert qty == 10


def test_sizing_nkd_at_30k_gives_at_least_1(at_equity):
    """At £30k with 0.5% = £150, NKD 14pt × £4 = £56 → 2 lots."""
    at_equity(30000)
    qty = position_size_contracts("NIKKEI_S1", "NIKKEI", 14, 4.0, 10)
    assert qty >= 1


def test_sizing_zero_stop_returns_minimum():
    """Edge case: zero stop distance should default to minimum, not crash."""
    qty = position_size_contracts("X", "X", stop_distance_pts=0,
                                    gbp_per_pt=1.0, max_contracts=10)
    assert qty == 0.5  # IG minimum £0.50/pt


def test_sizing_negative_stop_returns_minimum():
    """Negative stop (bug upstream) should default to minimum, not crash."""
    qty = position_size_contracts("X", "X", stop_distance_pts=-10,
                                    gbp_per_pt=1.0, max_contracts=10)
    assert qty == 0.5  # IG minimum £0.50/pt


# ─────────────────────────────────────────────────────────────────────
# RiskGateConfig env loading
# ─────────────────────────────────────────────────────────────────────

def test_config_defaults_are_conservative(monkeypatch):
    """Unset all env vars — defaults should be the safe paper values."""
    for k in ["STARTING_EQUITY_GBP", "DAILY_LOSS_LIMIT_PCT",
              "WEEKLY_LOSS_LIMIT_PCT", "MAX_CONCURRENT_POSITIONS",
              "CONSECUTIVE_LOSS_KILL", "RISK_PCT_PER_TRADE"]:
        monkeypatch.delenv(k, raising=False)
    cfg = RiskGateConfig()
    assert cfg.starting_equity_gbp == 5000.0
    assert cfg.daily_loss_limit_pct == 3.0
    assert cfg.weekly_loss_limit_pct == 6.0
    assert cfg.max_concurrent_positions == 3
    assert cfg.consecutive_loss_kill == 6
    assert cfg.risk_pct_per_trade == 0.5


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("STARTING_EQUITY_GBP", "100000")
    monkeypatch.setenv("RISK_PCT_PER_TRADE", "0.25")
    cfg = RiskGateConfig()
    assert cfg.starting_equity_gbp == 100000.0
    assert cfg.risk_pct_per_trade == 0.25
