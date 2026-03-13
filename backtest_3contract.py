"""
backtest_3contract.py — 3-Contract Partial Exit Strategy Backtest
═══════════════════════════════════════════════════════════════════

Strategy: Trade 3 contracts with staggered exits
  Contract 1: Exit at +20 pts profit (quick scalp)
  Contract 2: Exit at +50 pts profit (solid win)
  Contract 3: Ride full EMA trail (catch runners)

Compare vs baseline: 1 contract with full EMA trail (current strategy)
Also compare vs: 3 contracts all on EMA trail (pure scale-up)

All use bar 5 signal with 10 EMA trailing stop.
"""

import os
import sys
import logging
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("3CONTRACT")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class ContractExit:
    contract_num: int = 0
    exit_type: str = ""       # "FIXED_TP", "EMA_TRAIL", "INITIAL_STOP", "EOD"
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pts: float = 0.0
    held_bars: int = 0


@dataclass
class DayTrade:
    date: str = ""
    direction: str = ""
    entry_num: int = 0
    entry_price: float = 0.0
    contracts: list = field(default_factory=list)
    total_pnl: float = 0.0


def _calc_ema_series(closes: list[float], period: int) -> list[float]:
    """Calculate EMA for entire series."""
    if len(closes) < period:
        return [None] * len(closes)
    result = [None] * (period - 1)
    sma = sum(closes[:period]) / period
    result.append(sma)
    mult = 2 / (period + 1)
    ema = sma
    for price in closes[period:]:
        ema = (price - ema) * mult + ema
        result.append(round(ema, 1))
    return result


def candle_number(timestamp: pd.Timestamp) -> int:
    open_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def run_3contract_backtest(
    df: pd.DataFrame,
    all_df: pd.DataFrame = None,
    signal_bar: int = 5,
    tp1_pts: float = 20.0,
    tp2_pts: float = 50.0,
    num_contracts: int = 3,
) -> tuple[list, list, list]:
    """
    Run backtest with 3 contracts and staggered exits.

    Returns (baseline_1x, scaled_3x, partial_3x) — each a list of DayTrade objects.
    baseline_1x: 1 contract, full EMA trail (current strategy)
    scaled_3x: 3 contracts, all EMA trail (pure scale up)
    partial_3x: 3 contracts with partial exits at TP1, TP2, trail
    """
    from dax_bot import config
    from dax_bot.overnight import calculate_overnight_range

    be_trigger = config.TRAIL_BREAKEVEN_TRIGGER
    ema_trigger = config.TRAIL_EMA_TRIGGER
    ema_buffer = config.TRAIL_EMA_BUFFER
    ema_period = config.TRAIL_EMA_PERIOD
    max_entries = config.MAX_ENTRIES

    baseline_results = []
    scaled_results = []
    partial_results = []

    prev_close = 0

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx)
            if 1 <= cn <= 6:
                bars[cn] = {
                    "high": row["High"], "low": row["Low"],
                    "open": row["Open"], "close": row["Close"],
                }

        if signal_bar not in bars:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        sig = bars[signal_bar]
        buy_level = round(sig["high"] + config.BUFFER_PTS, 1)
        sell_level = round(sig["low"] - config.BUFFER_PTS, 1)

        # Build post-signal candles with EMA
        post_bars = []
        all_closes = []
        for idx, row in day_df.iterrows():
            all_closes.append(row["Close"])
            if candle_number(idx) > signal_bar:
                post_bars.append((idx, row, len(all_closes) - 1))

        ema_series = _calc_ema_series(all_closes, ema_period)

        # ── Simulate for all 3 modes in one pass ──
        for mode in ["baseline", "scaled", "partial"]:
            n_contracts = 1 if mode == "baseline" else num_contracts
            entries_used = 0
            direction = None
            entry_price = 0
            initial_stop = 0
            entry_bar_idx = 0
            max_fav = 0
            phase = "UNDERWATER"

            # Per-contract state for partial mode
            contract_active = [False] * n_contracts
            contract_tp_hit = [False] * n_contracts
            contract_exits = []
            trail_stop = 0

            day_trades = []

            for i, (idx, row, close_idx) in enumerate(post_bars):
                ema_val = ema_series[close_idx] if close_idx < len(ema_series) else None

                # ── Entry ──
                if direction is None and entries_used < max_entries:
                    if row["High"] >= buy_level:
                        direction = "LONG"
                        entry_price = buy_level
                        initial_stop = sell_level
                        trail_stop = sell_level
                        entries_used += 1
                        entry_bar_idx = i
                        max_fav = entry_price
                        phase = "UNDERWATER"
                        contract_active = [True] * n_contracts
                        contract_tp_hit = [False] * n_contracts
                        contract_exits = []
                    elif row["Low"] <= sell_level:
                        direction = "SHORT"
                        entry_price = sell_level
                        initial_stop = buy_level
                        trail_stop = buy_level
                        entries_used += 1
                        entry_bar_idx = i
                        max_fav = entry_price
                        phase = "UNDERWATER"
                        contract_active = [True] * n_contracts
                        contract_tp_hit = [False] * n_contracts
                        contract_exits = []

                if direction is None:
                    continue

                # ── Update MFE and phase ──
                if direction == "LONG":
                    max_fav = max(max_fav, row["High"])
                    favour = max_fav - entry_price
                else:
                    max_fav = min(max_fav, row["Low"])
                    favour = entry_price - max_fav

                # Phase uses current bar's favour (not running MFE)
                if direction == "LONG":
                    bar_favour = row["High"] - entry_price
                    above_ema = ema_val is not None and row["Close"] > ema_val
                else:
                    bar_favour = entry_price - row["Low"]
                    above_ema = ema_val is not None and row["Close"] < ema_val

                old_phase = phase
                if ema_val is not None and bar_favour >= ema_trigger and above_ema:
                    phase = "EMA_TRAIL"
                elif bar_favour >= be_trigger:
                    if phase == "UNDERWATER":
                        phase = "BREAKEVEN"
                # Never downgrade from EMA_TRAIL

                # ── Update trail stop (ratchet — never moves backward) ──
                if phase == "BREAKEVEN" and old_phase == "UNDERWATER":
                    if direction == "LONG":
                        trail_stop = max(trail_stop, entry_price)
                    else:
                        trail_stop = min(trail_stop, entry_price)
                elif phase == "EMA_TRAIL" and ema_val is not None:
                    if direction == "LONG":
                        raw = round(ema_val * (1 - ema_buffer), 1)
                        raw = max(raw, entry_price)
                        trail_stop = max(trail_stop, raw)
                    else:
                        raw = round(ema_val * (1 + ema_buffer), 1)
                        raw = min(raw, entry_price)
                        trail_stop = min(trail_stop, raw)

                # ── Partial exits (only for partial mode) ──
                if mode == "partial" and any(contract_active):
                    # Contract 1: TP at +tp1_pts
                    if contract_active[0] and not contract_tp_hit[0]:
                        if direction == "LONG" and row["High"] >= entry_price + tp1_pts:
                            contract_active[0] = False
                            contract_tp_hit[0] = True
                            exit_p = round(entry_price + tp1_pts, 1)
                            contract_exits.append(ContractExit(
                                contract_num=1, exit_type="FIXED_TP",
                                entry_price=entry_price, exit_price=exit_p,
                                pnl_pts=tp1_pts, held_bars=i - entry_bar_idx,
                            ))
                        elif direction == "SHORT" and row["Low"] <= entry_price - tp1_pts:
                            contract_active[0] = False
                            contract_tp_hit[0] = True
                            exit_p = round(entry_price - tp1_pts, 1)
                            contract_exits.append(ContractExit(
                                contract_num=1, exit_type="FIXED_TP",
                                entry_price=entry_price, exit_price=exit_p,
                                pnl_pts=tp1_pts, held_bars=i - entry_bar_idx,
                            ))

                    # Contract 2: TP at +tp2_pts
                    if n_contracts >= 2 and contract_active[1] and not contract_tp_hit[1]:
                        if direction == "LONG" and row["High"] >= entry_price + tp2_pts:
                            contract_active[1] = False
                            contract_tp_hit[1] = True
                            exit_p = round(entry_price + tp2_pts, 1)
                            contract_exits.append(ContractExit(
                                contract_num=2, exit_type="FIXED_TP",
                                entry_price=entry_price, exit_price=exit_p,
                                pnl_pts=tp2_pts, held_bars=i - entry_bar_idx,
                            ))
                        elif direction == "SHORT" and row["Low"] <= entry_price - tp2_pts:
                            contract_active[1] = False
                            contract_tp_hit[1] = True
                            exit_p = round(entry_price - tp2_pts, 1)
                            contract_exits.append(ContractExit(
                                contract_num=2, exit_type="FIXED_TP",
                                entry_price=entry_price, exit_price=exit_p,
                                pnl_pts=tp2_pts, held_bars=i - entry_bar_idx,
                            ))

                # ── Trail stop hit — EMA phase uses Close, others use Low/High ──
                stopped = False
                if phase == "EMA_TRAIL":
                    if direction == "LONG" and row["Close"] < trail_stop:
                        stopped = True
                    elif direction == "SHORT" and row["Close"] > trail_stop:
                        stopped = True
                else:
                    if direction == "LONG" and row["Low"] <= trail_stop:
                        stopped = True
                    elif direction == "SHORT" and row["High"] >= trail_stop:
                        stopped = True

                if stopped:
                    exit_price = trail_stop
                    if direction == "LONG":
                        base_pnl = round(exit_price - entry_price, 1)
                    else:
                        base_pnl = round(entry_price - exit_price, 1)

                    if mode == "baseline":
                        day_trades.append(DayTrade(
                            date=str(trade_date), direction=direction,
                            entry_num=entries_used, entry_price=entry_price,
                            total_pnl=base_pnl,
                        ))
                    elif mode == "scaled":
                        day_trades.append(DayTrade(
                            date=str(trade_date), direction=direction,
                            entry_num=entries_used, entry_price=entry_price,
                            total_pnl=base_pnl * n_contracts,
                        ))
                    elif mode == "partial":
                        # Exit remaining active contracts at trail stop
                        for c in range(n_contracts):
                            if contract_active[c]:
                                contract_exits.append(ContractExit(
                                    contract_num=c + 1, exit_type="EMA_TRAIL",
                                    entry_price=entry_price, exit_price=exit_price,
                                    pnl_pts=base_pnl, held_bars=i - entry_bar_idx,
                                ))
                                contract_active[c] = False

                        total = sum(ce.pnl_pts for ce in contract_exits)
                        day_trades.append(DayTrade(
                            date=str(trade_date), direction=direction,
                            entry_num=entries_used, entry_price=entry_price,
                            contracts=list(contract_exits),
                            total_pnl=total,
                        ))
                        contract_exits = []

                    direction = None

            # EOD: close any remaining positions
            if direction is not None:
                last_close = day_df.iloc[-1]["Close"]
                if direction == "LONG":
                    base_pnl = round(last_close - entry_price, 1)
                else:
                    base_pnl = round(entry_price - last_close, 1)

                if mode == "baseline":
                    day_trades.append(DayTrade(
                        date=str(trade_date), direction=direction,
                        entry_num=entries_used, entry_price=entry_price,
                        total_pnl=base_pnl,
                    ))
                elif mode == "scaled":
                    day_trades.append(DayTrade(
                        date=str(trade_date), direction=direction,
                        entry_num=entries_used, entry_price=entry_price,
                        total_pnl=base_pnl * n_contracts,
                    ))
                elif mode == "partial":
                    for c in range(n_contracts):
                        if contract_active[c]:
                            contract_exits.append(ContractExit(
                                contract_num=c + 1, exit_type="EOD",
                                entry_price=entry_price, exit_price=last_close,
                                pnl_pts=base_pnl, held_bars=len(post_bars) - entry_bar_idx,
                            ))
                    total = sum(ce.pnl_pts for ce in contract_exits)
                    day_trades.append(DayTrade(
                        date=str(trade_date), direction=direction,
                        entry_num=entries_used, entry_price=entry_price,
                        contracts=list(contract_exits),
                        total_pnl=total,
                    ))

            if mode == "baseline":
                baseline_results.extend(day_trades)
            elif mode == "scaled":
                scaled_results.extend(day_trades)
            elif mode == "partial":
                partial_results.extend(day_trades)

        prev_close = day_df.iloc[-1]["Close"]

    return baseline_results, scaled_results, partial_results


def print_results(baseline, scaled, partial, tp1, tp2):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("  3-CONTRACT PARTIAL EXIT BACKTEST")
    print(f"  TP1: +{tp1} pts (contract 1) | TP2: +{tp2} pts (contract 2) | Trail: 10 EMA (contract 3)")
    print("=" * 90)

    b_pnl = sum(t.total_pnl for t in baseline)
    s_pnl = sum(t.total_pnl for t in scaled)
    p_pnl = sum(t.total_pnl for t in partial)

    b_trades = len(baseline)
    s_trades = len(scaled)
    p_trades = len(partial)

    b_wins = sum(1 for t in baseline if t.total_pnl > 0)
    s_wins = sum(1 for t in scaled if t.total_pnl > 0)
    p_wins = sum(1 for t in partial if t.total_pnl > 0)

    # Drawdowns
    def calc_dd(trades):
        equity = []
        running = 0
        for t in trades:
            running += t.total_pnl
            equity.append(running)
        peak = 0
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = peak - e
            max_dd = max(max_dd, dd)
        return max_dd

    b_dd = calc_dd(baseline)
    s_dd = calc_dd(scaled)
    p_dd = calc_dd(partial)

    # Profit factor
    def calc_pf(trades):
        gross_win = sum(t.total_pnl for t in trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in trades if t.total_pnl <= 0))
        return round(gross_win / gross_loss, 1) if gross_loss > 0 else 999

    b_pf = calc_pf(baseline)
    s_pf = calc_pf(scaled)
    p_pf = calc_pf(partial)

    print(f"\n  {'Metric':<30} {'1x Baseline':>14} {'3x All Trail':>14} {'3x Partial':>14}")
    print("  " + "-" * 75)
    print(f"  {'Total P&L (pts)':<30} {b_pnl:>+13.1f} {s_pnl:>+13.1f} {p_pnl:>+13.1f}")
    print(f"  {'EUR at FDXS (x1/pt)':<30} {'':>5}EUR{b_pnl:>+8.0f} {'':>5}EUR{s_pnl:>+8.0f} {'':>5}EUR{p_pnl:>+8.0f}")
    print(f"  {'EUR at FDXM (x5/pt)':<30} {'':>4}EUR{b_pnl*5:>+9.0f} {'':>4}EUR{s_pnl*5:>+9.0f} {'':>4}EUR{p_pnl*5:>+9.0f}")
    print(f"  {'Trades':<30} {b_trades:>14} {s_trades:>14} {p_trades:>14}")
    print(f"  {'Win rate':<30} {b_wins/b_trades*100 if b_trades else 0:>13.0f}% {s_wins/s_trades*100 if s_trades else 0:>13.0f}% {p_wins/p_trades*100 if p_trades else 0:>13.0f}%")
    print(f"  {'Avg P&L per trade':<30} {b_pnl/b_trades if b_trades else 0:>+13.1f} {s_pnl/s_trades if s_trades else 0:>+13.1f} {p_pnl/p_trades if p_trades else 0:>+13.1f}")
    print(f"  {'Profit factor':<30} {b_pf:>14} {s_pf:>14} {p_pf:>14}")
    print(f"  {'Max drawdown (pts)':<30} {b_dd:>13.1f} {s_dd:>13.1f} {p_dd:>13.1f}")
    print(f"  {'P&L / Max DD ratio':<30} {b_pnl/b_dd if b_dd else 0:>13.1f} {s_pnl/s_dd if s_dd else 0:>13.1f} {p_pnl/p_dd if p_dd else 0:>13.1f}")

    # Partial exit breakdown
    print(f"\n  -- 3x Partial: Exit Breakdown --")
    all_contracts = [ce for t in partial for ce in t.contracts]
    if all_contracts:
        by_type = {}
        for ce in all_contracts:
            key = f"C{ce.contract_num} ({ce.exit_type})"
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(ce.pnl_pts)

        print(f"  {'Exit Type':<30} {'Count':>6} {'Total P&L':>10} {'Avg P&L':>10} {'Win%':>6}")
        print("  " + "-" * 65)
        for key in sorted(by_type.keys()):
            pnls = by_type[key]
            n = len(pnls)
            total = sum(pnls)
            avg = np.mean(pnls)
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n * 100 if n > 0 else 0
            print(f"  {key:<30} {n:>6} {total:>+9.1f} {avg:>+9.1f} {wr:>5.0f}%")

        # How often do TPs get hit?
        tp1_trades = [ce for ce in all_contracts if ce.contract_num == 1]
        tp1_fixed = [ce for ce in tp1_trades if ce.exit_type == "FIXED_TP"]
        tp2_trades = [ce for ce in all_contracts if ce.contract_num == 2]
        tp2_fixed = [ce for ce in tp2_trades if ce.exit_type == "FIXED_TP"]

        print(f"\n  TP hit rates:")
        print(f"    TP1 (+{tp1} pts): {len(tp1_fixed)}/{len(tp1_trades)} = {len(tp1_fixed)/len(tp1_trades)*100:.0f}%" if tp1_trades else "    TP1: N/A")
        print(f"    TP2 (+{tp2} pts): {len(tp2_fixed)}/{len(tp2_trades)} = {len(tp2_fixed)/len(tp2_trades)*100:.0f}%" if tp2_trades else "    TP2: N/A")

    # Verdict
    print(f"\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)

    if p_pnl > s_pnl:
        print(f"  3x PARTIAL beats 3x ALL TRAIL by {p_pnl - s_pnl:+.1f} pts")
        print(f"  Partial exits ADD value — early TPs bank profit on losers-that-could-have-been-winners")
    else:
        print(f"  3x ALL TRAIL beats 3x PARTIAL by {s_pnl - p_pnl:+.1f} pts")
        print(f"  Letting all 3 ride the EMA trail is better — the runners make up for the stops")

    # Risk-adjusted comparison
    if p_dd > 0 and s_dd > 0:
        p_ratio = p_pnl / p_dd
        s_ratio = s_pnl / s_dd
        if p_ratio > s_ratio:
            print(f"  But PARTIAL has better risk-adjusted return: {p_ratio:.1f}x vs {s_ratio:.1f}x (P&L/MaxDD)")
        else:
            print(f"  ALL TRAIL also better risk-adjusted: {s_ratio:.1f}x vs {p_ratio:.1f}x (P&L/MaxDD)")

    print(f"\n  Scale comparison: 1x baseline = {b_pnl:+.0f} pts, best 3x = {max(s_pnl, p_pnl):+.0f} pts")
    print(f"  That's a {max(s_pnl, p_pnl) / b_pnl:.1f}x improvement from 3 contracts" if b_pnl > 0 else "")
    print()


def main():
    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")

    if not os.path.exists(rth_path):
        logger.error("No cached DAX data. Run backtest.py --fetch first.")
        return

    df = pd.read_parquet(rth_path)
    all_df = pd.read_parquet(all_path) if os.path.exists(all_path) else None

    dates = pd.Series(df.index.date).unique()
    logger.info(f"Data: {len(df)} bars, {len(dates)} trading days ({dates[0]} to {dates[-1]})")

    # Run with default TP levels
    tp1, tp2 = 20.0, 50.0
    logger.info(f"Running 3-contract backtest: TP1=+{tp1}, TP2=+{tp2}, C3=EMA trail...")
    baseline, scaled, partial = run_3contract_backtest(df, all_df, signal_bar=5, tp1_pts=tp1, tp2_pts=tp2)
    print_results(baseline, scaled, partial, tp1, tp2)

    # Also test alternative TP levels
    for tp1_alt, tp2_alt in [(15, 40), (25, 60), (30, 80)]:
        logger.info(f"Testing TP1=+{tp1_alt}, TP2=+{tp2_alt}...")
        _, _, partial_alt = run_3contract_backtest(df, all_df, signal_bar=5, tp1_pts=tp1_alt, tp2_pts=tp2_alt)
        p_pnl = sum(t.total_pnl for t in partial_alt)
        p_trades = len(partial_alt)
        p_wins = sum(1 for t in partial_alt if t.total_pnl > 0)

        all_c = [ce for t in partial_alt for ce in t.contracts]
        tp1_trades = [ce for ce in all_c if ce.contract_num == 1]
        tp1_hit = [ce for ce in tp1_trades if ce.exit_type == "FIXED_TP"]
        tp2_trades = [ce for ce in all_c if ce.contract_num == 2]
        tp2_hit = [ce for ce in tp2_trades if ce.exit_type == "FIXED_TP"]

        print(f"  TP1=+{tp1_alt:>2}, TP2=+{tp2_alt:>2}: P&L={p_pnl:>+8.1f} | "
              f"WR={p_wins/p_trades*100:.0f}% | "
              f"TP1 hit={len(tp1_hit)/len(tp1_trades)*100:.0f}% | "
              f"TP2 hit={len(tp2_hit)/len(tp2_trades)*100:.0f}%"
              if tp1_trades and tp2_trades else f"  TP1=+{tp1_alt}, TP2=+{tp2_alt}: No data")


if __name__ == "__main__":
    main()
