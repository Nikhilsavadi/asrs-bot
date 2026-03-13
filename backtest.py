"""
backtest.py — ASRS Backtester & Situational Analyser
═══════════════════════════════════════════════════════════════════════════════

Two modes:
  1. BACKTEST  — Run ASRS rules over 2 years of data, output P&L stats
  2. SITUATIONAL — Slice results by conditions (day, gap, range, etc.)
                   to find WHEN the strategy works best/worst

Usage:
    python backtest.py                   Full backtest + summary
    python backtest.py --situational     Situational breakdown
    python backtest.py --export          Export trades to CSV
    python backtest.py --month 2025-06   Filter to specific month

Data:
    IBKR provides max 1Y of 5-min bars per request.
    We stitch 2 requests to get ~2 years of history.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime, timedelta, date, time as dtime
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from dax_bot import config
from dax_bot.broker import Broker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BACKTEST")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "dax")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING — Stitch 2 years from IBKR
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_2y_data(broker: Broker) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch ~2 years of 5-min DAX bars from IBKR.
    Returns (rth_bars, all_bars) — RTH for strategy, all for overnight range.
    IBKR limit: 1Y per request, so we make 2 sequential requests.
    """
    if not await broker.connect():
        logger.error("Cannot connect to IBKR")
        return pd.DataFrame(), pd.DataFrame()

    rth_frames = []
    all_frames = []

    for use_rth, label, target in [(True, "RTH", rth_frames), (False, "All hours", all_frames)]:
        logger.info(f"Fetching Year 1 ({label})...")
        try:
            bars = await broker.ib.reqHistoricalDataAsync(
                broker.contract,
                endDateTime="",
                durationStr="1 Y",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=2,
            )
            if bars:
                from ib_async import util
                df1 = util.df(bars)
                df1["date"] = pd.to_datetime(df1["date"], utc=True)
                df1 = df1.set_index("date")
                df1.index = df1.index.tz_convert(config.TZ_CET)
                df1.columns = [c.capitalize() for c in df1.columns]
                target.append(df1)
                logger.info(f"  Got {len(df1)} bars: {df1.index[0]} → {df1.index[-1]}")

                # Year 2
                end_dt = df1.index[0].strftime("%Y%m%d-%H:%M:%S")
                logger.info(f"Fetching Year 2 ({label}, ending {end_dt})...")
                bars2 = await broker.ib.reqHistoricalDataAsync(
                    broker.contract,
                    endDateTime=end_dt,
                    durationStr="1 Y",
                    barSizeSetting="5 mins",
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=2,
                )
                if bars2:
                    df2 = util.df(bars2)
                    df2["date"] = pd.to_datetime(df2["date"], utc=True)
                    df2 = df2.set_index("date")
                    df2.index = df2.index.tz_convert(config.TZ_CET)
                    df2.columns = [c.capitalize() for c in df2.columns]
                    target.append(df2)
                    logger.info(f"  Got {len(df2)} bars")
        except Exception as e:
            logger.warning(f"  Fetch failed ({label}): {e}")

    await broker.disconnect()

    rth_df = pd.concat(rth_frames).sort_index() if rth_frames else pd.DataFrame()
    all_df = pd.concat(all_frames).sort_index() if all_frames else pd.DataFrame()

    if not rth_df.empty:
        rth_df = rth_df[~rth_df.index.duplicated(keep='first')]
    if not all_df.empty:
        all_df = all_df[~all_df.index.duplicated(keep='first')]

    logger.info(f"RTH: {len(rth_df)} bars, All: {len(all_df)} bars")
    return rth_df, all_df


def load_cached_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load previously fetched data from disk."""
    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")
    rth_df = None
    all_df = None
    if os.path.exists(rth_path):
        rth_df = pd.read_parquet(rth_path)
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(rth_path))).days
        logger.info(f"Loaded cached RTH data ({len(rth_df)} bars, {age} days old)")
    if os.path.exists(all_path):
        all_df = pd.read_parquet(all_path)
        logger.info(f"Loaded cached all-hours data ({len(all_df)} bars)")
    return rth_df, all_df


def save_cached_data(rth_df: pd.DataFrame, all_df: pd.DataFrame = None):
    """Cache fetched data to avoid re-downloading."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rth_path = os.path.join(RESULTS_DIR, "historical_bars.parquet")
    rth_df.to_parquet(rth_path)
    logger.info(f"Cached {len(rth_df)} RTH bars")
    if all_df is not None and not all_df.empty:
        all_path = os.path.join(RESULTS_DIR, "historical_bars_all.parquet")
        all_df.to_parquet(all_path)
        logger.info(f"Cached {len(all_df)} all-hours bars")


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    date:          str = ""
    day_of_week:   str = ""
    direction:     str = ""
    entry_num:     int = 0
    entry_price:   float = 0.0
    exit_price:    float = 0.0
    pnl_pts:       float = 0.0
    mfe_pts:       float = 0.0     # Max favourable excursion
    mae_pts:       float = 0.0     # Max adverse excursion
    held_bars:     int = 0
    held_to_close: bool = False

    # Situational tags
    bar_range:     float = 0.0
    range_class:   str = ""        # NARROW / NORMAL / WIDE
    gap_dir:       str = ""        # GAP_UP / GAP_DOWN / FLAT
    gap_size:      float = 0.0
    context:       str = ""        # DIRECTIONAL / CHOPPY / OVERLAP / MIXED
    bar_bullish:   bool = False
    prev_day_dir:  str = ""        # UP / DOWN (prev day close vs open)

    # Overnight range (V58 theory)
    overnight_bias:    str = ""    # SHORT_ONLY / LONG_ONLY / STANDARD / NO_DATA
    bar4_vs_overnight: str = ""    # ABOVE / BELOW / INSIDE


@dataclass
class DayResult:
    date:          str = ""
    day_of_week:   str = ""
    bar4_high:     float = 0.0
    bar4_low:      float = 0.0
    bar4_range:    float = 0.0
    range_class:   str = ""
    buy_level:     float = 0.0
    sell_level:    float = 0.0
    gap_dir:       str = ""
    gap_size:      float = 0.0
    context:       str = ""
    overnight_bias: str = ""
    bar4_vs_overnight: str = ""
    trades:        list = field(default_factory=list)
    total_pnl:     float = 0.0
    triggered:     bool = False


def candle_number(timestamp: pd.Timestamp) -> int:
    open_time = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
    mins = int((timestamp - open_time).total_seconds() / 60)
    return (mins // 5) + 1


def classify_range(rng: float) -> str:
    if rng < config.NARROW_RANGE:
        return "NARROW"
    elif rng > config.WIDE_RANGE:
        return "WIDE"
    return "NORMAL"


def classify_gap(prev_close: float, today_open: float) -> tuple[str, float]:
    if prev_close == 0:
        return "FLAT", 0
    gap = round(today_open - prev_close, 1)
    gap_pct = round(gap / prev_close * 100, 3)
    if gap > 10:
        return "GAP_UP", gap
    elif gap < -10:
        return "GAP_DOWN", gap
    return "FLAT", gap


def analyse_context_bars(bars_1_3: list[dict]) -> str:
    if len(bars_1_3) < 3:
        return "MIXED"
    all_bull = all(b["close"] > b["open"] for b in bars_1_3)
    all_bear = all(b["close"] < b["open"] for b in bars_1_3)
    if all_bull or all_bear:
        return "DIRECTIONAL"

    highs = [b["high"] for b in bars_1_3]
    lows = [b["low"] for b in bars_1_3]
    total_rng = max(highs) - min(lows)
    avg_rng = np.mean([b["high"] - b["low"] for b in bars_1_3])
    if total_rng < avg_rng * 2:
        return "OVERLAP"

    wick_pcts = []
    for b in bars_1_3:
        body = abs(b["close"] - b["open"])
        rng = b["high"] - b["low"]
        wick_pcts.append((rng - body) / rng * 100 if rng > 0 else 0)
    if np.mean(wick_pcts) > 50:
        return "CHOPPY"

    return "MIXED"


def run_backtest(df: pd.DataFrame, all_df: pd.DataFrame = None) -> list[DayResult]:
    """
    Run ASRS backtest over all trading days in the dataframe.
    all_df contains non-RTH bars for overnight range calculation (V58 theory).
    Returns list of DayResult objects.
    """
    from dax_bot.overnight import calculate_overnight_range, OvernightBias

    results = []
    prev_close = 0

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        day_name = trade_date.strftime("%A")

        # ── Identify bars ──────────────────────────────────────────────
        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx)
            if 1 <= cn <= 6:
                bars[cn] = {
                    "high": row["High"], "low": row["Low"],
                    "open": row["Open"], "close": row["Close"],
                }

        if 4 not in bars:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        bar4 = bars[4]
        bar4_range = round(bar4["high"] - bar4["low"], 1)

        # Gap
        today_open = day_df.iloc[0]["Open"]
        gap_dir, gap_size = classify_gap(prev_close, today_open)

        # Context (bars 1-3)
        context_bars = [bars[i] for i in [1, 2, 3] if i in bars]
        context = analyse_context_bars(context_bars)

        # Range class
        range_class = classify_range(bar4_range)

        # Levels
        buy_level = round(bar4["high"] + config.BUFFER_PTS, 1)
        sell_level = round(bar4["low"] - config.BUFFER_PTS, 1)

        # Previous day direction
        prev_day_dir = ""
        if prev_close > 0:
            prev_day_dir = "UP" if day_df.iloc[0]["Open"] > prev_close else "DOWN"

        # ── Overnight range (V58) ──────────────────────────────────────
        overnight_bias_str = "NO_DATA"
        bar4_vs_overnight = ""
        if all_df is not None and not all_df.empty:
            # Get overnight bars for this date (00:00–06:00 CET)
            try:
                day_all = all_df.loc[str(trade_date)]
                overnight = day_all.between_time("00:00", "06:00")
                if not overnight.empty:
                    ov_result = calculate_overnight_range(
                        overnight, bar4["high"], bar4["low"]
                    )
                    overnight_bias_str = ov_result.bias.value
                    bar4_vs_overnight = ov_result.bar4_vs_range
            except (KeyError, Exception):
                pass  # No overnight data for this date

        day_result = DayResult(
            date=str(trade_date),
            day_of_week=day_name,
            bar4_high=round(bar4["high"], 1),
            bar4_low=round(bar4["low"], 1),
            bar4_range=bar4_range,
            range_class=range_class,
            buy_level=buy_level,
            sell_level=sell_level,
            gap_dir=gap_dir,
            gap_size=round(gap_size, 1),
            context=context,
            overnight_bias=overnight_bias_str,
            bar4_vs_overnight=bar4_vs_overnight,
        )

        # ── Simulate trades ────────────────────────────────────────────
        post_bar4 = day_df[day_df.index > day_df.index[0] + pd.Timedelta(minutes=19)]
        # Filter to candles after bar 4 (candle number > 4)
        post_bars = []
        for idx, row in day_df.iterrows():
            if candle_number(idx) > 4:
                post_bars.append((idx, row))

        entries_used = 0
        direction = None
        entry_price = 0
        trail_stop = 0
        mfe = 0
        mae = 0
        entry_bar_idx = 0

        for i, (idx, row) in enumerate(post_bars):
            if entries_used >= config.MAX_ENTRIES and direction is None:
                break

            # ── Entry check ────────────────────────────────────────
            if direction is None and entries_used < config.MAX_ENTRIES:
                if row["High"] >= buy_level:
                    direction = "LONG"
                    entry_price = buy_level
                    trail_stop = sell_level
                    entries_used += 1
                    mfe = 0
                    mae = 0
                    entry_bar_idx = i
                elif row["Low"] <= sell_level:
                    direction = "SHORT"
                    entry_price = sell_level
                    trail_stop = buy_level
                    entries_used += 1
                    mfe = 0
                    mae = 0
                    entry_bar_idx = i

            # ── Position management ────────────────────────────────
            if direction == "LONG":
                # MFE / MAE
                bar_mfe = row["High"] - entry_price
                bar_mae = entry_price - row["Low"]
                mfe = max(mfe, bar_mfe)
                mae = max(mae, bar_mae)

                # Trail: previous completed candle low
                if i > entry_bar_idx:
                    prev_low = round(post_bars[i - 1][1]["Low"], 1)
                    if prev_low > trail_stop:
                        trail_stop = prev_low

                # Stop check
                if row["Low"] <= trail_stop:
                    pnl = round(trail_stop - entry_price, 1)
                    trade = Trade(
                        date=str(trade_date), day_of_week=day_name,
                        direction="LONG", entry_num=entries_used,
                        entry_price=entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                        held_bars=i - entry_bar_idx,
                        bar_range=bar4_range, range_class=range_class,
                        gap_dir=gap_dir, gap_size=round(gap_size, 1),
                        context=context, bar_bullish=bar4["close"] > bar4["open"],
                        prev_day_dir=prev_day_dir,
                        overnight_bias=overnight_bias_str,
                        bar4_vs_overnight=bar4_vs_overnight,
                    )
                    day_result.trades.append(trade)
                    direction = None

            elif direction == "SHORT":
                bar_mfe = entry_price - row["Low"]
                bar_mae = row["High"] - entry_price
                mfe = max(mfe, bar_mfe)
                mae = max(mae, bar_mae)

                if i > entry_bar_idx:
                    prev_high = round(post_bars[i - 1][1]["High"], 1)
                    if prev_high < trail_stop:
                        trail_stop = prev_high

                if row["High"] >= trail_stop:
                    pnl = round(entry_price - trail_stop, 1)
                    trade = Trade(
                        date=str(trade_date), day_of_week=day_name,
                        direction="SHORT", entry_num=entries_used,
                        entry_price=entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                        held_bars=i - entry_bar_idx,
                        bar_range=bar4_range, range_class=range_class,
                        gap_dir=gap_dir, gap_size=round(gap_size, 1),
                        context=context, bar_bullish=bar4["close"] > bar4["open"],
                        prev_day_dir=prev_day_dir,
                        overnight_bias=overnight_bias_str,
                        bar4_vs_overnight=bar4_vs_overnight,
                    )
                    day_result.trades.append(trade)
                    direction = None

        # If still holding at EOD
        if direction is not None and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 1)
            if direction == "LONG":
                pnl = round(last_price - entry_price, 1)
            else:
                pnl = round(entry_price - last_price, 1)

            trade = Trade(
                date=str(trade_date), day_of_week=day_name,
                direction=direction, entry_num=entries_used,
                entry_price=entry_price, exit_price=last_price,
                pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                held_bars=len(post_bars) - entry_bar_idx,
                held_to_close=True,
                bar_range=bar4_range, range_class=range_class,
                gap_dir=gap_dir, gap_size=round(gap_size, 1),
                context=context, bar_bullish=bar4["close"] > bar4["open"],
                prev_day_dir=prev_day_dir,
                overnight_bias=overnight_bias_str,
                bar4_vs_overnight=bar4_vs_overnight,
            )
            day_result.trades.append(trade)

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)

        prev_close = day_df.iloc[-1]["Close"]

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  EMA TRAIL BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _calc_ema_series(closes: list[float], period: int) -> list[float | None]:
    """Calculate EMA for each bar. Returns list same length as closes."""
    result = [None] * len(closes)
    if len(closes) < period:
        return result
    ema = sum(closes[:period]) / period
    result[period - 1] = round(ema, 1)
    mult = 2 / (period + 1)
    for i in range(period, len(closes)):
        ema = (closes[i] - ema) * mult + ema
        result[i] = round(ema, 1)
    return result


def run_backtest_ema(
    df: pd.DataFrame,
    all_df: pd.DataFrame = None,
    enable_adds: bool = False,
    signal_bar: int = 4,
) -> list[DayResult]:
    """
    Run ASRS backtest with EMA-based trailing stop and optional add-to-winners.

    signal_bar: which bar to use for levels (4 or 5).

    3-phase trail:
      Phase 1 (Underwater): stop at original level
      Phase 2 (Breakeven):  stop at entry price (after TRAIL_BREAKEVEN_TRIGGER pts)
      Phase 3 (EMA Trail):  stop at EMA +/- TRAIL_EMA_BUFFER (after TRAIL_EMA_TRIGGER pts)

    Add-to-winners (if enable_adds=True):
      In Phase 3, if price touches EMA zone and bounces, add a second position.
    """
    from dax_bot.overnight import calculate_overnight_range

    be_trigger = config.TRAIL_BREAKEVEN_TRIGGER
    ema_trigger = config.TRAIL_EMA_TRIGGER
    ema_buffer = config.TRAIL_EMA_BUFFER
    ema_period = config.TRAIL_EMA_PERIOD
    touch_zone = config.ADD_EMA_TOUCH_ZONE
    max_adds = config.ADD_MAX_ENTRIES

    results = []
    prev_close = 0

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        day_name = trade_date.strftime("%A")

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
        sig_range = round(sig["high"] - sig["low"], 1)

        today_open = day_df.iloc[0]["Open"]
        gap_dir, gap_size = classify_gap(prev_close, today_open)
        context_bars = [bars[i] for i in [1, 2, 3] if i in bars]
        context = analyse_context_bars(context_bars)
        range_class = classify_range(sig_range)

        buy_level = round(sig["high"] + config.BUFFER_PTS, 1)
        sell_level = round(sig["low"] - config.BUFFER_PTS, 1)

        prev_day_dir = ""
        if prev_close > 0:
            prev_day_dir = "UP" if today_open > prev_close else "DOWN"

        overnight_bias_str = "NO_DATA"
        bar4_vs_overnight = ""
        if all_df is not None and not all_df.empty:
            try:
                day_all = all_df.loc[str(trade_date)]
                overnight = day_all.between_time("00:00", "06:00")
                if not overnight.empty:
                    ov_result = calculate_overnight_range(
                        overnight, sig["high"], sig["low"]
                    )
                    overnight_bias_str = ov_result.bias.value
                    bar4_vs_overnight = ov_result.bar4_vs_range
            except (KeyError, Exception):
                pass

        day_result = DayResult(
            date=str(trade_date), day_of_week=day_name,
            bar4_high=round(sig["high"], 1), bar4_low=round(sig["low"], 1),
            bar4_range=sig_range, range_class=range_class,
            buy_level=buy_level, sell_level=sell_level,
            gap_dir=gap_dir, gap_size=round(gap_size, 1),
            context=context, overnight_bias=overnight_bias_str,
            bar4_vs_overnight=bar4_vs_overnight,
        )

        # Build post-signal-bar candles with EMA
        post_bars = []
        all_closes = []
        for idx, row in day_df.iterrows():
            all_closes.append(row["Close"])
            if candle_number(idx) > signal_bar:
                post_bars.append((idx, row, len(all_closes) - 1))

        ema_series = _calc_ema_series(all_closes, ema_period)

        # -- Simulation state --
        # Support multiple concurrent positions for add-to-winners
        @dataclass
        class Position:
            direction: str = ""
            entry_price: float = 0.0
            trail_stop: float = 0.0
            initial_stop: float = 0.0
            mfe: float = 0.0
            mae: float = 0.0
            entry_bar_idx: int = 0
            phase: str = "UNDERWATER"
            is_add: bool = False
            ema_at_exit: float = 0.0
            bounce_count: int = 0

        positions: list[Position] = []
        entries_used = 0
        ema_touch_pending = False  # For add detection: touched but waiting for bounce

        for i, (idx, row, close_idx) in enumerate(post_bars):
            ema_val = ema_series[close_idx]

            # -- Check exits first (process all open positions) --
            closed_positions = []
            for pi, pos in enumerate(positions):
                if pos.direction == "LONG":
                    pos.mfe = max(pos.mfe, row["High"] - pos.entry_price)
                    pos.mae = max(pos.mae, pos.entry_price - row["Low"])
                elif pos.direction == "SHORT":
                    pos.mfe = max(pos.mfe, pos.entry_price - row["Low"])
                    pos.mae = max(pos.mae, row["High"] - pos.entry_price)

                # Determine phase
                if pos.direction == "LONG":
                    favour = row["High"] - pos.entry_price
                    above_ema = ema_val is not None and row["Close"] > ema_val
                else:
                    favour = pos.entry_price - row["Low"]
                    above_ema = ema_val is not None and row["Close"] < ema_val

                old_phase = pos.phase
                if ema_val is not None and favour >= ema_trigger and above_ema:
                    pos.phase = "EMA_TRAIL"
                elif favour >= be_trigger:
                    if pos.phase == "UNDERWATER":
                        pos.phase = "BREAKEVEN"
                # Don't downgrade from EMA_TRAIL

                # Update trail stop based on phase
                if pos.phase == "BREAKEVEN" and old_phase == "UNDERWATER":
                    if pos.direction == "LONG":
                        pos.trail_stop = max(pos.trail_stop, pos.entry_price)
                    else:
                        pos.trail_stop = min(pos.trail_stop, pos.entry_price)

                elif pos.phase == "EMA_TRAIL" and ema_val is not None:
                    if pos.direction == "LONG":
                        raw = round(ema_val * (1 - ema_buffer), 1)
                        raw = max(raw, pos.entry_price)  # Never below breakeven
                        pos.trail_stop = max(pos.trail_stop, raw)
                    else:
                        raw = round(ema_val * (1 + ema_buffer), 1)
                        raw = min(raw, pos.entry_price)
                        pos.trail_stop = min(pos.trail_stop, raw)

                # Check stop hit (use close for EMA trail, low/high for others)
                stopped = False
                if pos.phase == "EMA_TRAIL":
                    if pos.direction == "LONG" and row["Close"] < pos.trail_stop:
                        stopped = True
                    elif pos.direction == "SHORT" and row["Close"] > pos.trail_stop:
                        stopped = True
                else:
                    if pos.direction == "LONG" and row["Low"] <= pos.trail_stop:
                        stopped = True
                    elif pos.direction == "SHORT" and row["High"] >= pos.trail_stop:
                        stopped = True

                if stopped:
                    exit_price = pos.trail_stop
                    if pos.direction == "LONG":
                        pnl = round(exit_price - pos.entry_price, 1)
                    else:
                        pnl = round(pos.entry_price - exit_price, 1)

                    pos.ema_at_exit = ema_val or 0
                    trade = Trade(
                        date=str(trade_date), day_of_week=day_name,
                        direction=pos.direction, entry_num=pos.entry_bar_idx,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        pnl_pts=pnl, mfe_pts=round(pos.mfe, 1),
                        mae_pts=round(pos.mae, 1),
                        held_bars=i - pos.entry_bar_idx,
                        bar_range=sig_range, range_class=range_class,
                        gap_dir=gap_dir, gap_size=round(gap_size, 1),
                        context=context, bar_bullish=sig["close"] > sig["open"],
                        prev_day_dir=prev_day_dir,
                        overnight_bias=overnight_bias_str,
                        bar4_vs_overnight=bar4_vs_overnight,
                    )
                    day_result.trades.append(trade)
                    closed_positions.append(pi)

            # Remove closed positions (reverse order to preserve indices)
            for pi in reversed(closed_positions):
                positions.pop(pi)

            # -- Check for new entry (only if no positions open and below max) --
            if not positions and entries_used < config.MAX_ENTRIES:
                if row["High"] >= buy_level:
                    pos = Position(
                        direction="LONG", entry_price=buy_level,
                        trail_stop=sell_level, initial_stop=sell_level,
                        entry_bar_idx=i,
                    )
                    positions.append(pos)
                    entries_used += 1
                    ema_touch_pending = False
                elif row["Low"] <= sell_level:
                    pos = Position(
                        direction="SHORT", entry_price=sell_level,
                        trail_stop=buy_level, initial_stop=buy_level,
                        entry_bar_idx=i,
                    )
                    positions.append(pos)
                    entries_used += 1
                    ema_touch_pending = False

            # -- Add to winners check --
            if enable_adds and positions and len(positions) < max_adds:
                lead = positions[0]
                if lead.phase == "EMA_TRAIL" and ema_val is not None and entries_used < config.MAX_ENTRIES:
                    # Check EMA bounce
                    touch_dist = ema_val * touch_zone
                    if lead.direction == "LONG":
                        touched = row["Low"] <= ema_val + touch_dist
                        bounced = row["Close"] > ema_val
                    else:
                        touched = row["High"] >= ema_val - touch_dist
                        bounced = row["Close"] < ema_val

                    if touched and bounced:
                        add_entry = round(row["Close"], 1)
                        add_pos = Position(
                            direction=lead.direction,
                            entry_price=add_entry,
                            trail_stop=lead.trail_stop,
                            initial_stop=lead.trail_stop,
                            entry_bar_idx=i,
                            is_add=True,
                        )
                        positions.append(add_pos)
                        entries_used += 1
                        lead.bounce_count += 1

        # -- EOD: close all open positions --
        if positions and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 1)
            last_ema = ema_series[post_bars[-1][2]] if post_bars else None
            for pos in positions:
                if pos.direction == "LONG":
                    pnl = round(last_price - pos.entry_price, 1)
                else:
                    pnl = round(pos.entry_price - last_price, 1)

                trade = Trade(
                    date=str(trade_date), day_of_week=day_name,
                    direction=pos.direction, entry_num=pos.entry_bar_idx,
                    entry_price=pos.entry_price, exit_price=last_price,
                    pnl_pts=pnl, mfe_pts=round(pos.mfe, 1),
                    mae_pts=round(pos.mae, 1),
                    held_bars=len(post_bars) - pos.entry_bar_idx,
                    held_to_close=True,
                    bar_range=sig_range, range_class=range_class,
                    gap_dir=gap_dir, gap_size=round(gap_size, 1),
                    context=context, bar_bullish=sig["close"] > sig["open"],
                    prev_day_dir=prev_day_dir,
                    overnight_bias=overnight_bias_str,
                    bar4_vs_overnight=bar4_vs_overnight,
                )
                day_result.trades.append(trade)

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)
        prev_close = day_df.iloc[-1]["Close"]

    return results


def print_full_compare(orig: list[DayResult], ema_only: list[DayResult], ema_adds: list[DayResult]):
    """Print side-by-side comparison of Original, EMA Trail, EMA Trail + Adds."""
    def stats(results):
        trades = [t for r in results for t in r.trades]
        pnl = sum(t.pnl_pts for t in trades)
        wins = sum(1 for t in trades if t.pnl_pts > 0)
        losses = sum(1 for t in trades if t.pnl_pts < 0)
        wr = round(wins / len(trades) * 100) if trades else 0
        avg_w = round(np.mean([t.pnl_pts for t in trades if t.pnl_pts > 0]), 1) if wins else 0
        avg_l = round(np.mean([t.pnl_pts for t in trades if t.pnl_pts < 0]), 1) if losses else 0
        eq = []
        run = 0
        for r in results:
            run += r.total_pnl
            eq.append(run)
        pk = dd = 0
        for e in eq:
            pk = max(pk, e)
            dd = max(dd, pk - e)
        return {
            "n": len(trades), "pnl": pnl, "wr": wr,
            "avg_w": avg_w, "avg_l": avg_l, "dd": dd, "eq": eq,
        }

    s_orig = stats(orig)
    s_ema = stats(ema_only)
    s_adds = stats(ema_adds)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              FULL COMPARISON — Original vs EMA Trail vs EMA + Adds    ║
╠═══════════════════════════════════════════════════════════════════════╣
║               {'Original':>12}  {'EMA Trail':>12}  {'EMA + Adds':>12}     ║
║  Trades:      {s_orig['n']:>12}  {s_ema['n']:>12}  {s_adds['n']:>12}     ║
║  Win rate:    {s_orig['wr']:>11}%  {s_ema['wr']:>11}%  {s_adds['wr']:>11}%     ║
║  Avg win:     {s_orig['avg_w']:>+11.1f}  {s_ema['avg_w']:>+11.1f}  {s_adds['avg_w']:>+11.1f}     ║
║  Avg loss:    {s_orig['avg_l']:>+11.1f}  {s_ema['avg_l']:>+11.1f}  {s_adds['avg_l']:>+11.1f}     ║
║  Total P&L:   {s_orig['pnl']:>+11.1f}  {s_ema['pnl']:>+11.1f}  {s_adds['pnl']:>+11.1f}     ║
║  Max DD:      {s_orig['dd']:>11.1f}  {s_ema['dd']:>11.1f}  {s_adds['dd']:>11.1f}     ║
╠═══════════════════════════════════════════════════════════════════════╣
║  vs Original:              {s_ema['pnl'] - s_orig['pnl']:>+11.1f}  {s_adds['pnl'] - s_orig['pnl']:>+11.1f}     ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Monthly breakdown side by side
    print("── Monthly Breakdown ──────────────────────────────────────────────────")
    print(f"{'Month':<10} {'Orig':>8} {'EMA':>8} {'EMA+Add':>8} {'Best'}")
    print("─" * 55)

    monthly_orig = defaultdict(float)
    monthly_ema = defaultdict(float)
    monthly_adds = defaultdict(float)
    for r in orig:
        monthly_orig[r.date[:7]] += r.total_pnl
    for r in ema_only:
        monthly_ema[r.date[:7]] += r.total_pnl
    for r in ema_adds:
        monthly_adds[r.date[:7]] += r.total_pnl

    for m in sorted(monthly_orig.keys()):
        o, e, a = monthly_orig[m], monthly_ema.get(m, 0), monthly_adds.get(m, 0)
        best_val = max(o, e, a)
        best = "Orig" if best_val == o else ("EMA" if best_val == e else "EMA+A")
        print(f"{m:<10} {o:>+7.0f} {e:>+7.0f} {a:>+7.0f}  {best}")

    # Generate chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(14, 6))
        dates = [datetime.strptime(r.date, "%Y-%m-%d") for r in orig]

        ax.plot(dates, s_orig["eq"], color="#9E9E9E", linewidth=1, alpha=0.7,
                label=f"Original ({s_orig['pnl']:+.0f} pts)")
        ax.plot(dates, s_ema["eq"], color="#2196F3", linewidth=1.5,
                label=f"EMA Trail ({s_ema['pnl']:+.0f} pts)")
        ax.plot(dates, s_adds["eq"], color="#4CAF50", linewidth=2,
                label=f"EMA Trail + Adds ({s_adds['pnl']:+.0f} pts)")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("ASRS: Original vs EMA Trail vs EMA + Adds", fontsize=16, fontweight="bold")
        ax.set_ylabel("Cumulative P&L (points)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "charts", "full_comparison.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n📊 Chart saved to data/charts/full_comparison.png")
    except Exception as e:
        print(f"Chart error: {e}")


def run_monte_carlo(
    results: list[DayResult],
    num_sims: int = 10000,
    num_days: int = 252,
    starting_capital: float = 5000,
    pts_per_eur: float = 1.0,
):
    """
    Monte Carlo simulation using daily P&L resampling.

    Randomly resamples daily P&L (with replacement) to generate num_sims
    equity paths of num_days length. Measures:
      - Risk of ruin (equity hitting 0)
      - Drawdown distribution
      - Return distribution
      - Confidence intervals
    """
    # Extract daily P&L array
    daily_pnl = np.array([r.total_pnl * pts_per_eur for r in results if r.triggered])
    if len(daily_pnl) < 10:
        print("Not enough trading days for Monte Carlo simulation.")
        return

    rng = np.random.default_rng(42)

    # Run simulations
    all_final = np.zeros(num_sims)
    all_max_dd = np.zeros(num_sims)
    all_max_dd_pct = np.zeros(num_sims)
    ruin_count = 0
    half_ruin_count = 0
    all_paths = np.zeros((num_sims, num_days + 1))

    for s in range(num_sims):
        # Resample daily P&L with replacement
        sampled = rng.choice(daily_pnl, size=num_days, replace=True)
        equity = np.empty(num_days + 1)
        equity[0] = starting_capital
        ruined = False
        for d in range(num_days):
            equity[d + 1] = equity[d] + sampled[d]
            if equity[d + 1] <= 0:
                equity[d + 1:] = 0
                ruined = True
                break

        all_paths[s] = equity
        all_final[s] = equity[-1]

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = np.max(dd)
        all_max_dd[s] = max_dd
        peak_at_dd = peak[np.argmax(dd)]
        all_max_dd_pct[s] = (max_dd / peak_at_dd * 100) if peak_at_dd > 0 else 0

        if ruined:
            ruin_count += 1
        if equity[-1] < starting_capital * 0.5:
            half_ruin_count += 1

    # Statistics
    ruin_pct = ruin_count / num_sims * 100
    half_ruin_pct = half_ruin_count / num_sims * 100
    median_final = np.median(all_final)
    mean_final = np.mean(all_final)
    p5 = np.percentile(all_final, 5)
    p25 = np.percentile(all_final, 25)
    p75 = np.percentile(all_final, 75)
    p95 = np.percentile(all_final, 95)
    median_dd = np.median(all_max_dd)
    p95_dd = np.percentile(all_max_dd, 95)
    p99_dd = np.percentile(all_max_dd, 99)
    median_dd_pct = np.median(all_max_dd_pct)
    p95_dd_pct = np.percentile(all_max_dd_pct, 95)

    # Probability of profit
    profit_pct = np.sum(all_final > starting_capital) / num_sims * 100
    double_pct = np.sum(all_final > starting_capital * 2) / num_sims * 100

    # Annualised return estimate
    mean_daily = np.mean(daily_pnl)
    annual_est = mean_daily * num_days

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              MONTE CARLO SIMULATION — Risk of Ruin                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Simulations:      {num_sims:>10,}                                      ║
║  Days per sim:     {num_days:>10} (~1 year of trading)                  ║
║  Starting capital: {starting_capital:>10,.0f} EUR                                    ║
║  Daily P&L sample: {len(daily_pnl):>10} trading days                            ║
║  Avg daily P&L:    {mean_daily:>+10.1f} EUR                                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  RISK OF RUIN                                                         ║
║  Ruin (equity=0):         {ruin_pct:>6.2f}%                                  ║
║  50% drawdown:            {half_ruin_pct:>6.2f}%                                  ║
║  Probability of profit:   {profit_pct:>6.1f}%                                  ║
║  Probability of 2x:       {double_pct:>6.1f}%                                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  RETURN DISTRIBUTION (after {num_days} days)                               ║
║  5th percentile:  {p5:>+10,.0f} EUR  (worst case)                       ║
║  25th percentile: {p25:>+10,.0f} EUR                                    ║
║  Median:          {median_final:>+10,.0f} EUR                                    ║
║  Mean:            {mean_final:>+10,.0f} EUR                                    ║
║  75th percentile: {p75:>+10,.0f} EUR                                    ║
║  95th percentile: {p95:>+10,.0f} EUR  (best case)                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║  DRAWDOWN DISTRIBUTION                                                ║
║  Median max DD:      {median_dd:>8,.0f} EUR ({median_dd_pct:>5.1f}%)                       ║
║  95th pct max DD:    {p95_dd:>8,.0f} EUR ({p95_dd_pct:>5.1f}%)                       ║
║  99th pct max DD:    {p99_dd:>8,.0f} EUR                                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  SCALING ESTIMATES (annual, median)                                   ║
║  1 micro  (1 EUR/pt):  {(median_final - starting_capital):>+10,.0f} EUR/yr                       ║
║  5 micro  (5 EUR/pt):  {(median_final - starting_capital) * 5:>+10,.0f} EUR/yr                       ║
║  1 mini  (5 EUR/pt):   {(median_final - starting_capital) * 5:>+10,.0f} EUR/yr                       ║
║  10 mini (50 EUR/pt):  {(median_final - starting_capital) * 50:>+10,.0f} EUR/yr                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Generate chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Monte Carlo Simulation — {num_sims:,} runs, {num_days} days, "
                     f"starting {starting_capital:,.0f} EUR", fontsize=14, fontweight="bold")

        # 1. Equity paths (sample 200)
        ax = axes[0, 0]
        sample_idx = rng.choice(num_sims, size=min(200, num_sims), replace=False)
        for s in sample_idx:
            color = "#4CAF50" if all_final[s] > starting_capital else "#F44336"
            ax.plot(all_paths[s], color=color, alpha=0.05, linewidth=0.5)
        # Percentile bands
        p5_path = np.percentile(all_paths, 5, axis=0)
        p50_path = np.percentile(all_paths, 50, axis=0)
        p95_path = np.percentile(all_paths, 95, axis=0)
        ax.plot(p50_path, color="#2196F3", linewidth=2, label="Median")
        ax.fill_between(range(num_days + 1), p5_path, p95_path, alpha=0.15, color="#2196F3")
        ax.axhline(y=starting_capital, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title("Equity Paths (200 sample)")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("EUR")
        ax.legend()

        # 2. Final equity distribution
        ax = axes[0, 1]
        ax.hist(all_final, bins=100, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(x=starting_capital, color="gray", linestyle="--", linewidth=1, label="Starting")
        ax.axvline(x=median_final, color="#4CAF50", linewidth=2, label=f"Median: {median_final:,.0f}")
        ax.axvline(x=0, color="red", linewidth=1, label="Ruin")
        ax.set_title("Final Equity Distribution")
        ax.set_xlabel("EUR")
        ax.legend()

        # 3. Max drawdown distribution
        ax = axes[1, 0]
        ax.hist(all_max_dd, bins=100, color="#FF9800", alpha=0.7, edgecolor="white")
        ax.axvline(x=median_dd, color="#F44336", linewidth=2, label=f"Median: {median_dd:,.0f}")
        ax.axvline(x=p95_dd, color="red", linewidth=1, linestyle="--",
                   label=f"95th: {p95_dd:,.0f}")
        ax.set_title("Max Drawdown Distribution")
        ax.set_xlabel("EUR")
        ax.legend()

        # 4. Drawdown % distribution
        ax = axes[1, 1]
        ax.hist(all_max_dd_pct, bins=100, color="#9C27B0", alpha=0.7, edgecolor="white")
        ax.axvline(x=median_dd_pct, color="#F44336", linewidth=2,
                   label=f"Median: {median_dd_pct:.1f}%")
        ax.axvline(x=p95_dd_pct, color="red", linewidth=1, linestyle="--",
                   label=f"95th: {p95_dd_pct:.1f}%")
        ax.set_title("Max Drawdown % Distribution")
        ax.set_xlabel("%")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "charts", "monte_carlo.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Chart saved to data/charts/monte_carlo.png")
    except Exception as e:
        print(f"Chart error: {e}")


def run_backtest_bar5_fallback(
    df: pd.DataFrame,
    all_df: pd.DataFrame = None,
    target_dates: set[str] | None = None,
) -> list[DayResult]:
    """
    Re-run specific days using bar 5 levels instead of bar 4.
    If target_dates is None, defaults to days with 0 trades (skipped).
    Returns list of DayResult for just those days.
    """
    from dax_bot.overnight import calculate_overnight_range, OvernightBias

    if target_dates is None:
        bar4_results = run_backtest(df, all_df)
        target_dates = {r.date for r in bar4_results if not r.triggered}

    if not target_dates:
        logger.info("No target days — nothing to re-run with bar 5")
        return []

    logger.info(f"Re-running {len(target_dates)} days with bar 5 levels...")
    results = []
    prev_close = 0

    for trade_date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        # Track prev_close for all days
        today_open = day_df.iloc[0]["Open"]

        if str(trade_date) not in target_dates:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        day_name = trade_date.strftime("%A")

        bars = {}
        for idx, row in day_df.iterrows():
            cn = candle_number(idx)
            if 1 <= cn <= 6:
                bars[cn] = {
                    "high": row["High"], "low": row["Low"],
                    "open": row["Open"], "close": row["Close"],
                }

        if 5 not in bars:
            prev_close = day_df.iloc[-1]["Close"]
            continue

        bar5 = bars[5]
        bar4 = bars.get(4, bar5)
        bar5_range = round(bar5["high"] - bar5["low"], 1)

        gap_dir, gap_size = classify_gap(prev_close, today_open)
        context_bars = [bars[i] for i in [1, 2, 3] if i in bars]
        context = analyse_context_bars(context_bars)
        range_class = classify_range(bar5_range)

        buy_level = round(bar5["high"] + config.BUFFER_PTS, 1)
        sell_level = round(bar5["low"] - config.BUFFER_PTS, 1)

        prev_day_dir = ""
        if prev_close > 0:
            prev_day_dir = "UP" if today_open > prev_close else "DOWN"

        overnight_bias_str = "NO_DATA"
        bar4_vs_overnight = ""
        if all_df is not None and not all_df.empty:
            try:
                day_all = all_df.loc[str(trade_date)]
                overnight = day_all.between_time("00:00", "06:00")
                if not overnight.empty:
                    ov_result = calculate_overnight_range(
                        overnight, bar5["high"], bar5["low"]
                    )
                    overnight_bias_str = ov_result.bias.value
                    bar4_vs_overnight = ov_result.bar4_vs_range
            except (KeyError, Exception):
                pass

        day_result = DayResult(
            date=str(trade_date),
            day_of_week=day_name,
            bar4_high=round(bar5["high"], 1),
            bar4_low=round(bar5["low"], 1),
            bar4_range=bar5_range,
            range_class=range_class,
            buy_level=buy_level,
            sell_level=sell_level,
            gap_dir=gap_dir,
            gap_size=round(gap_size, 1),
            context=context,
            overnight_bias=overnight_bias_str,
            bar4_vs_overnight=bar4_vs_overnight,
        )

        # Simulate trades using candles after bar 5
        post_bars = []
        for idx, row in day_df.iterrows():
            if candle_number(idx) > 5:
                post_bars.append((idx, row))

        entries_used = 0
        direction = None
        entry_price = 0
        trail_stop = 0
        mfe = 0
        mae = 0
        entry_bar_idx = 0

        for i, (idx, row) in enumerate(post_bars):
            if entries_used >= config.MAX_ENTRIES and direction is None:
                break

            if direction is None and entries_used < config.MAX_ENTRIES:
                if row["High"] >= buy_level:
                    direction = "LONG"
                    entry_price = buy_level
                    trail_stop = sell_level
                    entries_used += 1
                    mfe = mae = 0
                    entry_bar_idx = i
                elif row["Low"] <= sell_level:
                    direction = "SHORT"
                    entry_price = sell_level
                    trail_stop = buy_level
                    entries_used += 1
                    mfe = mae = 0
                    entry_bar_idx = i

            if direction == "LONG":
                mfe = max(mfe, row["High"] - entry_price)
                mae = max(mae, entry_price - row["Low"])
                if i > entry_bar_idx:
                    prev_low = round(post_bars[i - 1][1]["Low"], 1)
                    if prev_low > trail_stop:
                        trail_stop = prev_low
                if row["Low"] <= trail_stop:
                    pnl = round(trail_stop - entry_price, 1)
                    day_result.trades.append(Trade(
                        date=str(trade_date), day_of_week=day_name,
                        direction="LONG", entry_num=entries_used,
                        entry_price=entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                        held_bars=i - entry_bar_idx,
                        bar_range=bar5_range, range_class=range_class,
                        gap_dir=gap_dir, gap_size=round(gap_size, 1),
                        context=context, bar_bullish=bar5["close"] > bar5["open"],
                        prev_day_dir=prev_day_dir,
                        overnight_bias=overnight_bias_str,
                        bar4_vs_overnight=bar4_vs_overnight,
                    ))
                    direction = None

            elif direction == "SHORT":
                mfe = max(mfe, entry_price - row["Low"])
                mae = max(mae, row["High"] - entry_price)
                if i > entry_bar_idx:
                    prev_high = round(post_bars[i - 1][1]["High"], 1)
                    if prev_high < trail_stop:
                        trail_stop = prev_high
                if row["High"] >= trail_stop:
                    pnl = round(entry_price - trail_stop, 1)
                    day_result.trades.append(Trade(
                        date=str(trade_date), day_of_week=day_name,
                        direction="SHORT", entry_num=entries_used,
                        entry_price=entry_price, exit_price=trail_stop,
                        pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                        held_bars=i - entry_bar_idx,
                        bar_range=bar5_range, range_class=range_class,
                        gap_dir=gap_dir, gap_size=round(gap_size, 1),
                        context=context, bar_bullish=bar5["close"] > bar5["open"],
                        prev_day_dir=prev_day_dir,
                        overnight_bias=overnight_bias_str,
                        bar4_vs_overnight=bar4_vs_overnight,
                    ))
                    direction = None

        # EOD close
        if direction is not None and post_bars:
            last_price = round(post_bars[-1][1]["Close"], 1)
            pnl = round((last_price - entry_price) if direction == "LONG" else (entry_price - last_price), 1)
            day_result.trades.append(Trade(
                date=str(trade_date), day_of_week=day_name,
                direction=direction, entry_num=entries_used,
                entry_price=entry_price, exit_price=last_price,
                pnl_pts=pnl, mfe_pts=round(mfe, 1), mae_pts=round(mae, 1),
                held_bars=len(post_bars) - entry_bar_idx, held_to_close=True,
                bar_range=bar5_range, range_class=range_class,
                gap_dir=gap_dir, gap_size=round(gap_size, 1),
                context=context, bar_bullish=bar5["close"] > bar5["open"],
                prev_day_dir=prev_day_dir,
                overnight_bias=overnight_bias_str,
                bar4_vs_overnight=bar4_vs_overnight,
            ))

        day_result.total_pnl = round(sum(t.pnl_pts for t in day_result.trades), 1)
        day_result.triggered = len(day_result.trades) > 0
        results.append(day_result)

        prev_close = day_df.iloc[-1]["Close"]

    return results


def print_bar5_fallback(bar4_results: list[DayResult], bar5_results: list[DayResult], label: str = "Skipped Days"):
    """Print comparison of target days re-run with bar 5."""
    bar4_by_date = {r.date: r for r in bar4_results}
    target_dates = {r.date for r in bar5_results}
    bar5_triggered = [r for r in bar5_results if r.triggered]
    bar5_not_triggered = [r for r in bar5_results if not r.triggered]

    all_b5_trades = [t for r in bar5_results for t in r.trades]
    b5_total_pnl = sum(t.pnl_pts for t in all_b5_trades)
    b5_winners = [t for t in all_b5_trades if t.pnl_pts > 0]
    b5_losers = [t for t in all_b5_trades if t.pnl_pts < 0]

    # Bar 4 P&L for just the target days
    b4_target_pnl = sum(bar4_by_date[d].total_pnl for d in target_dates if d in bar4_by_date)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         BAR 5 FALLBACK — {label:<35}║
╠══════════════════════════════════════════════════════════════╣
║  Days re-analysed:         {len(target_dates):<34}║
║  Triggered with bar 5:     {len(bar5_triggered):<34}║
║  Still no trigger:         {len(bar5_not_triggered):<34}║
╠══════════════════════════════════════════════════════════════╣
║  BAR 4 (these days only)    BAR 5 (replacement)              ║
║  Trades:  {sum(len(bar4_by_date[d].trades) for d in target_dates if d in bar4_by_date):>6}              Trades:  {len(all_b5_trades):>6}                ║
║  Winners: {sum(1 for d in target_dates if d in bar4_by_date for t in bar4_by_date[d].trades if t.pnl_pts > 0):>6}              Winners: {len(b5_winners):>6}                ║
║  Losers:  {sum(1 for d in target_dates if d in bar4_by_date for t in bar4_by_date[d].trades if t.pnl_pts < 0):>6}              Losers:  {len(b5_losers):>6}                ║
║  P&L:     {b4_target_pnl:>+7.1f}            P&L:     {b5_total_pnl:>+7.1f}              ║
╠══════════════════════════════════════════════════════════════╣
║  Avg win (bar5):   {round(np.mean([t.pnl_pts for t in b5_winners]), 1) if b5_winners else 0:+.1f} pts{' ' * 37}║
║  Avg loss (bar5):  {round(np.mean([t.pnl_pts for t in b5_losers]), 1) if b5_losers else 0:+.1f} pts{' ' * 37}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Day-by-day comparison
    print(f"{'Date':<12} {'Day':<4} {'B4 P&L':>7} {'B5 Range':>8} {'Buy':>8} {'Sell':>8} "
          f"{'B5 P&L':>7} {'Diff':>7} {'B5 Details'}")
    print("─" * 90)

    total_diff = 0
    for r5 in bar5_results:
        r4 = bar4_by_date.get(r5.date)
        b4_pnl = r4.total_pnl if r4 else 0
        diff = r5.total_pnl - b4_pnl
        total_diff += diff

        trades_str = " | ".join(
            f"{t.direction[0]}:{t.pnl_pts:+.0f}" for t in r5.trades
        ) if r5.trades else "— (no trigger)"

        icon = "🟢" if diff > 0 else ("🔴" if diff < 0 else "⬜")
        print(f"{r5.date:<12} {r5.day_of_week[:3]:<4} {b4_pnl:>+6.1f} {r5.bar4_range:>7.0f}p "
              f"{r5.buy_level:>8.1f} {r5.sell_level:>8.1f} "
              f"{r5.total_pnl:>+6.1f} {diff:>+6.1f} {icon} {trades_str}")

    # Summary
    bar4_total = sum(r.total_pnl for r in bar4_results)
    bar4_other = bar4_total - b4_target_pnl
    combined = bar4_other + b5_total_pnl
    print(f"\n── Impact (if bar 5 replaced bar 4 on these days) ─────")
    print(f"  Bar 4 total (all days):        {bar4_total:>+8.1f} pts")
    print(f"  Bar 4 on target days:          {b4_target_pnl:>+8.1f} pts")
    print(f"  Bar 5 on target days:          {b5_total_pnl:>+8.1f} pts")
    print(f"  Difference:                    {total_diff:>+8.1f} pts")
    print(f"  ─────────────────────────────────────────")
    print(f"  New total (bar4 kept + bar5):  {combined:>+8.1f} pts")
    better = "BETTER" if combined > bar4_total else "WORSE"
    print(f"  vs original:                   {combined - bar4_total:>+8.1f} pts ({better})")


def analyse_bar5_conditions(df: pd.DataFrame, all_df: pd.DataFrame = None):
    """
    Run bar 4 and bar 5 on ALL days, compare, and find bar 4 conditions
    that predict when bar 5 would be better.
    Outputs actionable rules: "When [condition], use bar 5 instead of bar 4."
    """
    # Run both
    bar4_results = run_backtest(df, all_df)
    all_dates = {r.date for r in bar4_results}
    bar5_results = run_backtest_bar5_fallback(df, all_df, target_dates=all_dates)

    bar4_by_date = {r.date: r for r in bar4_results}
    bar5_by_date = {r.date: r for r in bar5_results}

    # Build per-day comparison with bar 4 conditions
    @dataclass
    class DayComparison:
        date: str
        day_of_week: str
        bar4_pnl: float
        bar5_pnl: float
        diff: float  # bar5 - bar4 (positive = bar 5 better)
        bar4_range: float
        range_class: str
        gap_dir: str
        context: str
        overnight_bias: str
        bar4_vs_overnight: str
        bar4_bullish: bool

    comparisons = []
    for d in sorted(all_dates):
        r4 = bar4_by_date.get(d)
        r5 = bar5_by_date.get(d)
        if not r4 or not r5:
            continue

        bar4_bullish = False
        if r4.trades:
            bar4_bullish = r4.trades[0].bar_bullish

        comparisons.append(DayComparison(
            date=d,
            day_of_week=r4.day_of_week,
            bar4_pnl=r4.total_pnl,
            bar5_pnl=r5.total_pnl,
            diff=round(r5.total_pnl - r4.total_pnl, 1),
            bar4_range=r4.bar4_range,
            range_class=r4.range_class,
            gap_dir=r4.gap_dir,
            context=r4.context,
            overnight_bias=r4.overnight_bias,
            bar4_vs_overnight=r4.bar4_vs_overnight,
            bar4_bullish=bar4_bullish,
        ))

    if not comparisons:
        print("No comparison data available.")
        return

    total_b4 = sum(c.bar4_pnl for c in comparisons)
    total_b5 = sum(c.bar5_pnl for c in comparisons)
    b5_better_days = [c for c in comparisons if c.diff > 0]
    b4_better_days = [c for c in comparisons if c.diff < 0]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║      BAR 5 CONDITIONS ANALYSIS — When to use Bar 5?          ║
╠══════════════════════════════════════════════════════════════╣
║  Total days compared:    {len(comparisons):<36}║
║  Bar 5 better:           {len(b5_better_days)} days{' ' * 31}║
║  Bar 4 better:           {len(b4_better_days)} days{' ' * 31}║
║  Bar 4 total:            {total_b4:>+8.1f} pts{' ' * 28}║
║  Bar 5 total:            {total_b5:>+8.1f} pts{' ' * 28}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Define condition slicers (based on bar 4 properties)
    slicers = {
        "Day of Week": lambda c: c.day_of_week,
        "Gap Direction": lambda c: c.gap_dir,
        "Bar 4 Range": lambda c: c.range_class,
        "Context (Bars 1-3)": lambda c: c.context,
        "Bar 4 Character": lambda c: "BULLISH" if c.bar4_bullish else "BEARISH",
        "V58 Overnight Bias": lambda c: c.overnight_bias or "NO_DATA",
        "Bar4 vs Overnight": lambda c: c.bar4_vs_overnight or "NO_DATA",
    }

    print("── Single Conditions ──────────────────────────────────────────────")
    print(f"{'Condition':<22} {'Days':>5} {'B5>B4':>6} {'B4 P&L':>8} {'B5 P&L':>8} {'Diff':>8} {'Signal'}")
    print("─" * 80)

    for name, slicer in slicers.items():
        groups = defaultdict(list)
        for c in comparisons:
            groups[slicer(c)].append(c)

        for cat in sorted(groups.keys()):
            items = groups[cat]
            n = len(items)
            b5_wins = sum(1 for c in items if c.diff > 0)
            b4_pnl = sum(c.bar4_pnl for c in items)
            b5_pnl = sum(c.bar5_pnl for c in items)
            diff = b5_pnl - b4_pnl

            if n >= 3 and b5_wins / n >= 0.6 and diff > 0:
                sig = "→ USE BAR 5"
            elif n >= 3 and b5_wins / n <= 0.4 and diff < 0:
                sig = "→ KEEP BAR 4"
            else:
                sig = ""

            print(f"  {name[:10]+': '+cat:<20} {n:>5} {b5_wins:>5}/{n:<1} {b4_pnl:>+7.1f} {b5_pnl:>+7.1f} {diff:>+7.1f} {sig}")
        print()

    # Cross-tabulations
    combos = [
        ("Gap + Range", lambda c: f"{c.gap_dir} + {c.range_class}"),
        ("Gap + Context", lambda c: f"{c.gap_dir} + {c.context}"),
        ("Day + Gap", lambda c: f"{c.day_of_week[:3]} + {c.gap_dir}"),
        ("Context + Range", lambda c: f"{c.context} + {c.range_class}"),
        ("Context + Bar4 Char", lambda c: f"{c.context} + {'BULL' if c.bar4_bullish else 'BEAR'}"),
        ("Day + Range", lambda c: f"{c.day_of_week[:3]} + {c.range_class}"),
        ("Day + Context", lambda c: f"{c.day_of_week[:3]} + {c.context}"),
        ("V58 + Context", lambda c: f"{c.overnight_bias or 'N/A'} + {c.context}"),
        ("V58 + Range", lambda c: f"{c.overnight_bias or 'N/A'} + {c.range_class}"),
        ("V58 + Gap", lambda c: f"{c.overnight_bias or 'N/A'} + {c.gap_dir}"),
    ]

    print("\n── Cross-Tabulations (min 3 days) ─────────────────────────────────")
    print(f"{'Combo':<32} {'Days':>5} {'B5>B4':>6} {'B4 P&L':>8} {'B5 P&L':>8} {'Diff':>8} {'Signal'}")
    print("─" * 85)

    # Collect all "USE BAR 5" rules
    bar5_rules = []

    for name, slicer in combos:
        groups = defaultdict(list)
        for c in comparisons:
            groups[slicer(c)].append(c)

        sorted_groups = sorted(groups.items(), key=lambda x: sum(c.bar5_pnl - c.bar4_pnl for c in x[1]), reverse=True)

        for cat, items in sorted_groups:
            n = len(items)
            if n < 3:
                continue
            b5_wins = sum(1 for c in items if c.diff > 0)
            b4_pnl = sum(c.bar4_pnl for c in items)
            b5_pnl = sum(c.bar5_pnl for c in items)
            diff = b5_pnl - b4_pnl
            pct = b5_wins / n

            if pct >= 0.6 and diff > 0:
                sig = "✅ USE BAR 5"
                bar5_rules.append({
                    "combo": cat, "source": name, "sample": n,
                    "b5_win_pct": round(pct * 100), "diff": diff,
                    "b4_pnl": b4_pnl, "b5_pnl": b5_pnl,
                })
            elif pct <= 0.4 and diff < 0:
                sig = "🔒 KEEP BAR 4"
            else:
                sig = ""

            print(f"  {cat:<30} {n:>5} {b5_wins:>5}/{n:<1} {b4_pnl:>+7.1f} {b5_pnl:>+7.1f} {diff:>+7.1f} {sig}")

    # Actionable rules summary
    print(f"\n\n{'═' * 85}")
    print("ACTIONABLE RULES — When bar 4 has these conditions, use bar 5 instead")
    print(f"{'═' * 85}\n")

    if not bar5_rules:
        print("  No strong bar 5 rules found with sufficient sample size.\n")
    else:
        # Sort by diff descending
        bar5_rules.sort(key=lambda r: r["diff"], reverse=True)
        for i, rule in enumerate(bar5_rules, 1):
            print(f"  Rule {i}: When [{rule['combo']}]")
            print(f"          → USE BAR 5 ({rule['sample']} days, "
                  f"bar5 better {rule['b5_win_pct']}% of time, "
                  f"B4: {rule['b4_pnl']:+.0f} → B5: {rule['b5_pnl']:+.0f}, "
                  f"gain: {rule['diff']:+.0f} pts)\n")

    # Simulate combined: use bar 5 when any rule matches, else bar 4
    rule_combos = {r["combo"] for r in bar5_rules}
    combo_funcs = [(name, slicer) for name, slicer in combos]

    def day_matches_rule(c):
        for _, slicer in combo_funcs:
            if slicer(c) in rule_combos:
                return True
        return False

    combined_pnl = 0
    bar5_used = 0
    for c in comparisons:
        if day_matches_rule(c):
            combined_pnl += c.bar5_pnl
            bar5_used += 1
        else:
            combined_pnl += c.bar4_pnl

    print(f"── Simulated Impact (applying all rules above) ────────────────────")
    print(f"  Days using bar 5:    {bar5_used}/{len(comparisons)}")
    print(f"  Original (bar 4):    {total_b4:>+8.1f} pts")
    print(f"  With bar 5 rules:    {combined_pnl:>+8.1f} pts")
    print(f"  Improvement:         {combined_pnl - total_b4:>+8.1f} pts "
          f"({(combined_pnl/total_b4 - 1)*100 if total_b4 else 0:+.1f}%)\n")

    return bar5_rules


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[DayResult]):
    """Print overall backtest summary."""
    all_trades = [t for r in results for t in r.trades]
    triggered_days = [r for r in results if r.triggered]
    win_trades = [t for t in all_trades if t.pnl_pts > 0]
    lose_trades = [t for t in all_trades if t.pnl_pts < 0]
    flat_trades = [t for t in all_trades if t.pnl_pts == 0]

    total_pnl = sum(t.pnl_pts for t in all_trades)
    win_days = [r for r in results if r.total_pnl > 0]
    lose_days = [r for r in results if r.total_pnl < 0]

    # Streaks
    max_win_streak = max_lose_streak = 0
    cur_win = cur_lose = 0
    for r in triggered_days:
        if r.total_pnl > 0:
            cur_win += 1
            cur_lose = 0
            max_win_streak = max(max_win_streak, cur_win)
        elif r.total_pnl < 0:
            cur_lose += 1
            cur_win = 0
            max_lose_streak = max(max_lose_streak, cur_lose)
        else:
            cur_win = cur_lose = 0

    # Drawdown
    equity = []
    running = 0
    for r in results:
        running += r.total_pnl
        equity.append(running)
    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  ASRS BACKTEST RESULTS                       ║
╠══════════════════════════════════════════════════════════════╣
║  Period:     {results[0].date} → {results[-1].date:<30}║
║  Days:       {len(results)} total, {len(triggered_days)} triggered ({round(len(triggered_days)/len(results)*100)}%){' ' * 18}║
╠══════════════════════════════════════════════════════════════╣
║  TRADES                                                      ║
║  Total:      {len(all_trades):<47}║
║  Winners:    {len(win_trades)} ({round(len(win_trades)/len(all_trades)*100) if all_trades else 0}%){' ' * 38}║
║  Losers:     {len(lose_trades)} ({round(len(lose_trades)/len(all_trades)*100) if all_trades else 0}%){' ' * 38}║
║  Avg win:    {round(np.mean([t.pnl_pts for t in win_trades]), 1) if win_trades else 0:+.1f} pts{' ' * 37}║
║  Avg loss:   {round(np.mean([t.pnl_pts for t in lose_trades]), 1) if lose_trades else 0:+.1f} pts{' ' * 37}║
║  Best trade: {max(t.pnl_pts for t in all_trades) if all_trades else 0:+.1f} pts{' ' * 37}║
║  Worst trade:{min(t.pnl_pts for t in all_trades) if all_trades else 0:+.1f} pts{' ' * 37}║
╠══════════════════════════════════════════════════════════════╣
║  DAYS                                                        ║
║  Win days:   {len(win_days)} ({round(len(win_days)/len(triggered_days)*100) if triggered_days else 0}%){' ' * 38}║
║  Lose days:  {len(lose_days)} ({round(len(lose_days)/len(triggered_days)*100) if triggered_days else 0}%){' ' * 38}║
║  Best day:   {max(r.total_pnl for r in results):+.1f} pts{' ' * 37}║
║  Worst day:  {min(r.total_pnl for r in results):+.1f} pts{' ' * 37}║
╠══════════════════════════════════════════════════════════════╣
║  P&L                                                         ║
║  Total:      {total_pnl:+.1f} pts{' ' * 40}║
║  Daily avg:  {round(total_pnl / len(triggered_days), 1) if triggered_days else 0:+.1f} pts{' ' * 37}║
║  Monthly:    ~{round(total_pnl / max(1, (results[-1].date[:7] != results[0].date[:7]) + len(set(r.date[:7] for r in results))), 1):+.1f} pts/month{' ' * 30}║
╠══════════════════════════════════════════════════════════════╣
║  RISK                                                        ║
║  Max DD:     {max_dd:.1f} pts{' ' * 41}║
║  Win streak: {max_win_streak} days{' ' * 41}║
║  Lose streak:{max_lose_streak} days{' ' * 41}║
║  Avg MFE:    {round(np.mean([t.mfe_pts for t in all_trades]), 1) if all_trades else 0:.1f} pts (how far it went for you){' ' * 13}║
║  Avg MAE:    {round(np.mean([t.mae_pts for t in all_trades]), 1) if all_trades else 0:.1f} pts (how far it went against){' ' * 12}║
╠══════════════════════════════════════════════════════════════╣
║  SCALING                                                     ║
║  1 micro (€1/pt):   {total_pnl:+.0f} EUR{' ' * 33}║
║  5 micro (€5/pt):   {total_pnl * 5:+.0f} EUR{' ' * 32}║
║  1 mini  (€5/pt):   {total_pnl * 5:+.0f} EUR{' ' * 32}║
║  2 mini  (€10/pt):  {total_pnl * 10:+.0f} EUR{' ' * 31}║
║  10 mini (€50/pt):  {total_pnl * 50:+.0f} EUR{' ' * 30}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Monthly breakdown
    print("\n── Monthly Breakdown ──────────────────────────────────")
    print(f"{'Month':<10} {'Days':>5} {'Trades':>7} {'Win%':>6} {'P&L':>8} {'Cum':>8}")
    print("─" * 55)

    monthly = defaultdict(lambda: {"days": 0, "trades": 0, "wins": 0, "pnl": 0})
    for r in results:
        m = r.date[:7]
        monthly[m]["days"] += 1
        monthly[m]["trades"] += len(r.trades)
        monthly[m]["wins"] += sum(1 for t in r.trades if t.pnl_pts > 0)
        monthly[m]["pnl"] += r.total_pnl

    cum = 0
    for m in sorted(monthly.keys()):
        d = monthly[m]
        cum += d["pnl"]
        wr = round(d["wins"] / d["trades"] * 100) if d["trades"] else 0
        icon = "🟢" if d["pnl"] >= 0 else "🔴"
        print(f"{m:<10} {d['days']:>5} {d['trades']:>7} {wr:>5}% {d['pnl']:>+7.1f} {cum:>+7.1f} {icon}")


def print_situational(results: list[DayResult]):
    """
    Situational analysis — slice by every condition combination.
    Shows which conditions favour long, short, or staying out.
    """
    all_trades = [t for r in results for t in r.trades]
    if not all_trades:
        print("No trades to analyse.")
        return

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              SITUATIONAL ANALYSIS                            ║
║  "When does the strategy work best?"                         ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Define slicers ─────────────────────────────────────────────────
    slicers = {
        "Day of Week": lambda t: t.day_of_week,
        "Gap Direction": lambda t: t.gap_dir,
        "Bar 4 Range": lambda t: t.range_class,
        "Context (Bars 1-3)": lambda t: t.context,
        "Bar 4 Character": lambda t: "BULLISH" if t.bar_bullish else "BEARISH",
        "Trade Direction": lambda t: t.direction,
        "Entry Number": lambda t: f"Entry #{t.entry_num}",
        "Previous Day": lambda t: t.prev_day_dir or "N/A",
        "V58 Overnight Bias": lambda t: t.overnight_bias or "NO_DATA",
        "Bar4 vs Overnight": lambda t: t.bar4_vs_overnight or "NO_DATA",
    }

    for name, slicer in slicers.items():
        print(f"\n── {name} ─────────────────────────────────────────")
        print(f"{'Category':<16} {'Trades':>7} {'Win%':>6} {'Avg P&L':>8} {'Total':>8} {'Avg MFE':>8} {'Edge'}")
        print("─" * 75)

        groups = defaultdict(list)
        for t in all_trades:
            groups[slicer(t)].append(t)

        for cat in sorted(groups.keys()):
            trades = groups[cat]
            n = len(trades)
            wins = sum(1 for t in trades if t.pnl_pts > 0)
            avg_pnl = np.mean([t.pnl_pts for t in trades])
            total = sum(t.pnl_pts for t in trades)
            avg_mfe = np.mean([t.mfe_pts for t in trades])
            wr = round(wins / n * 100) if n else 0

            if avg_pnl > 3:
                edge = "🟢 STRONG"
            elif avg_pnl > 0:
                edge = "🟡 WEAK+"
            elif avg_pnl > -3:
                edge = "🟡 WEAK-"
            else:
                edge = "🔴 AVOID"

            print(f"{cat:<16} {n:>7} {wr:>5}% {avg_pnl:>+7.1f} {total:>+7.1f} {avg_mfe:>7.1f} {edge}")

    # ── Cross-tabulations (the gold) ───────────────────────────────────
    print(f"\n\n{'═' * 75}")
    print("CROSS-TABULATIONS — Condition Combos")
    print(f"{'═' * 75}")

    combos = [
        ("Day + Gap", lambda t: f"{t.day_of_week[:3]} + {t.gap_dir}"),
        ("Day + Direction", lambda t: f"{t.day_of_week[:3]} + {t.direction}"),
        ("Gap + Range", lambda t: f"{t.gap_dir} + {t.range_class}"),
        ("Gap + Context", lambda t: f"{t.gap_dir} + {t.context}"),
        ("Gap + Bar4 Char", lambda t: f"{t.gap_dir} + {'BULL' if t.bar_bullish else 'BEAR'} bar"),
        ("Context + Range", lambda t: f"{t.context} + {t.range_class}"),
        ("Context + Direction", lambda t: f"{t.context} + {t.direction}"),
        ("PrevDay + Gap", lambda t: f"Prev:{t.prev_day_dir or 'N/A'} + {t.gap_dir}"),
        # V58 overnight combos
        ("V58 Bias + Direction", lambda t: f"{t.overnight_bias or 'N/A'} + {t.direction}"),
        ("V58 Bias + Gap", lambda t: f"{t.overnight_bias or 'N/A'} + {t.gap_dir}"),
        ("V58 Bias + Day", lambda t: f"{t.overnight_bias or 'N/A'} + {t.day_of_week[:3]}"),
        ("V58 Bias + Context", lambda t: f"{t.overnight_bias or 'N/A'} + {t.context}"),
        ("V58 Bias + Range", lambda t: f"{t.overnight_bias or 'N/A'} + {t.range_class}"),
    ]

    for name, slicer in combos:
        print(f"\n── {name} ─────────────────────────────────────")
        print(f"{'Combo':<30} {'N':>4} {'Win%':>6} {'Avg':>7} {'Total':>7} {'Signal'}")
        print("─" * 70)

        groups = defaultdict(list)
        for t in all_trades:
            groups[slicer(t)].append(t)

        # Sort by total P&L
        sorted_groups = sorted(groups.items(), key=lambda x: sum(t.pnl_pts for t in x[1]), reverse=True)

        for cat, trades in sorted_groups:
            n = len(trades)
            if n < 3:
                continue  # Need minimum sample
            wins = sum(1 for t in trades if t.pnl_pts > 0)
            avg_pnl = np.mean([t.pnl_pts for t in trades])
            total = sum(t.pnl_pts for t in trades)
            wr = round(wins / n * 100)

            if wr >= 60 and avg_pnl > 3:
                sig = "✅ TRADE"
            elif wr <= 40 or avg_pnl < -3:
                sig = "❌ SKIP"
            else:
                sig = "⚠️  WEAK"

            print(f"{cat:<30} {n:>4} {wr:>5}% {avg_pnl:>+6.1f} {total:>+6.1f} {sig}")

    # ── Actionable rules ───────────────────────────────────────────────
    print(f"\n\n{'═' * 75}")
    print("ACTIONABLE RULES — Conditions with >60% win rate AND avg P&L > 3 pts")
    print(f"{'═' * 75}\n")

    rule_count = 0
    for name, slicer in combos:
        groups = defaultdict(list)
        for t in all_trades:
            groups[slicer(t)].append(t)

        for cat, trades in groups.items():
            n = len(trades)
            if n < 5:  # Minimum 5 trades for a rule
                continue
            wins = sum(1 for t in trades if t.pnl_pts > 0)
            avg_pnl = np.mean([t.pnl_pts for t in trades])
            wr = round(wins / n * 100)
            total = sum(t.pnl_pts for t in trades)

            if wr >= 60 and avg_pnl > 3:
                rule_count += 1
                print(f"  ✅ Rule {rule_count}: When [{cat}] → "
                      f"TRADE ({n} occurrences, {wr}% win, {avg_pnl:+.1f} avg, {total:+.1f} total)")

            elif wr <= 35 and avg_pnl < -3 and n >= 5:
                rule_count += 1
                print(f"  ❌ Rule {rule_count}: When [{cat}] → "
                      f"SKIP ({n} occurrences, {wr}% win, {avg_pnl:+.1f} avg, {total:+.1f} total)")

    if rule_count == 0:
        print("  No strong rules found with sufficient sample size.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════

def print_trade_log(results: list[DayResult], month_filter: str = ""):
    """Print detailed day-by-day log."""
    print(f"\n{'Date':<12} {'Day':<4} {'Range':>5} {'Gap':>8} {'Ctx':<6} "
          f"{'T':>2} {'P&L':>7} {'Details'}")
    print("─" * 80)

    for r in results:
        if month_filter and not r.date.startswith(month_filter):
            continue

        trades_str = " | ".join(
            f"{t.direction[0]}:{t.pnl_pts:+.0f}"
            for t in r.trades
        ) if r.trades else "—"

        icon = "🟢" if r.total_pnl > 0 else ("🔴" if r.total_pnl < 0 else "⬜")
        print(f"{r.date:<12} {r.day_of_week[:3]:<4} {r.bar4_range:>4.0f}p "
              f"{r.gap_dir:<8} {r.context[:6]:<6} "
              f"{len(r.trades):>2} {r.total_pnl:>+6.1f} {icon} {trades_str}")


def export_csv(results: list[DayResult]):
    """Export all trades to CSV for further analysis."""
    all_trades = [t for r in results for t in r.trades]
    if not all_trades:
        print("No trades to export.")
        return

    rows = []
    for t in all_trades:
        rows.append({
            "date": t.date, "day": t.day_of_week, "direction": t.direction,
            "entry_num": t.entry_num, "entry": t.entry_price, "exit": t.exit_price,
            "pnl_pts": t.pnl_pts, "mfe_pts": t.mfe_pts, "mae_pts": t.mae_pts,
            "held_bars": t.held_bars, "held_to_close": t.held_to_close,
            "bar4_range": t.bar_range, "range_class": t.range_class,
            "gap_dir": t.gap_dir, "gap_size": t.gap_size,
            "context": t.context, "bar4_bullish": t.bar_bullish,
            "prev_day": t.prev_day_dir,
            "overnight_bias": t.overnight_bias,
            "bar4_vs_overnight": t.bar4_vs_overnight,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "backtest_trades.csv")
    df.to_csv(path, index=False)
    print(f"Exported {len(rows)} trades to {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Try cached data first, otherwise fetch from IBKR
    rth_df, all_df = load_cached_data()
    if rth_df is None or "--refresh" in args:
        logger.info("Fetching data from IBKR...")
        b = Broker()
        rth_df, all_df = await fetch_2y_data(b)
        if rth_df.empty:
            print("No data available. Is IB Gateway running?")
            return
        save_cached_data(rth_df, all_df)

    # Run backtest (pass all_df for overnight range V58 analysis)
    logger.info("Running backtest...")
    results = run_backtest(rth_df, all_df)
    logger.info(f"Processed {len(results)} trading days")

    # Month filter
    month_filter = ""
    if "--month" in args:
        idx = args.index("--month")
        if idx + 1 < len(args):
            month_filter = args[idx + 1]

    # Generate charts if requested (or always with backtest)
    generate_charts = "--charts" in args or "--generate-rules" in args or len(args) == 0
    if generate_charts or "--charts" in args:
        from charts import generate_all_charts
        chart_paths = generate_all_charts(results)
        if chart_paths:
            print(f"\n📊 Generated {len(chart_paths)} charts in data/charts/")
            for p in chart_paths:
                print(f"   {os.path.basename(p)}")

    # Output — order matters: compound flags before simple ones
    if "--monte-carlo" in args:
        bar = 5 if "--bar5" in args else 4
        logger.info(f"Running EMA Trail + Adds backtest (bar {bar}) for Monte Carlo...")
        ema_adds = run_backtest_ema(rth_df, all_df, enable_adds=True, signal_bar=bar)
        capital = 5000
        if "--capital" in args:
            ci = args.index("--capital")
            if ci + 1 < len(args):
                capital = float(args[ci + 1])
        sims = 10000
        if "--sims" in args:
            si = args.index("--sims")
            if si + 1 < len(args):
                sims = int(args[si + 1])
        print_summary(ema_adds)
        run_monte_carlo(ema_adds, num_sims=sims, starting_capital=capital)

    elif "--bar5-full-compare" in args:
        logger.info("Running bar 5 full comparison (Original bar5 / EMA bar5 / EMA+Adds bar5)...")
        orig_b5 = run_backtest_ema(rth_df, all_df, enable_adds=False, signal_bar=5)
        ema_b5 = run_backtest_ema(rth_df, all_df, enable_adds=False, signal_bar=5)
        ema_adds_b5 = run_backtest_ema(rth_df, all_df, enable_adds=True, signal_bar=5)
        logger.info(f"Bar5 EMA: {len(ema_b5)} days, Bar5 EMA+Adds: {len(ema_adds_b5)} days")
        print_full_compare(orig_b5, ema_b5, ema_adds_b5)

    elif "--full-compare" in args:
        bar = 5 if "--bar5" in args else 4
        logger.info(f"Running all three variants (bar {bar})...")
        ema_only = run_backtest_ema(rth_df, all_df, enable_adds=False, signal_bar=bar)
        ema_adds = run_backtest_ema(rth_df, all_df, enable_adds=True, signal_bar=bar)
        logger.info(f"Original: {len(results)} days, EMA: {len(ema_only)}, EMA+Adds: {len(ema_adds)}")
        print_full_compare(results, ema_only, ema_adds)

    elif "--trail-compare" in args:
        logger.info("Running EMA trail backtest...")
        ema_results = run_backtest_ema(rth_df, all_df, enable_adds=False)
        logger.info(f"EMA: {len(ema_results)} days")
        print_full_compare(results, ema_results, ema_results)

    elif "--add-to-winners" in args:
        logger.info("Running EMA trail + adds backtest...")
        ema_only = run_backtest_ema(rth_df, all_df, enable_adds=False)
        ema_adds = run_backtest_ema(rth_df, all_df, enable_adds=True)
        logger.info(f"EMA: {len(ema_only)} days, EMA+Adds: {len(ema_adds)} days")
        print_full_compare(results, ema_only, ema_adds)

    elif "--bar5" in args:
        bar5_results = run_backtest_bar5_fallback(rth_df, all_df)
        print_summary(results)
        print_bar5_fallback(results, bar5_results, label="Skipped Days")

    elif "--bar5-losers" in args:
        loser_dates = {r.date for r in results if r.triggered and r.total_pnl < 0}
        bar5_results = run_backtest_bar5_fallback(rth_df, all_df, target_dates=loser_dates)
        print_summary(results)
        print_bar5_fallback(results, bar5_results, label="Losing Days")

    elif "--bar5-conditions" in args:
        analyse_bar5_conditions(rth_df, all_df)

    elif "--filtered" in args:
        # Re-run with skip rules applied retroactively
        all_trades = [t for r in results for t in r.trades]
        from rules import generate_rules_from_backtest, load_rules
        rules = generate_rules_from_backtest(all_trades)
        skip_combos = set(r["combo"] for r in rules.get("skip_rules", []))

        def trade_combos(t):
            """Generate all condition combos for a trade."""
            day = t.day_of_week[:3]
            bar_char = "BULL" if t.bar_bullish else "BEAR"
            ov = t.overnight_bias or "NO_DATA"
            return [
                f"{day} + {t.gap_dir}", f"{day} + {t.direction}",
                f"{t.gap_dir} + {t.range_class}", f"{t.gap_dir} + {t.context}",
                f"{t.gap_dir} + {bar_char} bar", f"{t.context} + {t.range_class}",
                f"{t.context} + {t.direction}",
                f"Prev:{t.prev_day_dir or 'N/A'} + {t.gap_dir}",
                f"{ov} + {t.direction}", f"{ov} + {t.gap_dir}",
                f"{ov} + {day}", f"{ov} + {t.context}", f"{ov} + {t.range_class}",
                f"{day} + {t.range_class}", f"{day} + {t.context}",
                f"Prev:{t.prev_day_dir or 'N/A'} + {t.context}",
                f"Prev:{t.prev_day_dir or 'N/A'} + {t.range_class}",
            ]

        def should_skip(t):
            combos = trade_combos(t)
            return any(c in skip_combos for c in combos)

        # Build filtered results
        filtered_results = []
        for r in results:
            kept = [t for t in r.trades if not should_skip(t)]
            skipped = [t for t in r.trades if should_skip(t)]
            from copy import deepcopy
            fr = deepcopy(r)
            fr.trades = kept
            fr.total_pnl = round(sum(t.pnl_pts for t in kept), 1)
            fr.triggered = len(kept) > 0
            filtered_results.append(fr)

        # Raw stats
        raw_trades = all_trades
        raw_pnl = sum(t.pnl_pts for t in raw_trades)
        raw_wins = sum(1 for t in raw_trades if t.pnl_pts > 0)

        # Filtered stats
        filt_trades = [t for r in filtered_results for t in r.trades]
        filt_pnl = sum(t.pnl_pts for t in filt_trades)
        filt_wins = sum(1 for t in filt_trades if t.pnl_pts > 0)
        skipped_trades = [t for t in raw_trades if should_skip(t)]
        skip_pnl = sum(t.pnl_pts for t in skipped_trades)

        # Equity curves
        raw_equity, filt_equity = [], []
        raw_run, filt_run = 0, 0
        for r, fr in zip(results, filtered_results):
            raw_run += r.total_pnl
            filt_run += fr.total_pnl
            raw_equity.append(raw_run)
            filt_equity.append(filt_run)

        # Max drawdowns
        def calc_dd(eq):
            pk, dd = 0, 0
            for e in eq:
                pk = max(pk, e)
                dd = max(dd, pk - e)
            return dd

        raw_wr = round(raw_wins / len(raw_trades) * 100) if raw_trades else 0
        filt_wr = round(filt_wins / len(filt_trades) * 100) if filt_trades else 0
        raw_avg = np.mean([t.pnl_pts for t in raw_trades]) if raw_trades else 0
        filt_avg = np.mean([t.pnl_pts for t in filt_trades]) if filt_trades else 0

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║         RAW vs FILTERED (Skip Rules Applied)                 ║
╠══════════════════════════════════════════════════════════════╣
║                    RAW           FILTERED        SKIPPED     ║
║  Trades:     {len(raw_trades):>6}         {len(filt_trades):>6}           {len(skipped_trades):>6}      ║
║  Win rate:   {raw_wr:>5}%        {filt_wr:>5}%                      ║
║  Avg P&L:    {raw_avg:>+7.1f}       {filt_avg:>+7.1f}         {np.mean([t.pnl_pts for t in skipped_trades]) if skipped_trades else 0:>+7.1f}     ║
║  Total P&L:  {raw_pnl:>+7.0f}       {filt_pnl:>+7.0f}         {skip_pnl:>+7.0f}     ║
║  Max DD:     {calc_dd(raw_equity):>6.0f}        {calc_dd(filt_equity):>6.0f}                       ║
╠══════════════════════════════════════════════════════════════╣
║  IMPROVEMENT                                                 ║
║  P&L boost:  {filt_pnl - raw_pnl:>+7.0f} pts ({(filt_pnl/raw_pnl - 1)*100 if raw_pnl else 0:>+.0f}%)                           ║
║  DD reduced: {calc_dd(raw_equity) - calc_dd(filt_equity):>+7.0f} pts                                    ║
║  Trades cut: {len(raw_trades) - len(filt_trades):>6} ({(len(raw_trades)-len(filt_trades))/len(raw_trades)*100 if raw_trades else 0:.0f}%)                                  ║
╚══════════════════════════════════════════════════════════════╝""")

        # Show what was skipped
        print(f"\n── Skipped Trades Detail ──")
        print(f"{'Date':<12} {'Dir':<6} {'P&L':>7} {'Reason skipped'}")
        print("─" * 60)
        for t in skipped_trades:
            matched = [c for c in trade_combos(t) if c in skip_combos]
            print(f"{t.date:<12} {t.direction:<6} {t.pnl_pts:>+6.1f} {matched[0] if matched else '?'}")

        # Generate filtered equity chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(14, 6))
            dates = [datetime.strptime(r.date, "%Y-%m-%d") for r in results]
            ax.plot(dates, raw_equity, color="#9E9E9E", linewidth=1, alpha=0.7, label=f"Raw ({raw_pnl:+.0f} pts)")
            ax.plot(dates, filt_equity, color="#4CAF50", linewidth=2, label=f"Filtered ({filt_pnl:+.0f} pts)")
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title("ASRS: Raw vs Filtered (Skip Rules Applied)", fontsize=16, fontweight="bold")
            ax.set_ylabel("Cumulative P&L (points)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.xticks(rotation=45)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = os.path.join(RESULTS_DIR, "charts", "filtered_equity.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n📊 Filtered equity chart saved to data/charts/filtered_equity.png")
        except Exception as e:
            print(f"Chart error: {e}")

        # Also show filtered summary
        print_summary(filtered_results)

    elif "--generate-rules" in args or "-g" in args:
        all_trades = [t for r in results for t in r.trades]
        from rules import generate_rules_from_backtest
        rules = generate_rules_from_backtest(all_trades)
        print_summary(results)
        print(f"\n✅ Generated {len(rules['trade_rules'])} TRADE rules, "
              f"{len(rules['skip_rules'])} SKIP rules")
        print(f"   Saved to data/situational_rules.json")
        print(f"\n   TRADE rules (auto-execute):")
        for r in rules["trade_rules"][:10]:
            print(f"     ✅ [{r['combo']}] — {r['sample']} trades, "
                  f"{r['win_rate']}% win, {r['avg_pnl']:+.1f} avg")
        print(f"\n   SKIP rules (auto-skip):")
        for r in rules["skip_rules"][:10]:
            print(f"     ❌ [{r['combo']}] — {r['sample']} trades, "
                  f"{r['win_rate']}% win, {r['avg_pnl']:+.1f} avg")
    elif "--situational" in args or "-s" in args:
        print_summary(results)
        print_situational(results)
    elif "--export" in args:
        export_csv(results)
        print_summary(results)
    elif "--log" in args:
        print_trade_log(results, month_filter)
    elif "--month" in args:
        filtered = [r for r in results if r.date.startswith(month_filter)]
        if filtered:
            print_summary(filtered)
            print_trade_log(filtered)
        else:
            print(f"No data for {month_filter}")
    else:
        print_summary(results)
        print_trade_log(results, month_filter)


if __name__ == "__main__":
    asyncio.run(main())
