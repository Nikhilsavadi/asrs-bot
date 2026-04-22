#!/usr/bin/env python3
"""
backtest_momentum_scalper.py
----------------------------
7-factor momentum scalping backtest on DAX (FDAX) + US30 (YM), 5-min bars,
last 5 years of firstrate futures data. Per-session indicator reset,
35-bar warmup, 1% fixed-fractional risk, ATR-based SL/TP/trail,
15% circuit breaker, 90-min time stop, no overnight positions.

Runs 2x2 = 4 passes:
    base / with-slippage   x   train (oldest 70%) / test (newest 30%)

Outputs (written alongside this script, prefix `scalper_`):
    scalper_trades_{pass}.csv           per-trade log
    scalper_equity_{inst}_{pass}.csv    timestamp,equity,drawdown_pct
    scalper_monthly_{pass}.csv          monthly % return + trade count
    scalper_summary.csv                 metric block for every pass
    scalper_factors_{pass}.csv          factor binding-constraint counts

Each instrument runs as an independent £10k sub-account (max 1 pos/inst,
max 2 total is therefore automatic since we run them in isolation).
"""

import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    print("ERROR: pandas_ta not installed. `pip install pandas_ta`", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================

STARTING_EQUITY = 10_000.0   # GBP per instrument
RISK_PER_TRADE = 0.01        # 1% of equity per trade
CIRCUIT_BREAKER_DD = 0.15    # halt day on 15% drawdown, reset next session
TIME_STOP_MIN = 90           # close after 90 minutes if not otherwise exited
WARMUP_BARS = 35             # per-session bars required before entries valid
YEARS_HISTORY = 5
TRAIN_FRAC = 0.70            # 70% oldest = train, 30% newest = test
TARGET_TZ = "Europe/Berlin"  # CET/CEST

OUT_DIR = Path(__file__).parent

INSTRUMENTS = {
    "DAX": {
        "file": "/root/asrs-bot/data/firstrate/FDAX_full_5min_continuous_ratio_adjusted.txt",
        "src_tz": "Europe/Berlin",
        "session_start_hm": (8, 0),
        "session_end_hm":   (16, 30),
        "entry_start_hm":   (8, 30),
        "entry_cutoff_before_close_min": 30,
        "slippage_pts": 2.0,
    },
    "US30": {
        "file": "/root/asrs-bot/data/firstrate/YM_full_5min_continuous_ratio_adjusted.txt",
        "src_tz": "America/New_York",
        "session_start_hm": (14, 30),
        "session_end_hm":   (21, 0),
        "entry_start_hm":   (14, 30),
        "entry_cutoff_before_close_min": 30,
        "slippage_pts": 3.0,
    },
}

FACTORS = ["ema_stack", "trend", "macd_cross", "rsi", "adx", "volume", "bb_break"]

# ============================================================================
# DATA LOAD
# ============================================================================

def load_firstrate(path: str, src_tz: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: data file not found: {p}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(
        path, header=None,
        names=["dt", "open", "high", "low", "close", "volume"],
    )
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").sort_index()
    df.index = df.index.tz_localize(ZoneInfo(src_tz)).tz_convert(ZoneInfo(TARGET_TZ))
    df = df[df.index.dayofweek < 5]
    return df


def slice_session(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sh, sm = cfg["session_start_hm"]
    eh, em = cfg["session_end_hm"]
    tmin = df.index.hour * 60 + df.index.minute
    return df[(tmin >= sh * 60 + sm) & (tmin <= eh * 60 + em)].copy()


# ============================================================================
# INDICATORS (per-session reset)
# ============================================================================

def _safe_series(res, index):
    if res is None:
        return pd.Series(np.nan, index=index)
    return res


def compute_indicators_per_session(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for i, (_d, g) in enumerate(df.groupby(df.index.date, sort=False)):
        g = g.copy()
        idx = g.index
        g["ema5"]  = _safe_series(ta.ema(g["close"], length=5), idx)
        g["ema13"] = _safe_series(ta.ema(g["close"], length=13), idx)
        g["ema34"] = _safe_series(ta.ema(g["close"], length=34), idx)
        g["ema50"] = _safe_series(ta.ema(g["close"], length=50), idx)
        g["rsi"]   = _safe_series(ta.rsi(g["close"], length=14), idx)

        macd = ta.macd(g["close"], fast=12, slow=26, signal=9)
        g["macd_hist"] = macd["MACDh_12_26_9"] if macd is not None else pd.Series(np.nan, index=idx)

        adx = ta.adx(g["high"], g["low"], g["close"], length=14)
        g["adx"] = adx["ADX_14"] if adx is not None else pd.Series(np.nan, index=idx)

        bb = ta.bbands(g["close"], length=20, std=2)
        if bb is not None:
            g["bb_upper"] = bb["BBU_20_2.0_2.0"]
            g["bb_lower"] = bb["BBL_20_2.0_2.0"]
            g["bb_width"] = bb["BBU_20_2.0_2.0"] - bb["BBL_20_2.0_2.0"]
        else:
            g["bb_upper"] = pd.Series(np.nan, index=idx)
            g["bb_lower"] = pd.Series(np.nan, index=idx)
            g["bb_width"] = pd.Series(np.nan, index=idx)

        g["bb_width_med20"] = g["bb_width"].rolling(20).median()
        g["atr"] = _safe_series(ta.atr(g["high"], g["low"], g["close"], length=14), idx)
        g["volma20"] = g["volume"].rolling(20).mean()
        g["session_bar_idx"] = np.arange(len(g))
        parts.append(g)
        if (i + 1) % 500 == 0:
            print(f"    indicators: {i + 1} sessions processed")
    return pd.concat(parts)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    ema_stack = (df["ema5"] > df["ema13"]) & (df["ema13"] > df["ema34"])
    trend = df["close"] > df["ema50"]

    h = df["macd_hist"]
    h_1 = h.shift(1)
    h_2 = h.shift(2)
    macd_cross = (h > 0) & ((h_1 <= 0) | (h_2 <= 0))

    rsi = (df["rsi"] >= 50) & (df["rsi"] <= 82)
    adx = df["adx"] >= 25
    volume = df["volume"] >= 2 * df["volma20"]

    squeezed = (df["bb_width"].shift(1) < df["bb_width_med20"].shift(1)) & \
               (df["bb_width"].shift(2) < df["bb_width_med20"].shift(2))
    bb_break = squeezed & (df["close"] > df["bb_upper"])

    return pd.DataFrame({
        "ema_stack":  ema_stack.fillna(False).astype(bool),
        "trend":      trend.fillna(False).astype(bool),
        "macd_cross": macd_cross.fillna(False).astype(bool),
        "rsi":        rsi.fillna(False).astype(bool),
        "adx":        adx.fillna(False).astype(bool),
        "volume":     volume.fillna(False).astype(bool),
        "bb_break":   bb_break.fillna(False).astype(bool),
    }, index=df.index)


# ============================================================================
# BACKTEST LOOP
# ============================================================================

def run_backtest(inst: str, df: pd.DataFrame, fac: pd.DataFrame,
                 slippage_pts: float, equity_start: float) -> tuple[list, list]:
    cfg = INSTRUMENTS[inst]
    sess_end_min = cfg["session_end_hm"][0] * 60 + cfg["session_end_hm"][1]
    entry_cutoff_min = sess_end_min - cfg["entry_cutoff_before_close_min"]

    idx = df.index
    closes = df["close"].values
    highs = df["high"].values
    lows  = df["low"].values
    atrs  = df["atr"].values
    volmas = df["volma20"].values
    vols  = df["volume"].values
    sbi   = df["session_bar_idx"].values

    fac_arr = fac[FACTORS].values.astype(bool)
    prev_fac_arr = np.vstack([np.zeros((1, 7), bool), fac_arr[:-1]])
    fac_all = fac_arr.all(axis=1)

    equity = equity_start
    peak   = equity_start
    curve  = []
    trades = []

    pos = None  # dict or None
    circuit_day = None
    current_day = None
    n = len(df)

    for i in range(n):
        t = idx[i]
        tmin = t.hour * 60 + t.minute
        day = t.date()

        if day != current_day:
            # safety net: close any open position at previous bar's close before day advances
            if pos is not None and i > 0:
                exit_price = closes[i - 1] - slippage_pts
                pnl_pts = exit_price - pos["entry_price"]
                pnl_gbp = pnl_pts * pos["size"]
                equity += pnl_gbp
                trades.append(_close_record(inst, pos, idx[i - 1], exit_price, pnl_pts, pnl_gbp, "SESSION_END"))
                pos = None
            current_day = day
            circuit_day = None  # reset circuit breaker on session change

        # equity / DD tracking
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        curve.append((t, equity, dd))

        # circuit breaker trigger
        if circuit_day != day and dd >= CIRCUIT_BREAKER_DD:
            circuit_day = day
            if pos is not None:
                exit_price = closes[i] - slippage_pts
                pnl_pts = exit_price - pos["entry_price"]
                pnl_gbp = pnl_pts * pos["size"]
                equity += pnl_gbp
                trades.append(_close_record(inst, pos, t, exit_price, pnl_pts, pnl_gbp, "CIRCUIT"))
                pos = None

        # manage open position
        if pos is not None:
            exit_reason = exit_price = None
            age_min = (t - pos["entry_time"]).total_seconds() / 60.0

            # last bar of session (firstrate labels 16:25 for 16:30 close → trigger 5min earlier)
            if tmin >= sess_end_min - 5:
                exit_reason, exit_price = "SESSION_END", closes[i] - slippage_pts
            elif age_min >= TIME_STOP_MIN:
                exit_reason, exit_price = "TIME", closes[i] - slippage_pts
            elif lows[i] <= pos["trail_stop"]:
                exit_reason = "TRAILING" if pos["trail_stop"] > pos["sl_init"] else "SL"
                exit_price = pos["trail_stop"] - slippage_pts
            elif highs[i] >= pos["tp"]:
                exit_reason, exit_price = "TP", pos["tp"] - slippage_pts
            else:
                # ratchet trail upward
                new_trail = closes[i] - pos["trail_mult"] * pos["atr_at_entry"]
                if new_trail > pos["trail_stop"]:
                    pos["trail_stop"] = new_trail

            if exit_reason is not None:
                pnl_pts = exit_price - pos["entry_price"]
                pnl_gbp = pnl_pts * pos["size"]
                equity += pnl_gbp
                trades.append(_close_record(inst, pos, t, exit_price, pnl_pts, pnl_gbp, exit_reason))
                pos = None

        # entry eligibility
        if pos is None and circuit_day != day:
            if sbi[i] >= WARMUP_BARS and tmin < entry_cutoff_min and fac_all[i]:
                atr = atrs[i]
                if not np.isnan(atr) and atr > 0:
                    entry_price = closes[i] + slippage_pts
                    risk_gbp = equity * RISK_PER_TRADE
                    size = risk_gbp / atr  # points-at-risk = 1 ATR
                    vol_regime = "high" if atr / closes[i] > 0.003 else "low"
                    trail_mult = 2.5 if vol_regime == "high" else 1.8

                    # factors that flipped True this bar (binding-constraint candidates)
                    flipped = [FACTORS[j] for j in range(7)
                               if fac_arr[i, j] and not prev_fac_arr[i, j]]

                    # marginal-volume flag: volume pass was borderline (< 2.5x MA)
                    vol_marginal = False
                    if not np.isnan(volmas[i]) and volmas[i] > 0:
                        vol_marginal = vols[i] < 2.5 * volmas[i]

                    pos = {
                        "entry_time":   t,
                        "entry_price":  entry_price,
                        "size":         size,
                        "sl_init":      entry_price - atr,
                        "tp":           entry_price + 3 * atr,
                        "trail_stop":   entry_price - atr,
                        "trail_mult":   trail_mult,
                        "atr_at_entry": atr,
                        "vol_regime":   vol_regime,
                        "factors":      {f: bool(fac_arr[i, j]) for j, f in enumerate(FACTORS)},
                        "binding":      flipped,
                        "vol_marginal": vol_marginal,
                    }

        if (i + 1) % 500 == 0:
            pass  # progress print handled at outer level

    # force-close anything still open at end of data
    if pos is not None:
        i = n - 1
        exit_price = closes[i] - slippage_pts
        pnl_pts = exit_price - pos["entry_price"]
        pnl_gbp = pnl_pts * pos["size"]
        equity += pnl_gbp
        trades.append(_close_record(inst, pos, idx[i], exit_price, pnl_pts, pnl_gbp, "SESSION_END"))

    return trades, curve


def _close_record(inst, pos, exit_time, exit_price, pnl_pts, pnl_gbp, reason):
    return {
        "instrument":         inst,
        "entry_time":         pos["entry_time"],
        "exit_time":          exit_time,
        "entry_price":        pos["entry_price"],
        "exit_price":         exit_price,
        "exit_reason":        reason,
        "pnl_pts":            pnl_pts,
        "pnl_pct":            pnl_pts / pos["entry_price"],
        "pnl_gbp":            pnl_gbp,
        "atr_at_entry":       pos["atr_at_entry"],
        "volatility_regime":  pos["vol_regime"],
        "all_7_factors_at_entry": json.dumps(pos["factors"]),
        "binding_constraint": json.dumps(pos["binding"]),
        "volume_marginal":    pos["vol_marginal"],
    }


# ============================================================================
# STATS
# ============================================================================

def compute_stats(trades: list, curve: list, start_eq: float) -> dict:
    if not trades:
        return {"trades": 0, "start_eq": start_eq, "end_eq": start_eq,
                "total_return_pct": 0, "cagr_pct": 0, "win_rate": 0,
                "avg_win_pct": 0, "avg_loss_pct": 0, "expectancy_gbp": 0,
                "profit_factor": float("nan"), "max_dd_pct": 0,
                "sharpe": 0, "calmar": 0}

    td = pd.DataFrame(trades)
    ec = pd.DataFrame(curve, columns=["t", "eq", "dd"])

    wins = td[td["pnl_gbp"] > 0]
    losses = td[td["pnl_gbp"] <= 0]
    n = len(td)
    win_rate = len(wins) / n if n else 0
    avg_win_pct = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss_pct = losses["pnl_pct"].mean() if len(losses) else 0
    expectancy = td["pnl_gbp"].mean()
    gw = wins["pnl_gbp"].sum() if len(wins) else 0
    gl = -losses["pnl_gbp"].sum() if len(losses) else 0
    pf = gw / gl if gl > 0 else float("inf")

    end_eq = ec["eq"].iloc[-1]
    max_dd = ec["dd"].max()
    total_ret = (end_eq - start_eq) / start_eq

    days = (pd.to_datetime(ec["t"].iloc[-1]) - pd.to_datetime(ec["t"].iloc[0])).days
    years = max(days / 365.25, 1e-6)
    cagr = (end_eq / start_eq) ** (1 / years) - 1 if end_eq > 0 else -1

    ec["date"] = pd.to_datetime(ec["t"]).dt.date
    daily_eq = ec.groupby("date")["eq"].last()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else float("inf")

    return {
        "trades": n,
        "start_eq": start_eq,
        "end_eq": round(end_eq, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "avg_win_pct": round(avg_win_pct * 100, 3),
        "avg_loss_pct": round(avg_loss_pct * 100, 3),
        "expectancy_gbp": round(expectancy, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else float("inf"),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
    }


def monthly_returns(curve: list, trades: list) -> pd.DataFrame:
    if not curve:
        return pd.DataFrame()
    ec = pd.DataFrame(curve, columns=["t", "eq", "dd"]).set_index("t")
    monthly_eq = ec["eq"].resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna() * 100

    if trades:
        td = pd.DataFrame(trades)
        td["month"] = pd.to_datetime(td["exit_time"]).dt.tz_localize(None).dt.to_period("M")
        counts = td.groupby("month").size()
    else:
        counts = pd.Series(dtype=int)

    out = pd.DataFrame({
        "month": monthly_ret.index.strftime("%Y-%m"),
        "return_pct": monthly_ret.round(2).values,
    })
    out["trade_count"] = out["month"].map(lambda m: int(counts.get(pd.Period(m, "M"), 0)))
    return out


def factor_binding_counts(trades: list) -> pd.DataFrame:
    counts = {f: 0 for f in FACTORS}
    no_flip = 0
    vol_marginal = 0
    for t in trades:
        bind = json.loads(t["binding_constraint"])
        if not bind:
            no_flip += 1
        for f in bind:
            counts[f] += 1
        if t["volume_marginal"]:
            vol_marginal += 1
    rows = [{"factor": f, "binding_count": c,
             "pct_of_trades": (c / len(trades) * 100) if trades else 0}
            for f, c in counts.items()]
    rows.append({"factor": "(none - all passed same bar as prev)",
                 "binding_count": no_flip,
                 "pct_of_trades": (no_flip / len(trades) * 100) if trades else 0})
    rows.append({"factor": "(volume was marginal, <2.5x MA)",
                 "binding_count": vol_marginal,
                 "pct_of_trades": (vol_marginal / len(trades) * 100) if trades else 0})
    return pd.DataFrame(rows)


# ============================================================================
# ORCHESTRATION
# ============================================================================

def prep_instrument(inst: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = INSTRUMENTS[inst]
    print(f"\n[{inst}] loading {cfg['file']}")
    raw = load_firstrate(cfg["file"], cfg["src_tz"])
    cutoff = raw.index.max() - pd.Timedelta(days=365 * YEARS_HISTORY)
    raw = raw[raw.index >= cutoff]
    print(f"[{inst}] {len(raw):,} raw bars, {raw.index.min().date()} → {raw.index.max().date()}")

    df = slice_session(raw, cfg)
    print(f"[{inst}] {len(df):,} session bars")

    print(f"[{inst}] computing indicators (per-session reset)…")
    df = compute_indicators_per_session(df)
    fac = compute_factors(df)
    return df, fac


def split_train_test(df: pd.DataFrame, fac: pd.DataFrame, frac: float):
    dates = sorted(set(df.index.date))
    split_i = int(len(dates) * frac)
    split_date = dates[split_i]
    train_mask = df.index.date < split_date
    test_mask  = ~train_mask
    return (df[train_mask], fac[train_mask],
            df[test_mask],  fac[test_mask],
            split_date)


def main():
    all_summary_rows = []

    per_inst = {}
    for inst in INSTRUMENTS:
        df, fac = prep_instrument(inst)
        df_tr, fac_tr, df_te, fac_te, split_date = split_train_test(df, fac, TRAIN_FRAC)
        print(f"[{inst}] split on {split_date}: "
              f"train {len(df_tr):,} bars / test {len(df_te):,} bars")
        per_inst[inst] = {
            "train": (df_tr, fac_tr),
            "test":  (df_te, fac_te),
            "split_date": split_date,
        }

    passes = [
        ("train_base",     "train", 0.0),
        ("train_slippage", "train", None),  # per-instrument slippage from config
        ("test_base",      "test",  0.0),
        ("test_slippage",  "test",  None),
    ]

    for pass_name, split, slip_override in passes:
        print(f"\n{'='*60}\n  PASS: {pass_name}\n{'='*60}")
        all_trades_pass = []
        curves_by_inst = {}

        for inst, data in per_inst.items():
            df, fac = data[split]
            slip = INSTRUMENTS[inst]["slippage_pts"] if slip_override is None else slip_override
            print(f"  [{inst}] running… slippage={slip}pt, {len(df):,} bars")
            trades, curve = run_backtest(inst, df, fac, slip, STARTING_EQUITY)
            print(f"  [{inst}] done: {len(trades)} trades")

            stats = compute_stats(trades, curve, STARTING_EQUITY)
            stats["instrument"] = inst
            stats["pass"] = pass_name
            all_summary_rows.append(stats)

            # per-instrument equity CSV
            ec_df = pd.DataFrame(curve, columns=["timestamp", "equity", "drawdown_pct"])
            ec_df["drawdown_pct"] = (ec_df["drawdown_pct"] * 100).round(3)
            ec_df.to_csv(OUT_DIR / f"scalper_equity_{inst}_{pass_name}.csv", index=False)

            all_trades_pass.extend(trades)
            curves_by_inst[inst] = ec_df.set_index(pd.to_datetime(ec_df["timestamp"]))["equity"]

        # combined: sum per-instrument equities (fill-forward on common index)
        if curves_by_inst:
            combined = pd.concat(curves_by_inst.values(), axis=1, sort=True).ffill().fillna(STARTING_EQUITY)
            combined_eq = combined.sum(axis=1)
            combined_peak = combined_eq.cummax()
            combined_dd = (combined_peak - combined_eq) / combined_peak
            combined_curve = [(t, e, d) for t, e, d in zip(combined_eq.index, combined_eq.values, combined_dd.values)]
            combined_start = STARTING_EQUITY * len(INSTRUMENTS)
            combined_stats = compute_stats(all_trades_pass, combined_curve, combined_start)
            combined_stats["instrument"] = "COMBINED"
            combined_stats["pass"] = pass_name
            all_summary_rows.append(combined_stats)

            ec_df = pd.DataFrame(combined_curve, columns=["timestamp", "equity", "drawdown_pct"])
            ec_df["drawdown_pct"] = (ec_df["drawdown_pct"] * 100).round(3)
            ec_df.to_csv(OUT_DIR / f"scalper_equity_COMBINED_{pass_name}.csv", index=False)

            monthly = monthly_returns(combined_curve, all_trades_pass)
            monthly.to_csv(OUT_DIR / f"scalper_monthly_{pass_name}.csv", index=False)

            factor_df = factor_binding_counts(all_trades_pass)
            factor_df.to_csv(OUT_DIR / f"scalper_factors_{pass_name}.csv", index=False)

        # trade log
        if all_trades_pass:
            trades_df = pd.DataFrame(all_trades_pass)
            trades_df.to_csv(OUT_DIR / f"scalper_trades_{pass_name}.csv", index=False)

    # summary table
    summary = pd.DataFrame(all_summary_rows)
    col_order = ["pass", "instrument", "trades", "start_eq", "end_eq",
                 "total_return_pct", "cagr_pct", "win_rate",
                 "avg_win_pct", "avg_loss_pct", "expectancy_gbp",
                 "profit_factor", "max_dd_pct", "sharpe", "calmar"]
    summary = summary[col_order]
    summary.to_csv(OUT_DIR / "scalper_summary.csv", index=False)

    # console print: side-by-side base vs slippage for train and test
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print()
    print("Files written to:", OUT_DIR)


if __name__ == "__main__":
    main()
