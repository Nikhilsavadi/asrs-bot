"""
analyse_pattern_strategies.py — test 3 simple strategies (ORB, Donchian,
Volatility-Squeeze) on gold/silver/oil/natgas/EURUSD with train/test.

All rules are brutally mechanical — no AI, no optimisation per instrument.
Parameters are auto-scaled from each instrument's median 5min range.

Strategies:
  A) ORB (baseline)          — session-open breakout (bars 1-4 hybrid).
  B) Donchian N=20 breakout  — long on close > prev 20-bar high; exit on
                                close < trailing 10-bar low; 1 trade at a time.
  C) Volatility squeeze      — when current bar range < 0.4 × 20-bar ATR AND
                                next bar breaks the squeeze bar's high/low,
                                enter. Stop at opposite side. Trail via 5-bar
                                low/high on each close.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path("/root/asrs-bot/data")
TZ = ZoneInfo("Europe/London")

INSTRUMENTS = {
    "EURUSD": "eurusd_m5.csv",
    "XAUUSD": "xauusd_m5.csv",
    "XAGUSD": "xagusd_m5.csv",
    "USOIL":  "usoil_m5.csv",
    "NGAS":   "ngas_m5.csv",
}

# Only trade during liquid hours (UK time) — keep it simple
LIQUID_START_HM = 8 * 60   # 08:00 UK
LIQUID_END_HM = 21 * 60    # 21:00 UK


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp": "utm", "open": "Open",
                            "high": "High", "low": "Low", "close": "Close"})
    df["dt"] = pd.to_datetime(df["utm"], unit="ms", utc=True).dt.tz_convert(TZ)
    df = df.sort_values("dt").reset_index(drop=True)
    df = df[df["dt"].dt.dayofweek < 5]
    # Drop flat-line synth bars
    df = df[~((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))]
    df["hm"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    df["date"] = df["dt"].dt.date
    df["year"] = df["dt"].dt.year
    # Liquid hours filter
    df = df[(df["hm"] >= LIQUID_START_HM) & (df["hm"] < LIQUID_END_HM)].reset_index(drop=True)
    return df


# ============================================================================
# A) ORB (reuse the simple implementation)
# ============================================================================

def run_orb(df: pd.DataFrame, session_open_hour: int) -> list:
    """Session-open ORB with bar 4 (narrow/wide hybrid)."""
    open_m = session_open_hour * 60
    sess_m = open_m + 8 * 60
    # Derive params
    sub = df[(df["hm"] >= open_m) & (df["hm"] < open_m + 20)]
    R = (sub["High"] - sub["Low"]).median()
    buffer_, narrow, wide, max_r = R * 0.2, R * 1.0, R * 3.0, R * 6.0

    trades = []
    for date_val, dday in df.groupby("date"):
        bars = dday[(dday["hm"] >= open_m) & (dday["hm"] < sess_m)].reset_index(drop=True)
        if len(bars) < 5: continue
        b4 = bars.iloc[3]
        b4r = b4["High"] - b4["Low"]
        if b4r <= wide:
            sh, sl = b4["High"], b4["Low"]; sig_i = 3
        else:
            if len(bars) < 5: continue
            b5 = bars.iloc[4]
            if (b5["High"] - b5["Low"]) > max_r or (b5["High"] - b5["Low"]) <= 0: continue
            sh, sl = b5["High"], b5["Low"]; sig_i = 4
        buy_lv, sell_lv = sh + buffer_, sl - buffer_
        t = simulate(bars.iloc[sig_i + 1:].reset_index(drop=True), buy_lv, sell_lv, sh, sl)
        if t:
            t["date"] = date_val; t["strategy"] = "ORB"
            trades.append(t)
    return trades


# ============================================================================
# B) Donchian 20-bar breakout
# ============================================================================

def run_donchian(df: pd.DataFrame, N: int = 20) -> list:
    """Long on close > prev N-bar high; short on close < prev N-bar low.
    Exit: trail 10-bar low (for long) / high (for short)."""
    trades = []
    for date_val, dday in df.groupby("date"):
        dday = dday.reset_index(drop=True)
        if len(dday) < N + 5: continue

        highs = dday["High"].values
        lows = dday["Low"].values
        closes = dday["Close"].values

        position = None
        entry = 0.0
        stop = 0.0

        for i in range(N, len(dday)):
            # Don't enter near EOD (leave 30 min)
            if i >= len(dday) - 6: break

            hh = highs[i - N:i].max()
            ll = lows[i - N:i].min()

            if position is None:
                if closes[i] > hh:
                    position = "LONG"; entry = closes[i]; stop = lows[i - N:i].min()
                elif closes[i] < ll:
                    position = "SHORT"; entry = closes[i]; stop = highs[i - N:i].max()
            else:
                # Trail
                if position == "LONG":
                    new_stop = lows[max(0, i - 10):i].min()
                    if new_stop > stop: stop = new_stop
                    if lows[i] <= stop:
                        trades.append({"pnl": stop - entry, "dir": "LONG",
                                       "entry": entry, "exit": stop, "reason": "TRAIL",
                                       "date": date_val, "strategy": "DONCHIAN"})
                        position = None
                else:
                    new_stop = highs[max(0, i - 10):i].max()
                    if new_stop < stop: stop = new_stop
                    if highs[i] >= stop:
                        trades.append({"pnl": entry - stop, "dir": "SHORT",
                                       "entry": entry, "exit": stop, "reason": "TRAIL",
                                       "date": date_val, "strategy": "DONCHIAN"})
                        position = None

        # EOD close
        if position is not None:
            exit_p = float(closes[-1])
            pnl = (exit_p - entry) if position == "LONG" else (entry - exit_p)
            trades.append({"pnl": pnl, "dir": position, "entry": entry,
                           "exit": exit_p, "reason": "EOD", "date": date_val,
                           "strategy": "DONCHIAN"})
    return trades


# ============================================================================
# C) Volatility squeeze breakout
# ============================================================================

def run_squeeze(df: pd.DataFrame, atr_n: int = 20, squeeze_ratio: float = 0.4) -> list:
    """When bar range < squeeze_ratio × ATR, take next bar's breakout."""
    trades = []
    for date_val, dday in df.groupby("date"):
        dday = dday.reset_index(drop=True)
        if len(dday) < atr_n + 10: continue

        highs = dday["High"].values
        lows = dday["Low"].values
        closes = dday["Close"].values
        tr = np.maximum.reduce([
            highs - lows,
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1)),
        ])

        position = None
        entry = 0.0
        stop = 0.0
        armed = False
        arm_high = arm_low = 0.0

        for i in range(atr_n, len(dday)):
            if i >= len(dday) - 6: break
            atr = tr[i - atr_n:i].mean()
            bar_range = highs[i] - lows[i]

            if position is None:
                if not armed and bar_range < squeeze_ratio * atr:
                    armed = True
                    arm_high = highs[i]
                    arm_low = lows[i]
                elif armed:
                    if highs[i] > arm_high:
                        position = "LONG"; entry = arm_high; stop = arm_low; armed = False
                    elif lows[i] < arm_low:
                        position = "SHORT"; entry = arm_low; stop = arm_high; armed = False
            else:
                if position == "LONG":
                    new_stop = lows[max(0, i - 5):i].min()
                    if new_stop > stop: stop = new_stop
                    if lows[i] <= stop:
                        trades.append({"pnl": stop - entry, "dir": "LONG",
                                       "entry": entry, "exit": stop, "reason": "TRAIL",
                                       "date": date_val, "strategy": "SQUEEZE"})
                        position = None
                else:
                    new_stop = highs[max(0, i - 5):i].max()
                    if new_stop < stop: stop = new_stop
                    if highs[i] >= stop:
                        trades.append({"pnl": entry - stop, "dir": "SHORT",
                                       "entry": entry, "exit": stop, "reason": "TRAIL",
                                       "date": date_val, "strategy": "SQUEEZE"})
                        position = None

        # EOD close
        if position is not None:
            exit_p = float(closes[-1])
            pnl = (exit_p - entry) if position == "LONG" else (entry - exit_p)
            trades.append({"pnl": pnl, "dir": position, "entry": entry,
                           "exit": exit_p, "reason": "EOD", "date": date_val,
                           "strategy": "SQUEEZE"})
    return trades


# ============================================================================
# Shared ORB engine helper
# ============================================================================

def simulate(bars: pd.DataFrame, buy_lv, sell_lv, sh, sl) -> dict | None:
    position = None; entry = 0.0; stop = 0.0
    for i in range(len(bars)):
        h, l, c = float(bars.iloc[i]["High"]), float(bars.iloc[i]["Low"]), float(bars.iloc[i]["Close"])
        if position == "LONG" and l <= stop:
            return {"pnl": stop - entry, "dir": "LONG", "entry": entry, "exit": stop, "reason": "STOP"}
        if position == "SHORT" and h >= stop:
            return {"pnl": entry - stop, "dir": "SHORT", "entry": entry, "exit": stop, "reason": "STOP"}
        if position is None:
            if h >= buy_lv: position = "LONG"; entry = buy_lv; stop = sl
            elif l <= sell_lv: position = "SHORT"; entry = sell_lv; stop = sh
        if position == "LONG" and i > 0:
            ns = float(bars.iloc[i - 1]["Low"])
            if ns > stop: stop = ns
        elif position == "SHORT" and i > 0:
            ns = float(bars.iloc[i - 1]["High"])
            if ns < stop: stop = ns
    if position is not None:
        exit_p = float(bars.iloc[-1]["Close"])
        pnl = (exit_p - entry) if position == "LONG" else (entry - exit_p)
        return {"pnl": pnl, "dir": position, "entry": entry, "exit": exit_p, "reason": "EOD"}
    return None


# ============================================================================
# Reporting
# ============================================================================

def stats(d: pd.DataFrame, label: str):
    if len(d) == 0:
        print(f"  {label:<35}  empty"); return
    w = d[d["pnl"] > 0]; l = d[d["pnl"] < 0]
    pf = w.pnl.sum() / max(abs(l.pnl.sum()), 1e-12)
    net = d.pnl.sum()
    wr = len(w) / len(d) * 100
    e = net / len(d)
    print(f"  {label:<35} n={len(d):>5}  PF={pf:>5.2f}  net={net:>+10,.4f}  WR={wr:>4.1f}%  E={e:>+7.5f}")


def run_all(df: pd.DataFrame, name: str) -> pd.DataFrame:
    all_trades = []
    # ORB: 2 sessions
    for oh in [8, 13]:
        ts = run_orb(df, oh)
        for t in ts: t["instrument"] = name; t["session"] = f"ORB_S{oh:02d}"
        all_trades.extend(ts)
    # Donchian
    ts = run_donchian(df)
    for t in ts: t["instrument"] = name; t["session"] = "DONCHIAN"
    all_trades.extend(ts)
    # Squeeze
    ts = run_squeeze(df)
    for t in ts: t["instrument"] = name; t["session"] = "SQUEEZE"
    all_trades.extend(ts)
    return pd.DataFrame(all_trades)


def main():
    all_results = {}
    for name, fname in INSTRUMENTS.items():
        p = DATA / fname
        if not p.exists():
            print(f"MISSING: {fname}"); continue
        print(f"\nLoading {name}...")
        df = load_csv(p)
        print(f"  {len(df):,} bars, {df['date'].min()} → {df['date'].max()}")
        res = run_all(df, name)
        res["year"] = pd.to_datetime(res["date"]).dt.year
        all_results[name] = res

    for period_label, yf in [
        ("FULL",  None),
        ("TRAIN 2020-2023", lambda y: y <= 2023),
        ("TEST  2024-2026", lambda y: y >= 2024),
    ]:
        print(f"\n{'#' * 95}\n  {period_label}\n{'#' * 95}")
        for name, t in all_results.items():
            tf = t if yf is None else t[t["year"].apply(yf)]
            print(f"\n  {name}:")
            for s in ["ORB_S08", "ORB_S13", "DONCHIAN", "SQUEEZE"]:
                stats(tf[tf["session"] == s], f"  {s}")


if __name__ == "__main__":
    main()
