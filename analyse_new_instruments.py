"""
analyse_new_instruments.py — test the ASRS ORB strategy on gold/silver/oil/EURUSD.

Rules (deliberately simple, no AI, no tuning per instrument):
  1. Session opens at fixed UK time (try 2 options: London 08:00, NY 13:30)
  2. Bars 1-4 form (5min × 4 = 20 min)
  3. Params auto-scaled from per-instrument median bar range R:
       buffer    = 0.2R
       narrow    = 1.0R
       wide      = 3.0R
       max_range = 6.0R
  4. If bar 4 range in [narrow, wide]: arm BUY @ h+buffer, SELL @ l-buffer
       Stop at bar 4 opposite side.
  5. If bar 4 > wide, use bar 5 (same rules).
  6. Single entry only, trail via prev bar low/high on each 5min close.
  7. EOD close at session end (default: session_open + 8 hours).
  8. No re-entry. No filters. Keep it brutally simple.

Train: 2020-2023 (4 years)
Test:  2024-2026 (2.3 years)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path("/root/asrs-bot/data")
TZ   = ZoneInfo("Europe/London")

# Instrument files — point-value approximated (used only for sanity scaling, not used in pts-based PF calc)
INSTRUMENTS = {
    "EURUSD": "eurusd_m5.csv",
    "XAUUSD": "xauusd_test.csv",    # may be partial
    "XAGUSD": "xagusd_m5.csv",
    "USOIL":  "usoil_m5.csv",
}

# Session open hours to try (UK local time)
SESSIONS = [
    ("S1_London", 8,  0),   # London open
    ("S2_NY",     13, 30),  # NY open
]
SESSION_LENGTH_H = 8        # force EOD after 8 hours


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp": "utm", "open": "Open",
                            "high": "High", "low": "Low", "close": "Close"})
    df["dt"] = pd.to_datetime(df["utm"], unit="ms", utc=True).dt.tz_convert(TZ)
    df = df.sort_values("dt").reset_index(drop=True)
    df = df[df["dt"].dt.dayofweek < 5]
    df = df[~((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]))]
    df["hm"]   = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    df["date"] = df["dt"].dt.date
    df["year"] = df["dt"].dt.year
    return df


def derive_params(df: pd.DataFrame, oh: int, om: int) -> dict:
    """Compute per-instrument scaled params from median 5min bar range at session open."""
    open_m = oh * 60 + om
    sub = df[(df["hm"] >= open_m) & (df["hm"] < open_m + 20)]
    R = (sub["High"] - sub["Low"]).median()
    return {
        "buffer":    round(R * 0.2, 5),
        "narrow":    round(R * 1.0, 5),
        "wide":      round(R * 3.0, 5),
        "max_range": round(R * 6.0, 5),
        "R":         R,
    }


def run_session(df_day: pd.DataFrame, oh: int, om: int, cfg: dict) -> dict | None:
    """Simulate one session on one day. Returns trade dict or None."""
    open_m = oh * 60 + om
    sess_m = open_m + SESSION_LENGTH_H * 60

    bars = df_day[(df_day["hm"] >= open_m) & (df_day["hm"] < sess_m)].reset_index(drop=True)
    if len(bars) < 5:
        return None

    bar4 = bars.iloc[3]
    bar4_range = bar4["High"] - bar4["Low"]

    if bar4_range < cfg["narrow"]:
        sig_h, sig_l = bar4["High"], bar4["Low"]
        sig_idx = 3
    elif bar4_range <= cfg["wide"]:
        sig_h, sig_l = bar4["High"], bar4["Low"]
        sig_idx = 3
    else:
        # Use bar 5
        if len(bars) < 5: return None
        bar5 = bars.iloc[4]
        b5r = bar5["High"] - bar5["Low"]
        if b5r > cfg["max_range"] or b5r <= 0:
            return None
        sig_h, sig_l = bar5["High"], bar5["Low"]
        sig_idx = 4

    buy_lv  = sig_h + cfg["buffer"]
    sell_lv = sig_l - cfg["buffer"]

    # Simulate from bar sig_idx+1 to EOD
    trading = bars.iloc[sig_idx + 1:].reset_index(drop=True)
    if len(trading) == 0: return None

    position = None
    entry = 0.0
    stop = 0.0

    for i in range(len(trading)):
        bar = trading.iloc[i]
        h, l, c = float(bar["High"]), float(bar["Low"]), float(bar["Close"])

        # Check stop first (priority)
        if position == "LONG" and l <= stop:
            return {"pnl": stop - entry, "dir": "LONG", "entry": entry,
                    "exit": stop, "reason": "STOP", "bars": i + 1}
        if position == "SHORT" and h >= stop:
            return {"pnl": entry - stop, "dir": "SHORT", "entry": entry,
                    "exit": stop, "reason": "STOP", "bars": i + 1}

        # Entry
        if position is None:
            if h >= buy_lv:
                position = "LONG"; entry = buy_lv; stop = sig_l
            elif l <= sell_lv:
                position = "SHORT"; entry = sell_lv; stop = sig_h

        # Trail on close (candle trail, prev bar's extremes)
        if position == "LONG" and i > 0:
            prev = trading.iloc[i - 1]
            new_stop = float(prev["Low"])
            if new_stop > stop: stop = new_stop
        elif position == "SHORT" and i > 0:
            prev = trading.iloc[i - 1]
            new_stop = float(prev["High"])
            if new_stop < stop: stop = new_stop

    # EOD close
    if position is not None:
        last = trading.iloc[-1]
        exit_price = float(last["Close"])
        pnl = (exit_price - entry) if position == "LONG" else (entry - exit_price)
        return {"pnl": pnl, "dir": position, "entry": entry, "exit": exit_price,
                "reason": "EOD", "bars": len(trading)}
    return None


def run_instrument(name: str, df: pd.DataFrame) -> pd.DataFrame:
    all_trades = []
    for sess_label, oh, om in SESSIONS:
        cfg = derive_params(df, oh, om)
        print(f"  {name} {sess_label}: R={cfg['R']:.5f}  narrow={cfg['narrow']}  wide={cfg['wide']}")

        for date_val, df_day in df.groupby("date"):
            tr = run_session(df_day, oh, om, cfg)
            if tr:
                tr["date"] = date_val
                tr["session"] = sess_label
                tr["instrument"] = name
                all_trades.append(tr)
    return pd.DataFrame(all_trades)


def stats(d: pd.DataFrame, label: str):
    if len(d) == 0:
        print(f"  {label:<45}  empty"); return
    w = d[d["pnl"] > 0]; l = d[d["pnl"] < 0]
    pf  = w.pnl.sum() / max(abs(l.pnl.sum()), 1e-12)
    net = d.pnl.sum()
    wr  = len(w) / len(d) * 100
    e   = net / len(d)
    print(f"  {label:<45} n={len(d):>5}  PF={pf:>5.2f}  net={net:>+10,.4f}  WR={wr:>4.1f}%  E={e:>+7.4f}")


def main():
    results = {}
    for name, fname in INSTRUMENTS.items():
        p = DATA / fname
        if not p.exists():
            print(f"MISSING: {p}"); continue
        print(f"\n{'=' * 95}\n  {name} — loading {fname}")
        df = load_csv(p)
        print(f"  {len(df):,} bars, {df['date'].min()} → {df['date'].max()}")
        trades = run_instrument(name, df)
        if len(trades) == 0:
            print(f"  NO TRADES"); continue
        trades["year"] = pd.to_datetime(trades["date"]).dt.year
        results[name] = trades

    # Report
    for period_label, year_filter in [
        ("FULL 2020-2026", None),
        ("TRAIN 2020-2023", lambda y: y <= 2023),
        ("TEST  2024-2026", lambda y: y >= 2024),
    ]:
        print(f"\n{'#' * 95}\n  {period_label}\n{'#' * 95}")
        for name, t in results.items():
            if year_filter:
                t_filt = t[t["year"].apply(year_filter)]
            else:
                t_filt = t
            print(f"\n  {name}:")
            stats(t_filt, "ALL (both sessions)")
            for s in t_filt["session"].unique():
                stats(t_filt[t_filt["session"] == s], f"  {s}")


if __name__ == "__main__":
    main()
