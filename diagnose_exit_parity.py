"""
diagnose_exit_parity.py — given live's actual entry time + price, simulate
the exit using tick_backtest's trail+stop logic and compare.

If exits match → divergence is purely entry-timing path-dependence.
If exits differ → something in the bot's exit logic differs from backtest.
"""
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import sys
sys.path.insert(0, "/app")

from tick_backtest import INSTRUMENTS, load_ticks, build_5min_bars


STOP_CONFIRM_SECS = 60


def simulate_exit_from_entry(
    ticks: pd.DataFrame, bars: pd.DataFrame, cfg: dict,
    entry_time: pd.Timestamp, entry_price: float, direction: str,
    initial_stop: float, eod_time: pd.Timestamp,
):
    """Run the tick_backtest exit logic starting from a known entry state."""
    bars_idx = bars.set_index("bar_start")
    current_stop = float(initial_stop)
    be_hit = False
    be_pts = cfg["be_pts"]; be_buf = cfg["be_buf"]; tight = cfg["tight"]
    stop_breach_since = 0.0
    last_bar_processed = None

    trading = ticks[(ticks["dt"] >= entry_time) & (ticks["dt"] < eod_time)].reset_index(drop=True)
    if len(trading) == 0:
        return None, "no ticks post-entry"

    bids = trading["bid"].values
    ofrs = trading["ofr"].values
    dts = trading["dt"].values

    for i in range(len(trading)):
        bid = bids[i]; ofr = ofrs[i]; dt = dts[i]
        bar_start = pd.Timestamp(dt).floor("5min")
        if pd.Timestamp(dt).tz is None:
            bar_start = bar_start.tz_localize(bars_idx.index.tz)

        # On new bar: apply trail update using previous bar's OHLC
        if last_bar_processed is not None and bar_start != last_bar_processed:
            if last_bar_processed in bars_idx.index:
                pb = bars_idx.loc[last_bar_processed]
                prev_h = float(pb["High"]); prev_l = float(pb["Low"]); prev_c = float(pb["Close"])
                if direction == "LONG":
                    unr = prev_c - entry_price
                    if not be_hit and unr >= be_pts:
                        be_hit = True
                        new_be = entry_price - be_buf
                        if new_be > current_stop: current_stop = new_be; stop_breach_since = 0
                    prof = prev_c - entry_price
                    ns = prev_c if prof >= tight else prev_l
                    if ns > current_stop: current_stop = round(ns, 1); stop_breach_since = 0
                else:
                    unr = entry_price - prev_c
                    if not be_hit and unr >= be_pts:
                        be_hit = True
                        new_be = entry_price + be_buf
                        if new_be < current_stop: current_stop = new_be; stop_breach_since = 0
                    prof = entry_price - prev_c
                    ns = prev_c if prof >= tight else prev_h
                    if ns < current_stop: current_stop = round(ns, 1); stop_breach_since = 0
        last_bar_processed = bar_start

        # 60s confirmation stop check
        breached = False
        if direction == "LONG" and bid <= current_stop: breached = True
        elif direction == "SHORT" and ofr >= current_stop: breached = True

        tick_ts = pd.Timestamp(dt).timestamp()
        if breached:
            if stop_breach_since == 0:
                stop_breach_since = tick_ts
            elif (tick_ts - stop_breach_since) >= STOP_CONFIRM_SECS:
                exit_p = float(bid) if direction == "LONG" else float(ofr)
                return (exit_p, f"STOP @ stop_lvl={current_stop:.1f} after 60s conf"), None
        else:
            stop_breach_since = 0

    # EOD
    exit_p = float(bids[-1]) if direction == "LONG" else float(ofrs[-1])
    return (exit_p, f"EOD stop_lvl={current_stop:.1f}"), None


def main():
    # Live DAX trades today — hardcoded from journal
    # (entry_price, direction, entry_time_cet, initial_stop_est)
    live_trades = [
        # DAX_S1 SHORTs — signal low 24046.4 + buffer 2 = sell_level 24044.4... no wait
        # Actually sell_level for SHORT entry = sig_l - buffer, and stop = sig_h
        # DAX_S1 sig: H=24074.2 L=24046.4 → buy_level=24076.2 sell_level=24044.4
        # But live entered at 24051.5 SHORT — that's ABOVE sell_level, so slippage was -7.1pt
        # Hmm the bar 4 was actually different. Let me use actual entries.
        # Live entries were SHORT 24051.5, 24050.9, 24050.0.
        # Actually to simplify: just trust the signal levels and reconstruct.
        {"sig": "DAX_S1", "entry_time": "09:26", "entry_price": 24051.5, "dir": "SHORT", "initial_stop": 24074.2, "live_exit": 24061.9, "live_pnl": -10.4},
        {"sig": "DAX_S1", "entry_time": "09:41", "entry_price": 24050.9, "dir": "SHORT", "initial_stop": 24074.2, "live_exit": 24048.0, "live_pnl": +2.9},
        {"sig": "DAX_S1", "entry_time": "10:16", "entry_price": 24050.0, "dir": "SHORT", "initial_stop": 24074.2, "live_exit": 24046.4, "live_pnl": +3.6},
        # DAX_S2 signal box from state file (let me just use bar high as stop estimate)
        {"sig": "DAX_S2", "entry_time": "14:26", "entry_price": 24069.7, "dir": "LONG", "initial_stop": 24052.0, "live_exit": 24049.8, "live_pnl": -19.9},
        {"sig": "DAX_S2", "entry_time": "14:47", "entry_price": 24070.3, "dir": "LONG", "initial_stop": 24052.0, "live_exit": 24052.8, "live_pnl": -17.5},
        {"sig": "DAX_S2", "entry_time": "15:08", "entry_price": 24066.0, "dir": "LONG", "initial_stop": 24052.0, "live_exit": 24056.8, "live_pnl": -9.2},
    ]

    cfg = INSTRUMENTS["DAX"]
    ticks = load_ticks(cfg["epic"], "2026-04-15", cfg["tz"])
    bars = build_5min_bars(ticks)
    tz = ZoneInfo(cfg["tz"])
    eod = pd.Timestamp("2026-04-15 17:30:00", tz=tz)

    print("DIAGNOSTIC: force backtest to use LIVE entry price+time, compare exits")
    print("=" * 90)
    print(f"{'Trade':<10} {'Dir':<5} {'Entry':>8} {'Live Exit':>10} "
          f"{'Sim Exit':>10} {'Live pnl':>9} {'Sim pnl':>8}  Note")
    for t in live_trades:
        entry_time_str = f"2026-04-15 {t['entry_time']}:00"
        entry_time = pd.Timestamp(entry_time_str, tz=tz)
        result, err = simulate_exit_from_entry(
            ticks, bars, cfg,
            entry_time, t["entry_price"], t["dir"],
            t["initial_stop"], eod,
        )
        if err:
            print(f"{t['sig']:<10} {t['dir']:<5} {t['entry_price']:>8.1f} "
                  f"{t['live_exit']:>10.1f} ERROR {err}")
            continue
        exit_p, note = result
        sim_pnl = (exit_p - t["entry_price"]) if t["dir"] == "LONG" else (t["entry_price"] - exit_p)
        diff = t["live_pnl"] - sim_pnl
        print(f"{t['sig']:<10} {t['dir']:<5} {t['entry_price']:>8.1f} "
              f"{t['live_exit']:>10.1f} {exit_p:>10.1f} "
              f"{t['live_pnl']:>+8.1f} {sim_pnl:>+7.1f}  {note}")


if __name__ == "__main__":
    main()
