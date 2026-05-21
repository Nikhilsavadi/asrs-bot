"""
replay_ticks.py — Event-driven replay of recorded tick CSVs through actual Signal class.

Mirrors the live bot's execution model:
  - Ticks fire callbacks (bracket trigger, stop monitor) synchronously
  - Bar completions fire on_bar_complete as async tasks (non-blocking)
  - morning_routine waits for bar 5 event asynchronously (not sequentially)
  - Trail updates on bar complete, re-entry gate on every tick

Usage:
    python3 replay_ticks.py                    # all instruments, all days
    python3 replay_ticks.py --date 2026-04-10  # specific day
    python3 replay_ticks.py --inst DAX         # specific instrument
"""
import asyncio
import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import deque

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from asrs.strategy import Signal, Phase
from asrs import config

# Bypass risk gate in replay (it reads live journal DB and blocks trigger
# if live has hit daily limit — irrelevant for a clean backtest).
from asrs import risk_gate as _rg
_rg.check_entry_allowed = lambda *_a, **_kw: (True, "")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
# Show strategy INFO for trade signals
logging.getLogger("asrs.strategy").setLevel(logging.INFO)
logger = logging.getLogger("replay")

TICK_DIR = Path("data/ticks")


class ReplayBroker:
    """Mock broker that simulates IG fills at market price during replay."""

    _epic_entry_locks: dict[str, asyncio.Lock] = {}

    def __init__(self, epic, currency, disaster_stop_pts, max_spread_pts):
        self.epic = epic
        self.currency = currency
        self._disaster_stop_pts = disaster_stop_pts
        self._max_spread_pts = max_spread_pts
        self.connected = True
        self._loop = None

        self._bid = 0.0
        self._ofr = 0.0
        self._mid = 0.0

        self._position = {"direction": "FLAT", "avg_cost": 0, "size": 0}
        self._stop_level = 0.0
        self._stop_active = False
        self._stop_direction = ""

        self._bars = deque(maxlen=300)
        self._current_bar = None

        self._pending_bracket = None
        self._tick_trigger_active = False
        self._on_trigger_callbacks = []
        self._on_stop_callbacks = []
        self._tick_rearm_callbacks = []

        self._position_deal_ids = {}
        self._consecutive_order_errors = 0
        self._max_consecutive_order_errors = 3

    @classmethod
    def _get_entry_lock(cls, epic):
        if epic not in cls._epic_entry_locks:
            cls._epic_entry_locks[epic] = asyncio.Lock()
        return cls._epic_entry_locks[epic]

    async def connect(self): return True
    async def ensure_connected(self): return True

    async def get_current_price(self):
        return self._mid

    def get_streaming_bars_df(self):
        if not self._bars:
            return pd.DataFrame()
        bars = list(self._bars)
        df = pd.DataFrame(bars)
        df.index = pd.DatetimeIndex(df["time"])
        return df[["Open", "High", "Low", "Close"]]

    def get_streaming_bar_count(self):
        return len(self._bars)

    async def get_position(self):
        return dict(self._position)

    async def place_market_order(self, action, qty, guaranteed_stop=False, stop_distance_pts=None):
        # guaranteed_stop + stop_distance_pts ignored in replay (no real broker)
        fill = self._ofr if action == "BUY" else self._bid
        direction = "BUY" if action == "BUY" else "SELL"
        self._position = {"direction": direction, "avg_cost": fill, "size": qty}
        deal_id = f"REPLAY_{id(self)}_{len(self._position_deal_ids)}"
        self._position_deal_ids[deal_id] = {"direction": direction, "size": qty, "level": fill}
        self._consecutive_order_errors = 0
        return {"order_id": deal_id, "avg_price": fill}

    async def close_position(self):
        self._position = {"direction": "FLAT", "avg_cost": 0, "size": 0}
        self._position_deal_ids.clear()
        return True

    async def place_oca_bracket(self, buy_price, sell_price, qty, oca_group,
                                  soft_arm=False):
        self._pending_bracket = {
            "buy_price": buy_price, "sell_price": sell_price,
            "qty": qty, "active": True, "oca_group": oca_group,
            "soft_arm": soft_arm,
        }
        return {"buy_id": "REPLAY_BUY", "sell_id": "REPLAY_SELL"}

    def activate_stop_monitor(self, direction, stop_level):
        self._stop_level = stop_level
        self._stop_active = True
        self._stop_direction = direction

    def deactivate_stop_monitor(self):
        self._stop_active = False

    def update_stop_level(self, new_level):
        self._stop_level = new_level

    def register_trigger_callback(self, cb):
        self._on_trigger_callbacks.append(cb)

    def register_stop_callback(self, cb):
        self._on_stop_callbacks.append(cb)

    def register_tick_rearm_callback(self, cb):
        self._tick_rearm_callbacks.append(cb)

    async def cancel_all_orders(self):
        if self._pending_bracket:
            self._pending_bracket["active"] = False

    async def check_trigger_levels(self):
        return None

    async def modify_stop_all(self, new_level):
        self._stop_level = new_level

    async def place_stop_order(self, action, qty, stop_price):
        self._stop_level = stop_price
        return {"order_id": "REPLAY_STOP"}

    def process_tick(self, bid, ofr, mid, dt):
        """Process a tick: update price, build bars, return (new_bar_completed, completed_bar)."""
        self._bid = bid
        self._ofr = ofr
        self._mid = mid

        bar_min = (dt.minute // 5) * 5
        bar_start = dt.replace(minute=bar_min, second=0, microsecond=0)
        completed_bar = None

        if self._current_bar is None or self._current_bar["time"] != bar_start:
            if self._current_bar is not None:
                completed_bar = dict(self._current_bar)
                self._bars.append(completed_bar)
            self._current_bar = {
                "time": bar_start,
                "Open": mid, "High": mid, "Low": mid, "Close": mid,
            }
        else:
            self._current_bar["High"] = max(self._current_bar["High"], mid)
            self._current_bar["Low"] = min(self._current_bar["Low"], mid)
            self._current_bar["Close"] = mid

        return completed_bar


async def replay_session(inst_name, sn, cfg, ticks_df, date_str):
    """Replay one session using event-driven async model."""
    tz = ZoneInfo(cfg["timezone"])
    oh = cfg[f"s{sn}_open_hour"]
    om = cfg[f"s{sn}_open_minute"]
    open_m = oh * 60 + om
    eod_m = cfg["session_end_hour"] * 60 + cfg["session_end_minute"]

    # Filter ticks to session
    session_ticks = ticks_df[
        (ticks_df["dt"].dt.hour * 60 + ticks_df["dt"].dt.minute >= open_m) &
        (ticks_df["dt"].dt.hour * 60 + ticks_df["dt"].dt.minute < eod_m)
    ].reset_index(drop=True)

    if len(session_ticks) < 50:
        return []

    broker = ReplayBroker(cfg["epic"], cfg["currency"], cfg["disaster_stop_pts"], cfg["max_spread"])
    broker._loop = asyncio.get_event_loop()

    alerts = []
    async def alert(msg):
        alerts.append(msg)

    signal = Signal(inst_name, sn, broker, None, alert)
    signal._replay_date = date.fromisoformat(date_str)
    # Register trigger callback (live main.py does this — was missing in replay).
    # Without this, broker._on_trigger_callbacks is empty and tick triggers
    # never invoke the Signal's on_tick_trigger → state.trades stays empty.
    broker.register_trigger_callback(signal.on_tick_trigger)

    # Process ticks with proper async task scheduling
    pending_tasks = []
    YIELD_EVERY = 10  # yield to event loop every N ticks (was 200 — too slow for
                       # morning_routine to complete before bar 5, missing triggers)

    for i, (_, tick) in enumerate(session_ticks.iterrows()):
        bid, ofr, mid_p = float(tick["bid"]), float(tick["ofr"]), float(tick["mid"])
        dt = tick["dt"]

        # Process tick (update price, build bars)
        completed_bar = broker.process_tick(bid, ofr, mid_p, dt)

        # If a 5-min bar completed, fire on_bar_complete as async task.
        if completed_bar is not None:
            task = asyncio.create_task(signal.on_bar_complete(completed_bar))
            pending_tasks.append(task)

            # Gather pending tasks at bar 4 AND bar 5 so morning_routine
            # completes (and bracket arms) BEFORE processing the next tick.
            # Otherwise the 2-sec sleep in on_bar_complete lets the main
            # loop race past bar 5 ticks without an armed bracket → no entries.
            bn = signal._bar_number(completed_bar["time"])
            if bn >= 4:
                await asyncio.sleep(0.1)
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    pending_tasks.clear()

        # Fire rearm callbacks (sync, on every tick)
        for cb in broker._tick_rearm_callbacks:
            try:
                cb(mid_p, bid, ofr)
            except Exception:
                pass

        # Tick-level stop check
        if broker._stop_active and not getattr(broker, '_stop_exit_active', False):
            hit = False
            if broker._stop_direction == "LONG" and bid > 0 and bid <= broker._stop_level:
                hit = True
            elif broker._stop_direction == "SHORT" and ofr > 0 and ofr >= broker._stop_level:
                hit = True

            if hit:
                broker._stop_exit_active = True
                exit_price = bid if broker._stop_direction == "LONG" else ofr
                broker._stop_active = False
                await broker.close_position()
                for cb in broker._on_stop_callbacks:
                    await cb({
                        "exit_price": exit_price,
                        "exit_intended": broker._stop_level,
                        "direction": broker._stop_direction,
                    })
                broker._stop_exit_active = False

        # Tick-level bracket trigger
        if (broker._pending_bracket and broker._pending_bracket.get("active")
                and not broker._tick_trigger_active):
            bracket = broker._pending_bracket
            if bid > 0 and ofr > 0:
                spread = ofr - bid
                if spread <= broker._max_spread_pts:
                    triggered_dir = None
                    if ofr >= bracket["buy_price"]:
                        triggered_dir = "BUY"
                    elif bid <= bracket["sell_price"]:
                        triggered_dir = "SELL"

                    if triggered_dir:
                        broker._tick_trigger_active = True
                        bracket["active"] = False
                        # SKIP no-arm path: don't place order, just notify
                        if bracket.get("soft_arm"):
                            trigger_price = ofr if triggered_dir == "BUY" else bid
                            trigger_result = {
                                "direction": "LONG" if triggered_dir == "BUY" else "SHORT",
                                "fill_price": trigger_price,
                                "order_id": "",
                                "soft_arm": True,
                                "trigger_bid": bid,
                                "trigger_ofr": ofr,
                                "trigger_spread": round(spread, 2),
                            }
                            for cb in broker._on_trigger_callbacks:
                                await cb(trigger_result)
                            broker._tick_trigger_active = False
                            continue
                        lock = broker._get_entry_lock(broker.epic)
                        try:
                            await lock.acquire()
                            pos = await broker.get_position()
                            if pos["direction"] != "FLAT":
                                broker._tick_trigger_active = False
                                continue
                            result = await broker.place_market_order(triggered_dir, bracket["qty"])
                            trigger_result = {
                                "direction": "LONG" if triggered_dir == "BUY" else "SHORT",
                                "fill_price": result["avg_price"],
                                "order_id": result["order_id"],
                                "trigger_bid": bid,
                                "trigger_ofr": ofr,
                                "trigger_spread": round(spread, 2),
                            }
                            for cb in broker._on_trigger_callbacks:
                                await cb(trigger_result)
                        finally:
                            broker._tick_trigger_active = False
                            if lock.locked():
                                lock.release()

        # Yield to event loop periodically to let async tasks (morning_routine, bar5 event) run
        if i % YIELD_EVERY == 0:
            await asyncio.sleep(0)

    # Let all pending bar tasks complete
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    # EOD force close
    if signal.state.phase in (Phase.LONG, Phase.SHORT):
        await signal.end_of_day()

    # Collect trades
    trades = []
    for t in signal.state.trades:
        t["instrument"] = inst_name
        t["session"] = f"S{sn}"
        t["date"] = date_str
        trades.append(t)

    return trades


async def replay_day(inst_name, date_str):
    """Replay all sessions for one instrument on one day."""
    cfg = config.INSTRUMENTS[inst_name]
    epic = cfg["epic"]
    tz = ZoneInfo(cfg["timezone"])

    path = TICK_DIR / f"{epic}_{date_str}.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    df = df.dropna(subset=["utm", "bid", "ofr"])
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df["utm"] = pd.to_numeric(df["utm"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["utm", "bid", "ofr", "mid"])
    if len(df) < 100:
        return []
    df["dt"] = pd.to_datetime(df["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(tz)
    df = df.sort_values("dt").reset_index(drop=True)

    all_trades = []
    sessions = [(sn, cfg[f"s{sn}_open_hour"], cfg[f"s{sn}_open_minute"])
                for sn in (1, 2, 3) if f"s{sn}_open_hour" in cfg]

    for sn, oh, om in sessions:
        trades = await replay_session(inst_name, sn, cfg, df, date_str)
        all_trades.extend(trades)

    return all_trades


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--inst", default=None, choices=["DAX", "US30", "NIKKEI"])
    args = ap.parse_args()

    files = sorted(TICK_DIR.glob("*.csv"))
    dates = sorted(set(f.stem.split("_")[-1] for f in files))
    if args.date:
        dates = [d for d in dates if d == args.date]

    instruments = [args.inst] if args.inst else ["DAX", "US30", "NIKKEI"]

    print("=" * 80)
    print("  REPLAY — actual Signal class, event-driven, on recorded tick data")
    print("=" * 80)

    all_trades = []
    for inst in instruments:
        for date_str in dates:
            trades = await replay_day(inst, date_str)
            all_trades.extend(trades)

    print(f"\n{'Signal':<15} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>8} {'Reason':<15}")
    print("-" * 70)
    for t in all_trades:
        entry = t.get("entry", t.get("entry_price", 0))
        exit_p = t.get("exit", t.get("exit_price", 0))
        pnl = t.get("pnl_pts", 0)
        reason = t.get("exit_reason", t.get("reason", "?"))
        print(f"{t['instrument']}_{t['session']:<8} {t.get('direction','?'):<6} "
              f"{entry:>9.1f} {exit_p:>9.1f} {pnl:>+8.1f} {reason:<15}")

    if all_trades:
        net = sum(t.get("pnl_pts", 0) for t in all_trades)
        wins = sum(1 for t in all_trades if t.get("pnl_pts", 0) > 0)
        print(f"\nTotal: {len(all_trades)} trades, net {net:+.0f}pts, {wins} winners")
    else:
        print("\nNo trades generated.")


if __name__ == "__main__":
    asyncio.run(main())
