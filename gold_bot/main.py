"""
main.py -- Strategy 3: Gold 15-min ORB + Weekly ORB
Aggregates 5-min Lightstreamer candles into 15-min bars, then runs strategy.

Exposes:
  - setup(shared, stream_mgr, tg_send) -- called from run_all.py at startup
  - graceful_shutdown()                 -- called on SIGTERM
"""

import asyncio
import logging
from datetime import datetime, timezone
from collections import deque

from gold_bot import config
from gold_bot.strategy import ORBStrategy
from shared import journal_db

logger = logging.getLogger("GOLD_ORB")

# Module-level state (injected by run_all.py)
broker_shared = None
stream_mgr = None
strategies = {}
_tg_send = None
_pending_entries = {}  # sess_key -> {direction, entry_price, stake, session, entry_time}

# 15-min candle aggregator per epic
_aggregators = {}


class CandleAggregator:
    """
    Aggregates 5-min candles from Lightstreamer into 15-min bars.
    Fires callback every 3 completed 5-min candles on 15-min boundaries.
    """

    def __init__(self, epic: str, tf_minutes: int = 15):
        self.epic = epic
        self.tf_minutes = tf_minutes
        self.bars_needed = tf_minutes // 5  # 3 for 15-min
        self._buffer: list = []
        self._callbacks: list = []
        self._aligned: bool = False  # True once first bar aligns to boundary

    def register_callback(self, cb):
        self._callbacks.append(cb)

    def on_5min_candle(self, candle: dict):
        """Called on each 5-min candle close from Lightstreamer."""
        bar = _parse_candle(candle)
        if bar is None:
            return

        ts = bar["timestamp"]
        if not isinstance(ts, datetime):
            ts = datetime.now(timezone.utc)

        minute = ts.minute if hasattr(ts, 'minute') else 0

        # Wait for alignment: first bar of a 15-min group starts at :00, :15, :30, :45
        if not self._aligned:
            if minute % self.tf_minutes == 0:
                self._aligned = True
                logger.info("[%s] Aggregator aligned at :%02d", self.epic, minute)
            else:
                logger.debug("[%s] Aggregator skipping unaligned bar at :%02d", self.epic, minute)
                return

        self._buffer.append(bar)

        # Emit when we have 3 bars (one complete 15-min group)
        if len(self._buffer) >= self.bars_needed:
            self._emit_aggregated()

    def _emit_aggregated(self):
        """Merge buffered 5-min bars into one 15-min bar and fire callbacks."""
        if len(self._buffer) < self.bars_needed:
            return

        # Take last N bars
        bars = self._buffer[-self.bars_needed:]
        self._buffer.clear()

        agg = {
            "Open": bars[0]["Open"],
            "High": max(b["High"] for b in bars),
            "Low": min(b["Low"] for b in bars),
            "Close": bars[-1]["Close"],
            "timestamp": bars[-1]["timestamp"],  # Use last bar's timestamp
        }

        logger.debug("[%s] 15m bar emitted: O=%.2f H=%.2f L=%.2f C=%.2f @ %s",
                    self.epic, agg["Open"], agg["High"], agg["Low"],
                    agg["Close"], agg["timestamp"])

        for cb in self._callbacks:
            try:
                asyncio.get_event_loop().create_task(cb(agg))
            except Exception as e:
                logger.error("Aggregator callback error: %s", e)


def _parse_candle(candle: dict) -> dict:
    """
    Parse IG Lightstreamer candle or IGStreamManager bar dict.
    IGStreamManager already gives us {time, Open, High, Low, Close}.
    """
    # IGStreamManager format (from shared/ig_stream.py)
    if "Open" in candle and "Close" in candle:
        ts = candle.get("time") or candle.get("timestamp")
        if ts is None:
            ts = datetime.now(timezone.utc)
        return {
            "Open": float(candle["Open"]),
            "High": float(candle["High"]),
            "Low": float(candle["Low"]),
            "Close": float(candle["Close"]),
            "timestamp": ts,
        }

    # Raw Lightstreamer format
    bid_open = _get_float(candle, "BID_OPEN", "bid_open", "OFR_OPEN")
    bid_high = _get_float(candle, "BID_HIGH", "bid_high", "OFR_HIGH")
    bid_low = _get_float(candle, "BID_LOW", "bid_low", "OFR_LOW")
    bid_close = _get_float(candle, "BID_CLOSE", "bid_close", "OFR_CLOSE")
    ofr_open = _get_float(candle, "OFR_OPEN", "ofr_open")
    ofr_close = _get_float(candle, "OFR_CLOSE", "ofr_close")

    o = (bid_open + ofr_open) / 2 if ofr_open else bid_open
    h = bid_high
    l = bid_low
    c = (bid_close + ofr_close) / 2 if ofr_close else bid_close

    ts = candle.get("UTM") or candle.get("utm") or candle.get("timestamp")
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    elif ts is None:
        ts = datetime.now(timezone.utc)

    return {"Open": o, "High": h, "Low": l, "Close": c, "timestamp": ts}


def _get_float(d: dict, *keys) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


async def setup(shared, stream, tg_send=None):
    """
    Initialize Strategy 3.
    Called once from run_all.py after IG session is established.
    """
    global broker_shared, stream_mgr, strategies, _tg_send
    broker_shared = shared
    stream_mgr = stream
    _tg_send = tg_send

    if config.ENABLE_GOLD:
        strategies["GOLD"] = ORBStrategy("GOLD")

        # Create 15-min aggregator
        agg = CandleAggregator(config.GOLD_EPIC, config.CANDLE_TF_MINUTES)
        agg.register_callback(lambda bar: _process_bar("GOLD", bar))
        _aggregators[config.GOLD_EPIC] = agg

        # Subscribe to 5-min candles + ticks
        await stream.subscribe_candles(config.GOLD_EPIC)
        await stream.subscribe_ticks(config.GOLD_EPIC)

        # Route 5-min candles through aggregator
        stream.register_candle_callback(config.GOLD_EPIC, _on_5min_candle)
        logger.info("Gold 15m ORB + Weekly enabled: %s", config.GOLD_EPIC)

    # Check for orphaned Gold positions on IG
    if config.ENABLE_GOLD:
        try:
            positions = await shared.rest_call(shared.ig.fetch_open_positions)
            gold_epic = config.GOLD_EPIC
            # IG returns a DataFrame — convert to list of dicts
            import pandas as pd
            if isinstance(positions, pd.DataFrame):
                positions = positions.to_dict("records") if not positions.empty else []
            for p in (positions or []):
                mkt = p.get("market", {})
                pos = p.get("position", {})
                if mkt.get("epic") == gold_epic and pos.get("size", 0) > 0:
                    direction = pos.get("direction", "?")
                    size = pos.get("size", 0)
                    level = pos.get("level", 0)
                    deal_id = pos.get("dealId", "?")
                    logger.warning("ORPHAN GOLD POSITION: %s x%s @ %s deal=%s",
                                   direction, size, level, deal_id)
                    if _tg_send:
                        await _tg_send(
                            f"⚠️ <b>ORPHAN GOLD POSITION</b>\n"
                            f"{direction} x{size} @ {level}\n"
                            f"Deal: {deal_id}\n"
                            f"Bot restarted — position not managed.\n"
                            f"Close manually or it will be picked up if session is active."
                        )
        except Exception as e:
            logger.error("Gold orphan check failed: %s", e)

    # S3 startup info is now included in the combined startup alert (shared/monitoring.py)


async def _on_5min_candle(candle: dict):
    """Callback from Lightstreamer — route to aggregator (must be async for IGStreamManager)."""
    logger.debug("5min candle → aggregator: %s buf=%d",
                candle.get("time", "?"),
                len(list(_aggregators.values())[0]._buffer) if _aggregators else -1)
    for epic, agg in _aggregators.items():
        agg.on_5min_candle(candle)
        break  # Only one epic per callback registration


async def _process_bar(instrument: str, bar: dict):
    """Process an aggregated 15-min bar through the strategy."""
    if instrument not in strategies:
        return

    strategy = strategies[instrument]

    # Snapshot state before to detect transitions
    prev_states = {}
    for key, sess in strategy.sessions.items():
        prev_states[key] = sess.state.value

    action = strategy.on_bar(bar)

    # Alert on state transitions
    if _tg_send:
        for key, sess in strategy.sessions.items():
            cur = sess.state.value
            prev = prev_states.get(key)
            if prev != cur:
                if cur == "WATCHING":
                    rng = sess.range_high - sess.range_low
                    await _tg_send(
                        f"📊 <b>RANGE SET</b> {instrument}/{sess.session_name}\n"
                        f"H: {sess.range_high:.2f} | L: {sess.range_low:.2f}\n"
                        f"Size: {rng:.2f}"
                    )
                elif prev == "WATCHING" and cur == "IDLE" and not action:
                    await _tg_send(
                        f"⏭ <b>SESSION SKIP</b> {instrument}/{sess.session_name}\n"
                        f"No breakout — range expired"
                    )

    if action is None:
        return

    await _execute_action(instrument, strategy, action)


async def _execute_action(instrument: str, strategy: ORBStrategy, action: dict):
    """Execute trade action via shared IG session."""
    cfg = config.INSTRUMENTS[instrument]

    if action["action"] == "OPEN":
        try:
            # Check spread
            market = await broker_shared.rest_call(
                broker_shared.ig.fetch_market_by_epic, cfg["epic"]
            )
            snapshot = market.get("snapshot", {})
            bid = snapshot.get("bid", 0)
            offer = snapshot.get("offer", 0)
            current_spread = offer - bid if bid and offer else 999
            max_spread = cfg["spread"] * 3

            if current_spread > max_spread:
                logger.warning("[%s] Spread too wide: %.4f > %.4f, skip",
                               instrument, current_spread, max_spread)
                if _tg_send:
                    await _tg_send(
                        f"⚠️ <b>SPREAD SKIP</b> {instrument}\n"
                        f"Spread: {current_spread:.2f} > max {max_spread:.2f}\n"
                        f"Signal: {action['direction']} | {action['session']}"
                    )
                return

            result = await broker_shared.rest_call(
                broker_shared.ig.create_open_position,
                currency_code="GBP",
                direction=action["direction"],
                epic=cfg["epic"],
                expiry="DFB",
                force_open=True,
                guaranteed_stop=False,
                level=None,
                limit_distance=action["limit_distance"],
                limit_level=None,
                order_type="MARKET",
                quote_id=None,
                size=action["stake"],
                stop_distance=action["stop_distance"],
                stop_level=None,
                trailing_stop=False,
                trailing_stop_increment=None,
            )

            deal_ref = result.get("dealReference")
            if deal_ref:
                confirm = await broker_shared.rest_call(
                    broker_shared.ig.fetch_deal_by_deal_reference, deal_ref
                )
                status = confirm.get("dealStatus")
                if status == "ACCEPTED":
                    deal_id = confirm.get("dealId")
                    level = confirm.get("level", 0)
                    strategy.register_fill(action["sess_key"], deal_id, level)

                    msg = (
                        f"<b>ORB {action['direction']}</b> {instrument}\n"
                        f"Entry: {level:.2f}\n"
                        f"Stop: {action['stop_distance']:.2f} pts\n"
                        f"Target: {action['limit_distance']:.2f} pts\n"
                        f"Stake: GBP{action['stake']:.2f}/pt\n"
                        f"Session: {action['session']}"
                    )
                    logger.info("[%s] Filled: %s @ %.2f deal=%s",
                                instrument, action["direction"], level, deal_id)
                    # Store entry for journal on close
                    _pending_entries[action["sess_key"]] = {
                        "direction": action["direction"],
                        "entry_price": level,
                        "stake": action["stake"],
                        "session": action["session"],
                        "entry_time": datetime.now(timezone.utc).strftime("%H:%M"),
                        "stop_distance": action["stop_distance"],
                        "limit_distance": action["limit_distance"],
                    }
                    if _tg_send:
                        await _tg_send(msg)
                else:
                    reason = confirm.get("reason", "UNKNOWN")
                    logger.error("[%s] Order rejected: %s", instrument, reason)
                    if _tg_send:
                        await _tg_send(f"ORB {instrument} REJECTED: {reason}")

        except Exception as e:
            logger.exception("[%s] Failed to open: %s", instrument, e)
            if _tg_send:
                await _tg_send(
                    f"❌ <b>ORB OPEN FAILED</b> {instrument}\n"
                    f"{action['direction']} | {action['session']}\n"
                    f"Error: {e}"
                )

    elif action["action"] == "AMEND_STOP":
        try:
            result = await broker_shared.rest_call(
                broker_shared.ig.update_open_position,
                limit_level=None,
                stop_level=action["new_stop"],
                deal_id=action["deal_id"],
            )
            logger.info("[%s] Breakeven stop amended: deal=%s stop=%.2f",
                        name, action["deal_id"], action["new_stop"])
            await _tg_send(
                f"[S2] BREAKEVEN GOLD\n"
                f"Deal: {action['deal_id']}\n"
                f"Stop moved to entry: {action['new_stop']:.2f}"
            )
        except Exception as e:
            logger.error("[%s] Amend stop failed: %s", name, e)
            await _tg_send(
                f"[S2] ❌ BREAKEVEN FAILED GOLD\n"
                f"Deal: {action['deal_id']}\n"
                f"Error: {e}"
            )

    elif action["action"] == "CLOSE":
        try:
            result = await broker_shared.rest_call(
                broker_shared.ig.close_open_position,
                deal_id=action["deal_id"],
                direction=action["direction"],
                epic=None, expiry="DFB", level=None,
                order_type="MARKET", quote_id=None,
                size=action["stake"],
            )
            deal_ref = result.get("dealReference")
            if deal_ref:
                confirm = await broker_shared.rest_call(
                    broker_shared.ig.fetch_deal_by_deal_reference, deal_ref
                )
                if confirm.get("dealStatus") == "ACCEPTED":
                    level = confirm.get("level", 0)
                    close_result = strategy.register_close(
                        action["sess_key"], level, action["reason"]
                    )

                    # Journal the trade
                    entry_info = _pending_entries.pop(action["sess_key"], {})
                    if close_result:
                        trade_dict = {
                            "direction": entry_info.get("direction", action["direction"]),
                            "entry_price": entry_info.get("entry_price", 0),
                            "exit_price": level,
                            "entry_intended": entry_info.get("entry_price", 0),
                            "exit_intended": level,
                            "entry_time": entry_info.get("entry_time", ""),
                            "exit_time": datetime.now(timezone.utc).strftime("%H:%M"),
                            "pnl_pts": close_result["pnl_pts"],
                            "pnl_gbp": close_result["pnl_gbp"],
                            "stake_per_point": entry_info.get("stake", action["stake"]),
                            "exit_reason": action["reason"],
                            "bar_range": entry_info.get("stop_distance", 0),
                            "range_flag": entry_info.get("session", ""),
                        }
                        try:
                            journal_db.insert_trade("GOLD", trade_dict, state=None)
                        except Exception as je:
                            logger.error("Gold journal write failed: %s", je)

                    if close_result and _tg_send:
                        sign = "+" if close_result["r_multiple"] > 0 else ""
                        await _tg_send(
                            f"<b>ORB CLOSE</b> {instrument}\n"
                            f"P&L: {sign}{close_result['pnl_pts']:.2f} pts "
                            f"({sign}{close_result['r_multiple']:.2f}R)\n"
                            f"GBP: {sign}{close_result['pnl_gbp']:.2f}\n"
                            f"Reason: {action['reason']}"
                        )

        except Exception as e:
            logger.exception("[%s] Failed to close: %s", instrument, e)
            if _tg_send:
                await _tg_send(
                    f"❌ <b>ORB CLOSE FAILED</b> {instrument}\n"
                    f"Deal: {action.get('deal_id', '?')}\n"
                    f"Error: {e}"
                )


async def graceful_shutdown():
    """Close all open positions (session ORB + weekly ORB)."""
    logger.info("Strategy 3 shutting down...")
    for name, strategy in strategies.items():
        # Close session positions
        for key, state in strategy.sessions.items():
            if state.state.value == "ENTERED" and state.deal_id:
                close_dir = "SELL" if state.direction == "BUY" else "BUY"
                try:
                    await broker_shared.rest_call(
                        broker_shared.ig.close_open_position,
                        deal_id=state.deal_id,
                        direction=close_dir,
                        epic=None, expiry="DFB", level=None,
                        order_type="MARKET", quote_id=None,
                        size=state.stake,
                    )
                    logger.info("[%s] Shutdown: closed session %s", name, state.deal_id)
                except Exception as e:
                    logger.error("[%s] Shutdown close failed: %s", name, e)

        # Close weekly position
        if strategy.weekly.deal_id:
            close_dir = "SELL" if strategy.weekly.direction == "BUY" else "BUY"
            try:
                await broker_shared.rest_call(
                    broker_shared.ig.close_open_position,
                    deal_id=strategy.weekly.deal_id,
                    direction=close_dir,
                    epic=None, expiry="DFB", level=None,
                    order_type="MARKET", quote_id=None,
                    size=strategy.weekly.stake,
                )
                logger.info("[%s] Shutdown: closed weekly %s",
                            name, strategy.weekly.deal_id)
            except Exception as e:
                logger.error("[%s] Shutdown weekly close failed: %s", name, e)

    logger.info("Strategy 3 shutdown complete")
