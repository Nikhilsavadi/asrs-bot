"""
US30 Bot main — S&P 500 ASRS strategy.
Reuses DAX bot logic with US30-specific config.
"""

import logging
import asyncio
from datetime import datetime

# Monkey-patch: make dax_bot code use spx_bot.config instead of dax_bot.config
# This is done by creating US30-specific instances with the right config.
from spx_bot import config
from dax_bot.strategy import (
    DailyState, Phase, calculate_levels, update_trail, process_stop_hit,
    check_add_to_winners, process_add_fill, classify_gap, get_bar,
    analyse_context, should_use_bar5,
)
from dax_bot.broker_ig import IGBroker
import httpx
from dax_bot.overnight import calculate_overnight_range, OvernightBias, OvernightResult

logger = logging.getLogger("US30_ASRS")

# Module-level state (initialized by run_all.py)
broker: IGBroker = None
_bar4_triggered = False
_tg_send = None  # Telegram send function, set by init()


async def _alert(text: str):
    """Send Telegram alert with [S3 US30] prefix."""
    if _tg_send:
        await _tg_send("[S3 US30] " + text)
    else:
        logger.info(f"ALERT (no TG): {text}")

# Override STATE_FILE for US30
import dax_bot.strategy as _strat
_US30_STATE_FILE = config.STATE_FILE


class US30DailyState(DailyState):
    """DailyState that uses US30 state file."""

    def save(self):
        import os, json
        from dataclasses import asdict
        os.makedirs(os.path.dirname(_US30_STATE_FILE), exist_ok=True)
        with open(_US30_STATE_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "US30DailyState":
        import os, json
        today = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
        try:
            if os.path.exists(_US30_STATE_FILE):
                with open(_US30_STATE_FILE) as f:
                    data = json.load(f)
                if data.get("date") == today:
                    state = cls()
                    for k, v in data.items():
                        if hasattr(state, k):
                            setattr(state, k, v)
                    return state
        except Exception as e:
            logger.warning(f"State load failed: {e}")
        state = cls()
        state.date = today
        return state


async def init(shared_session, stream_manager, tg_send=None):
    """Initialize US30 broker. Called by run_all.py."""
    global broker, _tg_send, _shared_session

    broker = IGBroker(shared_session, stream_manager, config.IG_EPIC, "GBP")
    broker.register_trigger_callback(on_tick_trigger)
    broker.register_order_handler(_handle_fill_event)
    _tg_send = tg_send
    _shared_session = shared_session
    logger.info(f"US30 bot initialized: {config.IG_EPIC}")


_shared_session = None
_prefetch_cache: dict = {}  # Pre-fetched data from warmup (gap, overnight bars)
_bar4_triggered = False  # Prevent double-trigger from callback + schedule

# Trade stream exit levels — used when IG stop fires server-side and /confirms returns 404
_last_stream_exit_levels: dict[str, float] = {}


async def _check_slippage(state: US30DailyState, fill_price: float) -> bool:
    """Check if fill slipped too far from trigger level. Returns True if OK, False if excessive.
    Slippage limit is proportional to the trade's initial risk (bar range), not a fixed number.
    If slippage < 50% of risk -> keep trade (slightly worse R:R but still valid).
    If slippage > 50% of risk -> close immediately (trade invalidated).
    """
    trigger_price = state.buy_level if state.direction == "LONG" else state.sell_level
    slippage = abs(fill_price - trigger_price)
    initial_risk = state.bar_range + config.BUFFER_PTS * 2  # bar range + both buffers
    max_slip = initial_risk * config.MAX_SLIPPAGE_PCT  # default 50% of risk
    if slippage > max_slip:
        logger.error(f"EXCESSIVE SLIPPAGE: fill={fill_price}, trigger={trigger_price}, "
                     f"slip={slippage:.1f}pts > {max_slip:.1f}pts ({config.MAX_SLIPPAGE_PCT*100:.0f}% of {initial_risk:.1f}pt risk)")
        await _alert(
            "EXCESSIVE SLIPPAGE -- CLOSING\n"
            f"Trigger: {trigger_price} | Fill: {fill_price}\n"
            f"Slippage: {slippage:.1f} pts (limit: {max_slip:.1f} = {config.MAX_SLIPPAGE_PCT*100:.0f}% of risk)\n"
            "Closing position immediately."
        )
        try:
            await broker.close_position()
            if fill_price:
                process_stop_hit(state, fill_price)
                if state.trades and state.trades[-1].get("exit"):
                    state.trades[-1]["exit_reason"] = "SLIPPAGE_CLOSE"

            # Slippage close = missed entry, not a failed trade.
            # Re-arm the bracket for another attempt if entries remain.
            if state.entries_used < config.MAX_ENTRIES:
                state.phase = Phase.ORDERS_PLACED
                state.direction = ""
                state.entry_price = 0
                logger.info(f"Slippage close -- re-arming bracket (entry {state.entries_used}/{config.MAX_ENTRIES})")
                await _alert("Re-arming bracket after slippage close")
                await broker.place_oca_bracket(
                    state.buy_level, state.sell_level,
                    qty=state.position_size or config.NUM_CONTRACTS,
                    oca_group=f"US30_{state.date}_{state.entries_used + 1}"
                )
            else:
                state.phase = Phase.DONE
                logger.info("Slippage close -- max entries reached, done for today")
            state.save()
        except Exception as e:
            logger.error(f"Slippage close failed: {e}")
            await _alert(f"Slippage close FAILED: {e}\nManual intervention required!")
        return False
    return True


def _check_daily_loss_limit() -> bool:
    """Return True if daily loss limit is breached."""
    from shared.journal_db import get_trades_for_date
    today_str = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
    trades = get_trades_for_date(today_str, instrument="US30")
    day_pnl = sum(t.get("pnl_gbp", 0) or 0 for t in trades)
    if day_pnl <= -config.MAX_DAILY_LOSS_GBP:
        logger.warning(f"Daily loss limit hit: {day_pnl:.2f} GBP (limit: -{config.MAX_DAILY_LOSS_GBP})")
        return True
    return False


async def _handle_fill_event(data):
    """
    Event-driven handler for IG streaming trade updates (OPU dict).
    Currently used for logging + storing exit levels for accurate PnL
    when IG server-side stops fire and /confirms returns 404.
    """
    if not isinstance(data, dict):
        return

    deal_status = data.get("dealStatus", data.get("status", ""))
    deal_id = data.get("dealId", "")
    level = data.get("level", 0)
    direction = data.get("direction", "")

    logger.info(f"[US30] Trade stream event: dealId={deal_id} status={deal_status} "
                f"direction={direction} level={level}")

    # Store last stream exit level for accurate PnL when IG stop fires
    if level and float(level) > 0:
        _last_stream_exit_levels[deal_id] = float(level)


async def on_candle_complete(bar: dict):
    """Called on every CONS_END bar. Triggers morning routine when bar 4 arrives."""
    global _bar4_triggered
    from zoneinfo import ZoneInfo
    cet = ZoneInfo("Europe/Berlin")

    bar_time = bar.get("time")  # CET datetime
    if not bar_time:
        return

    today_cet = datetime.now(cet).date()
    if bar_time.date() != today_cet:
        return

    # US30 opens at 14:30 CET. Bar number from that time.
    us_open_minutes = 14 * 60 + 30  # 14:30 CET
    bar_minutes = bar_time.hour * 60 + bar_time.minute
    minutes_since_open = bar_minutes - us_open_minutes
    if minutes_since_open < 0:
        return
    bar_number = minutes_since_open // 5 + 1

    if bar_number == 4 and not _bar4_triggered:
        state = US30DailyState.load()
        if state.phase == Phase.IDLE:
            _bar4_triggered = True
            logger.info(f"US30 Bar 4 complete — triggering morning routine (event-driven)")
            await morning_routine()

    # S2 event-driven trigger: bar 4 of S2 session (11:00 ET = 16:00 CET)
    if getattr(config, 'SESSION2_ENABLED', False):
        s2_open_minutes = getattr(config, 'SESSION2_HOUR_CET', 16) * 60  # 11:00 ET = 16:00 CET
        s2_bar_minutes = bar_minutes - s2_open_minutes
        if s2_bar_minutes >= 0:
            s2_bar_number = s2_bar_minutes // 5 + 1
            if s2_bar_number == 4:
                state = US30DailyState.load()
                if state.s2_phase == "IDLE":
                    logger.info("US30 S2 Bar 4 complete — triggering session2_routine (event-driven)")
                    await session2_routine()


async def on_tick_trigger(trigger: dict):
    """Called instantly when a tick crosses bracket levels (sub-second).
    Handles fill processing, stop placement, Telegram alert."""
    state = US30DailyState.load()
    if state.phase != Phase.ORDERS_PLACED:
        return

    direction = trigger["direction"]
    fill_price = trigger["fill_price"]
    order_id = trigger.get("order_id", "")

    # Update state
    state.phase = Phase.LONG_ACTIVE if direction == "LONG" else Phase.SHORT_ACTIVE
    state.direction = direction
    state.entry_price = fill_price
    # Stop at the opposite side of the bar range (not the OCA bracket level which may be 999999)
    state.initial_stop = state.bar_high + config.BUFFER_PTS if direction == "SHORT" else state.bar_low - config.BUFFER_PTS
    state.trailing_stop = state.initial_stop
    state.contracts_active = state.position_size
    state.entries_used += 1
    state.stop_order_id = order_id
    state.last_add_price = fill_price
    state.max_favourable = fill_price  # Will be updated by trail
    intended = state.sell_level if direction == "SHORT" else state.buy_level
    state.trades.append({
        "entry": fill_price,
        "exit": 0,
        "exit_intended": 0,
        "entry_intended": intended,
        "direction": direction,
        "entry_time": datetime.now(config.TZ_ET).strftime("%H:%M:%S"),
        "entry_slippage": round(abs(fill_price - intended), 1),
        "exit_slippage": 0,
        "slippage_total": round(abs(fill_price - intended), 1),
        "pnl_pts": 0,
        "pnl_per_contract": 0,
        "mfe": 0,
        "exit_reason": "",
        "contracts_stopped": 0,
        "tp1_filled": False,
        "tp2_filled": False,
    })
    state.save()
    logger.info(f"US30 tick trigger fill: {direction} @ {fill_price}")

    # Slippage guard
    if not await _check_slippage(state, fill_price):
        return

    # Place stop on IG
    try:
        await broker.modify_stop(order_id, state.trailing_stop)
        logger.info(f"US30 stop set @ {state.trailing_stop}")
    except Exception as e:
        logger.error(f"US30 stop placement failed: {e}")

    # Telegram alert
    await _alert(
        f"📈 <b>US30 {direction}</b>\n"
        f"Entry: {fill_price}\n"
        f"Stop: {state.trailing_stop}\n"
        f"Range: {state.bar_range:.1f}pts ({state.range_flag})\n"
    )


async def health_check():
    """14:00 UK — US30 health check before US open."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    logger.info("═══ US30 HEALTH CHECK ═══")
    mode = "DEMO" if config.IG_DEMO else "LIVE"

    price = None
    status = "Unknown"
    if broker:
        try:
            price = await broker.get_current_price()
            status = "Connected" if price else "No price"
        except Exception:
            status = "Unreachable"

    price_str = f"{price:.1f}" if price else "N/A"
    stream_bars = broker.get_streaming_bar_count() if broker else 0

    msg = (
        f"<b>US30 Health Check</b> [{mode}]\n"
        f"{now.strftime('%Y-%m-%d %H:%M')} ET\n"
        f"IG: {status}\n"
        f"Epic: {config.IG_EPIC}\n"
        f"US30: {price_str}\n"
        f"Streaming bars: {stream_bars}\n"
        f"Morning routine at {config.MORNING_HOUR}:{config.MORNING_MINUTE:02d} ET"
    )
    await _alert(msg)
    logger.info(f"Health check: IG={status}, US30={price_str}, bars={stream_bars}")


async def pre_trade_warmup():
    """14:20 UK (09:20 ET) — Verify connections before bar 4 window."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    logger.info("═══ US30 PRE-TRADE WARMUP ═══")
    issues = []

    # REST check
    if _shared_session:
        try:
            await _shared_session.keepalive()
            logger.info("Pre-warm: REST session alive")
        except Exception as e:
            issues.append(f"REST check failed: {e}")

    # Stream check
    if broker and _shared_session:
        try:
            stream_ok = await _shared_session.check_stream_health(
                broker._stream, config.IG_EPIC
            )
            if stream_ok:
                logger.info("Pre-warm: Lightstreamer alive")
            else:
                issues.append("Lightstreamer stale — resubscribed")
        except Exception as e:
            issues.append(f"Stream check failed: {e}")

    bar_count = broker.get_streaming_bar_count() if broker else 0
    logger.info(f"Pre-warm: {bar_count} streaming bars")

    # Pre-fetch gap and overnight data (cached for morning routine)
    try:
        prev_df = await broker.get_5min_bars("2 D")
        if not prev_df.empty:
            _prefetch_cache["prev_df"] = prev_df
            logger.info(f"Pre-warm: cached {len(prev_df)} bars for gap/overnight")
    except Exception as e:
        logger.warning(f"Pre-warm: bar pre-fetch failed: {e}")

    if issues:
        await _alert(
            "<b>Pre-trade warmup</b>\n"
            + "\n".join(f"- {i}" for i in issues) +
            "\nRecovery attempted. ~30 min to morning routine."
        )
    else:
        logger.info("Pre-warm: all systems OK")


async def stream_alive_check():
    """14:40 UK (09:40 ET) — Check bars are flowing before bar 4 closes."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    bar_count = broker.get_streaming_bar_count() if broker else 0
    if bar_count == 0:
        logger.warning("Stream check: No US30 bars — attempting recovery")
        if broker and _shared_session:
            recovered = await _shared_session.check_stream_health(
                broker._stream, config.IG_EPIC
            )
            if not recovered:
                await _alert(
                    "<b>Stream check FAILED</b>\n"
                    "No US30 bars + recovery failed.\n"
                    "Morning routine will use REST fallback."
                )
    else:
        logger.info(f"Stream check: {bar_count} US30 bars — OK")


_morning_running = False

async def morning_routine():
    """09:51 ET — Calculate levels from bar 4/5 and place orders."""
    global _morning_running
    if _morning_running:
        logger.warning("US30 morning routine already running — skipping duplicate")
        return
    _morning_running = True
    try:
        await _morning_routine_inner()
    finally:
        _morning_running = False

async def _morning_routine_inner():
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    from shared.holidays import is_holiday
    if is_holiday(now.date(), "US30"):
        logger.info(f"US30 market holiday ({now.date()}) — skipping")
        await _alert(f"📅 US30 market holiday today — no trading")
        return

    logger.info("═══ US30 MORNING ROUTINE ═══")
    state = US30DailyState.load()
    if state.phase != Phase.IDLE:
        logger.info("Already processed today")
        return

    if not broker:
        logger.error("Broker not initialized")
        return

    # Get today's bars
    df = broker.get_streaming_bars_df()
    if df.empty:
        try:
            df = await broker.get_5min_bars("1 D")
        except Exception as e:
            logger.error(f"Failed to get bars: {e}")

    if df.empty:
        await _alert("ASRS ERROR\nNo bar data available for US30.\nCheck logs.")
        return

    today_date = now.date()

    # FAST PATH: Get bar 4 from streaming FIRST, arm bracket immediately
    # Gap and overnight are done AFTER bracket is armed
    # Calculate levels directly — streaming bars are in CET, US open is 14:30 CET
    # We can't use DAX's candle_number() which is hardcoded to 09:00 CET
    from zoneinfo import ZoneInfo
    cet = ZoneInfo("Europe/Berlin")
    now_cet = datetime.now(cet)
    us_open_cet = now_cet.replace(hour=14, minute=30, second=0, microsecond=0)

    # Filter bars from US open onwards — use hour/minute comparison (avoids timezone bugs)
    import asyncio
    us_open_h = us_open_cet.hour
    us_open_m = us_open_cet.minute
    logger.info(f"Session open: 09:30 ET = {us_open_h:02d}:{us_open_m:02d} CET")
    for retry in range(4):
        df = broker.get_streaming_bars_df()
        if not df.empty:
            us_bars = df[
                (df.index.hour > us_open_h) |
                ((df.index.hour == us_open_h) & (df.index.minute >= us_open_m))
            ].sort_index().head(8)
        else:
            us_bars = pd.DataFrame()
        logger.info(f"US30 bars from {us_open_h:02d}:{us_open_m:02d} CET (attempt {retry+1}): {len(us_bars)} bars")
        if len(us_bars) >= 4:
            break
        if retry < 3:
            logger.info(f"Waiting 15s for bar 4...")
            await asyncio.sleep(15)

    if len(us_bars) < 4:
        await _alert(f"US30 ASRS ERROR\nOnly {len(us_bars)} bars since US open (need 4)\nRetried 4 times over 45s")
        return

    # Bar 4 = 4th bar from US open
    bar4 = us_bars.iloc[3]
    bar4_range = bar4["High"] - bar4["Low"]

    # Hybrid: bar 4 for narrow, bar 5 for normal/wide
    if bar4_range > config.NARROW_RANGE and len(us_bars) >= 5:
        signal_bar = us_bars.iloc[4]
        state.bar_number = 5
        logger.info(f"Using bar 5 (bar4 range {bar4_range:.1f} > NARROW {config.NARROW_RANGE})")
    else:
        signal_bar = bar4
        state.bar_number = 4

    state.bar_high = round(signal_bar["High"], 1)
    state.bar_low = round(signal_bar["Low"], 1)
    state.bar_range = round(signal_bar["High"] - signal_bar["Low"], 1)

    if state.bar_range < 3:
        await _alert(f"US30 ASRS: Bar range too small ({state.bar_range}pts)")
        return

    # Range flag
    if state.bar_range <= config.NARROW_RANGE:
        state.range_flag = "NARROW"
    elif state.bar_range >= config.WIDE_RANGE:
        state.range_flag = "WIDE"
    else:
        state.range_flag = "NORMAL"

    # Set levels
    state.buy_level = round(state.bar_high + config.BUFFER_PTS, 1)
    state.sell_level = round(state.bar_low - config.BUFFER_PTS, 1)
    state.phase = Phase.LEVELS_SET
    state.oca_group = f"US30_{state.date}_1"
    state.save()

    events = ["LEVELS_SET"]
    logger.info(f"Levels: Buy={state.buy_level} Sell={state.sell_level} "
                f"Bar={state.bar_number} Range={state.bar_range} ({state.range_flag})")

    # Risk cap: if bar range risk > MAX_RISK_GBP, tighten the stop
    risk_pts = state.bar_range + config.BUFFER_PTS * 2  # full stop distance
    risk_gbp = risk_pts * 1.0  # at £1/pt
    if risk_gbp > config.MAX_RISK_GBP:
        # Tighten stop to cap risk at MAX_RISK_GBP
        max_stop_distance = config.MAX_RISK_GBP  # at £1/pt, pts = £
        logger.info(f"Risk cap: {risk_gbp:.0f} > {config.MAX_RISK_GBP:.0f} — tightening stop from {risk_pts:.1f}pts to {max_stop_distance:.1f}pts")
        # Adjust levels: keep entry same, move stop closer
        state.bar_high = round(state.sell_level + max_stop_distance, 1)  # for SHORT: stop = entry + max_distance
        state.bar_low = round(state.buy_level - max_stop_distance, 1)   # for LONG: stop = entry - max_distance
        state.bar_range = max_stop_distance
        await _alert(f"[S3 US30] Risk capped: {risk_pts:.0f}pts → {max_stop_distance:.0f}pts (£{config.MAX_RISK_GBP:.0f} max)")

    # Position sizing
    state.position_size = min(config.NUM_CONTRACTS, config.MAX_CONTRACTS)
    if state.range_flag == "NARROW" and config.NARROW_STD_MULTIPLIER > 1:
        state.position_size = min(config.NUM_CONTRACTS * config.NARROW_STD_MULTIPLIER, config.MAX_CONTRACTS)
    state.save()

    # Daily loss circuit breaker
    if _check_daily_loss_limit():
        state.phase = Phase.DONE
        state.save()
        await _alert(
            "<b>DAILY LOSS LIMIT</b>\n"
            f"No entries today -- max daily loss ({config.MAX_DAILY_LOSS_GBP:.0f} GBP) reached."
        )
        return

    # ARM BRACKET IMMEDIATELY (both sides) — speed is critical at US open
    qty = min(config.NUM_CONTRACTS, config.MAX_CONTRACTS)
    state.position_size = qty
    result = await broker.place_oca_bracket(
        buy_price=state.buy_level, sell_price=state.sell_level,
        qty=qty, oca_group=f"US30_{state.date}_1",
    )

    if "error" not in result:
        state.buy_order_id = result.get("buy_order_id", "")
        state.sell_order_id = result.get("sell_order_id", "")
        state.phase = Phase.ORDERS_PLACED
        state.oca_group = f"US30_{state.date}_1"
        state.save()
        logger.info(f"Bracket armed: Buy={state.buy_level} Sell={state.sell_level}")
    else:
        await _alert(f"US30 ORDER FAILED: {result['error']}")
        return

    # Use pre-fetched data from warmup if available, else do REST call
    overnight_result = OvernightResult()
    try:
        prev_df = _prefetch_cache.pop("prev_df", None)
        if prev_df is None:
            prev_df = await broker.get_5min_bars("2 D")
        else:
            logger.info("Using pre-fetched bars from warmup (fast path)")
        if not prev_df.empty:
            prev_day = prev_df[prev_df.index.date < today_date]
            today_bars = df[df.index.date == today_date]
            if not prev_day.empty and not today_bars.empty:
                gap_dir, gap_size = classify_gap(prev_day["Close"].iloc[-1], today_bars["Open"].iloc[0])
                state.gap_dir = gap_dir
                state.gap_size = gap_size
                logger.info(f"Gap: {gap_dir} ({gap_size:+.1f})")
    except Exception as e:
        logger.warning(f"Gap computation failed: {e}")

    # For US30, "overnight" = pre-market bars before 14:30 CET (from streaming)
    # These are bars from when IG starts quoting US30 (~08:00 CET) to US open
    try:
        from zoneinfo import ZoneInfo
        cet = ZoneInfo("Europe/Berlin")
        pre_market = df[df.index < now_cet.replace(hour=14, minute=30)]
        if not pre_market.empty:
            overnight_result = calculate_overnight_range(
                pre_market, state.bar_high, state.bar_low
            )
            logger.info(f"US30 pre-market range: {overnight_result.range_high:.1f}-{overnight_result.range_low:.1f}, bias={overnight_result.bias.value}")
    except Exception as e:
        logger.warning(f"Pre-market range failed: {e}")

    state.overnight_high = overnight_result.range_high
    state.overnight_low = overnight_result.range_low
    state.overnight_range = overnight_result.range_size
    state.overnight_bias = overnight_result.bias.value

    # Apply bias filter: narrow to one side if bias is clear
    bias = overnight_result.bias
    if bias == OvernightBias.LONG_ONLY:
        logger.info("V58 bias: LONG ONLY — disabling sell side")
        broker._pending_bracket["sell_price"] = 0.01
    elif bias == OvernightBias.SHORT_ONLY:
        logger.info("V58 bias: SHORT ONLY — disabling buy side")
        broker._pending_bracket["buy_price"] = 999999.0

    state.save()

    # Send signal
    await _alert(
        f"<b>US30 ASRS Signal</b>\n"
        f"Bar {state.bar_number}: H={state.bar_high:.1f} L={state.bar_low:.1f}\n"
        f"Range: {state.bar_range:.1f} ({state.range_flag})\n"
        f"Buy: {state.buy_level:.1f} | Sell: {state.sell_level:.1f}\n"
        f"Bias: {state.overnight_bias}\n"
        f"Contracts: {state.position_size}"
    )
    logger.info("Orders placed — US30 morning routine complete")


async def monitor_cycle():
    """Every minute during RTH — check triggers, manage positions."""
    state = US30DailyState.load()
    if state.phase == Phase.IDLE or state.phase == Phase.DONE:
        return

    if not broker:
        return

    # Swap config for strategy calls
    original_config = _strat.config
    _strat.config = config

    try:
        if state.phase == Phase.ORDERS_PLACED:
            # Re-arm bracket if lost (e.g. after restart)
            pb = broker._pending_bracket
            if not pb or not pb.get("active"):
                logger.warning(f"US30: bracket lost — re-arming from state (buy={state.buy_level}, sell={state.sell_level})")
                await broker.place_oca_bracket(
                    state.buy_level, state.sell_level,
                    qty=state.position_size, oca_group=state.oca_group
                )
            # Check for trigger
            result = await broker.check_trigger_levels()
            if result:
                logger.info(f"US30 trigger: {result}")

        elif state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
            # Trail stop update
            df = broker.get_streaming_bars_df()
            if not df.empty:
                events = update_trail(state, df)
                if "TRAIL_UPDATED" in events or "TRAIL_TIGHT" in events:
                    # Update stop on ALL open deal IDs (original + adds)
                    all_deal_ids = [state.stop_order_id]
                    for add in state.add_positions:
                        if isinstance(add, dict) and add.get("deal_id"):
                            all_deal_ids.append(add["deal_id"])
                    for deal_id in all_deal_ids:
                        try:
                            await broker.modify_stop(deal_id, state.trailing_stop)
                        except Exception as e:
                            logger.warning(f"Stop update failed for {deal_id}: {e}")

            # Check if stopped out
            pos = await broker.get_position()
            if pos and pos["direction"] == "FLAT" and state.contracts_active > 0:
                exit_price = await broker.get_fill_price(state.stop_order_id)
                if not exit_price:
                    # IG server-side stop: /confirms returns 404. Use trade stream level.
                    for did in list(_last_stream_exit_levels.keys()):
                        exit_price = _last_stream_exit_levels.pop(did, None)
                        if exit_price:
                            logger.info(f"Using trade stream exit level: {exit_price} (deal {did})")
                            break
                if not exit_price:
                    exit_price = state.trailing_stop
                    logger.info(f"US30: /confirms failed — using trailing_stop as exit: {exit_price}")

                # Detect manual close: if current price is far from trailing_stop,
                # someone closed manually (not a stop hit)
                current_price = await broker.get_current_price() or exit_price
                if state.direction == "LONG":
                    stop_distance = abs(current_price - state.trailing_stop)
                else:
                    stop_distance = abs(current_price - state.trailing_stop)
                manual_close = stop_distance > state.bar_range * 0.5
                if manual_close:
                    logger.warning(f"US30: position closed far from trail stop ({current_price} vs {state.trailing_stop}) — likely manual close")
                    exit_price = current_price

                # Ensure contracts_active reflects what was actually stopped
                if state.contracts_active == 0:
                    state.contracts_active = 1 + len(state.add_positions)
                    logger.warning(f"contracts_active was 0 at stop processing — restored to {state.contracts_active}")
                logger.info(f"US30 stop processing: exit={exit_price}, trailing_stop={state.trailing_stop}, "
                            f"entry={state.entry_price}, contracts={state.contracts_active}, adds={len(state.add_positions)}")
                events = process_stop_hit(state, exit_price)
                if manual_close:
                    state.trades[-1]["exit_reason"] = "MANUAL_CLOSE"
                logger.info(f"US30 stop hit: {events}{' (MANUAL)' if manual_close else ''}")

                trade = state.trades[-1] if state.trades else {}
                await _alert(
                    f"✅ <b>US30 EXIT</b>{' (MANUAL)' if manual_close else ''}\n"
                    f"Direction: {trade.get('direction', '?')}\n"
                    f"Entry: {trade.get('entry', '?')} | Exit: {exit_price}\n"
                    f"PnL: {trade.get('pnl_pts', 0):.1f} pts\n"
                    f"MFE: {trade.get('mfe', 0):.1f} pts\n"
                    f"Reason: {trade.get('exit_reason', '?')}"
                )

                # Log to journal
                try:
                    from shared.journal_db import insert_trade
                    insert_trade("US30", trade, state=None)
                    logger.info(f"[US30] Trade logged to journal: {trade.get('pnl_pts', 0):.1f} pts")
                except Exception as e:
                    logger.warning(f"US30 journal write failed: {e}")

                # Re-entry: skip on manual close
                if manual_close:
                    state.phase = Phase.DONE
                    state.save()
                    logger.info("US30: skipping re-entry after manual close")
                elif "CAN_REENTER" in events:
                    if _check_daily_loss_limit():
                        state.phase = Phase.DONE
                        state.save()
                        await _alert("RE-ENTRY BLOCKED -- daily loss limit hit")
                    elif state.entries_used >= config.MAX_ENTRIES:
                        state.phase = Phase.DONE
                        state.save()
                        logger.info(f"Re-entry blocked: max entries reached ({state.entries_used}/{config.MAX_ENTRIES})")
                    else:
                        # Re-entry: stop at the trail stop level that just exited us
                        reentry_stop = state.trailing_stop
                        if state.reentry_direction == "LONG":
                            state.buy_level = state.reentry_price
                            state.sell_level = 0.01
                            state.bar_low = reentry_stop - config.BUFFER_PTS
                            state.bar_high = state.reentry_price + abs(state.reentry_price - reentry_stop)
                        else:
                            state.buy_level = 999999.0
                            state.sell_level = state.reentry_price
                            state.bar_high = reentry_stop + config.BUFFER_PTS
                            state.bar_low = state.reentry_price - abs(reentry_stop - state.reentry_price)
                        state.bar_range = abs(state.bar_high - state.bar_low)
                        state.save()
                        result = await broker.place_oca_bracket(
                            buy_price=state.buy_level, sell_price=state.sell_level,
                            qty=state.position_size or config.NUM_CONTRACTS,
                            oca_group=f"US30_{state.date}_{state.entries_used + 1}",
                        )
                        if "error" not in result:
                            state.buy_order_id = result.get("buy_order_id", "")
                            state.sell_order_id = result.get("sell_order_id", "")
                            state.phase = Phase.ORDERS_PLACED
                            state.save()
                        await _alert(
                            f"RE-ENTRY ARMED US30\n"
                            f"Direction: {state.reentry_direction}\n"
                            f"Trigger: {state.reentry_price}"
                        )

            # Add-to-winners
            if config.ADD_STRENGTH_ENABLED and state.adds_used < config.ADD_STRENGTH_MAX:
                price = await broker.get_current_price()
                if price:
                    add_events = check_add_to_winners(state, price)
                    if "ADD_TRIGGERED" in add_events:
                        add_action = "BUY" if state.direction == "LONG" else "SELL"
                        add_result = await broker.place_market_order(add_action, 1)
                        if "order_id" in add_result:
                            fill_price = add_result.get("avg_price", price)
                            deal_id = add_result.get("order_id", "")
                            process_add_fill(state, fill_price)
                            # Store deal_id on the add position for stop management
                            if state.add_positions:
                                state.add_positions[-1]["deal_id"] = deal_id
                                state.save()
                            # Set stop on the new position immediately
                            try:
                                await broker.modify_stop(deal_id, state.trailing_stop)
                                logger.info(f"US30 add stop set on {deal_id} @ {state.trailing_stop}")
                            except Exception as e:
                                logger.error(f"US30 add stop FAILED on {deal_id}: {e}")
                            await _alert(
                                f"ADD TO WINNERS US30\n"
                                f"{state.direction} +1 @ {fill_price}\n"
                                f"Add #{state.adds_used} | Total: {state.contracts_active}\n"
                                f"Stop: {state.trailing_stop}"
                            )
    finally:
        _strat.config = original_config


async def session2_routine():
    """11:21 ET — Session 2 continuation trade using bar 4 from 11:00-11:20 ET."""
    if not getattr(config, 'SESSION2_ENABLED', False):
        return
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    state = US30DailyState.load()

    # Get morning direction — fall back to calculating bias from live bars
    morning_dir = state.direction or state.reentry_direction
    if not morning_dir and state.trades:
        morning_dir = state.trades[0].get("direction", "")
    if not morning_dir:
        # S1 didn't trade — use state overnight bias first
        ob = state.overnight_bias
        if ob in ("LONG_ONLY", "LONG"):
            morning_dir = "LONG"
        elif ob in ("SHORT_ONLY", "SHORT"):
            morning_dir = "SHORT"
    if not morning_dir:
        # Still no direction — calculate from live bars (same as DAX S2)
        try:
            s2_df = broker.get_streaming_bars_df()
            if s2_df is not None and not s2_df.empty:
                from zoneinfo import ZoneInfo
                cet = ZoneInfo("Europe/Berlin")
                now_cet = datetime.now(cet)
                pre_s2 = s2_df[s2_df.index < now_cet.replace(hour=16, minute=0)]
                if len(pre_s2) >= 5:
                    on_mid = (pre_s2["High"].max() + pre_s2["Low"].min()) / 2
                    last_close = pre_s2.iloc[-1]["Close"]
                    morning_dir = "LONG" if last_close > on_mid else "SHORT"
                    logger.info(f"US30 S2: calculated bias from {len(pre_s2)} bars: {morning_dir}")
                else:
                    logger.warning(f"US30 S2: only {len(pre_s2)} pre-S2 bars, not enough")
        except Exception as e:
            logger.warning(f"US30 S2: live bar bias failed: {e}")
    if not morning_dir:
        logger.info("US30 S2: no direction available from any source — skipping")
        await _alert("US30 S2: skipped — no direction from S1, state, or live bars")
        return
    logger.info(f"US30 S2: direction = {morning_dir}")

    if state.s2_phase != "IDLE":
        logger.info(f"US30 S2: already processed (s2_phase={state.s2_phase})")
        return

    # Skip if daily loss limit hit
    if _check_daily_loss_limit():
        logger.info("US30 S2: daily loss limit — skipping")
        await _alert("US30 S2 SKIPPED: daily loss limit hit")
        return

    logger.info("═══ US30 SESSION 2 (11:00 ET) ═══")

    # Get streaming bars from 11:00 ET
    df = broker.get_streaming_bars_df()
    if df.empty:
        logger.warning("US30 S2: no streaming bars")
        return

    # Filter to 11:00-11:20 ET bars (in CET: 17:00-17:20 CET during EST, 16:00-16:20 during EDT)
    from zoneinfo import ZoneInfo
    cet = ZoneInfo("Europe/Berlin")
    now_cet = datetime.now(cet)
    # Convert 11:00 ET to CET
    s2_start_et = now.replace(hour=config.SESSION2_HOUR_ET, minute=0, second=0)
    s2_bar4_et = s2_start_et.replace(minute=20)

    # Convert to CET for bar filtering (bars are stored in CET)
    import pandas as pd
    s2_start_cet = pd.Timestamp(s2_start_et).tz_convert(cet)
    s2_bar4_cet = pd.Timestamp(s2_bar4_et).tz_convert(cet)

    # Use hour/minute comparison (avoids timezone object mismatch bugs)
    s2_h = s2_start_cet.hour
    s2_m = s2_start_cet.minute
    s2_bars = df[
        (df.index.hour > s2_h) |
        ((df.index.hour == s2_h) & (df.index.minute >= s2_m))
    ].sort_index().head(6)

    if len(s2_bars) < config.SESSION2_BAR_COUNT:
        logger.warning(f"US30 S2: only {len(s2_bars)} bars from {s2_h:02d}:{s2_m:02d} CET, need {config.SESSION2_BAR_COUNT}")
        return

    s2_bar = s2_bars.iloc[config.SESSION2_BAR_COUNT - 1]
    s2_high = s2_bar["High"]
    s2_low = s2_bar["Low"]
    s2_range = s2_high - s2_low

    if s2_range < 5 or s2_range > 200:
        logger.info(f"US30 S2: bar range {s2_range:.1f} out of bounds — skipping")
        return

    state.s2_bar_high = s2_high
    state.s2_bar_low = s2_low
    state.s2_bar_range = s2_range

    # Continuation bias from morning
    if morning_dir in ("LONG", "LONG_ACTIVE"):
        state.s2_buy_level = s2_high + config.BUFFER_PTS
        state.s2_sell_level = 0.01
        bias_dir = "LONG"
    else:
        state.s2_buy_level = 999999
        state.s2_sell_level = s2_low - config.BUFFER_PTS
        bias_dir = "SHORT"

    state.s2_phase = "ORDERS_PLACED"
    state.save()

    logger.info(f"US30 S2: Bar4 H={s2_high:.1f} L={s2_low:.1f} Range={s2_range:.1f} Bias={bias_dir}")

    await _alert(
        f"📊 <b>US30 SESSION 2 ARMED</b>\n"
        f"Bar 4 (11:00 ET): H={s2_high:.1f} L={s2_low:.1f}\n"
        f"Range: {s2_range:.1f}pts\n"
        f"Direction: {bias_dir} (continuation)\n"
        f"Buy: {state.s2_buy_level:.1f} | Sell: {state.s2_sell_level:.1f}"
    )
    logger.info("US30 S2: orders placed — routine complete")


async def _monitor_s2():
    """US30 Session 2 monitoring — polled every minute."""
    state = US30DailyState.load()
    if state.s2_phase in ("IDLE", "DONE"):
        return

    price = await broker.get_current_price()
    if not price or price <= 0:
        return

    # Swap config for strategy calls
    original_config = _strat.config
    _strat.config = config

    try:
        if state.s2_phase == "ORDERS_PLACED":
            # Check entry trigger
            triggered = None
            if state.s2_buy_level < 999999 and price >= state.s2_buy_level:
                triggered = "LONG"
            elif state.s2_sell_level > 0.02 and price <= state.s2_sell_level:
                triggered = "SHORT"

            if triggered:
                action = "BUY" if triggered == "LONG" else "SELL"
                qty = state.position_size or config.NUM_CONTRACTS
                result = await broker.place_market_order(action, qty)

                if "order_id" not in result:
                    logger.error(f"US30 S2 entry FAILED: {result}")
                    state.s2_phase = "DONE"
                    state.save()
                    return

                fill = result.get("avg_price", price)
                deal_id = result.get("order_id", "")

                state.s2_direction = triggered
                state.s2_entry_price = fill
                state.s2_phase = f"{triggered}_ACTIVE"
                state.s2_contracts_active = qty
                state.s2_entries_used = 1
                state.s2_last_add_price = fill

                if triggered == "LONG":
                    state.s2_initial_stop = state.s2_bar_low - config.BUFFER_PTS
                else:
                    state.s2_initial_stop = state.s2_bar_high + config.BUFFER_PTS
                state.s2_trailing_stop = state.s2_initial_stop

                try:
                    await broker.modify_stop(deal_id, state.s2_trailing_stop)
                except Exception as e:
                    logger.error(f"US30 S2 stop failed: {e}")

                state.save()
                slippage = abs(fill - (state.s2_buy_level if triggered == "LONG" else state.s2_sell_level))
                await _alert(
                    f"📊 <b>US30 S2 {triggered}</b>\n"
                    f"Entry: {fill:.1f}\n"
                    f"Stop: {state.s2_trailing_stop:.1f}\n"
                    f"Range: {state.s2_bar_range:.1f}pts\n"
                    f"Slippage: {slippage:.1f}pts"
                )
            return

        if state.s2_phase in ("LONG_ACTIVE", "SHORT_ACTIVE"):
            # Check exit
            s2_stopped = False
            if state.s2_direction == "LONG" and price <= state.s2_trailing_stop:
                s2_stopped = True
            elif state.s2_direction == "SHORT" and price >= state.s2_trailing_stop:
                s2_stopped = True

            if s2_stopped:
                exit_price = state.s2_trailing_stop
                pnl = (exit_price - state.s2_entry_price) if state.s2_direction == "LONG" else (state.s2_entry_price - exit_price)
                state.s2_phase = "DONE"
                state.save()
                await _alert(
                    f"📊 <b>US30 S2 EXIT</b> {state.s2_direction}\n"
                    f"Entry: {state.s2_entry_price:.1f} | Exit: {exit_price:.1f}\n"
                    f"P&L: {pnl:+.1f}pts | MFE: {state.s2_max_favourable:.1f}pts"
                )
                return

            # Update MFE
            unrealized = (price - state.s2_entry_price) if state.s2_direction == "LONG" else (state.s2_entry_price - price)
            if unrealized > state.s2_max_favourable:
                state.s2_max_favourable = unrealized

            # Breakeven
            if not state.s2_breakeven_hit and unrealized >= config.TRAIL_BREAKEVEN_PTS:
                state.s2_breakeven_hit = True
                state.s2_trailing_stop = state.s2_entry_price
                state.save()
                await _alert(f"📊 US30 S2 BREAKEVEN — stop → {state.s2_entry_price:.1f}")

            # Candle trail
            df = broker.get_streaming_bars_df()
            if not df.empty and len(df) >= 2:
                prev_bar = df.iloc[-2]
                use_tight = unrealized >= config.TRAIL_TIGHT_THRESHOLD
                if state.s2_direction == "LONG":
                    new_stop = prev_bar["Close"] if use_tight else prev_bar["Low"]
                    if new_stop > state.s2_trailing_stop:
                        old = state.s2_trailing_stop
                        state.s2_trailing_stop = new_stop
                        if abs(new_stop - old) >= config.TRAIL_MIN_MOVE:
                            await _alert(f"📊 US30 S2 TRAIL ↑\nStop: {old:.1f} → {new_stop:.1f}\n🔒 Locked: {new_stop - state.s2_entry_price:.1f}pts")
                else:
                    new_stop = prev_bar["Close"] if use_tight else prev_bar["High"]
                    if new_stop < state.s2_trailing_stop:
                        old = state.s2_trailing_stop
                        state.s2_trailing_stop = new_stop
                        if abs(new_stop - old) >= config.TRAIL_MIN_MOVE:
                            await _alert(f"📊 US30 S2 TRAIL ↓\nStop: {old:.1f} → {new_stop:.1f}\n🔒 Locked: {state.s2_entry_price - new_stop:.1f}pts")

            state.save()
    finally:
        _strat.config = original_config

    # Session 2 monitoring
    if getattr(config, 'SESSION2_ENABLED', False):
        await _monitor_s2()


async def end_of_day():
    """16:05 ET — Close everything."""
    now = datetime.now(config.TZ_ET)
    if now.weekday() >= 5:
        return

    state = US30DailyState.load()
    if state.phase in (Phase.IDLE, Phase.DONE) and state.s2_phase in ("IDLE", "DONE"):
        return

    logger.info("US30 end of day — closing positions")
    if state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        close_action = "SELL" if state.direction == "LONG" else "BUY"
        await broker.place_market_order(close_action, state.contracts_active)
        await _alert("US30 END OF DAY — S1 positions closed")

    global _bar4_triggered
    _bar4_triggered = False
    state.phase = Phase.DONE
    state.s2_phase = "DONE"
    state.save()

    # Combined daily summary (US30 EOD fires last — include DAX)
    try:
        from dax_bot.strategy import DailyState as DaxState
        dax_state = DaxState.load()
        dax_trades = dax_state.trades if dax_state else []
        dax_pnl = sum(t.get("pnl_pts", 0) for t in dax_trades)
        us30_trades = state.trades if state else []
        us30_pnl = sum(t.get("pnl_pts", 0) for t in us30_trades)
        total_pnl = dax_pnl + us30_pnl
        total_trades = len(dax_trades) + len(us30_trades)
        icon = "🟢" if total_pnl >= 0 else "🔴"
        sign = "+" if total_pnl >= 0 else ""

        await _alert(
            f"📊 <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 {state.date}\n"
            f"{icon} Net: <b>{sign}{round(total_pnl, 1)} pts</b> ({total_trades} trades)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"DAX S1: {len(dax_trades)} trades, {'+' if dax_pnl >= 0 else ''}{round(dax_pnl, 1)}pts\n"
            f"DAX S2: {dax_state.s2_phase if dax_state else 'N/A'}\n"
            f"US30 S1: {len(us30_trades)} trades, {'+' if us30_pnl >= 0 else ''}{round(us30_pnl, 1)}pts\n"
            f"US30 S2: {state.s2_phase}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
    except Exception as e:
        logger.warning(f"Combined summary failed: {e}")
