"""
bot.py — ASRS Trading Bot (IG Markets)
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python bot.py              → Run scheduled bot
    python bot.py --now        → Calculate & place orders immediately
    python bot.py --status     → Show current state + positions
    python bot.py --cancel     → Cancel all open orders
    python bot.py --close      → Close position + cancel orders
    python bot.py --test       → Test Telegram + IG connection
"""

import asyncio
import os
import sys
import signal
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from dax_bot import config

# ── Broker ───────────────────────────────────────────────────────────────────
from dax_bot.broker_ig import IGBroker as Broker

from dax_bot.strategy import (
    DailyState, Phase,
    calculate_levels, classify_gap, get_bar, process_fill, process_partial_fill,
    update_trail, process_stop_hit, check_add_to_winners, process_add_fill,
)
from dax_bot import alerts
from dax_bot import journal
from dax_bot.overnight import calculate_overnight_range, OvernightBias, OvernightResult
from dax_bot.dashboard import start_dashboard

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ASRS")

broker = None  # Set by run_all.py (shared session)
scheduler = AsyncIOScheduler(timezone=config.TZ_UK)

# Overnight bar cache — populated by hourly job, consumed by morning_routine
_overnight_bars_cache = {"date": "", "bars": []}


async def _check_slippage(state: DailyState, fill_price: float) -> bool:
    """Check if fill slipped too far from trigger level. Returns True if OK, False if excessive."""
    trigger_price = state.buy_level if state.direction == "LONG" else state.sell_level
    slippage = abs(fill_price - trigger_price)
    if slippage > config.MAX_SLIPPAGE_PTS:
        logger.error(f"EXCESSIVE SLIPPAGE: fill={fill_price}, trigger={trigger_price}, slip={slippage}pts")
        await alerts.send(
            "🚨 <b>EXCESSIVE SLIPPAGE — CLOSING</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Trigger: {trigger_price} | Fill: {fill_price}\n"
            f"Slippage: {slippage:.1f} pts (limit: {config.MAX_SLIPPAGE_PTS})\n"
            "Closing position immediately."
        )
        try:
            await broker.close_position()
            if fill_price:
                process_stop_hit(state, fill_price)
                if state.trades and state.trades[-1].get("exit"):
                    state.trades[-1]["exit_reason"] = "SLIPPAGE_CLOSE"
            state.phase = Phase.DONE
            state.save()
        except Exception as e:
            logger.error(f"Slippage close failed: {e}")
            await alerts.send(f"🚨🚨 Slippage close FAILED: {e}\nManual intervention required!")
        return False
    return True


def _check_daily_loss_limit() -> bool:
    """Return True if daily loss limit is breached."""
    from shared.journal_db import get_trades_for_date
    today_str = datetime.now(config.TZ_UK).strftime("%Y-%m-%d")
    trades = get_trades_for_date(today_str, instrument="DAX")
    day_pnl = sum(t.get("pnl_gbp", 0) or 0 for t in trades)
    if day_pnl <= -config.MAX_DAILY_LOSS_GBP:
        logger.warning(f"Daily loss limit hit: {day_pnl:.2f} GBP (limit: -{config.MAX_DAILY_LOSS_GBP})")
        return True
    return False


async def _check_flip_price(state: DailyState) -> bool:
    """Check if price has already blown past the flip level.
    If so, skip the flip to avoid entering with massive slippage.
    Returns True if flip is OK, False if blocked.
    """
    price = await broker.get_current_price()
    if price is None:
        return True  # Can't check — proceed cautiously

    # After a LONG stopped out, the flip will be SHORT at sell_level.
    # After a SHORT stopped out, the flip will be LONG at buy_level.
    # If price is already well past the level, skip.
    buy_gap = price - state.buy_level
    sell_gap = state.sell_level - price

    # Price already above buy_level — immediate LONG trigger
    if buy_gap > config.MAX_SLIPPAGE_PTS:
        logger.warning(f"FLIP BLOCKED: price {price} already {buy_gap:.1f} pts above "
                       f"buy_level {state.buy_level}")
        state.phase = Phase.DONE
        state.save()
        await alerts.send(
            "⏭ <b>FLIP SKIPPED — PRICE OVERSHOT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Price: {price} | Buy level: {state.buy_level}\n"
            f"Gap: {buy_gap:.1f} pts > limit {config.MAX_SLIPPAGE_PTS}\n"
            "Would fill with excessive slippage."
        )
        return False

    # Price already below sell_level — immediate SHORT trigger
    if sell_gap > config.MAX_SLIPPAGE_PTS:
        logger.warning(f"FLIP BLOCKED: price {price} already {sell_gap:.1f} pts below "
                       f"sell_level {state.sell_level}")
        state.phase = Phase.DONE
        state.save()
        await alerts.send(
            "⏭ <b>FLIP SKIPPED — PRICE OVERSHOT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Price: {price} | Sell level: {state.sell_level}\n"
            f"Gap: {sell_gap:.1f} pts > limit {config.MAX_SLIPPAGE_PTS}\n"
            "Would fill with excessive slippage."
        )
        return False

    return True


async def collect_overnight_bars():
    """Hourly job (00:00-07:00 CET) — sample current price to build overnight range.
    Falls back to REST if streaming price unavailable."""
    import pandas as pd
    from zoneinfo import ZoneInfo
    cet = ZoneInfo("Europe/Berlin")
    now_cet = datetime.now(cet)
    today_str = now_cet.strftime("%Y-%m-%d")

    # Reset cache at midnight
    if _overnight_bars_cache["date"] != today_str:
        _overnight_bars_cache["date"] = today_str
        _overnight_bars_cache["bars"] = []

    price = await broker.get_current_price() if broker else None
    if price and price > 0:
        _overnight_bars_cache["bars"].append({
            "time": now_cet,
            "Open": price, "High": price, "Low": price, "Close": price,
        })
        logger.info(f"Overnight bar cached: {price} ({len(_overnight_bars_cache['bars'])} samples)")
    else:
        logger.warning("Overnight bar collection: no price available")


def get_cached_overnight_df():
    """Convert cached overnight samples to a DataFrame for calculate_overnight_range()."""
    import pandas as pd
    from zoneinfo import ZoneInfo
    bars = _overnight_bars_cache.get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    cet = ZoneInfo("Europe/Berlin")
    df.index = pd.DatetimeIndex([b["time"] for b in bars], tz=cet)
    return df[["Open", "High", "Low", "Close"]]


async def connect_with_retry(max_attempts: int = 5, delay: int = 30, alert: bool = True) -> bool:
    """Try to connect to IG with retries."""
    for attempt in range(1, max_attempts + 1):
        ok = await broker.connect()
        if ok:
            if attempt > 1 and alert:
                await alerts.send(f"🔌 <b>IG Connected</b> (attempt {attempt}/{max_attempts})")
            return True
        logger.warning(f"IG connect attempt {attempt}/{max_attempts} failed")
        if attempt < max_attempts:
            if alert:
                await alerts.send(
                    f"⚠️ <b>IG Connection Failed</b>\n"
                    f"Attempt {attempt}/{max_attempts} — retrying in {delay}s"
                )
            await asyncio.sleep(delay)
    if alert:
        await alerts.send(
            f"❌ <b>IG Unreachable</b>\n"
            f"All {max_attempts} connection attempts failed.\n"
            f"Will retry at next scheduled cycle."
        )
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ROUTINES
# ══════════════════════════════════════════════════════════════════════════════

_bar4_triggered = False  # Prevent double-trigger from callback + schedule


async def on_tick_trigger(trigger: dict):
    """Called instantly when a tick crosses bracket levels (sub-second vs 5s polling).
    Handles fill processing, slippage check, stop placement — same as monitor_cycle entry path.
    """
    state = DailyState.load()
    if state.phase != Phase.ORDERS_PLACED:
        return  # Stale trigger — phase already changed

    direction = trigger["direction"]
    fill_price = trigger["fill_price"]
    events = process_fill(state, direction, fill_price)
    logger.info(f"Tick trigger fill: {direction} @ {fill_price}")

    # Slippage guard
    if not await _check_slippage(state, fill_price):
        return

    # Cancel the unfilled side
    if hasattr(broker, "cancel_oca_counterpart"):
        filled_id = state.buy_order_id if direction == "LONG" else state.sell_order_id
        await broker.cancel_oca_counterpart(str(filled_id))

    # Place trailing stop with retry (emergency close if all fail)
    if not await _place_stop_with_retry(state):
        return

    # Place TP1 and TP2 limit orders
    await place_tp_orders(state)

    await alerts.send(alerts.entry_triggered(state))
    logger.info(f"Tick trigger complete: {direction} @ {fill_price}, stop @ {state.trailing_stop}")


async def on_candle_complete(bar: dict):
    """
    Candle callback — fires every time a 5-min bar completes via Lightstreamer.
    Registered at startup for early stream-alive verification and event-driven
    morning routine trigger.

    Bar numbering: bar 1 starts at 09:00 CET (bar_number = minutes_since_0900 / 5 + 1)
    """
    global _bar4_triggered
    bar_time = bar["time"]  # CET datetime
    today_cet = datetime.now(config.TZ_CET).date()

    # Only care about today's bars during pre-market/morning
    if bar_time.date() != today_cet:
        return

    # Bar number: 09:00 = bar 1, 09:05 = bar 2, ..., 09:15 = bar 4
    market_open_minutes = bar_time.hour * 60 + bar_time.minute - (9 * 60)
    if market_open_minutes < 0:
        return  # Pre-market bar (before 09:00 CET)
    bar_number = market_open_minutes // 5 + 1

    if bar_number == 1:
        # Bar 1 complete at 09:05 CET — stream is alive
        logger.info(f"Bar 1 complete — streaming confirmed alive "
                    f"(O={bar['Open']} H={bar['High']} L={bar['Low']} C={bar['Close']})")
        await alerts.send(
            "📡 <b>Stream alive</b> — Bar 1 built\n"
            f"O={bar['Open']} H={bar['High']} L={bar['Low']} C={bar['Close']}"
        )

    elif bar_number == 4 and not _bar4_triggered:
        # Bar 4 complete at 09:20 CET — trigger morning routine immediately
        state = DailyState.load()
        if state.phase == Phase.IDLE:
            _bar4_triggered = True
            logger.info("Bar 4 complete — triggering morning routine (event-driven)")
            await morning_routine()
        else:
            logger.info(f"Bar 4 complete but phase={state.phase} — skipping trigger")


async def pre_trade_warmup():
    """08:00 UK — Verify REST + Lightstreamer are both alive before critical window.
    If either is down, attempt recovery with time to spare before 08:20.
    """
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        return

    logger.info("═══ PRE-TRADE WARMUP (08:00) ═══")
    issues = []

    # 1. REST session check
    rest_ok = await connect_with_retry(max_attempts=3, delay=10, alert=False)
    if rest_ok:
        logger.info("Pre-warm: REST session alive")
    else:
        issues.append("REST session unreachable after 3 attempts")
        logger.error("Pre-warm: REST session FAILED")

    # 2. Lightstreamer tick check
    if broker:
        from shared.ig_session import IGSharedSession
        shared = IGSharedSession.get_instance()
        stream_ok = await shared.check_stream_health(
            broker._stream, config.IG_EPIC
        )
        if stream_ok:
            logger.info("Pre-warm: Lightstreamer alive")
        else:
            issues.append("Lightstreamer stale/disconnected — resubscribed")
            logger.warning("Pre-warm: Lightstreamer required recovery")

    # 3. Check streaming bar count (at 08:00 UK = 09:00 CET, market just opened)
    bar_count = broker.get_streaming_bar_count() if broker else 0
    logger.info(f"Pre-warm: {bar_count} streaming bars so far")

    if issues:
        await alerts.send(
            "⚠️ <b>Pre-trade warmup (08:00)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(f"• {i}" for i in issues) +
            "\n━━━━━━━━━━━━━━━━━━━━━━\n"
            "Recovery attempted. 20 min to morning routine."
        )
    else:
        logger.info("Pre-warm: all systems OK")


async def stream_alive_check():
    """08:10 UK — Check if streaming bars have started arriving.
    If no bars yet, attempt stream recovery.
    """
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        return

    bar_count = broker.get_streaming_bar_count() if broker else 0
    if bar_count == 0:
        logger.warning("08:10 check: No streaming bars received — attempting recovery")

        # Try to recover the stream
        if broker:
            from shared.ig_session import IGSharedSession
            shared = IGSharedSession.get_instance()
            recovered = await shared.check_stream_health(
                broker._stream, config.IG_EPIC
            )
            if recovered:
                await alerts.send(
                    "⚠️ <b>Stream check (08:10)</b>\n"
                    "No bars received — stream resubscribed.\n"
                    "Waiting for bars. REST fallback available."
                )
            else:
                await alerts.send(
                    "🚨 <b>Stream check (08:10)</b>\n"
                    "No bars + recovery FAILED.\n"
                    "Morning routine will use REST fallback."
                )
    else:
        logger.info(f"08:10 check: {bar_count} streaming bars — OK")


async def health_check():
    """06:30 UK — Send diagnostic message. Does NOT disconnect broker."""
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        return

    logger.info("═══ HEALTH CHECK ═══")
    mode = "📄 DEMO" if config.IG_DEMO else "🔴 LIVE"

    # Test broker connection (retry up to 3 times silently)
    ok = await connect_with_retry(max_attempts=3, delay=20, alert=False)
    status = "✅ Connected" if ok else "❌ Unreachable"
    price = None
    if ok:
        price = await broker.get_current_price()
        # DO NOT disconnect — keep session alive for morning routine

    price_str = f"{price}" if price else "N/A"
    stream_bars = broker.get_streaming_bar_count() if broker else 0

    msg = (
        f"🩺 <b>ASRS Health Check</b> [{mode}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now.strftime('%Y-%m-%d %H:%M')} UK\n"
        f"🔌 IG: {status}\n"
        f"   Epic: {config.IG_EPIC}\n"
        f"💹 DAX: {price_str}\n"
        f"📡 Streaming bars today: {stream_bars}\n"
        f"📦 {config.INSTRUMENT} x{config.NUM_CONTRACTS}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ Morning routine at {config.MORNING_HOUR:02d}:{config.MORNING_MINUTE:02d}\n"
        f"<i>Session kept alive for morning routine</i>"
    )

    await alerts.send(msg)
    logger.info(f"Health check sent — IG: {status}, DAX: {price_str}")


async def _get_today_bars_with_fallback() -> "pd.DataFrame":
    """Get today's bars: streaming first, REST fallback with retries."""
    import pandas as pd

    # Step 1: Try streaming bars
    logger.info("Step 1: Checking streaming bars...")
    df = broker.get_streaming_bars_df()
    if not df.empty:
        today_date = datetime.now(config.TZ_CET).date()
        today = df[df.index.date == today_date]
        bar4 = get_bar(today, 4) if not today.empty else None
        if bar4:
            logger.info(f"Step 1: Stream has bar 4 — O={bar4['open']} H={bar4['high']} "
                        f"L={bar4['low']} C={bar4['close']}")
            return df

    # Step 2: Wait up to 60s for bar 4 to arrive via stream
    logger.info("Step 2: Bar 4 not in stream yet — waiting up to 60s...")
    for i in range(12):
        await asyncio.sleep(5)
        df = broker.get_streaming_bars_df()
        if not df.empty:
            today_date = datetime.now(config.TZ_CET).date()
            today = df[df.index.date == today_date]
            bar4 = get_bar(today, 4) if not today.empty else None
            if bar4:
                logger.info(f"Step 2: Bar 4 arrived after {(i+1)*5}s — "
                            f"O={bar4['open']} H={bar4['high']} L={bar4['low']} C={bar4['close']}")
                return df

    # Step 3: REST fallback with 3 retries
    logger.warning("Step 3: Stream failed — falling back to REST API...")
    for attempt in range(1, 4):
        logger.info(f"Step 3: REST attempt {attempt}/3...")
        df = await broker.get_5min_bars("1 D")
        if not df.empty:
            today_date = datetime.now(config.TZ_CET).date()
            today = df[df.index.date == today_date]
            bar4 = get_bar(today, 4) if not today.empty else None
            if bar4:
                logger.info(f"Step 3: REST got bar 4 on attempt {attempt}")
                return df
        if attempt < 3:
            logger.warning(f"Step 3: REST attempt {attempt} failed — retrying in 15s")
            await asyncio.sleep(15)

    logger.error("Step 3: All REST attempts failed — no bar 4 available")
    return pd.DataFrame()


async def morning_routine():
    """
    08:21 UK — Calculate levels from streaming bar 4 data.

    Flow:
      1. Get today's bars from stream (REST fallback)
      2. Compute gap + overnight bias
      3. Select signal bar (bar 4 default; bar 5 if BAR5_RULES match)
      4. Calculate levels, auto-trade
    """
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        logger.info("Weekend — skipping")
        return

    # Check if trading is paused via Telegram command
    import telegram_cmd
    if telegram_cmd.is_paused():
        logger.info("Trading paused via /pause command — skipping")
        await alerts.send("⏸ <b>DAX PAUSED</b> — morning routine skipped. Use /resume to resume.")
        return

    global _bar4_triggered
    logger.info("═══ MORNING ROUTINE ═══")
    state = DailyState.load()
    if state.phase != Phase.IDLE:
        logger.info("Already processed today")
        return

    # Ensure IG session is alive (Issue 2: auto-reconnect before major action)
    logger.info("Step 0: Ensuring IG connection...")
    if not await connect_with_retry(max_attempts=5, delay=30):
        await alerts.send(alerts.error_alert(
            "IG connection failed before morning routine.\n"
            "Will retry at 08:23 and 08:25."
        ))
        return
    logger.info("Step 0: IG connected")

    # ── Get today's bars (streaming first, REST fallback) ────────────
    df = await _get_today_bars_with_fallback()
    if df.empty:
        await alerts.send(alerts.error_alert(
            "No bar 4 data from stream OR REST.\n"
            "Stream failed + all 3 REST attempts failed.\n"
            "Market holiday? Check logs."
        ))
        return

    today_date = datetime.now(config.TZ_CET).date()

    # ── Pre-compute context for signal bar selection ──────────────────
    # 1. Gap direction (previous close vs today open) — REST for old data
    logger.info("Step 4: Computing gap direction...")
    try:
        prev_df = await broker.get_5min_bars("2 D")
        if not prev_df.empty:
            prev_day = prev_df[prev_df.index.date < today_date]
            today_bars = df[df.index.date == today_date]
            if not prev_day.empty and not today_bars.empty:
                gap_dir, gap_size = classify_gap(prev_day["Close"].iloc[-1], today_bars["Open"].iloc[0])
                state.gap_dir = gap_dir
                state.gap_size = gap_size
                logger.info(f"Step 4: Gap = {gap_dir} ({gap_size:+.1f} pts)")
    except Exception as e:
        logger.warning(f"Step 4: Gap computation failed (non-fatal): {e}")

    # 2. Overnight range (V58 theory) — REST first, cached bars fallback
    logger.info("Step 5: Computing overnight range...")
    overnight_result = OvernightResult()  # default: bias=NO_DATA
    try:
        overnight_df = await broker.get_overnight_bars()
        # Fallback to cached overnight samples if REST returned empty
        if overnight_df.empty:
            logger.info("REST overnight bars empty — using cached samples")
            overnight_df = get_cached_overnight_df()
        today_bars = df[df.index.date == today_date]
        bar4_temp = get_bar(today_bars, 4) if not today_bars.empty else None
        if bar4_temp:
            overnight_result = calculate_overnight_range(
                overnight_df, bar4_temp["high"], bar4_temp["low"]
            )
        else:
            overnight_result = calculate_overnight_range(overnight_df, 0, 0)
    except Exception as e:
        logger.warning(f"Step 5: Overnight range failed (non-fatal, using STANDARD): {e}")
    state.overnight_high = overnight_result.range_high
    state.overnight_low = overnight_result.range_low
    state.overnight_range = overnight_result.range_size
    state.overnight_bias = overnight_result.bias.value
    state.bar4_vs_overnight = overnight_result.bar4_vs_range
    state.save()
    logger.info(f"Step 5: Overnight {overnight_result.range_low}-{overnight_result.range_high}, "
                f"bias={overnight_result.bias.value}")

    # ── Calculate levels (bar selection uses gap_dir + overnight_bias) ──
    logger.info("Step 6: Calculating levels...")
    events = calculate_levels(state, df)
    logger.info(f"Step 6: Level events: {events}")

    if "LEVELS_SET" not in events and "WAITING_FOR_BAR5" not in events:
        await alerts.send(alerts.error_alert(f"Cannot calculate levels: {events}"))
        return

    # Wait for bar 5 only if conditions require it
    if "WAITING_FOR_BAR5" in events:
        logger.info("Waiting for bar 5 to close...")
        await asyncio.sleep(300)
        df = broker.get_streaming_bars_df()
        if df.empty:
            df = await broker.get_5min_bars("1 D")
        events = calculate_levels(state, df)
        if "LEVELS_SET" not in events:
            await alerts.send(alerts.error_alert(f"Bar 5 still not available: {events}"))
            return

    logger.info(f"Step 6: Levels — Buy={state.buy_level} Sell={state.sell_level} "
                f"Bar={state.bar_number} Range={state.bar_range}")

    # Build overnight info for alerts
    if overnight_result.bias == OvernightBias.NO_DATA:
        overnight_msg = (
            "\n⚠️ <b>Overnight data unavailable</b> — using STANDARD bracket (both sides)"
        )
    else:
        overnight_msg = (
            f"\n🌙 <b>Overnight Range:</b> {overnight_result.range_low:.0f}–"
            f"{overnight_result.range_high:.0f} ({overnight_result.range_size:.0f}pts)\n"
            f"Bar {state.bar_number} vs range: <b>{overnight_result.bar4_vs_range}</b>\n"
            f"V58 Bias: {overnight_result.emoji()} <b>{overnight_result.bias.value}</b>"
        )

    # Always trade — no skipping, no human input
    logger.info("Step 7: Sending Telegram signal...")
    await alerts.send(alerts.morning_levels(state) + overnight_msg)
    logger.info("Step 7: Telegram sent")

    # Daily loss circuit breaker
    if _check_daily_loss_limit():
        state.phase = Phase.DONE
        state.save()
        await alerts.send(
            "🛑 <b>DAILY LOSS LIMIT</b>\n"
            f"No entries today — max daily loss (£{config.MAX_DAILY_LOSS_GBP:.0f}) reached."
        )
        _bar4_triggered = True
        return

    # Position sizing: 2x on STANDARD+NARROW setups (validated out-of-sample)
    if (overnight_result.bias.value in ("STANDARD", "NO_DATA")
            and state.range_flag == "NARROW"
            and config.NARROW_STD_MULTIPLIER > 1):
        state.position_size = min(config.NUM_CONTRACTS * config.NARROW_STD_MULTIPLIER, config.MAX_CONTRACTS)
        logger.info(f"STANDARD+NARROW → sizing up to {state.position_size} contracts")
        await alerts.send(
            f"<b>NARROW RANGE + STANDARD</b> → {state.position_size}x contracts "
            f"(x{config.NARROW_STD_MULTIPLIER} multiplier)"
        )
    else:
        state.position_size = min(config.NUM_CONTRACTS, config.MAX_CONTRACTS)
    state.save()

    logger.info("Step 8: Placing orders...")
    await place_orders_with_bias(state, overnight_result.bias)
    _bar4_triggered = True
    logger.info("Step 8: Orders placed — morning routine complete")


async def failsafe_check():
    """08:25 UK — If phase is still IDLE, morning routine failed. Alert + retry."""
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        return

    state = DailyState.load()
    if state.phase == Phase.IDLE:
        logger.warning("FAILSAFE: Phase still IDLE at 08:25 — morning routine failed!")
        await alerts.send(
            "🚨 <b>WARNING: Morning routine failed</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Phase still IDLE at 08:25.\n"
            "No levels calculated. No orders placed.\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Attempting final retry now..."
        )
        # Final retry
        await morning_routine()

        # Check again
        state = DailyState.load()
        if state.phase == Phase.IDLE:
            await alerts.send(
                "❌ <b>MORNING ROUTINE FAILED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "All retries exhausted. No trading today.\n"
                "Check /logs for details."
            )
    else:
        logger.info(f"Failsafe check: phase={state.phase} — OK")




async def place_single_order(state: DailyState, direction: str):
    """Place only the buy-stop or sell-stop, not both.
    For IG spread bet, uses price-triggered approach (same as OCA bracket but one-sided).
    """
    qty = state.position_size or config.NUM_CONTRACTS
    if hasattr(broker, "_pending_bracket"):
        # IG price-triggered: store single-side trigger level
        if direction == "LONG":
            broker._pending_bracket = {
                "buy_price": state.buy_level,
                "sell_price": -999999,  # Will never trigger
                "qty": qty,
                "oca_group": state.oca_group,
                "active": True,
            }
            buy_id = f"pending_buy_{state.oca_group}"
            broker._orders[buy_id] = {
                "type": "pending", "direction": "BUY",
                "price": state.buy_level, "qty": qty,
            }
            state.buy_order_id = buy_id
        else:
            broker._pending_bracket = {
                "buy_price": 999999,  # Will never trigger
                "sell_price": state.sell_level,
                "qty": qty,
                "oca_group": state.oca_group,
                "active": True,
            }
            sell_id = f"pending_sell_{state.oca_group}"
            broker._orders[sell_id] = {
                "type": "pending", "direction": "SELL",
                "price": state.sell_level, "qty": qty,
            }
            state.sell_order_id = sell_id

        state.phase = Phase.ORDERS_PLACED
        state.save()
        label = "LONG" if direction == "LONG" else "SHORT"
        level = state.buy_level if direction == "LONG" else state.sell_level
        stop = state.sell_level if direction == "LONG" else state.buy_level
        emoji = "🟩" if direction == "LONG" else "🟥"
        await alerts.send(
            f"{emoji} <b>{label} ONLY order placed</b> (price-triggered)\n"
            f"{'Buy' if direction == 'LONG' else 'Sell'} stop: {level}\n"
            f"Stop: {stop}"
        )
        logger.info(f"IG single order (price-triggered): {label} @ {level}")
        return

    # Should not reach here — IG is the only broker
    logger.error("place_single_order: unexpected code path")


async def place_orders_with_bias(state: DailyState, bias: OvernightBias):
    """
    Place orders respecting overnight range bias.

    SHORT_ONLY → Only place sell stop (fade the overnight up move)
    LONG_ONLY  → Only place buy stop (fade the overnight down move)
    STANDARD   → Normal OCA bracket (both sides)
    """
    if bias == OvernightBias.SHORT_ONLY:
        logger.info("V58 bias: SHORT ONLY — placing sell stop only")
        await place_single_order(state, "SHORT")
        await alerts.send(
            f"🔴 <b>V58: SHORT ONLY</b>\n"
            f"Signal bar above overnight range → fading the up move\n"
            f"Sell stop: {state.sell_level}"
        )

    elif bias == OvernightBias.LONG_ONLY:
        logger.info("V58 bias: LONG ONLY — placing buy stop only")
        await place_single_order(state, "LONG")
        await alerts.send(
            f"🟢 <b>V58: LONG ONLY</b>\n"
            f"Signal bar below overnight range → fading the down move\n"
            f"Buy stop: {state.buy_level}"
        )

    else:
        # STANDARD or NO_DATA — normal OCA bracket
        await place_bracket_orders(state)


async def _place_stop_with_retry(state: DailyState, max_attempts: int = 3) -> bool:
    """Place trailing stop after entry, with retries. Emergency close if all fail."""
    stop_action = "SELL" if state.direction == "LONG" else "BUY"
    for attempt in range(1, max_attempts + 1):
        try:
            result = await broker.place_stop_order(
                action=stop_action,
                qty=state.contracts_active or config.NUM_CONTRACTS,
                stop_price=state.trailing_stop,
            )
            if "order_id" in result:
                state.stop_order_id = result["order_id"]
                state.save()
                if attempt > 1:
                    logger.info(f"Stop placed on attempt {attempt}")
                return True
            logger.warning(f"Stop placement attempt {attempt}/{max_attempts} failed: {result}")
        except Exception as e:
            logger.error(f"Stop placement attempt {attempt}/{max_attempts} error: {e}")
        if attempt < max_attempts:
            await asyncio.sleep(2 * attempt)

    # All retries exhausted — EMERGENCY: close position to avoid naked exposure
    logger.error("EMERGENCY: All stop placement attempts failed — closing position")
    await alerts.send(
        "🚨 <b>EMERGENCY CLOSE</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Stop placement failed {max_attempts}x after {state.direction} entry.\n"
        "Closing position to prevent unprotected exposure.\n"
        "Manual review required."
    )
    try:
        await broker.close_position()
        price = await broker.get_current_price()
        if price:
            process_stop_hit(state, price)
            if state.trades and state.trades[-1].get("exit"):
                state.trades[-1]["exit_reason"] = "EMERGENCY_CLOSE"
        state.save()
    except Exception as e:
        logger.error(f"Emergency close also failed: {e}")
        await alerts.send(f"🚨🚨 <b>EMERGENCY CLOSE FAILED</b>: {e}\nManual intervention required!")
    return False


async def place_bracket_orders(state: DailyState):
    """Place OCA buy-stop + sell-stop bracket."""
    qty = state.position_size or config.NUM_CONTRACTS
    result = await broker.place_oca_bracket(
        buy_price=state.buy_level,
        sell_price=state.sell_level,
        qty=qty,
        oca_group=state.oca_group,
    )

    if "error" in result:
        await alerts.send(alerts.error_alert(f"Order failed: {result['error']}"))
        return

    state.buy_order_id = result["buy_order_id"]
    state.sell_order_id = result["sell_order_id"]
    state.phase = Phase.ORDERS_PLACED
    state.save()

    await alerts.send(alerts.orders_placed(state))
    logger.info(f"Orders live — Buy: {state.buy_order_id}, Sell: {state.sell_order_id}")


async def place_tp_orders(state: DailyState):
    """Place TP1 and TP2 limit orders for partial exits (3-contract mode)."""
    if not config.PARTIAL_EXIT:
        return

    tp_action = "SELL" if state.direction == "LONG" else "BUY"

    # TP1: 1 contract at +20 pts
    try:
        tp1_result = await broker.place_limit_order(
            action=tp_action, qty=1, limit_price=state.tp1_price,
        )
        if "order_id" in tp1_result:
            state.tp1_order_id = tp1_result["order_id"]
            logger.info(f"TP1 limit: {tp_action} 1 @ {state.tp1_price}")
        elif "error" in tp1_result:
            logger.error(f"TP1 placement failed: {tp1_result['error']}")
            await alerts.send(f"TP1 FAILED: {tp1_result.get('error', 'unknown')}")
    except Exception as e:
        logger.error(f"TP1 placement exception: {e}")
        await alerts.send(f"TP1 EXCEPTION: {e}")

    # TP2: 1 contract at +50 pts
    try:
        tp2_result = await broker.place_limit_order(
            action=tp_action, qty=1, limit_price=state.tp2_price,
        )
        if "order_id" in tp2_result:
            state.tp2_order_id = tp2_result["order_id"]
            logger.info(f"TP2 limit: {tp_action} 1 @ {state.tp2_price}")
        elif "error" in tp2_result:
            logger.error(f"TP2 placement failed: {tp2_result['error']}")
            await alerts.send(f"TP2 FAILED: {tp2_result.get('error', 'unknown')}")
    except Exception as e:
        logger.error(f"TP2 placement exception: {e}")
        await alerts.send(f"TP2 EXCEPTION: {e}")

    state.save()


async def check_tp_fills(state: DailyState) -> list[str]:
    """Check if TP1/TP2 limit orders have filled using actual order status."""
    if not config.PARTIAL_EXIT:
        return []

    events = []

    if not state.tp1_filled and state.tp1_order_id:
        status = await broker.get_order_status(state.tp1_order_id)
        if status == "Filled":
            fill_price = await broker.get_fill_price(state.tp1_order_id) or state.tp1_price
            events.extend(process_partial_fill(state, 1, fill_price))
            await alerts.send(alerts.tp_filled(state, 1))
            logger.info(f"TP1 filled @ {fill_price} — {state.contracts_active} contracts remaining")
        elif status == "Cancelled":
            if state.tp1_replaces < 3:
                state.tp1_replaces += 1
                logger.warning(f"TP1 cancelled — re-placing ({state.tp1_replaces}/3)")
                tp_action = "SELL" if state.direction == "LONG" else "BUY"
                tp1 = await broker.place_limit_order(tp_action, 1, state.tp1_price)
                if "order_id" in tp1:
                    state.tp1_order_id = tp1["order_id"]
                    state.save()
            else:
                logger.error("TP1 cancelled 3x — giving up on re-placement")
                await alerts.send("⚠️ TP1 cancelled 3x — no more re-placements")

    if not state.tp2_filled and state.tp2_order_id:
        status = await broker.get_order_status(state.tp2_order_id)
        if status == "Filled":
            fill_price = await broker.get_fill_price(state.tp2_order_id) or state.tp2_price
            events.extend(process_partial_fill(state, 2, fill_price))
            await alerts.send(alerts.tp_filled(state, 2))
            logger.info(f"TP2 filled @ {fill_price} — {state.contracts_active} contracts remaining")
        elif status == "Cancelled":
            if state.tp2_replaces < 3:
                state.tp2_replaces += 1
                logger.warning(f"TP2 cancelled — re-placing ({state.tp2_replaces}/3)")
                tp_action = "SELL" if state.direction == "LONG" else "BUY"
                tp2 = await broker.place_limit_order(tp_action, 1, state.tp2_price)
                if "order_id" in tp2:
                    state.tp2_order_id = tp2["order_id"]
                    state.save()
            else:
                logger.error("TP2 cancelled 3x — giving up on re-placement")
                await alerts.send("⚠️ TP2 cancelled 3x — no more re-placements")

    return events


async def _adjust_stop_after_tp(state: DailyState):
    """Adjust stop order quantity after a TP fill."""
    if state.contracts_active <= 0:
        return
    success = await broker.modify_stop_qty(state.stop_order_id, state.contracts_active)
    if not success:
        await broker.cancel_order(state.stop_order_id)
        stop_action = "SELL" if state.direction == "LONG" else "BUY"
        result = await broker.place_stop_order(
            action=stop_action, qty=state.contracts_active, stop_price=state.trailing_stop,
        )
        if "order_id" in result:
            state.stop_order_id = result["order_id"]
            state.save()
    logger.info(f"Stop qty adjusted to {state.contracts_active}")


async def _handle_fill_event(data):
    """
    Event-driven handler for IG streaming trade updates (OPU dict).
    IG OPU fields: dealId, dealStatus, direction, level, size, status, etc.
    Currently used for logging only — actual fill detection is via price-triggered
    polling in monitor_cycle. This prevents the handler from crashing on every event.
    """
    if not isinstance(data, dict):
        return

    deal_status = data.get("dealStatus", data.get("status", ""))
    deal_id = data.get("dealId", "")
    level = data.get("level", 0)
    direction = data.get("direction", "")

    # Log all trade events for debugging
    logger.info(f"[DAX] Trade stream event: dealId={deal_id} status={deal_status} "
                f"direction={direction} level={level}")
    return
    state = DailyState.load()

    try:
        # ── Entry fill (OCA bracket triggered) ──────────────────────────
        if state.phase == Phase.ORDERS_PLACED and order_id in (state.buy_order_id, state.sell_order_id):
            direction = "LONG" if order_id == state.buy_order_id else "SHORT"
            events = process_fill(state, direction, fill_price)
            logger.info(f"⚡ INSTANT ENTRY: {direction} @ {fill_price}")

            # IG OCA simulation: cancel the unfilled side
            if hasattr(broker, "cancel_oca_counterpart"):
                await broker.cancel_oca_counterpart(str(order_id))

            # Place trailing stop immediately
            stop_action = "SELL" if direction == "LONG" else "BUY"
            stop_result = await broker.place_stop_order(
                action=stop_action, qty=state.contracts_active or config.NUM_CONTRACTS, stop_price=state.trailing_stop,
            )
            if "order_id" in stop_result:
                state.stop_order_id = stop_result["order_id"]
                state.save()

            await place_tp_orders(state)
            await alerts.send(alerts.entry_triggered(state))
            return

        # ── TP1 fill ────────────────────────────────────────────────────
        if (state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE)
                and order_id == state.tp1_order_id and not state.tp1_filled):
            process_partial_fill(state, 1, fill_price)
            logger.info(f"⚡ INSTANT TP1 @ {fill_price} — {state.contracts_active} left")
            await alerts.send(alerts.tp_filled(state, 1))
            await _adjust_stop_after_tp(state)
            return

        # ── TP2 fill ────────────────────────────────────────────────────
        if (state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE)
                and order_id == state.tp2_order_id and not state.tp2_filled):
            process_partial_fill(state, 2, fill_price)
            logger.info(f"⚡ INSTANT TP2 @ {fill_price} — {state.contracts_active} left")
            await alerts.send(alerts.tp_filled(state, 2))
            await _adjust_stop_after_tp(state)
            return

        # ── Stop hit ────────────────────────────────────────────────────
        if (state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE)
                and order_id == state.stop_order_id):
            logger.info(f"⚡ INSTANT STOP HIT @ {fill_price}")

            # Cancel any remaining TP orders
            if config.PARTIAL_EXIT:
                if not state.tp1_filled and state.tp1_order_id:
                    await broker.cancel_order(state.tp1_order_id)
                if not state.tp2_filled and state.tp2_order_id:
                    await broker.cancel_order(state.tp2_order_id)

            events = process_stop_hit(state, fill_price)
            trade_log = state.trades[-1] if state.trades else {}
            await alerts.send(alerts.exit_stopped(state, trade_log))

            # Log completed trade to journal
            if trade_log.get("exit"):
                journal.append_trade(trade_log, state)
                trade_log["journaled"] = True

            if "CAN_FLIP" in events:
                if _check_daily_loss_limit():
                    state.phase = Phase.DONE
                    state.save()
                    await alerts.send(
                        "🛑 <b>FLIP BLOCKED — DAILY LOSS LIMIT</b>\n"
                        f"Max daily loss (£{config.MAX_DAILY_LOSS_GBP:.0f}) reached."
                    )
                    return
                if not await _check_flip_price(state):
                    return
                try:
                    await place_bracket_orders(state)
                    state = DailyState.load()
                    if state.phase != Phase.ORDERS_PLACED:
                        raise RuntimeError(f"Flip bracket failed — phase is {state.phase}")
                except Exception as flip_err:
                    logger.error(f"Flip bracket placement failed: {flip_err}")
                    state.phase = Phase.DONE
                    state.save()
                    await alerts.send(
                        "🚨 <b>FLIP FAILED</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Error: <code>{flip_err}</code>\n"
                        "Phase set to DONE — no further entries today.\n"
                        "Manual review required."
                    )
            return

    except Exception as e:
        logger.error(f"Fill event handler error: {e}", exc_info=True)


async def monitor_cycle():
    """Every 5 min — backup fill check + trail stops.
    When in ORDERS_PLACED phase on IG, polls price every 5s for fast entry detection.
    """
    try:
        state = DailyState.load()
        # Fast polling for IG price-triggered entries
        if (state.phase == Phase.ORDERS_PLACED
                and config.BROKER == "ig"
                and hasattr(broker, "check_trigger_levels")):
            if not await broker.ensure_connected():
                return
            for _ in range(12):  # 12 x 5s = 60s (one monitor cycle)
                state = DailyState.load()
                if state.phase != Phase.ORDERS_PLACED:
                    break
                trigger = await broker.check_trigger_levels()
                if trigger:
                    direction = trigger["direction"]
                    fill_price = trigger["fill_price"]
                    events = process_fill(state, direction, fill_price)
                    logger.info(f"Price trigger fill: {direction} @ {fill_price}")

                    # Slippage guard
                    if not await _check_slippage(state, fill_price):
                        return  # Excessive slippage — position closed

                    # IG OCA simulation: cancel the unfilled side
                    if hasattr(broker, "cancel_oca_counterpart"):
                        filled_id = state.buy_order_id if direction == "LONG" else state.sell_order_id
                        await broker.cancel_oca_counterpart(str(filled_id))

                    # Place trailing stop with retry (emergency close if all fail)
                    if not await _place_stop_with_retry(state):
                        return  # Emergency close happened

                    # Place TP1 and TP2 limit orders (3-contract mode)
                    await place_tp_orders(state)

                    await alerts.send(alerts.entry_triggered(state))
                    return
                await asyncio.sleep(5)
            return

        await _monitor_cycle_inner()
    except Exception as e:
        logger.error(f"Monitor cycle error: {e}", exc_info=True)
        state = DailyState.load()
        await alerts.send(
            f"🚨 <b>MONITOR CYCLE ERROR</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Phase: {state.phase}\n"
            f"Error: <code>{e}</code>\n"
            f"<i>Will retry next cycle.</i>"
        )


async def _monitor_cycle_inner():
    """Inner monitor logic — wrapped by monitor_cycle for error safety."""
    state = DailyState.load()

    if state.phase in (Phase.IDLE, Phase.DONE):
        return
    if not await broker.ensure_connected():
        now = datetime.now(config.TZ_UK)
        await alerts.send(
            f"🚨 <b>IG CONNECTION LOST</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Time: {now.strftime('%H:%M:%S')} UK\n"
            f"Phase: {state.phase}\n"
            f"Auto-reconnect failed.\n"
            f"Will retry next cycle (1 min).\n"
            f"<i>Check /logs for details.</i>"
        )
        return

    # ── Check for fills ────────────────────────────────────────────────
    if state.phase == Phase.ORDERS_PLACED:
        pos = await broker.get_position()
        if pos["direction"] != "FLAT":
            events = process_fill(state, pos["direction"], pos["avg_cost"])
            logger.info(f"Fill: {events}")

            # Slippage guard
            if not await _check_slippage(state, pos["avg_cost"]):
                return  # Excessive slippage — position closed

            # IG OCA simulation: cancel the unfilled side
            if hasattr(broker, "cancel_oca_counterpart"):
                filled_id = state.buy_order_id if pos["direction"] == "LONG" else state.sell_order_id
                await broker.cancel_oca_counterpart(str(filled_id))

            # Place trailing stop with retry (emergency close if all fail)
            if not await _place_stop_with_retry(state):
                return  # Emergency close happened

            # Place TP1 and TP2 limit orders (3-contract mode)
            await place_tp_orders(state)

            await alerts.send(alerts.entry_triggered(state))
            return

    # ── Trail stop / detect exit ───────────────────────────────────────
    if state.phase in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        pos = await broker.get_position()

        # ── Position reconciliation ──────────────────────────────────
        if pos["direction"] != "FLAT":
            # Direction mismatch (bot says LONG, IG says SHORT or vice versa)
            if pos["direction"] != state.direction:
                logger.error(f"POSITION MISMATCH: bot={state.direction}, IG={pos['direction']}")
                await alerts.send(
                    "🚨 <b>POSITION MISMATCH</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Bot state: {state.direction}\n"
                    f"IG actual: {pos['direction']}\n"
                    "Setting phase to DONE — manual review required."
                )
                state.phase = Phase.DONE
                state.save()
                return

            # Qty mismatch (sync contracts_active to IG reality)
            ig_qty = int(abs(pos["position"]))
            if ig_qty != state.contracts_active and ig_qty > 0:
                logger.warning(
                    f"QTY MISMATCH: bot={state.contracts_active}, IG={ig_qty} — syncing"
                )
                await alerts.send(
                    f"⚠️ <b>QTY SYNC</b>: bot had {state.contracts_active} contracts, "
                    f"IG has {ig_qty}. Syncing."
                )
                state.contracts_active = ig_qty
                state.save()

        if pos["direction"] == "FLAT":
            # All contracts closed — event handler may have already processed this
            if state.phase == Phase.DONE or state.phase == Phase.LEVELS_SET:
                return  # Already handled by event handler

            # Cancel any remaining TP orders
            if config.PARTIAL_EXIT:
                if not state.tp1_filled and state.tp1_order_id:
                    await broker.cancel_order(state.tp1_order_id)
                if not state.tp2_filled and state.tp2_order_id:
                    await broker.cancel_order(state.tp2_order_id)

            # Use actual fill price from stop order if available
            exit_price = await broker.get_fill_price(state.stop_order_id) or state.trailing_stop
            events = process_stop_hit(state, exit_price)
            logger.info(f"Stop hit: {events}")

            trade = state.trades[-1] if state.trades else {}
            await alerts.send(alerts.exit_stopped(state, trade))

            # Log completed trade to journal
            if trade.get("exit"):
                journal.append_trade(trade, state)
                trade["journaled"] = True

            if "CAN_FLIP" in events:
                if _check_daily_loss_limit():
                    state.phase = Phase.DONE
                    state.save()
                    await alerts.send(
                        "🛑 <b>FLIP BLOCKED — DAILY LOSS LIMIT</b>\n"
                        f"Max daily loss (£{config.MAX_DAILY_LOSS_GBP:.0f}) reached."
                    )
                    return
                if not await _check_flip_price(state):
                    return
                try:
                    await place_bracket_orders(state)
                    state = DailyState.load()
                    if state.phase != Phase.ORDERS_PLACED:
                        raise RuntimeError(f"Flip bracket failed — phase is {state.phase}")
                except Exception as flip_err:
                    logger.error(f"Flip bracket placement failed: {flip_err}")
                    state.phase = Phase.DONE
                    state.save()
                    await alerts.send(
                        "🚨 <b>FLIP FAILED</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Error: <code>{flip_err}</code>\n"
                        "Phase set to DONE — no further entries today.\n"
                        "Manual review required."
                    )
            return

        # Still in — check TP fills (backup for event handler)
        tp_events = await check_tp_fills(state)

        # Update stop qty if TPs reduced position
        if tp_events and state.contracts_active > 0:
            await _adjust_stop_after_tp(state)

        # Update trail — use streaming bars (no REST call needed)
        df = broker.get_streaming_bars_df()
        if df.empty:
            # Fallback to REST if stream has no bars
            df = await broker.get_5min_bars("1 D")
        if df.empty:
            return

        old_stop = state.trailing_stop
        events = update_trail(state, df)

        if "BREAKEVEN_HIT" in events:
            success = await broker.modify_stop(state.stop_order_id, state.trailing_stop)
            if not success:
                if not await _place_stop_with_retry(state):
                    return  # Emergency close happened
            await alerts.send(f"BREAKEVEN — stop moved to entry {state.entry_price}")
            logger.info(f"Breakeven hit: stop → {state.entry_price}")

        if "TRAIL_UPDATED" in events:
            success = await broker.modify_stop(state.stop_order_id, state.trailing_stop)
            if not success:
                if not await _place_stop_with_retry(state):
                    return  # Emergency close happened

            if abs(state.trailing_stop - old_stop) >= config.TRAIL_MIN_MOVE:
                await alerts.send(alerts.trail_updated(state, old_stop))
            logger.info(f"Trail: {old_stop} → {state.trailing_stop}")

        # ── Add-to-winners check ───────────────────────────────────────
        if config.ADD_STRENGTH_ENABLED and state.adds_used < config.ADD_STRENGTH_MAX:
            price = await broker.get_current_price()
            if price:
                add_events = check_add_to_winners(state, price)
                if "ADD_TRIGGERED" in add_events:
                    # Place market order for 1 contract in same direction
                    add_action = "BUY" if state.direction == "LONG" else "SELL"
                    add_result = await broker.place_market_order(add_action, 1)
                    if "order_id" not in add_result:
                        logger.error(f"Add-to-winners order FAILED: {add_result}")
                    else:
                        fill_price = add_result.get("avg_price", price)
                        process_add_fill(state, fill_price)

                        # Update stop order qty to include the new contract
                        success = await broker.modify_stop_qty(
                            state.stop_order_id, state.contracts_active
                        )
                        if not success:
                            # Re-place stop with new qty
                            await broker.cancel_order(state.stop_order_id)
                            stop_action = "SELL" if state.direction == "LONG" else "BUY"
                            result = await broker.place_stop_order(
                                action=stop_action,
                                qty=state.contracts_active,
                                stop_price=state.trailing_stop,
                            )
                            if "order_id" in result:
                                state.stop_order_id = result["order_id"]
                                state.save()

                        await alerts.send(
                            f"➕ <b>ADD TO WINNERS</b>\n"
                            f"{state.direction} +1 contract @ {fill_price}\n"
                            f"Add #{state.adds_used} | Total contracts: {state.contracts_active}\n"
                            f"Trail stop: {state.trailing_stop}"
                        )
                        logger.info(f"Add #{state.adds_used}: {add_action} 1 @ {fill_price}, "
                                     f"total contracts={state.contracts_active}")


async def end_of_day():
    """17:35 UK — Close everything, send summary."""
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        return

    state = DailyState.load()
    if state.phase == Phase.IDLE:
        return

    logger.info("═══ END OF DAY ═══")

    # Cancel all pending orders first
    if await broker.ensure_connected():
        await broker.cancel_all_orders()

    # Close any open position with retries
    position_closed = False
    for attempt in range(1, 4):
        try:
            if not await broker.ensure_connected():
                logger.warning(f"EOD close attempt {attempt}/3: connection failed")
                await asyncio.sleep(5 * attempt)
                continue
            pos = await broker.get_position()
            if pos["direction"] == "FLAT":
                position_closed = True
                break
            price = await broker.get_current_price()
            if price:
                process_stop_hit(state, price)
                # Mark as EOD close (overrides the stop-derived reason)
                if state.trades and state.trades[-1].get("exit"):
                    state.trades[-1]["exit_reason"] = "EOD_CLOSE"
            await broker.close_position()
            # Verify close
            await asyncio.sleep(2)
            pos = await broker.get_position()
            if pos["direction"] == "FLAT":
                position_closed = True
                logger.info(f"EOD position closed on attempt {attempt}")
                break
            logger.warning(f"EOD close attempt {attempt}/3: position still open")
        except Exception as e:
            logger.error(f"EOD close attempt {attempt}/3 error: {e}")
        await asyncio.sleep(5 * attempt)

    if not position_closed:
        await alerts.send(
            "🚨 <b>EOD CLOSE FAILED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Position may still be open after 3 attempts.\n"
            "Manual intervention required!"
        )

    # Log any completed trades not already journaled during the day
    for t in state.trades:
        if t.get("exit") and not t.get("journaled"):
            journal.append_trade(t, state)
            t["journaled"] = True

    await alerts.send(alerts.day_summary(state))

    global _bar4_triggered
    state.phase = Phase.DONE
    state.save()
    _bar4_triggered = False  # Reset for next trading day
    await broker.disconnect()
    logger.info("Day complete")


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

async def graceful_shutdown():
    """Cancel open orders and close connections before exit."""
    logger.info("═══ GRACEFUL SHUTDOWN ═══")
    state = DailyState.load()

    try:
        if await broker.connect():
            # Cancel all open orders to avoid orphans
            cancelled = await broker.cancel_all_orders()
            if cancelled:
                logger.info(f"Shutdown: cancelled {cancelled} open orders")
                await alerts.send(
                    f"🛑 <b>BOT SHUTDOWN</b>\n"
                    f"Cancelled {cancelled} open orders\n"
                    f"Phase: {state.phase}\n"
                    f"<i>Bot will resume on restart</i>"
                )

            # If we have an active position, DON'T close it — just notify
            pos = await broker.get_position()
            if pos["direction"] != "FLAT":
                await alerts.send(
                    f"⚠️ <b>Position still open!</b>\n"
                    f"{pos['direction']} {abs(pos['position'])} contracts\n"
                    f"<i>Stop orders cancelled — monitor manually until restart</i>"
                )

            await broker.disconnect()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


async def startup_recovery():
    """
    On startup, check if we have an existing position from a previous session.
    If so, resume monitoring by restoring the stop order.
    Also checks IG for orphaned positions even when state shows no active trade.
    """
    state = DailyState.load()

    # Check for orphaned positions even when state says IDLE/DONE
    if state.phase not in (Phase.LONG_ACTIVE, Phase.SHORT_ACTIVE):
        try:
            if await broker.ensure_connected():
                pos = await broker.get_position()
                if pos["direction"] != "FLAT":
                    logger.warning(f"ORPHAN POSITION detected: {pos['direction']} "
                                   f"x{pos['position']} @ {pos['avg_cost']}")
                    await alerts.send(
                        "⚠️ <b>ORPHAN POSITION FOUND</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"IG has {pos['direction']} x{pos['position']} "
                        f"@ {pos['avg_cost']}\n"
                        f"But bot state is {state.phase.value}\n"
                        "Manual review required — close via IG or restart with state fix."
                    )
        except Exception as e:
            logger.error(f"Orphan check failed: {e}")
        return

    # Don't attempt recovery on weekends — market is closed
    now = datetime.now(config.TZ_UK)
    if now.weekday() >= 5:
        logger.info(f"Startup recovery skipped — weekend (phase={state.phase})")
        return

    logger.info("═══ STARTUP RECOVERY ═══")
    logger.info(f"Resuming {state.direction} position from {state.entry_price}")

    if not await broker.connect():
        logger.error("Cannot connect for recovery — will retry on next monitor cycle")
        return

    pos = await broker.get_position()
    if pos["direction"] == "FLAT":
        # Position was closed while we were down
        logger.info("Position already closed — marking as stopped out")
        process_stop_hit(state, state.trailing_stop)
        trade = state.trades[-1] if state.trades else {}
        await alerts.send(alerts.exit_stopped(state, trade))
        return

    # Position still open — re-place the trailing stop
    stop_action = "SELL" if state.direction == "LONG" else "BUY"
    qty = state.contracts_active or config.NUM_CONTRACTS

    result = await broker.place_stop_order(
        action=stop_action, qty=qty, stop_price=state.trailing_stop,
    )
    if "order_id" in result:
        state.stop_order_id = result["order_id"]
        state.save()
        logger.info(f"Recovery: stop re-placed @ {state.trailing_stop} (qty={qty})")

    # Re-place TP orders if not yet filled
    if config.PARTIAL_EXIT:
        tp_action = "SELL" if state.direction == "LONG" else "BUY"
        if not state.tp1_filled and state.tp1_price:
            tp1 = await broker.place_limit_order(tp_action, 1, state.tp1_price)
            if "order_id" in tp1:
                state.tp1_order_id = tp1["order_id"]
        if not state.tp2_filled and state.tp2_price:
            tp2 = await broker.place_limit_order(tp_action, 1, state.tp2_price)
            if "order_id" in tp2:
                state.tp2_order_id = tp2["order_id"]
        state.save()

    await alerts.send(
        f"🔄 <b>BOT RECOVERED</b>\n"
        f"{state.direction} @ {state.entry_price}\n"
        f"Stop: {state.trailing_stop} (qty={qty})\n"
        f"Contracts active: {state.contracts_active}\n"
        f"<i>Monitoring resumed</i>"
    )


async def run_bot():
    # Write PID for Docker healthcheck
    with open("/tmp/asrs.pid", "w") as f:
        f.write(str(os.getpid()))

    # Graceful shutdown on SIGTERM (docker stop) and SIGINT (Ctrl+C)
    loop = asyncio.get_event_loop()
    shutdown_triggered = False

    def _handle_signal():
        nonlocal shutdown_triggered
        if shutdown_triggered:
            return
        shutdown_triggered = True
        logger.info("Shutdown signal received")
        loop.create_task(_shutdown_and_exit())

    async def _shutdown_and_exit():
        await graceful_shutdown()
        scheduler.shutdown(wait=False)
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    # Recovery: check for existing positions from previous session
    await startup_recovery()

    # Register event-driven fill handler (fires instantly on any order fill)
    broker.register_order_handler(_handle_fill_event)

    # Start web dashboard on port 8080
    start_dashboard(port=8080)

    # Start Telegram command handler (shared with FTSE bot)
    import telegram_cmd
    ftse_broker_ref = None
    try:
        from ftse_bot.main import broker as _ftse_broker
        ftse_broker_ref = _ftse_broker
    except Exception:
        pass
    asyncio.get_event_loop().create_task(
        telegram_cmd.poll_commands(dax_broker=broker, ftse_broker=ftse_broker_ref)
    )

    scheduler.add_job(health_check, "cron",
        day_of_week="mon-fri", hour=6, minute=30,
        id="health", misfire_grace_time=120)

    scheduler.add_job(morning_routine, "cron",
        day_of_week="mon-fri", hour=config.MORNING_HOUR, minute=config.MORNING_MINUTE,
        id="morning", misfire_grace_time=120)

    scheduler.add_job(monitor_cycle, "cron",
        day_of_week="mon-fri", hour="8-17", minute="*",
        id="monitor", misfire_grace_time=30)

    scheduler.add_job(end_of_day, "cron",
        day_of_week="mon-fri", hour=config.SUMMARY_HOUR, minute=config.SUMMARY_MINUTE,
        id="eod", misfire_grace_time=120)

    scheduler.start()

    mode = "DEMO" if config.IG_DEMO else "LIVE"
    tg = "✅" if config.TG_TOKEN else "❌"
    exit_mode = f"Partial (TP1={config.TP1_PTS}, TP2={config.TP2_PTS})" if config.PARTIAL_EXIT else "Full EMA trail"
    add_str = f"+{config.ADD_STRENGTH_MAX} @ +{config.ADD_STRENGTH_TRIGGER}pts" if config.ADD_STRENGTH_ENABLED else "OFF"
    print(f"""
╔══════════════════════════════════════════════╗
║         ASRS DAX Trading Bot                 ║
║         IG Markets Edition                   ║
╠══════════════════════════════════════════════╣
║  Mode:        {mode:<31}║
║  IG Epic:     {config.IG_EPIC:<31}║
║  Telegram:    {tg:<31}║
║  Contracts:   {config.NUM_CONTRACTS:<31}║
║  Exit mode:   {exit_mode:<31}║
║  Max entries: {config.MAX_ENTRIES:<31}║
║  Add-to-win:  {add_str:<31}║
║                                              ║
║  06:30  Health check diagnostic               ║
║  08:20  Levels + OCA bracket orders          ║
║  08:25–17:30  Monitor & trail every 5m       ║
║  17:35  Close all + summary                  ║
╚══════════════════════════════════════════════╝
""")

    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if not shutdown_triggered:
            await graceful_shutdown()
            scheduler.shutdown()
            await broker.disconnect()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_now():
    print("Running ASRS now...\n")
    await morning_routine()
    await monitor_cycle()
    state = DailyState.load()
    print(f"\nPhase: {state.phase} | Buy: {state.buy_level} | Sell: {state.sell_level}")

async def cmd_status():
    state = DailyState.load()
    print(f"Date:     {state.date}")
    print(f"Phase:    {state.phase}")
    print(f"Entries:  {state.entries_used}/{config.MAX_ENTRIES}")
    print(f"Buy:      {state.buy_level}")
    print(f"Sell:     {state.sell_level}")
    print(f"Bar:      #{state.bar_number} ({state.range_flag}, {state.bar_range} pts)")
    if state.direction:
        print(f"Position: {state.direction} @ {state.entry_price}")
        print(f"Trail:    {state.trailing_stop}")
    if state.trades:
        total = 0
        for t in state.trades:
            pnl = t.get("pnl_pts", "open")
            total += pnl if isinstance(pnl, (int, float)) else 0
            print(f"  #{t['num']} {t['direction']}: {t.get('entry','?')} → {t.get('exit','open')} = {pnl}")
        print(f"  Total: {total:+.1f} pts")
    if await broker.connect():
        pos = await broker.get_position()
        orders = await broker.get_open_orders()
        print(f"\nIG: {pos['direction']} ({pos['position']}), {len(orders)} open orders")
        await broker.disconnect()

async def cmd_cancel():
    if await broker.connect():
        print(f"Cancelled {await broker.cancel_all_orders()} orders")
        await broker.disconnect()

async def cmd_close():
    if await broker.connect():
        await broker.cancel_all_orders()
        await broker.close_position()
        print("Done")
        await broker.disconnect()

async def cmd_test():
    tg_ok = await alerts.send(f"✅ <b>ASRS Bot Test</b>\nEpic: {config.IG_EPIC}")
    print(f"Telegram: {'✅' if tg_ok else '❌'}")
    ig_ok = await broker.connect()
    if ig_ok:
        price = await broker.get_current_price()
        print(f"IG: ✅ | DAX: {price}")
        await broker.disconnect()
    else:
        print("IG: ❌ (check credentials)")


async def cmd_backtest():
    from backtest import main as bt_main
    await bt_main()

async def cmd_situational():
    sys.argv = ["backtest.py", "--situational"]
    from backtest import main as bt_main
    await bt_main()


def main():
    cmd = sys.argv[1].lower().strip("-") if len(sys.argv) > 1 else "run"
    cmds = {"run": run_bot, "now": cmd_now, "status": cmd_status,
            "cancel": cmd_cancel, "close": cmd_close, "test": cmd_test,
            "backtest": cmd_backtest, "situational": cmd_situational}
    if cmd == "help":
        print("Commands: --now | --status | --cancel | --close | --test | --backtest | --situational | --help")
        return
    if cmd in cmds:
        asyncio.run(cmds[cmd]())
    else:
        print(f"Unknown: {cmd}. Try --help")

if __name__ == "__main__":
    main()
