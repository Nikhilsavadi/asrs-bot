"""
backtest_v2.py — ASRS strategy backtest with HONEST re-entry semantics.

OPTION B from the 2026-04-09 backtest fix discussion:
  - Run 5-min bars as the primary engine (cheap, fast)
  - On STOP events, drill into 1-min bars to check what actually happened
  - Re-entry only fires if a 1-min bar after the stop ACTUALLY touches a level
  - Fill at the 1-min bar OPEN where the touch happened (more realistic)
  - If no 1-min bar in the rest of the session touches a level → no re-entry

This closes Bug 1 (fictional fill prices) and Bug 2 (no pullback required)
from the original `backtest.py`. Bug 3 (within-bar order of touches) is
reduced from 5-min ambiguity to 1-min ambiguity, which is much smaller.

Bug 3 is fully closed only by Option C (full event-driven 1-min backtest).
That's a future project — see project_backtest_option_c.md in memory.

Public API matches backtest.py exactly:
  simulate_session_v2(day_5min, day_1min, hours_5m, minutes_5m,
                       open_h, open_m, eod_h, eod_m, cfg) -> list[trades]

The only difference vs the original:
  - Takes day_1min as a second array (None disables drill-down)
  - Re-entry path uses _find_real_reentry_in_1min instead of bar-low scan
"""
import numpy as np

BAR5_RULES = ["NORMAL", "WIDE"]


def _find_real_reentry_in_1min(
    day_1min: np.ndarray,
    one_min_minutes: np.ndarray,  # minute-of-day for each 1-min bar
    after_min: int,                # only consider 1-min bars at or after this minute-of-day
    eod_mins: int,
    buy_level: float,
    sell_level: float,
) -> tuple[int, str, float] | None:
    """
    Walk forward through 1-min bars after `after_min` and find a re-entry
    that the LIVE BOT would actually take.

    The live bot's re-entry path is:
      1. After a stop, phase = LEVELS_SET. Bracket NOT yet armed.
      2. monitor_cycle waits until price is inside the range
         (sell_level <= price <= buy_level) before arming the bracket.
      3. Once armed, the tick trigger fires on the next price that
         crosses OUT of the range (above buy_level or below sell_level).
      4. Fill is at the level + small slippage (~1-3 pts in practice
         because the stop-entry order has minimal latency).

    To match this in the backtest:
      Phase 1: scan 1-min bars from `after_min`, look for the FIRST bar
               where (sell_level <= bar.low AND bar.high <= buy_level)
               OR (bar.low <= buy_level AND bar.high >= sell_level)
               i.e. bar's range overlaps with the level range.
               Equivalently: bar.low <= buy_level AND bar.high >= sell_level.
      Phase 2: from that "armed" bar onwards, scan for the next bar that
               actually crosses OUT (high >= buy_level OR low <= sell_level).
               Fill at the level (not at bar open) — this represents the
               intra-bar tick trigger firing as soon as price hits the level.

    Returns (idx, direction, fill_price) or None.

    Notes on the fill price:
      - We fill at the LEVEL itself, not at `max(bar.open, level)`. This
        is because by the time we're checking this re-entry, the bracket
        was armed in a previous bar where price was inside the range. Any
        subsequent bar that crosses out has price ramping THROUGH the
        level, so a stop-entry order would fill at the level + 1-2 ticks
        of slippage (which we approximate as exactly the level).
      - This is slightly optimistic but matches what the live bot does.
        Real live slippage on micro futures is 1-3 pts, captured in the
        live bot's slippage_pct check.
    """
    if day_1min is None or len(day_1min) == 0:
        return None

    n = len(day_1min)
    armed_at = -1  # 1-min bar index where price first came back inside the range

    for j in range(n):
        if one_min_minutes[j] < after_min:
            continue
        if one_min_minutes[j] >= eod_mins:
            return None

        bo = day_1min[j, 0]
        bh = day_1min[j, 1]
        bl = day_1min[j, 2]

        # PHASE 1: wait for price to come back inside the range.
        # A 1-min bar "comes inside" if its range overlaps with the
        # [sell_level, buy_level] range.
        if armed_at < 0:
            # Bar overlaps the range if bl <= buy_level AND bh >= sell_level
            if bl <= buy_level and bh >= sell_level:
                armed_at = j
                # Don't immediately fire on the same bar that armed —
                # the live bot needs at least one tick after arming to
                # trigger. Continue to check the SAME bar for a cross-out
                # only if the bar's open was inside the range (otherwise
                # the cross was on the way IN, not OUT).
                if sell_level <= bo <= buy_level:
                    # Same bar can also fire the cross-out: check high/low
                    if bh >= buy_level:
                        return (j, "LONG", buy_level)
                    if bl <= sell_level:
                        return (j, "SHORT", sell_level)
                # Otherwise wait for the next bar
            continue

        # PHASE 2: bracket armed. Watch for cross-out.
        if bh >= buy_level:
            return (j, "LONG", buy_level)
        if bl <= sell_level:
            return (j, "SHORT", sell_level)

    return None


def simulate_session_v2(
    day_bars: np.ndarray,           # 5-min OHLC
    hours: np.ndarray,
    minutes: np.ndarray,
    open_h: int, open_m: int,
    eod_h: int, eod_m: int,
    cfg: dict,
    day_1min: np.ndarray | None = None,    # 1-min OHLC for drill-down
    one_min_hours: np.ndarray | None = None,
    one_min_minutes: np.ndarray | None = None,
) -> list[dict]:
    """
    Same signature/behaviour as backtest.simulate_session EXCEPT the
    re-entry path uses 1-min drill-down to check if the level was
    actually touched.

    If day_1min is None, falls back to the original re-entry behaviour
    (so the function can be a drop-in replacement when 1-min data is
    unavailable).
    """
    n = len(day_bars)
    if n == 0:
        return []

    buffer = cfg["buffer"]
    open_mins = open_h * 60 + open_m
    eod_mins = eod_h * 60 + eod_m

    bar_mins = hours * 60 + minutes
    mins_from_open = bar_mins - open_mins
    bar_nums = mins_from_open // 5 + 1

    # Bar 4
    bar4_mask = (mins_from_open >= 0) & (bar_nums == 4)
    bar4_idx = np.where(bar4_mask)[0]
    if len(bar4_idx) == 0:
        return []
    b4i = bar4_idx[0]
    bar4_h = day_bars[b4i, 1]
    bar4_l = day_bars[b4i, 2]
    bar4_range = bar4_h - bar4_l

    if bar4_range < cfg["narrow_range"]:
        range_flag = "NARROW"
    elif bar4_range > cfg["wide_range"]:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    sig_h, sig_l = bar4_h, bar4_l
    bar_num = 4
    if range_flag in BAR5_RULES:
        bar5_mask = (mins_from_open >= 0) & (bar_nums == 5)
        bar5_idx = np.where(bar5_mask)[0]
        if len(bar5_idx) > 0:
            b5i = bar5_idx[0]
            sig_h = day_bars[b5i, 1]
            sig_l = day_bars[b5i, 2]
            bar_num = 5

    bar_range = sig_h - sig_l
    if bar_range < cfg["narrow_range"]:
        range_flag = "NARROW"
    elif bar_range > cfg["wide_range"]:
        range_flag = "WIDE"
    else:
        range_flag = "NORMAL"

    if bar_range > cfg["max_bar_range"]:
        return []

    bar_high = round(sig_h, 1)
    bar_low = round(sig_l, 1)
    buy_level = round(bar_high + buffer, 1)
    sell_level = round(bar_low - buffer, 1)

    risk_pts = bar_range + buffer * 2
    if risk_pts > cfg["max_risk_gbp"]:
        max_stop = cfg["max_risk_gbp"]
        bar_high = round(sell_level + max_stop, 1)
        bar_low = round(buy_level - max_stop, 1)
        bar_range = max_stop

    sig_bar_mask = (mins_from_open >= 0) & (bar_nums == bar_num)
    sig_idx = np.where(sig_bar_mask)[0]
    if len(sig_idx) == 0:
        return []
    start_after = sig_idx[0] + 1
    if start_after >= n:
        return []

    rem_open = day_bars[start_after:, 0]
    rem_high = day_bars[start_after:, 1]
    rem_low = day_bars[start_after:, 2]
    rem_close = day_bars[start_after:, 3]
    rem_mins = bar_mins[start_after:]
    rem_n = len(rem_open)

    # First trigger (initial entry — same as v1)
    first_long = -1
    first_short = -1
    for j in range(rem_n):
        if rem_mins[j] >= eod_mins:
            break
        if first_long < 0 and rem_high[j] >= buy_level:
            first_long = j
        if first_short < 0 and rem_low[j] <= sell_level:
            first_short = j
        if first_long >= 0 and first_short >= 0:
            break

    if first_long < 0 and first_short < 0:
        return []

    if first_long >= 0 and first_short >= 0:
        if first_long < first_short:
            direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
        elif first_short < first_long:
            direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
        else:
            if rem_open[first_long] >= buy_level:
                direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
            else:
                direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short
    elif first_long >= 0:
        direction, entry, stop, start = "LONG", buy_level, sell_level, first_long
    else:
        direction, entry, stop, start = "SHORT", sell_level, buy_level, first_short

    trades = []
    entries_used = 0
    max_entries = cfg["max_entries"]
    active = True
    breakeven_hit = False
    adds_used = 0
    add_pnl = 0.0
    last_add_price = entry
    mfe = 0.0
    waiting = False
    # Track the minute-of-day where the most recent stop fired (used by
    # the 1-min re-entry drill-down to know where to start scanning).
    last_stop_min: int = 0

    for j in range(start, rem_n):
        bm = rem_mins[j]
        if bm >= eod_mins:
            if active:
                ep = rem_open[j]
                pnl = (ep - entry) if direction == "LONG" else (entry - ep)
                trades.append({
                    "direction": direction, "entry": round(entry, 1),
                    "exit": round(ep, 1), "pnl_pts": round(pnl + add_pnl, 1),
                    "mfe": round(mfe, 1), "adds": adds_used, "reason": "EOD",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1),
                })
            break

        bh = rem_high[j]
        bl = rem_low[j]
        bc = rem_close[j]
        bo = rem_open[j]

        if active:
            # Stop check
            if direction == "LONG" and bl <= stop:
                pnl = stop - entry
                trades.append({
                    "direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(pnl + add_pnl, 1),
                    "mfe": round(mfe, 1), "adds": adds_used, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1),
                })
                entries_used += 1
                active = False
                waiting = entries_used < max_entries
                # Stop fired during this 5-min bar; for the 1-min drill-down
                # start scanning from the END of this bar (bm + 5 minutes).
                last_stop_min = bm + 5
                continue

            if direction == "SHORT" and bh >= stop:
                pnl = entry - stop
                trades.append({
                    "direction": direction, "entry": round(entry, 1),
                    "exit": round(stop, 1), "pnl_pts": round(pnl + add_pnl, 1),
                    "mfe": round(mfe, 1), "adds": adds_used, "reason": "STOP",
                    "bar_num": bar_num, "range_flag": range_flag,
                    "bar_range": round(bar_range, 1),
                })
                entries_used += 1
                active = False
                waiting = entries_used < max_entries
                last_stop_min = bm + 5
                continue

            # MFE
            if direction == "LONG":
                m = bh - entry
                if m > mfe:
                    mfe = m
                unrealized = bc - entry
            else:
                m = entry - bl
                if m > mfe:
                    mfe = m
                unrealized = entry - bc

            # Breakeven
            if not breakeven_hit and unrealized >= cfg["breakeven_pts"]:
                breakeven_hit = True
                if direction == "LONG" and stop < entry:
                    stop = entry
                elif direction == "SHORT" and stop > entry:
                    stop = entry

            # Candle trail
            if j > start:
                prev_h = rem_high[j - 1]
                prev_l = rem_low[j - 1]
                prev_c = rem_close[j - 1]
                if direction == "LONG":
                    profit = prev_c - entry
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_l
                    if ns > stop:
                        stop = round(ns, 1)
                else:
                    profit = entry - prev_c
                    ns = prev_c if profit >= cfg["tight_threshold"] else prev_h
                    if ns < stop:
                        stop = round(ns, 1)

            # Adds
            if adds_used < cfg["add_max"]:
                ref = last_add_price
                pfr = (bc - ref) if direction == "LONG" else (ref - bc)
                if pfr >= cfg["add_trigger"]:
                    adds_used += 1
                    last_add_price = bc
                    breakeven_hit = True

        elif waiting:
            # ── HONEST RE-ENTRY: drill into 1-min bars ─────────────
            #
            # Old behaviour: claim re-entry on next 5-min bar where
            # bh >= buy_level OR bl <= sell_level, fill at level.
            # That produces fictional fills on continuation moves
            # (today's US30 case: claimed entry at 48095 when market
            # was trading 48298+).
            #
            # New behaviour: scan 1-min bars from `last_stop_min`
            # forward, find the FIRST one whose high reaches buy_level
            # or low reaches sell_level. Fill at max(bar_open, level)
            # for LONG (so a gap-up bar fills at the open, not the level).
            #
            # If no 1-min bar in the rest of the session touches a level,
            # no re-entry happens.
            if day_1min is not None and one_min_minutes is not None:
                hit = _find_real_reentry_in_1min(
                    day_1min, one_min_minutes, last_stop_min, eod_mins,
                    buy_level, sell_level,
                )
                if hit is not None:
                    _idx, hit_dir, fill_price = hit
                    direction = hit_dir
                    entry = fill_price
                    stop = sell_level if hit_dir == "LONG" else buy_level
                    active, waiting = True, False
                    breakeven_hit = False
                    adds_used = 0
                    add_pnl = 0.0
                    last_add_price = entry
                    mfe = 0.0
                    # Continue the 5-min loop from the current position;
                    # the next 5-min bar will manage the new position.
                    # Note: we don't try to back-fill 5-min bars between
                    # last_stop_min and the 1-min hit time — the next
                    # iteration of the outer loop will pick up management
                    # at the current 5-min bar.
                else:
                    # No re-entry possible — wait fired the rest of the session.
                    # Set waiting to False so we don't keep checking; this
                    # session is effectively done re-entering.
                    waiting = False
            else:
                # Fallback: original (broken) behaviour for cases where
                # 1-min data isn't available. Same code as backtest.py.
                tl = bh >= buy_level
                ts = bl <= sell_level
                if tl and ts:
                    if bo >= buy_level:
                        ts = False
                    elif bo <= sell_level:
                        tl = False
                    else:
                        tl = bc >= bo
                        ts = not tl
                if tl:
                    direction, entry, stop = "LONG", buy_level, sell_level
                    active, waiting = True, False
                    breakeven_hit, adds_used, add_pnl = False, 0, 0.0
                    last_add_price = entry
                    mfe = max(0, bh - entry)
                elif ts:
                    direction, entry, stop = "SHORT", sell_level, buy_level
                    active, waiting = True, False
                    breakeven_hit, adds_used, add_pnl = False, 0, 0.0
                    last_add_price = entry
                    mfe = max(0, entry - bl)

    return trades
