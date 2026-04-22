// =========================================================================
// ASRS Opening Range Breakout — ProRealCode port of Python v2 backtest
// Matches asrs/strategy.py logic 1:1 where possible
//
// Rules ported:
//   R1:  Bar 4 signal trigger
//   R2:  Hybrid bar 5 (NARROW uses bar 4, NORMAL/WIDE uses bar 5)
//   R3:  Buy = sig_high + buffer, Sell = sig_low - buffer
//   R5:  OCA bracket via simultaneous BUY STOP + SELLSHORT STOP
//   R6:  Max bar range filter (skip days with crazy bars)
//   R11: Max entries per session (default 3)
//   R12: Initial stop at sig_low (LONG) / sig_high (SHORT)
//   R13: Breakeven + buffer after +N points profit
//   R14: Candle trail on previous 5-min bar extreme
//   R15: Tight trail (prev close) when deep in profit
//   R18: Stop exit detection
//   R19: EOD force close at session end
//   R20: Re-entry allowed when price back inside [sell_level, buy_level]
//
// Rules NOT ported (simplifications):
//   - Adds (add_max = 0 in live config anyway)
//   - Slippage check (bar-level execution can't model this)
//   - Position sizing (ProBacktest uses fixed lot)
//   - Risk cap via max_risk_gbp (uses raw bar range)
//
// Usage:
//   1. Open the appropriate chart (Germany 40 DFB for DAX, etc.)
//   2. Set timeframe to 5 minutes
//   3. Go to ProBacktest → New strategy
//   4. Paste this code
//   5. EDIT the PARAMS section below for the signal you want to test
//   6. Click Backtest → view PF, net profit, equity curve
//
// Run this 8 times, once per signal:
//   DAX_S1, DAX_S2, US30_S1, US30_S2, US30_S3, NIKKEI_S1, NIKKEI_S2, NIKKEI_S3
//
// PARAMS for each signal at the very bottom of this file.
// =========================================================================

DEFPARAM CumulateOrders = False
DEFPARAM PreLoadBars = 200

// ==================== EDIT THESE FOR EACH SIGNAL ====================
// Default: DAX_S1 (Germany 40 chart, 5min, Berlin time)

sessOpenH = 9              // Session open hour (chart local time)
sessOpenM = 0              // Session open minute
sessEndH  = 17             // Session end hour
sessEndM  = 30             // Session end minute

buf         = 2            // Entry level buffer (pts)
narrowR     = 15           // Bar range < narrowR → NARROW (use bar 4)
wideR       = 40           // Bar range > wideR → WIDE (still use bar 5)
maxR        = 120          // Skip day if bar range > maxR
bePts       = 99999        // Breakeven DISABLED (set high so never triggers)
beBuf       = 5            // BE stop buffer (stop = entry -/+ beBuf)
tightThresh = 100          // Trail uses prev_close when profit > tightThresh
maxEntries  = 1            // Max entries per session (SIMPLIFIED: 1 for testing)

// ====================================================================

// -- Time bookkeeping --
curMinOfDay  = Hour * 60 + Minute
sessOpenMin  = sessOpenH * 60 + sessOpenM
sessEndMin   = sessEndH * 60 + sessEndM

// In-session flag (only run trading logic during session)
inSess = curMinOfDay >= sessOpenMin AND curMinOfDay < sessEndMin

// Session bar number (1 = first 5-min bar of session)
sessBarNum = 0
IF inSess THEN
    sessBarNum = (curMinOfDay - sessOpenMin) / 5 + 1
ENDIF

// -- Day change detection (reset state on new trading day at session open) --
// We detect first bar of session as "new session start"
IF inSess AND sessBarNum = 1 THEN
    bar4H = 0
    bar4L = 0
    bar5H = 0
    bar5L = 0
    sigH = 0
    sigL = 0
    buyLv = 0
    sellLv = 0
    initStop = 0
    currStop = 0
    entries = 0
    beHit = 0
    direction = 0
    entryPrice = 0
    waitingReentry = 0
    reEntryClear = 1
    wasLong = 0
    wasShort = 0
    sessionDone = 0
ENDIF

// -- Capture bar 4 --
IF inSess AND sessBarNum = 4 AND bar4H = 0 THEN
    bar4H = High
    bar4L = Low
ENDIF

// -- Classify bar 4 and set signal levels if NARROW --
IF inSess AND sessBarNum = 4 AND bar4H > 0 AND sigH = 0 THEN
    rng4 = bar4H - bar4L
    IF rng4 > 0 AND rng4 < narrowR AND rng4 <= maxR THEN
        // NARROW → use bar 4
        sigH = bar4H
        sigL = bar4L
        buyLv = sigH + buf
        sellLv = sigL - buf
        initStop = 0  // set per-direction on fill
    ENDIF
ENDIF

// -- Capture bar 5 and use it (NORMAL/WIDE cases, or fallback) --
IF inSess AND sessBarNum = 5 AND sigH = 0 THEN
    bar5H = High
    bar5L = Low
    rng5 = bar5H - bar5L
    IF rng5 > 0 AND rng5 <= maxR THEN
        sigH = bar5H
        sigL = bar5L
        buyLv = sigH + buf
        sellLv = sigL - buf
    ELSE
        // Bar 5 still too wide → abort session
        sessionDone = 1
    ENDIF
ENDIF

// -- Re-entry gate: after a stop-out, wait for price back inside range --
// Only check the gate if we're NOT in a position AND we didn't just exit this bar
// (wasLong=0 AND wasShort=0 means the bar started flat — we're not mid-exit)
IF waitingReentry = 1 AND sigH > 0 AND wasLong = 0 AND wasShort = 0 THEN
    // Clear gate if current bar close is inside the range
    IF Close >= sellLv AND Close <= buyLv THEN
        reEntryClear = 1
        waitingReentry = 0
    ENDIF
ENDIF

// -- Arm bracket (stop order) --
// Guards:
//   wasLong/wasShort = 0 → we were FLAT at start of bar (prevents same-bar
//     re-entry after stop-out, which was causing 90% of trades to hit BE)
//   Use IF/ELSIF for LONG priority (matches Python v2 engine for wide bars
//     that cross both levels in a single bar — don't flip position)
IF inSess AND sigH > 0 AND OnMarket = 0 AND wasLong = 0 AND wasShort = 0 AND entries < maxEntries AND reEntryClear = 1 AND sessionDone = 0 AND sessBarNum > 4 AND curMinOfDay < sessEndMin THEN
    IF High >= buyLv THEN
        BUY 1 CONTRACT AT buyLv STOP
        entries = entries + 1
        sessionDone = 1
    ELSIF Low <= sellLv THEN
        SELLSHORT 1 CONTRACT AT sellLv STOP
        entries = entries + 1
        sessionDone = 1
    ENDIF
ENDIF

// -- Detect new fill (set initial stop, don't double-count entries) --
// entries is already incremented in the arm block above
IF LongOnMarket > 0 AND wasLong = 0 THEN
    currStop = sigL
    beHit = 0
    direction = 1
    entryPrice = PositionPrice
    stopDistInit = entryPrice - currStop
    IF stopDistInit > 0 THEN
        SET STOP LOSS stopDistInit
    ENDIF
ENDIF

IF ShortOnMarket > 0 AND wasShort = 0 THEN
    currStop = sigH
    beHit = 0
    direction = -1
    entryPrice = PositionPrice
    stopDistInit = currStop - entryPrice
    IF stopDistInit > 0 THEN
        SET STOP LOSS stopDistInit
    ENDIF
ENDIF

// -- Detect exit (was in trade, now flat) --
// Trigger re-entry wait state
IF (wasLong = 1 OR wasShort = 1) AND NOT OnMarket THEN
    IF entries < maxEntries AND inSess THEN
        waitingReentry = 1
        reEntryClear = 0
    ENDIF
    direction = 0
    beHit = 0
ENDIF

// -- Trail management (only if in position) --
IF OnMarket AND direction <> 0 THEN
    prevH = High[1]
    prevL = Low[1]
    prevC = Close[1]

    IF direction = 1 THEN
        // LONG: breakeven
        unrL = prevC - entryPrice
        IF beHit = 0 AND unrL >= bePts THEN
            beHit = 1
            newBeL = entryPrice - beBuf
            IF newBeL > currStop THEN
                currStop = newBeL
            ENDIF
        ENDIF
        // LONG: candle trail (prev close if deep profit, else prev low)
        profL = prevC - entryPrice
        IF profL >= tightThresh THEN
            nsL = prevC
        ELSE
            nsL = prevL
        ENDIF
        IF nsL > currStop THEN
            currStop = nsL
        ENDIF
        // Apply as distance from entry
        stopDistL = entryPrice - currStop
        IF stopDistL > 0 THEN
            SET STOP LOSS stopDistL
        ENDIF
    ENDIF

    IF direction = -1 THEN
        // SHORT: breakeven
        unrS = entryPrice - prevC
        IF beHit = 0 AND unrS >= bePts THEN
            beHit = 1
            newBeS = entryPrice + beBuf
            IF newBeS < currStop THEN
                currStop = newBeS
            ENDIF
        ENDIF
        // SHORT: candle trail (prev close if deep profit, else prev high)
        profS = entryPrice - prevC
        IF profS >= tightThresh THEN
            nsS = prevC
        ELSE
            nsS = prevH
        ENDIF
        IF nsS < currStop THEN
            currStop = nsS
        ENDIF
        stopDistS = currStop - entryPrice
        IF stopDistS > 0 THEN
            SET STOP LOSS stopDistS
        ENDIF
    ENDIF
ENDIF

// -- EOD force close (session end reached, close anything open) --
IF curMinOfDay >= sessEndMin AND OnMarket THEN
    IF LongOnMarket THEN
        SELL AT MARKET
    ENDIF
    IF ShortOnMarket THEN
        EXITSHORT AT MARKET
    ENDIF
    sessionDone = 1
ENDIF

// -- State trackers for next bar's transition detection --
wasLong  = LongOnMarket
wasShort = ShortOnMarket


// =========================================================================
// PARAMETER PRESETS — copy the right block into the PARAMS section above
// =========================================================================
//
// === DAX_S1 (Germany 40 chart, 5min) ===
//   sessOpenH = 9   sessOpenM = 0
//   sessEndH  = 17  sessEndM  = 30
//   buf = 2   narrowR = 15   wideR = 40   maxR = 120
//   bePts = 15   beBuf = 5   tightThresh = 100
//   maxEntries = 3
//
// === DAX_S2 (Germany 40 chart, 5min) ===
//   sessOpenH = 14  sessOpenM = 0
//   sessEndH  = 17  sessEndM  = 30
//   buf = 2   narrowR = 15   wideR = 40   maxR = 120
//   bePts = 15   beBuf = 5   tightThresh = 100
//   maxEntries = 3
//
// === US30_S1 (Wall Street chart, 5min, ET) ===
//   sessOpenH = 9   sessOpenM = 30
//   sessEndH  = 16  sessEndM  = 0
//   buf = 5   narrowR = 30   wideR = 100   maxR = 300
//   bePts = 20   beBuf = 5   tightThresh = 80
//   maxEntries = 3
//
// === US30_S2 (Wall Street chart, 5min, ET) ===
//   sessOpenH = 11  sessOpenM = 0
//   sessEndH  = 16  sessEndM  = 0
//   buf = 5   narrowR = 30   wideR = 100   maxR = 300
//   bePts = 20   beBuf = 5   tightThresh = 80
//   maxEntries = 3
//
// === US30_S3 (Wall Street chart, 5min, ET) ===
//   sessOpenH = 13  sessOpenM = 0
//   sessEndH  = 16  sessEndM  = 0
//   buf = 5   narrowR = 30   wideR = 100   maxR = 300
//   bePts = 20   beBuf = 5   tightThresh = 80
//   maxEntries = 3
//
// === NIKKEI_S1 (Japan 225 chart, 5min, JST) ===
//   sessOpenH = 10  sessOpenM = 0
//   sessEndH  = 15  sessEndM  = 0
//   buf = 2   narrowR = 50   wideR = 150   maxR = 250
//   bePts = 50   beBuf = 10  tightThresh = 300
//   maxEntries = 3
//
// === NIKKEI_S2 (Japan 225 chart, 5min, JST) ===
//   sessOpenH = 12  sessOpenM = 0
//   sessEndH  = 15  sessEndM  = 0
//   buf = 2   narrowR = 50   wideR = 150   maxR = 250
//   bePts = 50   beBuf = 10  tightThresh = 300
//   maxEntries = 3
//
// === NIKKEI_S3 (Japan 225 chart, 5min, JST) ===
//   sessOpenH = 13  sessOpenM = 0
//   sessEndH  = 15  sessEndM  = 0
//   buf = 2   narrowR = 50   wideR = 150   maxR = 250
//   bePts = 50   beBuf = 10  tightThresh = 300
//   maxEntries = 3
