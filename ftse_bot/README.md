# FTSE 1BN/1BP Trading Bot

Automated IBKR trading bot implementing Tom Hougaard's **First Bar Negative / First Bar Positive** strategy on FTSE 100.

## Strategy

Every weekday at **08:05 UK time**, the bot reads the completed 08:00-08:05 five-minute bar:

| Bar Type | Condition | Orders Placed |
|----------|-----------|---------------|
| **1BN** (First Bar Negative) | Close < Open | Buy stop 1pt below low **AND** sell stop 1pt above high (OCO) |
| **1BP** (First Bar Positive) | Close > Open | Sell stop 1pt below low only |
| **DOJI** | Close == Open | Configurable: SKIP (default), TREAT_AS_1BN, TREAT_AS_1BP |

### Position Sizing
- Default: £1/point (configurable via `STAKE_PER_POINT`)
- Auto-halved when bar width > 30 points

### Stop Management (3 phases)
1. **INITIAL** — Stop at opposite side of entry (bar width distance)
2. **BREAKEVEN** — Stop moves to entry price when profit reaches +10 pts
3. **TRAILING** — Trail by bar width in favour direction (never widens)

### Session Rules
- No profit target — ride until stopped out or 16:30 hard close
- Mon-Fri only, respects UK bank holidays
- Max 1 trade per day

## Environment Variables

```env
# IBKR (shared with ASRS via same gateway)
IB_HOST=ibgateway
IB_PORT=4002
FTSE_IB_CLIENT_ID=2         # Must differ from ASRS (1)

# Instrument
FTSE_INSTRUMENT=IBGB100     # FTSE 100 CFD

# Strategy
STAKE_PER_POINT=1            # GBP per point
DOJI_ACTION=SKIP             # SKIP | TREAT_AS_1BN | TREAT_AS_1BP
FTSE_BAR_WIDTH_THRESHOLD=30  # Halve stake above this
FTSE_BREAKEVEN_TRIGGER=10    # Pts to move stop to breakeven

# Telegram (separate bot from ASRS)
FTSE_TELEGRAM_TOKEN=
FTSE_TELEGRAM_CHAT_ID=
```

## Running

### With Docker Compose (alongside ASRS)
```bash
# Start both bots
docker compose up -d

# Start FTSE only
docker compose up -d ftse

# View logs
docker compose logs ftse -f

# Rebuild after changes
docker compose up -d --build ftse
```

### Standalone
```bash
python -m ftse_bot.main              # Run scheduled bot
python -m ftse_bot.main --test       # Test Telegram + IBKR
python -m ftse_bot.main --status     # Show current state
python -m ftse_bot.main --cancel     # Cancel all open orders
python -m ftse_bot.main --close      # Close position + cancel
```

## Schedule (UK time)

| Time | Action |
|------|--------|
| 07:00 | Health check |
| 08:05 | Read 1st bar, classify, place orders |
| 08:05-16:30 | Monitor every 1 minute (stop management) |
| 16:30 | Hard close session |
| 17:00 | Daily summary + weekly P&L |

## Telegram Alerts

- 🇬🇧 Bar detected (type, OHLC, width, stake)
- 📥 Orders placed (direction, entry, stop, stake)
- ✅ Entry filled (price, time)
- 🔄 Stop moved to breakeven
- 📈 Trailing stop update (every 5pt move)
- 🛑 Stopped out (P&L in pts and £)
- ⏰ Session close (16:30 forced exit)
- ⚠️ No trade (doji skip, no trigger)
- ❌ Error alerts

## Data Files

- `data/ftse_trades.csv` — Trade journal
- `data/ftse_state.json` — Intraday state (reset daily)
- `data/ftse_rth.parquet` — Historical RTH bars (for backtesting)
- `data/ftse_all.parquet` — Historical all-hours bars
