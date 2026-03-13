# ASRS Bot — Automated DAX Trading via IBKR

Fully automated implementation of Tom Hougaard's **Advanced School Run Strategy** on DAX 40, connected to Interactive Brokers for real-time data and order execution.

## What It Does

Every trading morning, completely hands-free:

1. **08:20 UK** — Fetches real-time 5-min bars from IBKR, identifies bar 4, calculates buy/sell levels
2. **08:20 UK** — Places OCA bracket on IBKR: buy-stop above + sell-stop below (when one fills, the other auto-cancels)
3. **08:25–17:30** — Monitors position every 5 minutes, trails stop to last 5-min candle low/high
4. **On fill** — Sends Telegram alert, places trailing stop order on IBKR
5. **On stop** — Closes position, places new OCA bracket if flip available
6. **17:35 UK** — Sends day summary, closes any remaining position

You wake up, check Telegram. That's it.

## Strategy Rules

| Rule | Implementation |
|---|---|
| Entry | 4th bar high + 2 pts (buy) / low - 2 pts (sell) |
| Initial stop | Opposite entry level |
| Trailing stop | Low of last completed 5-min candle (longs) / High (shorts) |
| Trail direction | Only moves in your favour |
| Max entries | 2 per day (1 flip allowed) |
| Profit target | None — let it run |
| Order type | OCA bracket (auto-cancel opposite on fill) |

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  IBKR Servers   │◄───►│  IB Gateway  │◄───►│   ASRS Bot   │
│  (Eurex/DAX)    │     │  (Docker)    │     │  (Python)    │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                               ┌─────▼─────┐
                                               │  Telegram  │
                                               │  (Alerts)  │
                                               └───────────┘
```

## Files

```
asrs-bot/
├── bot.py              Main orchestrator — scheduler, CLI
├── broker.py           IBKR connection, data, orders via ib_async
├── strategy.py         ASRS state machine, levels, trailing stop
├── alerts.py           Telegram formatting & sending
├── config.py           All settings from .env
├── requirements.txt    Python dependencies
├── Dockerfile          Bot container
├── docker-compose.yml  Bot + IB Gateway together
├── .env.example        Configuration template
└── data/
    └── daily_state.json    Auto-created daily state
```

## Setup

### 1. Prerequisites

- IBKR account (even paper trading account works)
- Docker & Docker Compose installed
- Telegram account

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your IBKR credentials, Telegram token, etc.
```

### 3. Telegram Bot Setup

1. Open Telegram → search `@BotFather`
2. Send `/newbot`, follow prompts → copy the **token**
3. Send any message to your new bot
4. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Find `"chat":{"id":123456789}` → that's your **CHAT_ID**

### 4. Start (Docker)

```bash
# Start IB Gateway + bot
docker compose up -d

# Check logs
docker compose logs -f asrs

# Test connections
docker compose exec asrs python bot.py --test
```

### 5. Start (Local — for development)

```bash
# Start IB Gateway separately (or use TWS)
# Then:
pip install -r requirements.txt
python bot.py --test      # Test connections
python bot.py --now       # Run morning routine immediately
python bot.py             # Start scheduled bot
```

## CLI Commands

```bash
python bot.py              # Run scheduled bot (24/5)
python bot.py --now        # Calculate levels + place orders immediately
python bot.py --status     # Show current state + IBKR position
python bot.py --test       # Test Telegram + IBKR connections
python bot.py --cancel     # Cancel all open DAX orders
python bot.py --close      # Close any open position at market
python bot.py --help       # Show commands
```

## IBKR Ports

| Port | Mode | Use |
|------|------|-----|
| 4002 | Paper (Gateway) | Testing — no real money |
| 4001 | Live (Gateway) | Real trading |
| 7497 | Paper (TWS) | If using TWS desktop instead |
| 7496 | Live (TWS) | If using TWS desktop instead |

The bot auto-detects paper vs live from the port and shows it in every alert.

## Instrument Options

| Config | Instrument | Point Value | Commission | Best For |
|--------|-----------|-------------|------------|----------|
| `FDXS` | Micro DAX Future | €1/pt | €0.38 | Starting (Month 1-6) |
| `FDXM` | Mini DAX Future | €5/pt | €0.75 | Scaling (Month 7+) |
| `IBDE30` + `SEC_TYPE=CFD` | DAX CFD | €1/pt | ~0.01% | Alternative |

Change in `.env`:
```
INSTRUMENT=FDXM
NUM_CONTRACTS=1
```

## Paper Trading First

The bot defaults to paper trading (port 4002). Run it for at least 1 month:

1. Watch the Telegram alerts daily
2. Check if levels match what you'd calculate manually
3. Verify trailing stop moves correctly
4. Review day summaries for edge validation

When ready for live:
```
# In .env or docker-compose.yml:
TRADING_MODE=live    # IB Gateway
IB_PORT=4001         # Live port
```

## Deployment (VPS)

### Hetzner (Recommended — €4/month)

```bash
# On your VPS:
apt update && apt install docker.io docker-compose-plugin -y

# Clone your bot
git clone <your-repo> asrs-bot && cd asrs-bot

# Configure
cp .env.example .env && nano .env

# Start
docker compose up -d

# Auto-restart on reboot
docker compose up -d  # restart: unless-stopped handles this
```

### Oracle Cloud (Free tier)

Same steps — Oracle's Always Free ARM instance handles this easily.

## Scaling Path

| Phase | Config | €/point | Daily avg | Monthly |
|-------|--------|---------|-----------|---------|
| Month 1-3 | 1x FDXS | €1 | ~€15 | ~€300 |
| Month 4-6 | 2x FDXS | €2 | ~€30 | ~€600 |
| Month 7-12 | 1x FDXM | €5 | ~€75 | ~€1,500 |
| Year 2 | 2x FDXM | €10 | ~€150 | ~€3,000 |
| Year 3-5 | 10x FDXM | €50 | ~€750 | ~€15,000 |

## Troubleshooting

**Bot can't connect to IBKR**
- Check IB Gateway is running: `docker compose logs ibgateway`
- Verify API is enabled in Gateway settings
- Check port matches (4002 for paper)

**No bars returned**
- Market may be closed (holiday, weekend)
- DAX futures market data subscription needed on IBKR
- Check IBKR account has Eurex permissions

**Orders rejected**
- Check margin requirements
- Verify futures trading permissions enabled
- Paper account may need manual data subscription setup

**Telegram not sending**
- Verify token and chat_id in .env
- Test with: `python bot.py --test`

## IBKR Market Data

You need DAX market data subscription. In IBKR:
1. Login to Client Portal
2. Settings → Market Data Subscriptions
3. Add: **Eurex** (covers DAX futures)
4. Paper accounts: may need to request data separately

## Going Live Checklist

- [ ] Paper traded for 30+ days
- [ ] Win rate and P&L match expectations
- [ ] Trailing stop behaves correctly
- [ ] Reviewed all error scenarios
- [ ] Set `TRADING_MODE=live` and `IB_PORT=4001`
- [ ] Start with 1 Micro DAX contract (€1/pt)
- [ ] Scale only after 2+ months of live consistency
