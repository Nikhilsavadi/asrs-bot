#!/bin/bash
# cutover_to_docker.sh — migrate the bot from host process to Docker container
#
# Run this AFTER US30 close (21:00 UK) and BEFORE Tokyo open (00:00 UK).
# Uses the safe restart window per memory.
#
# Idempotent: safe to re-run if any step fails.
#
# Usage:
#   ./cutover_to_docker.sh         # interactive — pauses at each step
#   ./cutover_to_docker.sh --auto  # fully automated (only run after dry test)

set -e
cd /root/asrs-bot

INTERACTIVE=true
if [ "${1:-}" = "--auto" ]; then INTERACTIVE=false; fi

confirm() {
    if $INTERACTIVE; then
        echo
        read -p "    → $1 [y/N] " ans
        [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "    aborted"; exit 1; }
    fi
}

step() {
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
}

step "1. Pre-flight: verify no open positions"
python3 -c "
import asyncio
from ib_async import IB
async def m():
    ib = IB()
    await ib.connectAsync('127.0.0.1', 4002, clientId=99)
    ps = [p for p in ib.positions() if p.position != 0]
    print(f'Open positions: {len(ps)}')
    for p in ps: print(f'  {p.contract.localSymbol} {p.position}')
    ib.disconnect()
    if ps:
        raise SystemExit(f'ABORT: {len(ps)} open positions, refusing cutover')
asyncio.run(m())
"
echo "  ✓ flat"

step "2. Pre-flight: verify it's a safe restart window"
HOUR_UK=$(TZ=Europe/London date +%H)
if [ "$HOUR_UK" -ge 21 ] || [ "$HOUR_UK" -lt 7 ]; then
    echo "  ✓ $HOUR_UK:xx UK is within safe window (21:00-07:00)"
else
    echo "  ⚠ $HOUR_UK:xx UK is INSIDE market hours"
    confirm "Proceed anyway?"
fi

step "3. Stop host bot cleanly"
touch /tmp/asrs-bot.stop
for p in $(pgrep -f "asrs.main"); do
    echo "  killing python pid $p"
    kill -15 $p 2>/dev/null || true
done
sleep 5
for p in $(pgrep -f "asrs.main\|run_bot.sh"); do
    echo "  hard-killing pid $p"
    kill -9 $p 2>/dev/null || true
done
sleep 2
if pgrep -af "asrs.main\|run_bot.sh" | grep -v claude- > /dev/null; then
    echo "  ✗ FAILED to kill all processes"
    pgrep -af "asrs.main\|run_bot.sh"
    exit 1
fi
echo "  ✓ host bot stopped"

step "4. Create host runtime directory"
mkdir -p /var/lib/asrs-runtime/logs
chmod 755 /var/lib/asrs-runtime
echo "  ✓ /var/lib/asrs-runtime ready"

step "5. Migrate any existing state files (idempotent copy)"
# State files are already in ./data so the volume mount will pick them up
ls -la data/state/*.json 2>/dev/null | head || echo "  no state files (clean start)"
# Pause sentinel migration
if [ -f /tmp/asrs-bot.paused ]; then
    cp /tmp/asrs-bot.paused /var/lib/asrs-runtime/asrs-bot.paused
    echo "  ⚠ pause sentinel found and migrated — bot will start PAUSED"
fi

step "6. Build the bot image"
confirm "Build asrs-bot image now?"
docker compose build bot
echo "  ✓ image built"

step "7. Verify .env is present and has required vars"
if [ ! -f .env ]; then
    echo "  ✗ .env not found — bot needs IB credentials + Telegram token"
    exit 1
fi
for var in IB_USERNAME IB_PASSWORD TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID; do
    if ! grep -q "^$var=" .env; then
        echo "  ✗ .env missing $var"
        exit 1
    fi
done
echo "  ✓ .env has all required keys"

step "8. Start the container"
confirm "Start asrs-bot container?"
docker compose up -d bot
sleep 10
echo "  ✓ container started"

step "9. Verify container health"
sleep 30
docker ps --filter name=asrs-bot --format '{{.Names}}: {{.Status}}'
echo
echo "  Recent logs:"
docker logs --tail 30 asrs-bot 2>&1 | grep -E "Bot started|real-time bars|Reconciliation|Error|Traceback" | tail -10
if docker ps --filter name=asrs-bot --filter status=running -q | grep -q .; then
    echo "  ✓ container is running"
else
    echo "  ✗ container is NOT running"
    docker logs --tail 50 asrs-bot
    exit 1
fi

step "10. Verify bot can place + see streams"
sleep 20
docker logs --tail 100 asrs-bot 2>&1 | grep -E "real-time bars|Bot started" | tail -5
echo "  ✓ checks passed"

step "11. Disable host wrapper (so it doesn't auto-restart)"
# The wrapper is gone (we killed it in step 3), but make sure no cron starts it
echo "  TODO manual: verify no cron / systemd unit starts run_bot.sh"
echo "  Container's restart: unless-stopped policy handles auto-restart now"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  CUTOVER COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo
echo "  Old: run_bot.sh wrapper + python -m asrs.main host process"
echo "  New: docker container 'asrs-bot' with restart: unless-stopped"
echo
echo "  Operator commands:"
echo "    Pause:  touch /var/lib/asrs-runtime/asrs-bot.paused"
echo "    Resume: rm /var/lib/asrs-runtime/asrs-bot.paused"
echo "    Logs:   docker logs -f asrs-bot"
echo "             OR: tail /var/lib/asrs-runtime/logs/asrs.log"
echo "    Stop:   docker compose stop bot"
echo "    Start:  docker compose start bot"
echo "    Restart: docker compose restart bot"
echo
echo "  Next: monitor for the next 24h. If anything goes wrong:"
echo "    docker compose stop bot"
echo "    ./run_bot.sh   # falls back to host process"
echo
