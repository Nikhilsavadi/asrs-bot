#!/bin/bash
# run_bot.sh — Auto-restart wrapper for asrs.main running outside Docker.
#
# Restarts the bot if it crashes, with cooldown to avoid restart loops.
# Logs to /tmp/asrs-logs/asrs.log.
#
# Usage:
#   nohup ./run_bot.sh > /tmp/asrs-bot-stdout.log 2>&1 &
#   disown
#
# Stop:
#   touch /tmp/asrs-bot.stop  (creates stop sentinel — wrapper exits cleanly)
#   pkill -f "asrs.main"      (also kills the wrapper if it's waiting)

set -u
cd /root/asrs-bot

LOG_DIR=/tmp/asrs-logs
mkdir -p "$LOG_DIR"
mkdir -p /app/data

STOP_SENTINEL=/tmp/asrs-bot.stop
COOLDOWN=10
MIN_RUNTIME=30   # if bot dies in <30s, increase cooldown
MAX_COOLDOWN=300

cool=$COOLDOWN

while true; do
    if [ -f "$STOP_SENTINEL" ]; then
        echo "[$(date '+%F %T')] Stop sentinel found at $STOP_SENTINEL — exiting wrapper" \
            | tee -a "$LOG_DIR/wrapper.log"
        rm -f "$STOP_SENTINEL"
        exit 0
    fi

    echo "[$(date '+%F %T')] Starting bot" | tee -a "$LOG_DIR/wrapper.log"
    started=$(date +%s)

    LOG_DIR="$LOG_DIR" \
    IB_CLIENT_ID=42 \
    BROKER_TYPE=ib \
    STARTING_EQUITY_GBP="${STARTING_EQUITY_GBP:-5000}" \
    RISK_GATE_START_DATE="${RISK_GATE_START_DATE:-2026-04-09}" \
    DAILY_LOSS_LIMIT_PCT="${DAILY_LOSS_LIMIT_PCT:-3.0}" \
    WEEKLY_LOSS_LIMIT_PCT="${WEEKLY_LOSS_LIMIT_PCT:-6.0}" \
    MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-3}" \
    CONSECUTIVE_LOSS_KILL="${CONSECUTIVE_LOSS_KILL:-6}" \
    RISK_PCT_PER_TRADE="${RISK_PCT_PER_TRADE:-0.5}" \
    MAX_CONTRACTS="${MAX_CONTRACTS:-5}" \
    IB_ADAPTIVE_PRIORITY="${IB_ADAPTIVE_PRIORITY:-Urgent}" \
        python3 -m asrs.main 2>&1 | tee -a "$LOG_DIR/asrs.stderr.log"

    exitcode=$?
    runtime=$(( $(date +%s) - started ))
    echo "[$(date '+%F %T')] Bot exited code=$exitcode after ${runtime}s" \
        | tee -a "$LOG_DIR/wrapper.log"

    if [ -f "$STOP_SENTINEL" ]; then
        rm -f "$STOP_SENTINEL"
        exit 0
    fi

    # Adaptive cooldown: short runtimes mean a startup bug, back off
    if [ $runtime -lt $MIN_RUNTIME ]; then
        cool=$(( cool * 2 ))
        if [ $cool -gt $MAX_COOLDOWN ]; then cool=$MAX_COOLDOWN; fi
        echo "[$(date '+%F %T')] Short runtime, backoff cooldown=${cool}s" \
            | tee -a "$LOG_DIR/wrapper.log"
    else
        cool=$COOLDOWN
    fi

    sleep $cool
done
