#!/bin/bash
# run_bot_live.sh — LIVE wrapper for asrs.main on REAL money account.
#
# Differs from run_bot.sh (paper) in:
#   - Real risk gate limits enabled (3% daily / 6% weekly)
#   - Lower starting equity (£5k initial)
#   - NIKKEI disabled (NKD margin requires £30k+, see scaling plan)
#   - IB_PORT=4001 (live gateway) instead of 4002 (paper)
#   - IB_CLIENT_ID=43 to avoid clashing with paper bot if both run
#
# Usage:
#   nohup ./run_bot_live.sh > /tmp/asrs-live-stdout.log 2>&1 &
#   disown
#
# Stop:
#   touch /tmp/asrs-live.stop
#   pkill -f "asrs.main"
#
# IMPORTANT: do NOT run paper and live wrappers in parallel against
# the same IB Gateway. They will fight for clientId. Run one or the
# other, OR use separate Gateway containers.

set -u
cd /root/asrs-bot

LOG_DIR=/tmp/asrs-live-logs
mkdir -p "$LOG_DIR"
mkdir -p /app/data

STOP_SENTINEL=/tmp/asrs-live.stop
COOLDOWN=10
MIN_RUNTIME=30
MAX_COOLDOWN=300

cool=$COOLDOWN

while true; do
    if [ -f "$STOP_SENTINEL" ]; then
        echo "[$(date '+%F %T')] Stop sentinel found — exiting wrapper" \
            | tee -a "$LOG_DIR/wrapper.log"
        rm -f "$STOP_SENTINEL"
        exit 0
    fi

    echo "[$(date '+%F %T')] Starting LIVE bot" | tee -a "$LOG_DIR/wrapper.log"
    started=$(date +%s)

    # LIVE MODE — real money. Conservative risk gate engaged.
    LOG_DIR="$LOG_DIR" \
    IB_CLIENT_ID=43 \
    IB_PORT=4001 \
    BROKER_TYPE=ib \
    STARTING_EQUITY_GBP="${STARTING_EQUITY_GBP:-5000}" \
    RISK_GATE_START_DATE="${RISK_GATE_START_DATE:-2026-04-23}" \
    DAILY_LOSS_LIMIT_PCT="${DAILY_LOSS_LIMIT_PCT:-3.0}" \
    WEEKLY_LOSS_LIMIT_PCT="${WEEKLY_LOSS_LIMIT_PCT:-6.0}" \
    MAX_CONCURRENT_POSITIONS="${MAX_CONCURRENT_POSITIONS:-3}" \
    CONSECUTIVE_LOSS_KILL="${CONSECUTIVE_LOSS_KILL:-6}" \
    RISK_PCT_PER_TRADE="${RISK_PCT_PER_TRADE:-0.5}" \
    MAX_CONTRACTS="${MAX_CONTRACTS:-5}" \
    IB_ADAPTIVE_PRIORITY="${IB_ADAPTIVE_PRIORITY:-Urgent}" \
    DISABLE_INSTRUMENTS="${DISABLE_INSTRUMENTS:-NIKKEI}" \
        python3 -m asrs.main 2>&1 | tee -a "$LOG_DIR/asrs.stderr.log"

    exitcode=$?
    runtime=$(( $(date +%s) - started ))
    echo "[$(date '+%F %T')] LIVE bot exited code=$exitcode after ${runtime}s" \
        | tee -a "$LOG_DIR/wrapper.log"

    if [ -f "$STOP_SENTINEL" ]; then
        rm -f "$STOP_SENTINEL"
        exit 0
    fi

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
