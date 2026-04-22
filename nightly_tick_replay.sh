#!/bin/bash
# Nightly tick-level replay: runs the actual Signal class through recorded
# tick CSVs for today's date. Compares to live journal trade-by-trade.
# Any mismatch = execution bug OR missing tick data.
cd /root/asrs-bot || exit 1
DATE=$(date -u +%Y-%m-%d)
/usr/bin/python3 /root/asrs-bot/replay_ticks.py --date "$DATE"
