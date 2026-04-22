#!/bin/bash
# Runs the proper Variant B + fade + milestone backtest against live trades
# for today's date. Posts per-instrument delta to Telegram. Uses local IG
# tick CSVs (no historical-allowance hit).
cd /root/asrs-bot || exit 1
exec /usr/bin/python3 /root/asrs-bot/nightly_backtest_vs_live.py
