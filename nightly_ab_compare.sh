#!/bin/bash
# Wrapper for nightly_ab_compare.py (env loaded by python-dotenv).
cd /root/asrs-bot || exit 1
exec /usr/bin/python3 /root/asrs-bot/nightly_ab_compare.py
