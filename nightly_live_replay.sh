#!/bin/bash
# Wrapper for nightly_live_replay.py.
cd /root/asrs-bot || exit 1
exec /usr/bin/python3 /root/asrs-bot/nightly_live_replay.py
