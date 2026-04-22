#!/bin/bash
cd /root/asrs-bot || exit 1
exec /usr/bin/python3 /root/asrs-bot/daily_ig_reconcile.py
