#!/bin/bash
# Authoritative IG-vs-journal reconcile. Runs AFTER market close (21:30 UTC /
# 22:30 BST) so the fresh IG login can't disrupt the live bot. Rolling 12-day
# window covers the post-2026-05-21-fix audit period and ongoing drift checks.
cd /root/asrs-bot || exit 1
exec /usr/bin/python3 /root/asrs-bot/ig_journal_reconcile.py --days 12
