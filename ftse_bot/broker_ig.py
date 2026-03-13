"""
broker_ig.py — FTSE IG Markets Broker Adapter
═══════════════════════════════════════════════════════════════════════════════

Re-exports the shared IGBroker from dax_bot.broker_ig.
Both bots use the exact same adapter class — the only difference
is the epic and currency passed at construction time.
"""

from dax_bot.broker_ig import IGBroker  # noqa: F401
