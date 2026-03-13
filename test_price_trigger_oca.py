"""
test_price_trigger_oca.py -- End-to-end test of the price-triggered OCA bracket flow.

Steps:
  1. Connect to IG demo (DAX epic IX.D.DAX.DAILY.IP)
  2. Get current mid price
  3. Place price-triggered bracket via broker.place_oca_bracket() +/- 3 pts
  4. Poll broker.check_trigger_levels() every 3s for 60s
  5. If triggered: confirm fill, place protective stop, then close everything
  6. If not triggered in 60s: cancel and exit
"""

import asyncio
import sys
import os
import logging

# Ensure .env is loaded from the project root
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from dax_bot.broker_ig import IGBroker
from dax_bot import config

# Set up logging so we see broker-level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_price_trigger_oca")


async def main():
    broker = IGBroker(epic="IX.D.DAX.DAILY.IP", currency=config.CURRENCY)

    # ── Step 1: Connect ──────────────────────────────────────────────────
    print("\n[STEP 1] Connecting to IG demo...")
    ok = await broker.connect()
    if not ok:
        print("FAILED to connect to IG. Check credentials in .env")
        sys.exit(1)
    print("[STEP 1] Connected to IG demo.\n")

    try:
        # ── Step 2: Get current mid price ────────────────────────────────
        print("[STEP 2] Fetching current DAX mid price...")
        mid = await broker.get_current_price()
        if mid is None:
            print("FAILED to fetch price.")
            return
        print(f"[STEP 2] Current DAX mid price: {mid}\n")

        # ── Step 3: Place OCA bracket +/- 3 pts ─────────────────────────
        buy_level = round(mid + 3, 1)
        sell_level = round(mid - 3, 1)
        qty = 1  # 1 GBP/pt

        print("[STEP 3] Placing price-triggered OCA bracket:")
        print(f"         BUY  trigger @ {buy_level}  (mid + 3)")
        print(f"         SELL trigger @ {sell_level}  (mid - 3)")

        bracket = await broker.place_oca_bracket(
            buy_price=buy_level,
            sell_price=sell_level,
            qty=qty,
            oca_group="PRICE_TRIGGER_TEST",
        )
        print(f"[STEP 3] Bracket placed: {bracket}\n")

        # ── Step 4: Poll check_trigger_levels() every 3s for 60s ────────
        print("[STEP 4] Polling check_trigger_levels() every 3s (60s timeout)...")
        trigger_result = None

        for i in range(20):  # 20 x 3s = 60s
            await asyncio.sleep(3)
            elapsed = (i + 1) * 3

            result = await broker.check_trigger_levels()
            current = await broker.get_current_price()

            if result is not None:
                trigger_result = result
                print(f"         [{elapsed:>2}s] TRIGGERED! {result}")
                break
            else:
                print(
                    f"         [{elapsed:>2}s] price={current}  |  "
                    f"need >={buy_level} or <={sell_level}  |  no trigger"
                )

        if trigger_result is None:
            # ── Step 6 (no trigger): Cancel and exit ─────────────────────
            print("\n[STEP 6] No trigger in 60s. Cancelling pending bracket and exiting.")
            broker._pending_bracket = None
            broker._orders.clear()
            print("[STEP 6] Bracket cancelled. Done.\n")
            return

        # ── Step 5: Triggered -- confirm fill, place stop, close ────────
        direction = trigger_result["direction"]
        fill_price = trigger_result["fill_price"]
        order_id = trigger_result["order_id"]

        print(f"\n[STEP 5a] Fill confirmed: {direction} @ {fill_price}  (deal={order_id})")

        # Cancel OCA counterpart
        await broker.cancel_oca_counterpart(order_id)
        print("[STEP 5a] OCA counterpart deactivated.")

        # Place protective stop 6 pts from fill (= bracket width)
        if direction == "LONG":
            stop_price = round(fill_price - 6, 1)
            stop_action = "SELL"
        else:
            stop_price = round(fill_price + 6, 1)
            stop_action = "BUY"

        print(f"\n[STEP 5b] Placing protective stop: {stop_action} @ {stop_price}")
        stop_result = await broker.place_stop_order(
            action=stop_action,
            qty=qty,
            stop_price=stop_price,
        )
        stop_id = stop_result.get("order_id", "unknown")
        if "error" in stop_result:
            print(f"[STEP 5b] Stop order failed: {stop_result['error']}")
        else:
            print(f"[STEP 5b] Stop placed: {stop_id}")

        # Close everything
        print("\n[STEP 5c] Closing position...")
        await asyncio.sleep(1)
        closed = await broker.close_position()
        print(f"[STEP 5c] Position closed: {closed}")

        # Cancel any remaining working orders
        cancelled = await broker.cancel_all_orders()
        print(f"[STEP 5c] Cancelled {cancelled} remaining working order(s).")

        # Final P&L estimate
        exit_price = await broker.get_current_price()
        if exit_price and fill_price:
            if direction == "LONG":
                pnl = round(exit_price - fill_price, 1)
            else:
                pnl = round(fill_price - exit_price, 1)
            print(f"\n[RESULT] Approx P&L: {'+' if pnl >= 0 else ''}{pnl} pts")
        print("[RESULT] Test complete.\n")

    finally:
        await broker.disconnect()
        print("[CLEANUP] Disconnected from IG.\n")


if __name__ == "__main__":
    asyncio.run(main())
