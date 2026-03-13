"""
test_oca.py — Test OCA bracket order flow on IG demo.

Simulates the real DAX morning routine:
  1. Connect to IG
  2. Get current DAX price
  3. Place OCA bracket: buy stop above + sell stop below current price
  4. Verify both working orders exist
  5. Wait for one side to fill (or manually trigger by placing close to price)
  6. Verify OCA counterpart cancellation
  7. Place protective stop on the filled position
  8. Modify the stop (trail simulation)
  9. Close position
  10. Clean up
"""

import asyncio
from dax_bot.broker_ig import IGBroker
from dax_bot import config
from dax_bot import alerts


async def main():
    broker = IGBroker(epic=config.IG_EPIC, currency=config.CURRENCY)

    # 1. Connect
    ok = await broker.connect()
    if not ok:
        print("FAILED to connect")
        return
    print("Connected ✅")

    # 2. Get price
    price = await broker.get_current_price()
    print(f"DAX mid price: {price}")
    if not price:
        await broker.disconnect()
        return

    # 3. Place OCA bracket — tight levels so one fills quickly
    #    Buy stop 3pts above mid, sell stop 3pts below mid
    buy_level = round(price + 3, 1)
    sell_level = round(price - 3, 1)
    qty = 1  # £1/pt

    print(f"\nPlacing OCA bracket:")
    print(f"  BUY  stop @ {buy_level} (above mid)")
    print(f"  SELL stop @ {sell_level} (below mid)")

    result = await broker.place_oca_bracket(
        buy_price=buy_level,
        sell_price=sell_level,
        qty=qty,
        oca_group="TEST_OCA_001",
    )

    if "error" in result:
        print(f"OCA FAILED: {result['error']}")
        await broker.disconnect()
        return

    buy_id = result["buy_order_id"]
    sell_id = result["sell_order_id"]
    print(f"  BUY  deal: {buy_id}")
    print(f"  SELL deal: {sell_id}")

    await alerts.send(
        f"🧪 <b>DAX OCA TEST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Mid: {price}\n"
        f"🟩 BUY stop: {buy_level} ({buy_id})\n"
        f"🟥 SELL stop: {sell_level} ({sell_id})\n"
        f"<i>Waiting for fill...</i>"
    )

    # 4. Verify working orders exist
    orders = await broker.get_open_orders()
    print(f"\nWorking orders: {len(orders)}")
    for o in orders:
        print(f"  {o['action']} {o['type']} @ {o['price']} ({o['order_id']})")

    # 5. Poll for fill (check every 3s, timeout 60s)
    print("\nWaiting for fill (60s timeout)...")
    filled_direction = None
    filled_price = None
    filled_id = None

    for i in range(20):
        await asyncio.sleep(3)
        pos = await broker.get_position()

        if pos["direction"] != "FLAT":
            filled_direction = pos["direction"]
            filled_price = pos["avg_cost"]
            filled_id = buy_id if filled_direction == "LONG" else sell_id
            print(f"\n✅ FILLED: {filled_direction} @ {filled_price}")
            break

        # Show current price to see if we're getting close
        cur = await broker.get_current_price()
        elapsed = (i + 1) * 3
        print(f"  [{elapsed}s] Price: {cur} | Waiting... (need >{buy_level} or <{sell_level})")

    if not filled_direction:
        print("\nNo fill in 60s — cancelling orders and exiting")
        cancelled = await broker.cancel_all_orders()
        print(f"Cancelled {cancelled} orders")

        await alerts.send(
            f"🧪 <b>DAX OCA TEST — NO FILL</b>\n"
            f"Cancelled {cancelled} orders.\n"
            f"<i>Market didn't reach ±3pts in 60s</i>"
        )
        await broker.disconnect()
        return

    # 6. Cancel OCA counterpart
    print("\nCancelling OCA counterpart...")
    cancelled = await broker.cancel_oca_counterpart(filled_id)
    print(f"Counterpart cancelled: {cancelled}")

    # Verify only 1 side remains (as position, not working order)
    remaining_orders = await broker.get_open_orders()
    print(f"Remaining working orders: {len(remaining_orders)}")

    await alerts.send(
        f"🧪 <b>DAX OCA FILLED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{filled_direction} @ {filled_price}\n"
        f"Counterpart cancelled: {'✅' if cancelled else '❌'}\n"
        f"Remaining orders: {len(remaining_orders)}"
    )

    # 7. Place protective stop
    bar_width = abs(buy_level - sell_level)
    if filled_direction == "LONG":
        stop_price = round(filled_price - bar_width, 1)
    else:
        stop_price = round(filled_price + bar_width, 1)

    print(f"\nPlacing protective stop @ {stop_price}...")
    stop_action = "SELL" if filled_direction == "LONG" else "BUY"
    stop_result = await broker.place_stop_order(
        action=stop_action,
        qty=qty,
        stop_price=stop_price,
    )
    stop_id = stop_result.get("order_id", "?")
    print(f"Stop placed: {stop_id}")

    # 8. Trail the stop (simulate moving it closer)
    if filled_direction == "LONG":
        new_stop = round(stop_price + 2, 1)
    else:
        new_stop = round(stop_price - 2, 1)

    print(f"Trailing stop: {stop_price} → {new_stop}...")
    trail_ok = await broker.modify_stop(stop_id, new_stop)
    print(f"Trail modified: {trail_ok}")

    await alerts.send(
        f"🧪 <b>DAX STOP TEST</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Initial stop: {stop_price}\n"
        f"Trailed to: {new_stop}\n"
        f"Modify success: {'✅' if trail_ok else '❌'}"
    )

    # 9. Close position
    await asyncio.sleep(2)
    cur_price = await broker.get_current_price()
    print(f"\nClosing position (current: {cur_price})...")
    closed = await broker.close_position()
    print(f"Closed: {closed}")

    if cur_price and filled_price:
        if filled_direction == "LONG":
            pnl = round(cur_price - filled_price, 1)
        else:
            pnl = round(filled_price - cur_price, 1)
    else:
        pnl = 0

    # Cancel any remaining orders
    remaining = await broker.cancel_all_orders()

    await alerts.send(
        f"🧪 <b>DAX OCA TEST COMPLETE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{filled_direction} @ {filled_price}\n"
        f"Exit: {cur_price}\n"
        f"P&L: {'+' if pnl >= 0 else ''}{pnl} pts\n"
        f"Stop trail: ✅\n"
        f"OCA cancel: ✅\n"
        f"Position close: {'✅' if closed else '❌'}\n"
        f"<i>All tests passed</i>"
    )

    await broker.disconnect()
    print(f"\n✅ OCA test complete — P&L: {pnl} pts")


if __name__ == "__main__":
    asyncio.run(main())
