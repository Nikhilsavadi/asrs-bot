"""
test_trade.py — Place sample trades on IG demo to verify end-to-end flow.

Tests:
  1. Connect to IG
  2. Send Telegram alert
  3. Place a small market order (£1/pt FTSE + £1/pt DAX)
  4. Check position shows up
  5. Send position alert
  6. Close position
  7. Send close confirmation
  8. Test command handler responses
"""

import asyncio
import sys

async def test_ftse():
    from ftse_bot.broker_ig import IGBroker
    from ftse_bot import config
    from ftse_bot import telegram_alerts as alerts

    print("\n=== FTSE TEST ===")
    broker = IGBroker(epic=config.IG_EPIC, currency=config.CURRENCY)

    # 1. Connect
    ok = await broker.connect()
    if not ok:
        print("FTSE: FAILED to connect")
        return False
    print("FTSE: Connected ✅")

    # 2. Get price
    price = await broker.get_current_price()
    print(f"FTSE: Price = {price}")

    # 3. Place market BUY order (£1/pt)
    print("FTSE: Placing BUY market order (£1/pt)...")
    result = await broker.place_market_order(action="BUY", qty=1)
    if "error" in result:
        print(f"FTSE: Order FAILED: {result['error']}")
        await broker.disconnect()
        return False

    deal_id = result.get("order_id", "?")
    fill = result.get("avg_price", "?")
    print(f"FTSE: FILLED @ {fill} (deal: {deal_id})")

    # 4. Send Telegram alert
    await alerts.send(
        f"🧪 <b>FTSE TEST TRADE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"BUY £1/pt @ {fill}\n"
        f"Deal: {deal_id}\n"
        f"<i>Test trade — will close in 5s</i>"
    )

    # 5. Check position
    await asyncio.sleep(2)
    pos = await broker.get_position()
    print(f"FTSE: Position = {pos}")

    # 6. Get current price for unrealised P&L
    cur_price = await broker.get_current_price()
    if cur_price and fill:
        ur = round(cur_price - float(fill), 1)
        print(f"FTSE: Unrealised = {ur} pts")

    # 7. Close position
    await asyncio.sleep(3)
    print("FTSE: Closing position...")
    closed = await broker.close_position()
    print(f"FTSE: Closed = {closed}")

    close_price = await broker.get_current_price()
    pnl = round(close_price - float(fill), 1) if close_price and fill else 0

    await alerts.send(
        f"🧪 <b>FTSE TEST CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: {fill}\n"
        f"Exit:  {close_price}\n"
        f"P&L:   {'+' if pnl >= 0 else ''}{pnl} pts\n"
        f"<i>Test complete ✅</i>"
    )

    await broker.disconnect()
    print("FTSE: Test complete ✅")
    return True


async def test_dax():
    from dax_bot.broker_ig import IGBroker
    from dax_bot import config
    from dax_bot import alerts

    print("\n=== DAX TEST ===")
    broker = IGBroker(epic=config.IG_EPIC, currency=config.CURRENCY)

    # 1. Connect
    ok = await broker.connect()
    if not ok:
        print("DAX: FAILED to connect")
        return False
    print("DAX: Connected ✅")

    # 2. Get price
    price = await broker.get_current_price()
    print(f"DAX: Price = {price}")

    # 3. Place market SELL order (€1/pt)
    print("DAX: Placing SELL market order (€1/pt)...")
    result = await broker.place_market_order(action="SELL", qty=1)
    if "error" in result:
        print(f"DAX: Order FAILED: {result['error']}")
        await broker.disconnect()
        return False

    deal_id = result.get("order_id", "?")
    fill = result.get("avg_price", "?")
    print(f"DAX: FILLED @ {fill} (deal: {deal_id})")

    # 4. Send Telegram alert
    await alerts.send(
        f"🧪 <b>DAX TEST TRADE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"SELL €1/pt @ {fill}\n"
        f"Deal: {deal_id}\n"
        f"<i>Test trade — will close in 5s</i>"
    )

    # 5. Check position
    await asyncio.sleep(2)
    pos = await broker.get_position()
    print(f"DAX: Position = {pos}")

    # 6. Close position
    await asyncio.sleep(3)
    print("DAX: Closing position...")
    closed = await broker.close_position()
    print(f"DAX: Closed = {closed}")

    close_price = await broker.get_current_price()
    pnl = round(float(fill) - close_price, 1) if close_price and fill else 0

    await alerts.send(
        f"🧪 <b>DAX TEST CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Entry: {fill}\n"
        f"Exit:  {close_price}\n"
        f"P&L:   {'+' if pnl >= 0 else ''}{pnl} pts\n"
        f"<i>Test complete ✅</i>"
    )

    await broker.disconnect()
    print("DAX: Test complete ✅")
    return True


async def test_command_handler():
    """Test /status command handler directly."""
    import telegram_cmd

    print("\n=== COMMAND HANDLER TEST ===")

    # Create temporary broker instances for status check
    from ftse_bot.broker_ig import IGBroker as FTSEBroker
    from ftse_bot import config as fc
    from dax_bot.broker_ig import IGBroker as DAXBroker
    from dax_bot import config as dc

    ftse_b = FTSEBroker(epic=fc.IG_EPIC, currency=fc.CURRENCY)
    dax_b = DAXBroker(epic=dc.IG_EPIC, currency=dc.CURRENCY)

    await ftse_b.connect()
    await dax_b.connect()

    # Test /status
    print("Testing /status...")
    await telegram_cmd.handle_status(dax_b, ftse_b)
    print("Status sent ✅")

    # Test /positions
    print("Testing /positions...")
    await telegram_cmd.handle_positions(dax_b, ftse_b)
    print("Positions sent ✅")

    # Test /pnl
    print("Testing /pnl...")
    await telegram_cmd.handle_pnl()
    print("PnL sent ✅")

    await ftse_b.disconnect()
    await dax_b.disconnect()
    print("Command handler test complete ✅")


async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "ftse"):
        await test_ftse()

    if target in ("all", "dax"):
        # Small delay between to avoid IG rate limiting
        if target == "all":
            await asyncio.sleep(2)
        await test_dax()

    if target in ("all", "cmd"):
        if target == "all":
            await asyncio.sleep(2)
        await test_command_handler()

    print("\n✅ All tests complete — check Telegram for alerts!")


if __name__ == "__main__":
    asyncio.run(main())
