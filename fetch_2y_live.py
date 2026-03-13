"""
fetch_2y_live.py — Fetch 2Y DAX data using 1-month chunks from the current FDAX contract.
IBKR provides up to 2Y of historical data for futures via HMDS.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from ib_async import IB, Future, util
import pandas as pd
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("FETCH_2Y")

TZ_CET = ZoneInfo("Europe/Berlin")

# Quarterly FDAX expiries going back 2+ years (3rd Friday of Mar/Jun/Sep/Dec)
EXPIRIES = [
    "20260320",  # Mar 2026 (current)
    "20251219",  # Dec 2025
    "20250919",  # Sep 2025
    "20250620",  # Jun 2025
    "20250321",  # Mar 2025
    "20241220",  # Dec 2024
    "20240920",  # Sep 2024
    "20240621",  # Jun 2024
    "20240315",  # Mar 2024
]


async def fetch_1m_chunk(ib, contract, end_dt, use_rth, label=""):
    """Fetch 1 month of 5-min bars ending at end_dt."""
    end_str = end_dt.strftime("%Y%m%d-%H:%M:%S") if end_dt else ""
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime=end_str, durationStr="1 M",
            barSizeSetting="5 mins", whatToShow="TRADES",
            useRTH=use_rth, formatDate=2
        )
        if bars:
            df = util.df(bars)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
            df.index = df.index.tz_convert(TZ_CET)
            df.columns = [c.capitalize() for c in df.columns]
            logger.info(f"  {label}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
            return df
        else:
            logger.info(f"  {label}: no data")
            return None
    except Exception as e:
        logger.warning(f"  {label}: error {e}")
        return None


async def fetch_contract_chunks(ib, expiry, use_rth=True):
    """Fetch all data for one contract in 1-month chunks."""
    contract = Future(symbol="DAX", lastTradeDateOrContractMonth=expiry,
                      exchange="EUREX", currency="EUR", tradingClass="FDAX",
                      includeExpired=True)
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            logger.warning(f"  {expiry}: cannot qualify")
            return None
        contract = qualified[0]
    except Exception as e:
        logger.warning(f"  {expiry}: qualify error: {e}")
        return None

    rth_label = "RTH" if use_rth else "All"
    frames = []

    # Fetch in 1-month chunks, stepping backward from contract expiry (or now)
    exp_date = datetime.strptime(expiry, "%Y%m%d")
    now = datetime.now()
    end = min(exp_date, now)

    # Go back up to 6 months per contract (contracts overlap ~3 months)
    for i in range(6):
        chunk_end = end - timedelta(days=30 * i)
        if chunk_end < exp_date - timedelta(days=200):
            break

        label = f"{expiry} {rth_label} chunk-{i}"
        df = await fetch_1m_chunk(ib, contract, chunk_end, use_rth, label)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            # No more data available for this contract going back
            break
        await asyncio.sleep(3)

    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        return combined
    return None


async def main():
    ib = IB()
    logger.info("Connecting to live IBKR...")
    await ib.connectAsync("172.18.0.1", 4003, clientId=95, timeout=60)
    logger.info("Connected")

    all_rth = []
    all_full = []

    for expiry in EXPIRIES:
        logger.info(f"Fetching {expiry}...")

        # RTH data
        rth = await fetch_contract_chunks(ib, expiry, use_rth=True)
        if rth is not None and not rth.empty:
            all_rth.append(rth)
            logger.info(f"  {expiry} RTH subtotal: {len(rth)} bars")

        # All-hours data
        full = await fetch_contract_chunks(ib, expiry, use_rth=False)
        if full is not None and not full.empty:
            all_full.append(full)
            logger.info(f"  {expiry} All subtotal: {len(full)} bars")

    ib.disconnect()
    logger.info("Disconnected")

    # Combine and deduplicate
    if all_rth:
        rth = pd.concat(all_rth).sort_index()
        rth = rth[~rth.index.duplicated(keep="first")]
        dates = pd.Series(rth.index.date).unique()
        logger.info(f"RTH total: {len(rth)} bars, {len(dates)} trading days, {dates[0]} to {dates[-1]}")
        rth.to_parquet("data/historical_bars.parquet")
        logger.info("Saved RTH data")
    else:
        logger.error("No RTH data fetched")

    if all_full:
        full = pd.concat(all_full).sort_index()
        full = full[~full.index.duplicated(keep="first")]
        logger.info(f"All-hours total: {len(full)} bars")
        full.to_parquet("data/historical_bars_all.parquet")
        logger.info("Saved all-hours data")
    else:
        logger.warning("No all-hours data fetched")


if __name__ == "__main__":
    asyncio.run(main())
