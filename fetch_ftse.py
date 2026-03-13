"""
fetch_ftse.py -- Download FTSE 100 5-min bars from IBKR
Uses IBGB100 CFD (MIDPOINT) + Z future (TRADES) for max coverage.
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from ib_async import IB, CFD, ContFuture, util
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FTSE_FETCH")

TZ_UK = ZoneInfo("Europe/London")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ftse")
os.makedirs(DATA_DIR, exist_ok=True)


async def fetch_chunks(ib, contract, what_to_show, use_rth, label):
    """Fetch data in 6-month chunks going backward."""
    frames = []
    end_dt = ""  # empty = now

    for i in range(6):  # Up to 3 years back in 6M chunks
        dur = "6 M"
        chunk_label = f"{label} chunk-{i}"
        logger.info(f"  {chunk_label}: fetching {dur} ending {end_dt or 'now'}...")

        try:
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime=end_dt, durationStr=dur,
                barSizeSetting="5 mins", whatToShow=what_to_show,
                useRTH=use_rth, formatDate=2,
            )
        except Exception as e:
            logger.warning(f"  {chunk_label}: error {e}")
            break

        if not bars:
            logger.info(f"  {chunk_label}: no data, stopping")
            break

        df = util.df(bars)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        df.index = df.index.tz_convert(TZ_UK)
        df.columns = [c.capitalize() for c in df.columns]
        frames.append(df)
        logger.info(f"  {chunk_label}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

        # Next chunk ends where this one started
        end_dt = df.index[0].strftime("%Y%m%d-%H:%M:%S")
        await asyncio.sleep(5)

    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        return combined
    return None


async def fetch():
    ib = IB()
    await ib.connectAsync("172.18.0.2", 4004, clientId=99)
    logger.info("Connected to IBKR")

    # Source 1: IBGB100 CFD with MIDPOINT (up to ~6 months)
    cfd = CFD(symbol="IBGB100", exchange="SMART", currency="GBP")
    q1 = await ib.qualifyContractsAsync(cfd)
    if not q1:
        logger.error("Could not qualify IBGB100 CFD")
        ib.disconnect()
        return
    cfd_contract = q1[0]
    logger.info(f"CFD contract: {cfd_contract}")

    # Source 2: Z continuous future with TRADES
    cf = ContFuture(symbol="Z", exchange="ICEEU", currency="GBP")
    q2 = await ib.qualifyContractsAsync(cf)
    fut_contract = q2[0] if q2 else None
    if fut_contract:
        logger.info(f"Future contract: {fut_contract}")
    else:
        logger.warning("Could not qualify Z future, will use CFD only")

    for use_rth, label, filename in [
        (True, "RTH", "ftse_rth.parquet"),
        (False, "All hours", "ftse_all.parquet"),
    ]:
        all_frames = []

        # Fetch CFD data (MIDPOINT)
        logger.info(f"Fetching CFD {label}...")
        cfd_data = await fetch_chunks(ib, cfd_contract, "MIDPOINT", use_rth, f"CFD-{label}")
        if cfd_data is not None:
            all_frames.append(cfd_data)
            logger.info(f"CFD {label}: {len(cfd_data)} bars")
        await asyncio.sleep(5)

        # Fetch Future data (TRADES) — may extend further back
        if fut_contract:
            logger.info(f"Fetching Future {label}...")
            fut_data = await fetch_chunks(ib, fut_contract, "TRADES", use_rth, f"FUT-{label}")
            if fut_data is not None:
                all_frames.append(fut_data)
                logger.info(f"Future {label}: {len(fut_data)} bars")
            await asyncio.sleep(5)

        if all_frames:
            combined = pd.concat(all_frames).sort_index()
            # Prefer CFD data where overlapping (more accurate for CFD trading)
            combined = combined[~combined.index.duplicated(keep="last")]
            path = os.path.join(DATA_DIR, filename)
            combined.to_parquet(path)
            dates = pd.Series(combined.index.date).unique()
            logger.info(f"Saved {label}: {len(combined)} bars, {len(dates)} days "
                        f"({dates[0]} to {dates[-1]}) -> {path}")
        else:
            logger.error(f"No data for {label}")

    ib.disconnect()
    logger.info("Done")


if __name__ == "__main__":
    asyncio.run(fetch())
