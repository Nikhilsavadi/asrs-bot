"""
fetch_ftse_ig.py — Download 3 years of FTSE 100 5-min bars from IG Markets
===========================================================================

IG allows historical price data via REST API. We fetch in 1-month chunks
to stay within API limits, then combine into a single parquet file.

Usage:
    python fetch_ftse_ig.py                  # Fetch 3 years
    python fetch_ftse_ig.py --years 1        # Fetch 1 year
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from trading_ig import IGService
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("FTSE_IG")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ftse")
os.makedirs(DATA_DIR, exist_ok=True)

TZ_UK = ZoneInfo("Europe/London")

EPIC = os.getenv("FTSE_IG_EPIC", "IX.D.FTSE.DAILY.IP")


def connect_ig(force_live: bool = False) -> IGService:
    username = os.getenv("IG_USERNAME", "")
    password = os.getenv("IG_PASSWORD", "")
    api_key = os.getenv("IG_API_KEY", "")
    if force_live:
        acc_type = "LIVE"
        logger.info("Using LIVE IG credentials for data fetch")
    else:
        acc_type = "DEMO" if os.getenv("IG_DEMO", "true").lower() == "true" else "LIVE"

    logger.info(f"Connecting to IG ({acc_type})...")
    # Disable return_dataframe to avoid pandas freq bug in trading_ig
    ig = IGService(username, password, api_key, acc_type, return_dataframe=False)
    ig.create_session()
    logger.info("Connected")
    return ig


def parse_prices(raw_prices: list[dict]) -> pd.DataFrame:
    """Parse raw IG price data into a clean DataFrame (bypassing broken trading_ig pandas code)."""
    rows = []
    for p in raw_prices:
        ts = p.get("snapshotTime") or p.get("snapshotTimeUTC")
        bid = p.get("closePrice", {})
        # Use bid mid or bid prices
        o = p.get("openPrice", {}).get("bid")
        h = p.get("highPrice", {}).get("bid")
        l = p.get("lowPrice", {}).get("bid")
        c = p.get("closePrice", {}).get("bid")
        v = p.get("lastTradedVolume", 0)
        if o is not None and h is not None:
            rows.append({"date": ts, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v or 0})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Parse timestamps — IG v2 format: "2025-01-15 08:00:00"
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")
    df.index = df.index.tz_convert(TZ_UK)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_chunk(ig: IGService, epic: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Fetch one chunk of historical data."""
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    try:
        result = ig.fetch_historical_prices_by_epic_and_date_range(
            epic=epic,
            resolution="MINUTE_5",
            start_date=start_str,
            end_date=end_str,
            version="2",
        )
    except Exception as e:
        err = str(e)
        if "exceeded" in err.lower() or "allowance" in err.lower():
            logger.warning(f"Rate limit hit, waiting 5 minutes...")
            time.sleep(300)
            try:
                result = ig.fetch_historical_prices_by_epic_and_date_range(
                    epic=epic, resolution="MINUTE_5",
                    start_date=start_str, end_date=end_str, version="2",
                )
            except Exception as e2:
                logger.error(f"  Retry failed: {e2}")
                return None
        else:
            logger.error(f"  Error: {e}")
            return None

    if result is None or "prices" not in result:
        logger.warning(f"  No data for {start.date()} to {end.date()}")
        return None

    prices = result["prices"]
    if not prices:
        return None

    # Parse raw JSON price list into DataFrame
    df = parse_prices(prices)
    if df.empty:
        return None

    return df


def main():
    years = 1  # IG DFB only keeps ~10-11 months of 5-min data
    if "--years" in sys.argv:
        idx = sys.argv.index("--years")
        if idx + 1 < len(sys.argv):
            years = int(sys.argv[idx + 1])

    force_live = "--live" in sys.argv
    ig = connect_ig(force_live=force_live)

    # Calculate date range — start from ~11 months ago (IG 5-min limit)
    end_date = datetime.now(TZ_UK)
    start_date = end_date - timedelta(days=years * 365)

    logger.info(f"Fetching FTSE data: {start_date.date()} to {end_date.date()}")
    logger.info(f"Epic: {EPIC}")

    # Use 1-week chunks with generous rate limiting
    # IG allows ~10k data points per week on historical API
    chunk_days = 7
    frames = []
    partial_path = os.path.join(DATA_DIR, "ftse_fetch_progress.parquet")

    # Load any existing progress
    if os.path.exists(partial_path):
        existing = pd.read_parquet(partial_path)
        frames.append(existing)
        last_date = existing.index[-1]
        start_date = last_date.to_pydatetime() + timedelta(minutes=5)
        logger.info(f"Resuming from {start_date.date()} ({len(existing)} bars already fetched)")

    current_start = start_date
    total_chunks = int((end_date - start_date).days / chunk_days) + 1
    chunk_num = 0
    consecutive_empty = 0
    consecutive_errors = 0

    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
        chunk_num += 1

        logger.info(f"Chunk {chunk_num}/{total_chunks}: {current_start.date()} to {chunk_end.date()}")

        df = fetch_chunk(ig, EPIC, current_start, chunk_end)
        if df is not None and not df.empty:
            frames.append(df)
            logger.info(f"  Got {len(df)} bars")
            consecutive_empty = 0
            consecutive_errors = 0

            # Save progress after every successful chunk
            progress = pd.concat(frames).sort_index()
            progress = progress[~progress.index.duplicated(keep="first")]
            progress.to_parquet(partial_path)
            total_bars = len(progress)
            total_days = len(set(progress.index.date))
            logger.info(f"  Progress: {total_bars} bars, {total_days} days total")
        else:
            consecutive_empty += 1
            logger.info(f"  No data ({consecutive_empty} empty in a row)")

        current_start = chunk_end

        # Rate limiting: wait 10s between chunks to stay well within limits
        # IG's historical data allowance resets gradually
        time.sleep(10)

        # If we hit rate limit (403), wait much longer
        if consecutive_errors >= 2:
            logger.info("Multiple errors, waiting 5 minutes for rate limit reset...")
            time.sleep(300)
            consecutive_errors = 0

    if not frames:
        logger.error("No data fetched!")
        return

    # Combine and deduplicate
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.dropna(how="all")

    days = len(set(combined.index.date))
    logger.info(f"Total: {len(combined)} bars, {days} trading days")
    logger.info(f"Range: {combined.index[0]} to {combined.index[-1]}")

    # Save all-hours data
    all_path = os.path.join(DATA_DIR, "ftse_all.parquet")
    combined.to_parquet(all_path)
    logger.info(f"Saved all-hours to {all_path}")

    # Also save RTH only (08:00 to 16:30 UK)
    rth = combined[(combined.index.hour >= 8) &
                   ((combined.index.hour < 16) |
                    ((combined.index.hour == 16) & (combined.index.minute <= 25)))]
    rth_path = os.path.join(DATA_DIR, "ftse_rth.parquet")
    rth.to_parquet(rth_path)
    rth_days = len(set(rth.index.date))
    logger.info(f"Saved RTH ({len(rth)} bars, {rth_days} days) to {rth_path}")

    # Clean up progress file
    if os.path.exists(partial_path):
        os.remove(partial_path)
        logger.info("Cleaned up progress file")


if __name__ == "__main__":
    main()
