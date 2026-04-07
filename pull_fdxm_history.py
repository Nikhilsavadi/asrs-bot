"""
pull_fdxm_history.py — Download N years of 5-min FDXM bars from IBKR.

Walks through quarterly contracts (Mar/Jun/Sep/Dec, 3rd Friday) using
includeExpired=True so we can resolve historical contracts. For each
contract pulls its liquid period (typically last ~120 days before
expiry). Stitches into a single CSV sorted by timestamp.

Usage:
    python3 pull_fdxm_history.py [--years 2] [--out data/ibkr/FDXM_5min.csv]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone, date
from pathlib import Path

from ib_async import Future

from shared.ib_session import IBSharedSession


CHUNK_DAYS = 30
SLEEP_BETWEEN = 10


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_friday + 14)


def quarterly_expiries(years_back: int, look_forward_days: int = 90) -> list[str]:
    """Quarterly expiry strings (YYYYMMDD), oldest → newest."""
    today = date.today()
    earliest = today - timedelta(days=years_back * 365 + 60)
    latest = today + timedelta(days=look_forward_days)
    out = []
    for year in range(earliest.year - 1, latest.year + 2):
        for month in (3, 6, 9, 12):
            exp = third_friday(year, month)
            if earliest <= exp <= latest:
                out.append(exp.strftime("%Y%m%d"))
    return sorted(set(out))


async def fetch_chunk(session, contract, end_dt: datetime, days: int):
    end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")
    try:
        bars = await session.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_str,
            durationStr=f"{days} D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )
        return bars or []
    except Exception as e:
        print(f"  fetch error @ {end_str}: {e}", file=sys.stderr, flush=True)
        return []


async def pull_one_contract(session, expiry: str, max_days: int,
                             symbol="DAX", exchange="EUREX", currency="EUR",
                             trading_class="FDXM"):
    """Pull liquid history for one expired/active contract."""
    kwargs = dict(symbol=symbol, exchange=exchange, currency=currency,
                  lastTradeDateOrContractMonth=expiry)
    if trading_class:
        kwargs["tradingClass"] = trading_class
    c = Future(**kwargs)
    c.includeExpired = True

    details = await session.ib.reqContractDetailsAsync(c)
    if not details:
        return [], None

    contract = details[0].contract
    contract.includeExpired = True

    # Walk back from expiry date in chunks
    end_date = datetime.strptime(expiry, "%Y%m%d").replace(
        hour=23, minute=59, second=0, tzinfo=timezone.utc,
    )
    chunks = (max_days + CHUNK_DAYS - 1) // CHUNK_DAYS
    bars_out: dict[datetime, dict] = {}

    for i in range(chunks):
        bars = await fetch_chunk(session, contract, end_date, CHUNK_DAYS)
        if not bars:
            break
        new_count = 0
        oldest = None
        for b in bars:
            ts = b.date if isinstance(b.date, datetime) else datetime.fromisoformat(str(b.date))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts in bars_out:
                continue
            bars_out[ts] = {
                "open": float(b.open), "high": float(b.high),
                "low":  float(b.low),  "close": float(b.close),
                "volume": float(getattr(b, "volume", 0) or 0),
            }
            new_count += 1
            if oldest is None or ts < oldest:
                oldest = ts
        if new_count == 0 or oldest is None:
            break
        end_date = oldest - timedelta(seconds=1)
        await asyncio.sleep(SLEEP_BETWEEN)

    return list(bars_out.items()), contract.conId


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--out", default="data/ibkr/FDXM_5min.csv")
    ap.add_argument("--symbol", default="DAX")
    ap.add_argument("--exchange", default="EUREX")
    ap.add_argument("--currency", default="EUR")
    ap.add_argument("--trading-class", default="FDXM")
    ap.add_argument("--per-contract-days", type=int, default=120,
                    help="Max days to pull from each contract (~liquid window)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    s = IBSharedSession.get_instance()
    print(f"Connecting to IB Gateway (port {s._port}, clientId {s._client_id})...", flush=True)
    if not await s.connect():
        print("Connection failed", file=sys.stderr)
        sys.exit(1)
    print("Connected.", flush=True)

    expiries = quarterly_expiries(args.years)
    print(f"\nWalking {len(expiries)} quarterly contracts:", flush=True)
    for e in expiries:
        print(f"  {e}", flush=True)

    all_bars: dict[datetime, dict] = {}

    # Process newest → oldest (most liquid first; we may bail early if quota hit)
    for expiry in reversed(expiries):
        print(f"\n--- {args.trading_class} {expiry} ---", flush=True)
        items, cid = await pull_one_contract(
            s, expiry, args.per_contract_days,
            symbol=args.symbol, exchange=args.exchange,
            currency=args.currency, trading_class=args.trading_class,
        )
        if not items:
            print(f"  no bars (contract may not exist yet)", flush=True)
            continue
        added = 0
        for ts, row in items:
            if ts in all_bars:
                continue
            all_bars[ts] = row
            added += 1
        ts_min = min(t for t, _ in items)
        ts_max = max(t for t, _ in items)
        print(f"  conId={cid}  bars={len(items):,}  +{added} new  "
              f"range={ts_min.date()}→{ts_max.date()}  total={len(all_bars):,}",
              flush=True)
        await asyncio.sleep(SLEEP_BETWEEN)

    # Write CSV
    print(f"\nWriting {len(all_bars):,} bars to {args.out}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for ts in sorted(all_bars.keys()):
            r = all_bars[ts]
            w.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                r["open"], r["high"], r["low"], r["close"], int(r["volume"]),
            ])

    if all_bars:
        ts_min = min(all_bars.keys())
        ts_max = max(all_bars.keys())
        print(f"\nDone. Date range: {ts_min} → {ts_max}", flush=True)
        print(f"Total unique 5-min bars: {len(all_bars):,}", flush=True)
    await s.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
