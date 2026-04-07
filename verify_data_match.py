"""
verify_data_match.py — Compare live IBKR historical bars against firstrate
backtest data for the same recent dates. Confirms the validation transfers.

For each instrument:
  1. Fetch last 7 days of 5-min bars from IBKR (live, current contract)
  2. Load firstrate backtest data for same date range
  3. Compare bar-by-bar: count, range, alignment, mean abs error
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from ib_async import Future

from shared.ib_session import IBSharedSession


# Compare against the firstrate ratio-adjusted continuous series
INSTRUMENTS = {
    "DAX": {
        "ib_symbol": "DAX", "ib_exchange": "EUREX", "ib_currency": "EUR",
        "ib_trading_class": "FDXM",   # Mini DAX €5/pt — matches firstrate FDAX ratio-adj scale roughly
        "fr_file": "data/firstrate/FDAX_full_5min_continuous_ratio_adjusted.txt",
        "fr_tz": "Europe/Berlin",
    },
    "US30": {
        "ib_symbol": "MYM", "ib_exchange": "CBOT", "ib_currency": "USD",
        "ib_trading_class": "",
        "fr_file": "data/firstrate/YM_full_5min_continuous_ratio_adjusted.txt",
        "fr_tz": "America/New_York",
    },
    "NIKKEI": {
        "ib_symbol": "NKD", "ib_exchange": "CME", "ib_currency": "USD",
        "ib_trading_class": "",
        "fr_file": "data/firstrate/NKD_full_5min_continuous_ratio_adjusted.txt",
        "fr_tz": "America/New_York",
    },
}


def load_firstrate(filepath, src_tz):
    df = pd.read_csv(filepath, header=None,
                     names=["dt", "Open", "High", "Low", "Close", "Volume"])
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt")
    df.index = df.index.tz_localize(ZoneInfo(src_tz)).tz_convert("UTC")
    return df[["Open", "High", "Low", "Close"]]


async def fetch_ibkr_bars(session, spec, days=5):
    """Fetch last `days` of 5-min bars for the front-month contract."""
    kwargs = dict(symbol=spec["ib_symbol"], exchange=spec["ib_exchange"],
                  currency=spec["ib_currency"])
    if spec["ib_trading_class"]:
        kwargs["tradingClass"] = spec["ib_trading_class"]

    # First find the front-month
    from asrs.contract_resolver import resolve_front_month
    # Hack: use the resolver but with explicit symbol/exchange via SPECS override
    # Easier path: just query and pick first quarterly future
    from ib_async import Contract
    search = Contract(secType="FUT", **kwargs)
    details = await session.ib.reqContractDetailsAsync(search)
    if not details:
        return None, None
    # Pick nearest quarterly
    today = datetime.now().date()
    cands = []
    for d in details:
        try:
            exp = datetime.strptime(d.contract.lastTradeDateOrContractMonth, "%Y%m%d").date()
        except Exception:
            continue
        if exp > today and exp.month in (3, 6, 9, 12):
            cands.append((exp, d.contract))
    if not cands:
        return None, None
    cands.sort()
    contract = cands[0][1]

    # Fetch bars
    bars = await session.ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr=f"{days} D",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
    )
    if not bars:
        return contract, pd.DataFrame()
    df = pd.DataFrame([{
        "Open": float(b.open), "High": float(b.high),
        "Low": float(b.low), "Close": float(b.close),
    } for b in bars])
    idx = pd.DatetimeIndex([
        b.date if isinstance(b.date, datetime)
        else pd.to_datetime(b.date, utc=True)
        for b in bars
    ])
    df.index = idx
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return contract, df


def compare_bars(ibkr_df, fr_df, label):
    """Align by timestamp and compute price-level differences."""
    if ibkr_df is None or ibkr_df.empty:
        print(f"  {label}: IBKR has no bars")
        return
    if fr_df is None or fr_df.empty:
        print(f"  {label}: firstrate has no bars")
        return

    # Find overlapping dates
    ib_start = ibkr_df.index.min()
    ib_end = ibkr_df.index.max()
    fr_start = fr_df.index.min()
    fr_end = fr_df.index.max()
    overlap_start = max(ib_start, fr_start)
    overlap_end = min(ib_end, fr_end)

    if overlap_start >= overlap_end:
        print(f"  {label}: NO DATE OVERLAP")
        print(f"    IBKR:      {ib_start} → {ib_end}")
        print(f"    firstrate: {fr_start} → {fr_end}")
        return

    ib_sub = ibkr_df[(ibkr_df.index >= overlap_start) & (ibkr_df.index <= overlap_end)]
    fr_sub = fr_df[(fr_df.index >= overlap_start) & (fr_df.index <= overlap_end)]

    # Inner join on index
    merged = ib_sub.join(fr_sub, how="inner", lsuffix="_ib", rsuffix="_fr")
    if merged.empty:
        print(f"  {label}: NO TIMESTAMP MATCHES")
        print(f"    IBKR sample: {list(ib_sub.index[:3])}")
        print(f"    firstrate sample: {list(fr_sub.index[:3])}")
        return

    # Sample 5 random rows for visual comparison
    print(f"  {label}: overlap {overlap_start.date()} → {overlap_end.date()}")
    print(f"    IBKR bars in overlap: {len(ib_sub)}")
    print(f"    firstrate bars in overlap: {len(fr_sub)}")
    print(f"    matched timestamps: {len(merged)}")

    # Diffs
    for col in ["Open", "High", "Low", "Close"]:
        diff = (merged[f"{col}_ib"] - merged[f"{col}_fr"]).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        ib_level = merged[f"{col}_ib"].mean()
        rel = (mean_diff / ib_level) * 100 if ib_level else 0
        print(f"    {col}: mean abs diff {mean_diff:.2f} pts ({rel:.3f}%), max {max_diff:.2f}")

    # Sample comparison
    print(f"\n    Sample (first 3 matched bars):")
    for ts in merged.index[:3]:
        ib = merged.loc[ts]
        print(f"      {ts}  IB: O={ib['Open_ib']:.1f} H={ib['High_ib']:.1f} L={ib['Low_ib']:.1f} C={ib['Close_ib']:.1f}")
        print(f"      {ts}  FR: O={ib['Open_fr']:.1f} H={ib['High_fr']:.1f} L={ib['Low_fr']:.1f} C={ib['Close_fr']:.1f}")
        print()


async def main():
    s = IBSharedSession.get_instance()
    await s.connect()
    print("=" * 70)
    print("  DATA MATCH: IBKR live vs firstrate backtest (last 5 trading days)")
    print("=" * 70)

    for inst_name, spec in INSTRUMENTS.items():
        print(f"\n--- {inst_name} ---")
        contract, ib_df = await fetch_ibkr_bars(s, spec, days=5)
        if contract is None:
            print(f"  Failed to qualify {inst_name}")
            continue
        print(f"  Contract: {contract.localSymbol or contract.symbol} expiry={contract.lastTradeDateOrContractMonth}")
        try:
            fr_df = load_firstrate(spec["fr_file"], spec["fr_tz"])
        except Exception as e:
            print(f"  firstrate load failed: {e}")
            continue
        compare_bars(ib_df, fr_df, inst_name)

    await s.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
