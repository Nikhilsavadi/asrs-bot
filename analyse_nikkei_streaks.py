"""
analyse_nikkei_streaks.py — how common are 9-day drawdowns like current live?

Current live (Apr 6-17): 41 trades, WR 29%, PF 0.48, -554pt.
Question: does the 18yr backtest have similar or worse 9-day periods?
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import build_cache, run, filter_combined

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def main():
    pkl = "/root/asrs-bot/data/variant_b_trades.parquet"
    if os.path.exists(pkl):
        print(f"Loading cached trades from {pkl}", flush=True)
        trades = pd.read_parquet(pkl)
    else:
        print("Loading Variant B trades (cache miss)...", flush=True)
        cache = build_cache()
        trades = filter_combined(pd.DataFrame(run(cache, trail_close=True, mfe_mode=False)))
        trades["date_dt"] = pd.to_datetime(trades["date"])
        trades.to_parquet(pkl)
        print(f"Cached → {pkl}")
    if "date_dt" not in trades.columns:
        trades["date_dt"] = pd.to_datetime(trades["date"])

    # NIKKEI-only
    nik = trades[trades["instrument"] == "NIKKEI"].copy()
    print(f"NIKKEI backtest total: {len(nik):,} trades")

    # Daily aggregation
    daily = nik.groupby(nik["date_dt"].dt.date).agg(
        n=("pnl_pts", "size"),
        net=("pnl_pts", "sum"),
        wins=("pnl_pts", lambda x: (x > 0).sum()),
        wpts=("pnl_pts", lambda x: x[x > 0].sum()),
        lpts=("pnl_pts", lambda x: x[x < 0].sum()),
    ).reset_index()
    daily["wr"] = daily["wins"] / daily["n"].replace(0, 1)

    # Current live reference
    LIVE_DAYS = 9; LIVE_NET = -554.6; LIVE_PF = 0.48; LIVE_WR = 0.293
    print(f"\nLive reference: {LIVE_DAYS}d, net {LIVE_NET:+.0f}pt, PF {LIVE_PF:.2f}, WR {LIVE_WR*100:.1f}%")

    # Rolling 9-day windows over NIKKEI backtest
    n = len(daily)
    rolling_stats = []
    for i in range(n - LIVE_DAYS + 1):
        w = daily.iloc[i:i + LIVE_DAYS]
        wpts = w["wpts"].sum(); lpts = w["lpts"].sum()
        pf = wpts / abs(lpts) if lpts < 0 else float("inf")
        rolling_stats.append({
            "start": w.iloc[0]["date_dt"],
            "end": w.iloc[-1]["date_dt"],
            "n_trades": int(w["n"].sum()),
            "net": float(w["net"].sum()),
            "pf": pf,
            "wr": float(w["wins"].sum() / max(w["n"].sum(), 1)),
        })
    roll = pd.DataFrame(rolling_stats)

    # How many 9-day windows were as bad or worse than current live?
    worse_net = roll[roll["net"] <= LIVE_NET]
    worse_pf = roll[roll["pf"] <= LIVE_PF]
    worse_both = roll[(roll["net"] <= LIVE_NET) & (roll["pf"] <= LIVE_PF)]

    print(f"\nRolling 9-day windows analysed: {len(roll):,}")
    print(f"  Windows with net ≤ {LIVE_NET:.0f}pt: {len(worse_net):,} ({len(worse_net)/len(roll)*100:.1f}%)")
    print(f"  Windows with PF ≤ {LIVE_PF:.2f}:    {len(worse_pf):,} ({len(worse_pf)/len(roll)*100:.1f}%)")
    print(f"  Windows meeting BOTH criteria:  {len(worse_both):,} ({len(worse_both)/len(roll)*100:.1f}%)")

    print(f"\n10 worst 9-day windows by net pts (all NIKKEI backtest history):")
    print(f"  {'start':<12} {'end':<12} {'trades':<7} {'net':<10} {'PF':<6} {'WR':<6}")
    for _, w in roll.nsmallest(10, "net").iterrows():
        print(f"  {str(w['start']):<12} {str(w['end']):<12} "
              f"{w['n_trades']:<7} {w['net']:>+7.0f}   {w['pf']:<5.2f}  {w['wr']*100:>4.1f}%")

    # Also: worst calendar-month NIKKEI periods
    monthly = nik.groupby(nik["date_dt"].dt.to_period("M")).agg(
        n=("pnl_pts", "size"),
        net=("pnl_pts", "sum"),
        wpts=("pnl_pts", lambda x: x[x > 0].sum()),
        lpts=("pnl_pts", lambda x: x[x < 0].sum()),
        wins=("pnl_pts", lambda x: (x > 0).sum()),
    ).reset_index()
    monthly["pf"] = monthly["wpts"] / monthly["lpts"].abs().replace(0, 0.001)
    monthly["wr"] = monthly["wins"] / monthly["n"]
    worst_months = monthly[monthly["net"] < 0].sort_values("net")
    print(f"\nAll losing NIKKEI months in backtest ({len(worst_months)} of {len(monthly)}):")
    print(f"  {'month':<10} {'trades':<7} {'net':<10} {'PF':<6} {'WR':<6}")
    for _, m in worst_months.head(15).iterrows():
        print(f"  {str(m['date_dt']):<10} {m['n']:<7} {m['net']:>+7.0f}   "
              f"{m['pf']:<5.2f}  {m['wr']*100:>4.1f}%")

    print(f"\n  Losing months: {len(worst_months)}/{len(monthly)} ({len(worst_months)/len(monthly)*100:.1f}%)")


if __name__ == "__main__":
    main()
