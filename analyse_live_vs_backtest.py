"""
analyse_live_vs_backtest.py — compare live trades vs Variant B backtest on same dates.

Pulls live trades from journal, filters backtest trades to same date range,
compares aggregate stats per instrument and overall.

Also breaks down by exit_reason, entry slippage, and signal to identify
systematic gaps between live execution and backtest assumption.
"""
import os, sys, numpy as np, pandas as pd, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB = "/root/asrs-bot/data/trade_journal.db"
BACKTEST_PARQUET = "/root/asrs-bot/data/variant_b_trades.parquet"


def pf(pnl):
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def fmt_stats(pnl, label):
    if len(pnl) == 0:
        return f"  {label:<30} n=0"
    w = (pnl > 0).sum()
    return (f"  {label:<30} n={len(pnl):>4} "
            f"PF={pf(pnl):>5.2f} "
            f"net={pnl.sum():>+7.1f} "
            f"WR={w/len(pnl)*100:>5.1f}% "
            f"avgW={pnl[pnl>0].mean() if w else 0:>+6.1f} "
            f"avgL={pnl[pnl<0].mean() if (pnl<0).any() else 0:>+6.1f} "
            f"per-trade={pnl.mean():+.1f}")


def main():
    # Load live trades
    conn = sqlite3.connect(DB)
    live = pd.read_sql(
        "SELECT * FROM trades WHERE mode='live'",
        conn
    )
    conn.close()
    live["date"] = pd.to_datetime(live["date"])
    live_dates = sorted(live["date"].dt.date.unique())
    print(f"Live trades: {len(live):,} across {len(live_dates)} days "
          f"({live_dates[0]} → {live_dates[-1]})")

    # Load backtest trades (Variant B, cached)
    if not os.path.exists(BACKTEST_PARQUET):
        print(f"ERROR: {BACKTEST_PARQUET} missing. Run analyse_nikkei_streaks.py first to cache.")
        return
    bt = pd.read_parquet(BACKTEST_PARQUET)
    bt["date_dt"] = pd.to_datetime(bt["date"])
    # Filter to same date range
    start, end = min(live_dates), max(live_dates)
    bt_win = bt[(bt["date_dt"].dt.date >= start) & (bt["date_dt"].dt.date <= end)]
    # FirstRate data may not include weekends, map only to live-traded dates
    bt_win = bt_win[bt_win["date_dt"].dt.date.isin(live_dates)]

    print(f"\nBacktest (Variant B) trades on same live dates: {len(bt_win):,}")

    # --- AGGREGATE: live vs backtest ---
    print(f"\n{'='*100}")
    print(f"  OVERALL (all 3 instruments, same date range)")
    print(f"{'='*100}")
    print(fmt_stats(live["pnl_pts"].values, "LIVE    (all trades)"))
    print(fmt_stats(bt_win["pnl_pts"].values, "BACKTEST (Variant B, same dates)"))

    # --- PER INSTRUMENT ---
    print(f"\n{'='*100}")
    print(f"  PER INSTRUMENT")
    print(f"{'='*100}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        l = live[live["instrument"] == inst]
        b = bt_win[bt_win["instrument"] == inst]
        print(f"\n  {inst}:")
        print(fmt_stats(l["pnl_pts"].values, "LIVE"))
        print(fmt_stats(b["pnl_pts"].values, "BACKTEST"))
        delta = (l["pnl_pts"].mean() if len(l) else 0) - (b["pnl_pts"].mean() if len(b) else 0)
        print(f"  PER-TRADE GAP: {delta:+.1f}pt live minus backtest")

    # --- EXIT REASON MIX (live only) ---
    print(f"\n{'='*100}")
    print(f"  LIVE EXIT REASON DISTRIBUTION")
    print(f"{'='*100}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        l = live[live["instrument"] == inst]
        if len(l) == 0: continue
        mix = l.groupby("exit_reason").agg(
            n=("pnl_pts", "size"),
            mean=("pnl_pts", "mean"),
            net=("pnl_pts", "sum"),
        ).sort_values("n", ascending=False)
        print(f"\n  {inst}:")
        for r, row in mix.iterrows():
            print(f"    {str(r):<15} n={row['n']:>3} mean={row['mean']:+6.1f} net={row['net']:+7.1f}")

    # --- BACKTEST EXIT REASON MIX (same window) ---
    print(f"\n{'='*100}")
    print(f"  BACKTEST EXIT REASON DISTRIBUTION (same window)")
    print(f"{'='*100}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        b = bt_win[bt_win["instrument"] == inst]
        if len(b) == 0: continue
        mix = b.groupby("reason").agg(
            n=("pnl_pts", "size"),
            mean=("pnl_pts", "mean"),
            net=("pnl_pts", "sum"),
        ).sort_values("n", ascending=False)
        print(f"\n  {inst}:")
        for r, row in mix.iterrows():
            print(f"    {str(r):<15} n={row['n']:>3} mean={row['mean']:+6.1f} net={row['net']:+7.1f}")

    # --- ENTRY / EXIT SLIPPAGE (live) ---
    print(f"\n{'='*100}")
    print(f"  LIVE ENTRY/EXIT SLIPPAGE")
    print(f"{'='*100}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        l = live[live["instrument"] == inst]
        if len(l) == 0: continue
        print(f"\n  {inst}:")
        print(f"    entry_slippage  mean={l['entry_slippage'].mean():+.2f}pt  "
              f"max={l['entry_slippage'].max():+.2f}pt  "
              f"p95={l['entry_slippage'].quantile(0.95):+.2f}pt")
        print(f"    exit_slippage   mean={l['exit_slippage'].mean():+.2f}pt  "
              f"max={l['exit_slippage'].max():+.2f}pt")
        print(f"    entry_spread    mean={l['entry_spread'].mean():+.2f}pt  "
              f"max={l['entry_spread'].max():+.2f}pt")

    # --- TRADE COUNT MISMATCH ---
    print(f"\n{'='*100}")
    print(f"  DAILY TRADE COUNT (LIVE vs BACKTEST)")
    print(f"{'='*100}")
    print(f"  {'date':<12} {'inst':<8} {'live n':<8} {'bt n':<8} {'live net':<10} {'bt net':<10} {'delta':<10}")
    for d in live_dates:
        for inst in ["DAX", "US30", "NIKKEI"]:
            l = live[(live["date"].dt.date == d) & (live["instrument"] == inst)]
            b = bt_win[(bt_win["date_dt"].dt.date == d) & (bt_win["instrument"] == inst)]
            if len(l) == 0 and len(b) == 0: continue
            l_net = l["pnl_pts"].sum(); b_net = b["pnl_pts"].sum()
            flag = " *" if abs(len(l) - len(b)) > 1 or abs(l_net - b_net) > 50 else ""
            print(f"  {d}   {inst:<8} {len(l):<8} {len(b):<8} "
                  f"{l_net:>+7.1f}   {b_net:>+7.1f}   {l_net - b_net:>+7.1f}{flag}")


if __name__ == "__main__":
    main()
