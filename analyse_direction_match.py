"""
analyse_direction_match.py — do live and backtest take the same DIRECTION?

For each (instrument, date, session) with both a live trade and a backtest
trade, compare first_direction. Structural divergence = different direction
= path-dependent first-cross chose opposite level.
"""
import os, sys, sqlite3, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB = "/root/asrs-bot/data/trade_journal.db"
BT = "/root/asrs-bot/data/variant_b_trades.parquet"


def infer_session(entry_time_str, inst):
    h = int(entry_time_str.split(":")[0]); m = int(entry_time_str.split(":")[1])
    if inst == "US30":
        if h*60+m < 11*60: return "S1"
        if h*60+m < 13*60: return "S2"
        return "S3"
    if inst == "NIKKEI":
        if h*60+m < 12*60: return "S1"
        if h*60+m < 13*60: return "S2"
        return "S3"
    if inst == "DAX":
        if h*60+m < 14*60: return "S1"
        return "S2"


def main():
    conn = sqlite3.connect(DB)
    live = pd.read_sql("SELECT instrument,date,direction,entry_time,pnl_pts "
                       "FROM trades WHERE mode='live' ORDER BY date, instrument, entry_time", conn)
    conn.close()
    live["session"] = live.apply(lambda r: infer_session(r["entry_time"], r["instrument"]), axis=1)
    # First trade of each (inst, date, session)
    live_first = live.groupby(["instrument","date","session"]).first().reset_index()
    live_first = live_first.rename(columns={"direction":"live_dir", "pnl_pts":"live_first_pnl"})
    # Also total for session
    live_totals = live.groupby(["instrument","date","session"]).agg(
        live_n=("direction","size"),
        live_net=("pnl_pts","sum"),
    ).reset_index()
    live_summary = live_first[["instrument","date","session","live_dir","live_first_pnl"]].merge(
        live_totals, on=["instrument","date","session"])

    # Backtest trades (Variant B)
    bt = pd.read_parquet(BT)
    bt["date"] = bt["date"].astype(str)
    # signal column is like "DAX_S1"
    bt["session"] = bt["signal"].str.split("_").str[-1]
    bt_first = bt.groupby(["instrument","date","session"]).first().reset_index()
    bt_first = bt_first.rename(columns={"direction":"bt_dir", "pnl_pts":"bt_first_pnl"})
    bt_totals = bt.groupby(["instrument","date","session"]).agg(
        bt_n=("direction","size"),
        bt_net=("pnl_pts","sum"),
    ).reset_index()
    bt_summary = bt_first[["instrument","date","session","bt_dir","bt_first_pnl"]].merge(
        bt_totals, on=["instrument","date","session"])

    # Join on same session
    joined = live_summary.merge(bt_summary, on=["instrument","date","session"], how="inner")
    if len(joined) == 0:
        print("No overlapping sessions between live and backtest.")
        return

    joined["match"] = joined["live_dir"] == joined["bt_dir"]
    print(f"Overlapping sessions (live date in backtest range): {len(joined)}")
    print(f"Direction match: {joined['match'].sum()}/{len(joined)} "
          f"({joined['match'].mean()*100:.1f}%)\n")

    print(f"{'date':<12} {'inst':<7} {'sess':<4} "
          f"{'live_dir':<9} {'bt_dir':<9} {'match':<6} "
          f"{'live_n':<6} {'bt_n':<6} {'live_net':<10} {'bt_net':<10} {'gap':<10}")
    print("-" * 100)
    for _, r in joined.sort_values(["date","instrument","session"]).iterrows():
        flag = "✓" if r["match"] else "✗"
        gap = r["live_net"] - r["bt_net"]
        print(f"  {r['date']:<10} {r['instrument']:<7} {r['session']:<4} "
              f"{r['live_dir']:<9} {r['bt_dir']:<9} {flag:<6} "
              f"{r['live_n']:<6} {r['bt_n']:<6} "
              f"{r['live_net']:>+7.1f}   {r['bt_net']:>+7.1f}   {gap:>+7.1f}")

    print(f"\n{'='*100}\n  AGGREGATE\n{'='*100}")
    matched = joined[joined["match"]]
    mismatched = joined[~joined["match"]]
    print(f"\n  Matched direction ({len(matched)} sessions):")
    print(f"    live_net total: {matched['live_net'].sum():+.1f}")
    print(f"    bt_net total:   {matched['bt_net'].sum():+.1f}")
    print(f"    gap:            {(matched['live_net']-matched['bt_net']).sum():+.1f}")
    print(f"\n  Mismatched direction ({len(mismatched)} sessions):")
    print(f"    live_net total: {mismatched['live_net'].sum():+.1f}")
    print(f"    bt_net total:   {mismatched['bt_net'].sum():+.1f}")
    print(f"    gap:            {(mismatched['live_net']-mismatched['bt_net']).sum():+.1f}")


if __name__ == "__main__":
    main()
