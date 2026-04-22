"""
Re-segment fade trades by ORIGINAL WINNER pnl (not bar 4 range).
Hypothesis: bigger base winners → more extended price → bigger fade edge.
"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"


def pf(pnl):
    if len(pnl) == 0: return 0
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def main():
    df = pd.read_parquet(TRADES_CACHE)
    df["is_fade"] = df.get("is_fade", False)
    if df["is_fade"].dtype != bool:
        df["is_fade"] = df["is_fade"].fillna(False).astype(bool)

    # Within each (date, signal), order is sim-generated — base trades first, then their fades
    # Try to pair each fade with its preceding TRAIL_WIN base trade
    # Use stable row order
    df = df.reset_index(drop=True)

    # Fade signals are tagged "DAX_S1_FADE" etc — strip suffix for matching
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)

    # Within each (date, base_signal) group, the sim appends fades AFTER base trades
    # Multiple TRAIL_WINs → multiple fades, in FIFO order.
    orig_win_pts = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        trail_win_queue = []  # list of (pnl, idx) in order
        for idx, row in group.iterrows():
            if not row["is_fade"] and row["reason"] == "TRAIL_WIN" and row["pnl_pts"] > 0:
                trail_win_queue.append(row["pnl_pts"])
        # Now match fades in order — first fade → first trail_win, etc.
        fade_count = 0
        for idx, row in group.iterrows():
            if row["is_fade"]:
                if fade_count < len(trail_win_queue):
                    orig_win_pts[idx] = trail_win_queue[fade_count]
                    fade_count += 1

    fade = df[df["is_fade"]].copy()
    fade["orig_win"] = fade.index.map(orig_win_pts)
    unmatched = fade["orig_win"].isna().sum()
    if unmatched:
        print(f"  warning: {unmatched}/{len(fade)} fade trades unmatched to base winner — dropping")
    fade = fade.dropna(subset=["orig_win"]).copy()
    fade["pnl_pts_net"] = fade["pnl_pts"] - fade["instrument"].map(SPREAD)
    fade["date_dt"] = pd.to_datetime(fade["date"])
    fade["year"] = fade["date_dt"].dt.year

    print(f"FADE with orig_win matched: {len(fade):,}  "
          f"PF={pf(fade['pnl_pts_net']):.2f}  net={fade['pnl_pts_net'].sum():+,.0f}")
    print(f"\n  orig_win distribution: min={fade['orig_win'].min():.0f}  "
          f"p25={fade['orig_win'].quantile(0.25):.0f}  "
          f"p50={fade['orig_win'].quantile(0.5):.0f}  "
          f"p75={fade['orig_win'].quantile(0.75):.0f}  "
          f"p95={fade['orig_win'].quantile(0.95):.0f}  "
          f"max={fade['orig_win'].max():.0f}")

    # Bucket by orig_win size
    for inst in ["ALL", "DAX", "US30", "NIKKEI"]:
        d = fade if inst == "ALL" else fade[fade.instrument == inst]
        if len(d) == 0: continue
        d = d.copy()
        bins = [0, 10, 20, 30, 50, 100, 10000]
        labels = ["<10", "10-20", "20-30", "30-50", "50-100", ">100"]
        d["win_bucket"] = pd.cut(d["orig_win"], bins=bins, labels=labels, include_lowest=True)
        print(f"\n  By orig_win ({inst}):")
        print(f"    {'bucket':<10} {'n':<7} {'PF':<7} {'net':<10} {'TR PF':<7} {'TE PF':<7} {'Δ':<6}")
        for b, dd in d.groupby("win_bucket", observed=True):
            if len(dd) < 50: continue
            p = pf(dd["pnl_pts_net"])
            net = dd["pnl_pts_net"].sum()
            tr = dd[dd.year <= 2017]; te = dd[dd.year >= 2018]
            tr_pf = pf(tr["pnl_pts_net"]) if len(tr) > 10 else 0
            te_pf = pf(te["pnl_pts_net"]) if len(te) > 10 else 0
            star = " ⭐" if p >= 2.0 and te_pf >= 1.7 else ("  ✗" if p < 1.3 else "")
            print(f"    {str(b):<10} {len(dd):<7,} {p:<5.2f}   {net:>+7,.0f}   "
                  f"{tr_pf:<5.2f}   {te_pf:<5.2f}   {p-1.54:>+.2f}{star}")


if __name__ == "__main__":
    main()
