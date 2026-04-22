"""
Verify concentration risk: apples-to-apples outlier stress across base/fade/combined
POST-SPREAD. Uses trade cache so runs in seconds.
"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3


def pf(pnl):
    if len(pnl) == 0: return 0
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def apply_spread(df):
    df = df.copy()
    df["pnl_pts_net"] = df["pnl_pts"] - df["instrument"].map(SPREAD)
    return df


def outlier(d, col, label):
    print(f"\n  {label}: n={len(d):,} PF={pf(d[col]):.2f} net={d[col].sum():+,.0f}")
    for p_pct in [0.5, 1, 2, 5, 10]:
        k = int(len(d) * p_pct / 100)
        ds = d.sort_values(col, ascending=False)
        d_top = ds.iloc[k:]
        d_both = ds.iloc[k:-k] if k > 0 else ds
        print(f"    -{p_pct:>4.1f}% top (k={k:>5}):  PF={pf(d_top[col]):>5.2f}  net={d_top[col].sum():>+9,.0f}"
              f"   |  -{p_pct:>4.1f}% both: PF={pf(d_both[col]):>5.2f}  net={d_both[col].sum():>+9,.0f}")


def main():
    df = pd.read_parquet(TRADES_CACHE)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False)
    base = df[~df["is_fade"]].copy()
    fade = df[df["is_fade"]].copy()
    base_f = filter_combined(base)
    combined = pd.concat([base_f, fade], ignore_index=True)

    base_f = apply_spread(base_f)
    fade = apply_spread(fade)
    combined = apply_spread(combined)

    print("Outlier concentration — POST-SPREAD net pnl")
    outlier(base_f, "pnl_pts_net", "BASE alone")
    outlier(fade, "pnl_pts_net", "FADE alone")
    outlier(combined, "pnl_pts_net", "COMBINED")

    print(f"\n\n  Top-trade composition (combined, post-spread):")
    ds = combined.sort_values("pnl_pts_net", ascending=False).reset_index(drop=True)
    for top_pct in [1, 5, 10, 25]:
        k = int(len(ds) * top_pct / 100)
        top = ds.head(k)
        fade_pct = (top["is_fade"] == True).mean() * 100
        print(f"    top {top_pct:>3}% (k={k:>5}): {fade_pct:.1f}% are FADE trades, "
              f"avg size {top['pnl_pts_net'].mean():+.1f}pt")


if __name__ == "__main__":
    main()
