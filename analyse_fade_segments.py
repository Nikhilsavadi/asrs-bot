"""
Find sub-filters that improve fade PF. TRAIN 2008-17 / TEST 2018-26 to
avoid curve-fitting. Goal: identify segments with PF ≥ 2.0 that hold OOS.
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


def show(df, col, label, min_n=200):
    print(f"\n  {label} (segmented by {col}):")
    print(f"    {'bucket':<25} {'n':<7} {'PF':<7} {'net':<10} {'TR PF':<7} {'TE PF':<7} {'Δ vs 1.54':<10}")
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    for bucket, d in df.groupby(col, observed=True):
        if len(d) < min_n: continue
        p = pf(d["pnl_pts_net"])
        net = d["pnl_pts_net"].sum()
        tr = d[d.year <= 2017]; te = d[d.year >= 2018]
        tr_pf = pf(tr["pnl_pts_net"]) if len(tr) > 10 else 0
        te_pf = pf(te["pnl_pts_net"]) if len(te) > 10 else 0
        lift = p - 1.54
        star = " ⭐" if p >= 2.0 and te_pf >= 1.7 else ("  ✗" if p < 1.3 else "")
        print(f"    {str(bucket):<25} {len(d):<7,} {p:<5.2f}   {net:>+7,.0f}   "
              f"{tr_pf:<5.2f}   {te_pf:<5.2f}   {lift:>+5.2f}{star}")


def main():
    df = pd.read_parquet(TRADES_CACHE)
    df["is_fade"] = df.get("is_fade", False)
    if df["is_fade"].dtype != bool:
        df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    fade = df[df["is_fade"]].copy()
    fade["pnl_pts_net"] = fade["pnl_pts"] - fade["instrument"].map(SPREAD)
    fade["date_dt"] = pd.to_datetime(fade["date"])

    print(f"FADE trades: {len(fade):,}  PF={pf(fade['pnl_pts_net']):.2f}  "
          f"net={fade['pnl_pts_net'].sum():+,.0f}  "
          f"WR={(fade['pnl_pts_net']>0).mean()*100:.1f}%")
    print("\n  Goal: find sub-groups with PF ≥ 2.0 that hold on TEST (2018-26)")
    print("  ⭐ = strong + OOS-valid  |  ✗ = clear avoid")

    show(fade, "instrument", "A) By instrument")
    show(fade, "range_flag", "B) By signal bar range flag")
    show(fade, "bar_num", "C) By signal bar number (4 vs 5)")
    show(fade, "direction", "D) By fade direction (LONG fade = was SHORT winner)")

    fade["range_bucket"] = pd.cut(fade["bar_range"].astype(float),
                                   bins=[0, 15, 30, 50, 100, 1000],
                                   labels=["<15", "15-30", "30-50", "50-100", ">100"])
    show(fade, "range_bucket", "E) By signal bar range (pts)")

    fade["dow"] = fade["date_dt"].dt.day_name()
    show(fade, "dow", "F) By day of week")

    fade["month"] = fade["date_dt"].dt.month_name()
    show(fade, "month", "G) By month")

    fade["inst_flag"] = fade["instrument"] + "/" + fade["range_flag"].astype(str)
    show(fade, "inst_flag", "H) Instrument × range_flag", min_n=500)

    fade["inst_bar"] = fade["instrument"] + "/bar" + fade["bar_num"].astype(str)
    show(fade, "inst_bar", "I) Instrument × bar number", min_n=200)

    fade["inst_dir"] = fade["instrument"] + "/" + fade["direction"]
    show(fade, "inst_dir", "J) Instrument × fade direction", min_n=500)

    fade["year"] = fade["date_dt"].dt.year
    fade["decade"] = (fade["year"] // 5) * 5
    show(fade, "decade", "K) By 5-year regime")


if __name__ == "__main__":
    main()
