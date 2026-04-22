"""
Correlation risk: does simultaneous same-direction exposure across instruments
cluster big losses? Test: what happens if 2+/3 instruments signal same direction on a day?
"""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def main():
    df = pd.read_parquet(TRADES_CACHE)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    from backtest_v2_atr import filter_combined
    df = df.reset_index(drop=True)
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        fades = group[group.is_fade]
        for i, (idx, _) in enumerate(fades.iterrows()):
            if i < len(q): orig_win[idx] = q[i]
    fade = df[df.is_fade].copy()
    fade["orig_win"] = fade.index.map(orig_win)
    fade_f = fade[fade["orig_win"].fillna(0) >= 20].copy()
    base = df[~df.is_fade].copy()
    base_f = filter_combined(base)
    c = pd.concat([base_f, fade_f], ignore_index=True)
    c["spread_cost"] = c["instrument"].map(SPREAD)
    c["slip_cost"] = np.where(c["pnl_pts"] < 0, c["instrument"].map(SLIP), 0)
    c["pnl_net"] = c["pnl_pts"] - c["spread_cost"] - c["slip_cost"]
    c["date_dt"] = pd.to_datetime(c["date"]).dt.date
    c["year"] = pd.to_datetime(c["date"]).dt.year
    c["era"] = c["year"].apply(lambda y: "TRAIN" if y <= 2017 else "TEST")

    # For each day, determine dominant direction per instrument (count LONG vs SHORT trades)
    day_dir = c.groupby(["date_dt", "instrument", "direction"]).size().unstack(fill_value=0)
    # Simplify: per (date, instrument), get primary direction
    def primary_dir(row):
        return "LONG" if row.get("LONG", 0) > row.get("SHORT", 0) else ("SHORT" if row.get("SHORT", 0) > 0 else "")
    day_dir["primary"] = day_dir.apply(primary_dir, axis=1)
    primary = day_dir[["primary"]].reset_index().pivot(index="date_dt", columns="instrument", values="primary").fillna("")

    # Count instruments with LONG vs SHORT per day
    def count_same_dir(row):
        dirs = [v for v in row if v]
        if len(dirs) < 2:
            return "solo"
        long_ct = dirs.count("LONG")
        short_ct = dirs.count("SHORT")
        if long_ct >= 2:
            return f"{long_ct}×LONG"
        if short_ct >= 2:
            return f"{short_ct}×SHORT"
        return "mixed"
    primary["regime"] = primary[["DAX", "US30", "NIKKEI"]].apply(count_same_dir, axis=1)
    print("Day regime distribution:")
    print(primary["regime"].value_counts())

    # Merge back and segment pnl
    c2 = c.merge(primary[["regime"]], left_on="date_dt", right_index=True, how="left")
    print(f"\n{'='*80}\n  PnL BY CORRELATION REGIME (what to do on 'all aligned' days)\n{'='*80}")
    print(f"\n  {'regime':<15} {'n':<7} {'PF':<7} {'net':<12} {'per-trade':<10} {'worst':<10}")
    for regime, d in c2.groupby("regime"):
        worst_day = d.groupby(d.date_dt)["pnl_net"].sum().min()
        print(f"  {str(regime):<15} {len(d):<7,} {pf(d['pnl_net']):<5.2f}   "
              f"{d['pnl_net'].sum():>+8,.0f}   {d['pnl_net'].mean():>+6.1f}pt   "
              f"{worst_day:>+6.0f}")

    # What if we halved stake on "all-aligned" days?
    print(f"\n{'='*80}\n  SCENARIO: halve stake on 2+ same-direction days\n{'='*80}")
    cut_regimes = ["2×LONG", "2×SHORT", "3×LONG", "3×SHORT"]
    c2["pnl_adjusted"] = np.where(c2["regime"].isin(cut_regimes),
                                    c2["pnl_net"] * 0.5, c2["pnl_net"])
    print(f"  Baseline: n={len(c2):,} PF={pf(c2['pnl_net']):.2f} net={c2['pnl_net'].sum():+,.0f}")
    print(f"  Halved:   n={len(c2):,} PF={pf(c2['pnl_adjusted']):.2f} net={c2['pnl_adjusted'].sum():+,.0f}")
    # Worst-day impact
    daily_base = c2.groupby("date_dt")["pnl_net"].sum()
    daily_half = c2.groupby("date_dt")["pnl_adjusted"].sum()
    print(f"  Worst day (baseline): {daily_base.min():+.0f}pt")
    print(f"  Worst day (halved):   {daily_half.min():+.0f}pt")


if __name__ == "__main__":
    main()
