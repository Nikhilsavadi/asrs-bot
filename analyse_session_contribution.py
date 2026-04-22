"""
Per-session contribution: does each session (S1/S2/S3 per instrument) earn edge,
or do some drag the total? Segment + train/test.
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

    # Apply filters (COMBINED + fade ≥20pt)
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
    c["year"] = pd.to_datetime(c["date"]).dt.year
    c["era"] = c["year"].apply(lambda y: "TRAIN" if y <= 2017 else "TEST")
    c["session"] = c["_base_signal"].str.extract(r"_S(\d+)")

    print(f"{'='*85}\n  PER-SESSION CONTRIBUTION (post-friction, 18yr)\n{'='*85}")
    print(f"\n  {'signal':<15} {'n':<7} {'PF':<7} {'net':<12} {'TR PF':<7} {'TE PF':<7} {'per-trade':<10}")
    for sig in sorted(c["_base_signal"].unique()):
        d = c[c._base_signal == sig]
        tr = d[d.era == "TRAIN"]
        te = d[d.era == "TEST"]
        pt = d["pnl_net"].mean()
        print(f"  {sig:<15} {len(d):<7,} {pf(d['pnl_net']):<5.2f}   "
              f"{d['pnl_net'].sum():>+8,.0f}   {pf(tr['pnl_net']):<5.2f}   "
              f"{pf(te['pnl_net']):<5.2f}   {pt:>+6.1f}")

    # Per-instrument: which session dominates?
    print(f"\n{'='*85}\n  PER-INSTRUMENT SESSION BREAKDOWN\n{'='*85}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        d = c[c.instrument == inst]
        print(f"\n  {inst}: {len(d):,} trades, net={d['pnl_net'].sum():+,.0f}")
        for sig in sorted(d._base_signal.unique()):
            ds = d[d._base_signal == sig]
            pct = ds["pnl_net"].sum() / d["pnl_net"].sum() * 100 if d["pnl_net"].sum() else 0
            tr = ds[ds.era == "TRAIN"]; te = ds[ds.era == "TEST"]
            print(f"    {sig:<15} n={len(ds):>5,} net={ds['pnl_net'].sum():>+7,.0f} ({pct:>5.1f}% of {inst}) "
                  f"TR PF={pf(tr['pnl_net']):.2f} TE PF={pf(te['pnl_net']):.2f}")


if __name__ == "__main__":
    main()
