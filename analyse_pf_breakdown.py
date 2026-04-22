"""
Anatomy of profits/losses: who contributes, worst losses, concentration.
Segmented by train/test, base/fade.
"""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades.parquet"


def pf(pnl):
    w = pnl[pnl > 0].sum()
    l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def main():
    df = pd.read_parquet(TRADES_CACHE)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["era"] = df["year"].apply(lambda y: "TRAIN" if y <= 2017 else "TEST")

    from backtest_v2_atr import filter_combined
    base = df[~df.is_fade].copy()
    base_f = filter_combined(base)
    base_f["layer"] = "BASE"

    df_idx = df.reset_index(drop=True).copy()
    df_idx["_base_signal"] = df_idx["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df_idx.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        fades = group[group.is_fade]
        for i, (idx, _) in enumerate(fades.iterrows()):
            if i < len(q): orig_win[idx] = q[i]
    fade = df_idx[df_idx.is_fade].copy()
    fade["orig_win"] = fade.index.map(orig_win)
    fade_f = fade[fade["orig_win"].fillna(0) >= 20].copy()
    fade_f["layer"] = "FADE"

    c = pd.concat([base_f, fade_f], ignore_index=True)
    c["year"] = pd.to_datetime(c["date"]).dt.year
    c["era"] = c["year"].apply(lambda y: "TRAIN" if y <= 2017 else "TEST")
    c["spread_cost"] = c["instrument"].map(SPREAD)
    c["slip_cost"] = np.where(c["pnl_pts"] < 0, c["instrument"].map(SLIP), 0)
    c["pnl_net"] = c["pnl_pts"] - c["spread_cost"] - c["slip_cost"]

    print(f"{'='*90}\n  PNL DISTRIBUTION (post-friction)\n{'='*90}")
    for era in ["TRAIN", "TEST"]:
        for inst in ["DAX", "US30", "NIKKEI"]:
            d = c[(c.era == era) & (c.instrument == inst)]["pnl_net"]
            if len(d) == 0: continue
            w = d[d > 0]; l = d[d < 0]
            print(f"\n  {era} {inst}: n={len(d):,}  mean={d.mean():+.1f}pt  PF={pf(d):.2f}")
            if len(w):
                print(f"    wins   n={len(w):>5,} mean=+{w.mean():.1f}  p95=+{w.quantile(0.95):.0f}  p99=+{w.quantile(0.99):.0f}  max=+{w.max():.0f}")
            if len(l):
                print(f"    losses n={len(l):>5,} mean={l.mean():.1f}  p5={l.quantile(0.05):.0f}  p1={l.quantile(0.01):.0f}  min={l.min():.0f}")

    print(f"\n\n{'='*90}\n  BIG LOSS ANATOMY — top 10 worst trades by era\n{'='*90}")
    for era in ["TRAIN", "TEST"]:
        print(f"\n  {era} worst 10:")
        d = c[c.era == era].nsmallest(10, "pnl_net")
        print(f"    {'date':<12} {'inst':<7} {'layer':<5} {'dir':<6} {'flag':<8} {'bar_rng':<8} {'pnl_net':<9} {'reason':<15}")
        for _, r in d.iterrows():
            print(f"    {r['date']:<12} {r['instrument']:<7} {r['layer']:<5} "
                  f"{r['direction']:<6} {str(r.get('range_flag','-')):<8} "
                  f"{float(r.get('bar_range', 0)):<8.0f} {r['pnl_net']:>+7.1f}  {r['reason']:<15}")

    print(f"\n\n{'='*90}\n  BIG WIN ANATOMY — top 10 best trades by era\n{'='*90}")
    for era in ["TRAIN", "TEST"]:
        print(f"\n  {era} best 10:")
        d = c[c.era == era].nlargest(10, "pnl_net")
        print(f"    {'date':<12} {'inst':<7} {'layer':<5} {'dir':<6} {'flag':<8} {'bar_rng':<8} {'pnl_net':<9} {'reason':<15}")
        for _, r in d.iterrows():
            print(f"    {r['date']:<12} {r['instrument']:<7} {r['layer']:<5} "
                  f"{r['direction']:<6} {str(r.get('range_flag','-')):<8} "
                  f"{float(r.get('bar_range', 0)):<8.0f} {r['pnl_net']:>+7.1f}  {r['reason']:<15}")

    print(f"\n\n{'='*90}\n  CONCENTRATION — what fraction of trades carry the edge\n{'='*90}")
    for era in ["TRAIN", "TEST"]:
        d = c[c.era == era].copy()
        wins = d[d.pnl_net > 0].sort_values("pnl_net", ascending=False)
        losses = d[d.pnl_net < 0].sort_values("pnl_net", ascending=True)
        tw = wins.pnl_net.sum()
        tl = abs(losses.pnl_net.sum())
        print(f"\n  {era}:")
        for pct in [1, 5, 10, 25, 50]:
            kw = max(1, int(len(wins) * pct / 100))
            kl = max(1, int(len(losses) * pct / 100))
            w_top = wins.head(kw).pnl_net.sum()
            l_top = abs(losses.head(kl).pnl_net.sum())
            print(f"    top {pct:>3}%: wins {w_top/max(tw,1)*100:>5.1f}% of gross wins  "
                  f"| losses {l_top/max(tl,1)*100:>5.1f}% of gross losses")

    print(f"\n\n{'='*90}\n  BIG WIN PROFILE — wins >100pt net\n{'='*90}")
    big_wins = c[c.pnl_net > 100].copy()
    print(f"\n  n={len(big_wins):,}/{len(c):,} ({len(big_wins)/len(c)*100:.2f}%)")
    for col in ["range_flag", "instrument", "layer", "bar_num", "era"]:
        print(f"\n  by {col}:")
        for v, d in big_wins.groupby(col, observed=True):
            pct = len(d) / len(big_wins) * 100
            print(f"    {str(v):<12} n={len(d):>4,}  {pct:>5.1f}% of big wins  sum={d['pnl_net'].sum():>+7,.0f}")

    print(f"\n\n{'='*90}\n  BIG LOSS PROFILE — losses <-50pt net\n{'='*90}")
    big_losses = c[c.pnl_net < -50].copy()
    print(f"\n  n={len(big_losses):,}/{len(c):,} ({len(big_losses)/len(c)*100:.2f}%)")
    for col in ["range_flag", "instrument", "layer", "bar_num", "era"]:
        print(f"\n  by {col}:")
        for v, d in big_losses.groupby(col, observed=True):
            pct = len(d) / len(big_losses) * 100
            print(f"    {str(v):<12} n={len(d):>4,}  {pct:>5.1f}% of big losses  sum={d['pnl_net'].sum():>+7,.0f}")


if __name__ == "__main__":
    main()
