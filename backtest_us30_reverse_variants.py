"""
backtest_us30_reverse_variants.py — test relaxed reverse-reentry policies on US30.

Current: US30 locks to first_direction ("never" policy). Blocks opposite re-entry.
18yr baseline: opposite re-entries PF 0.81 → locking is net +32k pts.

Today (2026-04-21) S1 LONG stopped quick, price reversed 400pt, we couldn't SHORT.
User question: would `after_loss` or `small_win_threshold` variants capture this
without breaking the 18yr edge?

Variants tested on US30 only (DAX + NIKKEI unchanged, 18yr):
  A  current (never)
  B  after_loss (like NIKKEI: allow opp if first LOST)
  C  small_loss_or_win<5  (allow opp if first lost OR first won ≤5pt)
  D  small_loss_or_win<10 (≤10pt)
  E  small_loss_or_win<20 (≤20pt)
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import stats
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
PARQUET = "/root/asrs-bot/data/variant_b_fade_trades_s3.parquet"
UPLIFT = 1.06


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def filter_with_policy(df, us30_policy, us30_small_win_thresh=0):
    """
    us30_policy: "never" | "after_loss" | "small"
    us30_small_win_thresh: if policy="small", allow opposite if fp <= threshold
    DAX stays "never", NIKKEI stays "after_loss"
    """
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"}
    )
    df = df.merge(first, on=["date", "signal"], how="left")
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    df["first_won"] = df["fp"] > 0
    # Opp re-entry = is_re AND not same
    opp = df["is_re"] & ~df["same"]
    # DAX: always drop opp
    drop_dax = opp & (df["instrument"] == "DAX")
    # NIKKEI: drop opp if first_won (same as before)
    drop_nikkei = opp & (df["instrument"] == "NIKKEI") & df["first_won"]
    # US30: varies by policy
    if us30_policy == "never":
        drop_us30 = opp & (df["instrument"] == "US30")
    elif us30_policy == "after_loss":
        drop_us30 = opp & (df["instrument"] == "US30") & df["first_won"]
    elif us30_policy == "small":
        # Drop opp if first won by MORE than threshold (trending market → don't reverse)
        drop_us30 = (opp & (df["instrument"] == "US30")
                     & df["first_won"]
                     & (df["fp"] > us30_small_win_thresh))
    else:
        drop_us30 = opp & (df["instrument"] == "US30")
    drop = drop_dax | drop_nikkei | drop_us30
    return df[~drop].copy()


def fade_filter(df, fade_min_winner=20):
    """Fades require ≥20pt TRAIL_WIN from base. Mirrors MC."""
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        fades = group[group.is_fade]
        for i, (idx, _) in enumerate(fades.iterrows()):
            if i < len(q): orig_win[idx] = q[i]
    df["orig_win"] = df.index.map(orig_win)
    keep_base = ~df["is_fade"]
    keep_fade = df["is_fade"] & (df["orig_win"].fillna(0) >= fade_min_winner)
    return df[keep_base | keep_fade].copy()


def main():
    df_raw = pd.read_parquet(PARQUET)
    df_raw = df_raw.reset_index(drop=True)
    df_faded = fade_filter(df_raw, fade_min_winner=20)

    variants = [
        ("A current (US30 never)",          "never", 0),
        ("B US30 after_loss (like NIKKEI)", "after_loss", 0),
        ("C US30 small-win ≤5pt",           "small", 5),
        ("D US30 small-win ≤10pt",          "small", 10),
        ("E US30 small-win ≤20pt",          "small", 20),
    ]

    print(f"{'='*110}")
    print(f"  US30 reverse-reentry variants (18yr, Variant B + fade, friction applied)")
    print(f"{'='*110}")

    for lbl, pol, thresh in variants:
        # Apply reverse-reentry policy
        filt = filter_with_policy(df_faded, pol, thresh)
        # Apply friction
        filt = apply_friction(filt)
        filt["year"] = pd.to_datetime(filt["date"]).dt.year
        # Rename pnl_net → pnl_pts for stats compatibility
        filt["pnl_pts"] = filt["pnl_net"]

        print(f"\n  {lbl}")
        stats(filt, "    FULL 18yr  ")
        stats(filt[filt.year <= 2017], "    TRAIN 2008-17")
        stats(filt[filt.year >= 2018], "    TEST 2018-26 ")
        # Per instrument
        for inst in ["DAX", "US30", "NIKKEI"]:
            sub = filt[filt.instrument == inst]
            w = sub[sub.pnl_pts > 0].pnl_pts.sum()
            l = abs(sub[sub.pnl_pts < 0].pnl_pts.sum())
            pf = w/max(l, 0.001)
            wr = (sub.pnl_pts > 0).mean() * 100
            print(f"      {inst:<8} n={len(sub):>5,}  PF={pf:.2f}  net={sub.pnl_pts.sum():>+10,.0f}  WR={wr:>5.1f}%")

    # Focus: US30 only comparison across variants
    print(f"\n{'='*110}")
    print(f"  US30 ONLY — side-by-side")
    print(f"{'='*110}")
    print(f"  {'Variant':<40} {'n':>6} {'PF':>6} {'net':>10} {'avg_yr':>10} {'WR':>6}  {'vs A':>8}")
    baseline_net = None
    for lbl, pol, thresh in variants:
        filt = filter_with_policy(df_faded, pol, thresh)
        filt = apply_friction(filt)
        us = filt[filt.instrument == "US30"]
        w = us[us.pnl_net > 0].pnl_net.sum()
        l = abs(us[us.pnl_net < 0].pnl_net.sum())
        pf = w/max(l, 0.001)
        wr = (us.pnl_net > 0).mean() * 100
        net = us.pnl_net.sum()
        n_years = pd.to_datetime(us["date"]).dt.year.nunique()
        avg_per_year = net / n_years if n_years else 0
        delta = "—" if baseline_net is None else f"{net - baseline_net:+,.0f}"
        if baseline_net is None: baseline_net = net
        print(f"  {lbl:<40} {len(us):>6,} {pf:>6.2f} {net:>+10,.0f} {avg_per_year:>+10,.0f} {wr:>5.1f}%  {delta:>8}")


if __name__ == "__main__":
    main()
