"""
backtest_vol_and_time.py — test volatility regime and time-of-day filters.

Uses the existing variant_b_fade_trades_s3.parquet. Infers vol regime from
bar_range (signal bar range) and entry time from exit_idx - bars_held.

Variants:
  1. Vol regime: classify signal bar range into LOW/MID/HIGH buckets per
     instrument. See if any bucket is consistently unprofitable.
  2. Time of day: derive entry bar position (bars after session open), split
     into EARLY/MID/LATE. Check PF per bucket.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
PARQUET = "/root/asrs-bot/data/variant_b_fade_trades_s3.parquet"
UPLIFT = 1.06

bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["session_end_hour"] = 15
bt.INSTRUMENTS["NIKKEI"]["session_end_minute"] = 0


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def fade_filter_df(df, fade_min_winner=20):
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        for i, (idx, _) in enumerate(group[group.is_fade].iterrows()):
            if i < len(q): orig_win[idx] = q[i]
    df["orig_win"] = df.index.map(orig_win)
    keep_base = ~df["is_fade"]
    keep_fade = df["is_fade"] & (df["orig_win"].fillna(0) >= fade_min_winner)
    return df[keep_base | keep_fade].copy()


def session_open_bar_idx(inst, sig):
    """Derive absolute bar index of session open from signal name."""
    cfg = bt.INSTRUMENTS[inst]
    s = int(sig.split("_S")[1][0])
    h = cfg[f"s{s}_open_hour"]; m = cfg[f"s{s}_open_minute"]
    return (h * 60 + m) // 5


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def bucket_stats(df, col, buckets, label):
    print(f"\n  --- {label} ---")
    print(f"  {'bucket':<20} {'n':>8} {'PF':>6} {'net':>12} {'avg':>8} {'WR':>6}")
    for b in buckets:
        sub = df[df[col] == b]
        if len(sub) == 0: continue
        p = pf(sub["pnl_net"])
        wr = (sub.pnl_net > 0).mean() * 100
        flag = "⚠️" if p < 1.3 else ("•" if p < 1.6 else "✓")
        print(f"  {str(b):<20} {len(sub):>8,} {p:>6.2f} {sub.pnl_net.sum():>+12,.0f} {sub.pnl_net.mean():>+8.2f} {wr:>5.1f}%  {flag}")


def main():
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    df = fade_filter_df(df, 20)
    df = filter_combined(df)
    df = apply_friction(df)

    # Only BASE trades for vol/time analysis (fades are post-base)
    base = df[~df.is_fade].copy()

    # === #1 VOLATILITY REGIME ===
    # Classify signal bar range into per-instrument quartiles
    print("="*100)
    print("  #1 VOLATILITY REGIME (signal bar range quartile, per instrument)")
    print("="*100)
    base["range_q"] = (
        base.groupby("instrument")["bar_range"]
        .transform(lambda s: pd.qcut(s, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop"))
    )
    for inst in ["ALL", "DAX", "US30", "NIKKEI"]:
        sub = base if inst == "ALL" else base[base.instrument == inst]
        bucket_stats(sub, "range_q", ["Q1_low", "Q2", "Q3", "Q4_high"],
                     f"{inst} — by signal bar range quartile")

    # === #2 TIME OF DAY ===
    # Derive entry bar position from exit_idx and rough duration (exit_idx - bar_num)
    # Actually: approximate entry_bar_idx = session_open_bar_idx + bar_num (signal
    # bar is bar 4 or 5 → entry happens ~bar_num or +1)
    # The absolute bar index scale: day starts at bar 0 = 00:00.
    # So entry_bar_idx = ss_open + bar_num
    base["session_open_idx"] = base.apply(
        lambda r: session_open_bar_idx(r["instrument"], r["signal"].replace("_FADE", "")), axis=1)
    # bars since session open = bar_num (signal bar 4 means entry could be bar 5+)
    # Actually: for the FIRST entry, entry_bar = session_open + bar_num + 1 at earliest.
    # For re-entries, it's exit_idx (approximate — happens a few bars later).
    # Simpler: use exit_idx - session_open_idx as "bars into session"
    base["bars_into_session"] = base["exit_idx"] - base["session_open_idx"]
    base["time_bucket"] = pd.cut(
        base["bars_into_session"], bins=[-1, 10, 20, 30, 1000],
        labels=["0-10bars (50min)", "10-20 (50-100min)", "20-30 (100-150min)", "30+ (150min+)"],
    )
    print("\n" + "="*100)
    print("  #2 TIME OF DAY (exit bar index relative to session open)")
    print("="*100)
    for inst in ["ALL", "DAX", "US30", "NIKKEI"]:
        sub = base if inst == "ALL" else base[base.instrument == inst]
        bucket_stats(sub, "time_bucket", ["0-10bars (50min)", "10-20 (50-100min)", "20-30 (100-150min)", "30+ (150min+)"],
                     f"{inst} — by bars-into-session")

    # Also: first entry vs re-entries
    base["is_re"] = base.groupby(["date", "signal"])["pnl_pts"].cumcount() > 0
    print("\n" + "="*100)
    print("  FIRST ENTRY vs RE-ENTRY")
    print("="*100)
    for inst in ["ALL", "DAX", "US30", "NIKKEI"]:
        sub = base if inst == "ALL" else base[base.instrument == inst]
        for is_re in [False, True]:
            s = sub[sub.is_re == is_re]
            if len(s) == 0: continue
            p = pf(s["pnl_net"])
            wr = (s.pnl_net > 0).mean() * 100
            lbl = "RE-ENTRY" if is_re else "1st entry"
            print(f"  {inst:<8} {lbl:<10} n={len(s):>6,} PF={p:.2f} net={s.pnl_net.sum():>+10,.0f} WR={wr:.1f}%")


if __name__ == "__main__":
    main()
