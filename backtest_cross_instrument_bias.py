"""
backtest_cross_instrument_bias.py — does NIKKEI (earlier) or prior-day direction
predict DAX/US30 performance same day?

NIKKEI closes ~15:00 JST (06:00 UTC) before DAX opens 9:00 CET (07:00 UTC).
DAX/US30 overlap but DAX comes first.

Tests:
  1. DAX daily pnl | NIKKEI same-day pnl bucket
  2. DAX daily pnl | NIKKEI prev-day pnl bucket
  3. DAX daily pnl | DAX prev-day pnl bucket
  4. US30 daily pnl | NIKKEI+DAX same-day pnl bucket (both precede US30)
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


def bucket_pnl(pnl):
    if pnl < -50: return "1_big_loss"
    if pnl < -10: return "2_loss"
    if pnl < 10:  return "3_flat"
    if pnl < 50:  return "4_win"
    return "5_big_win"


def pf(s):
    w = s[s > 0].sum(); l = abs(s[s < 0].sum())
    return w / max(l, 0.001)


def main():
    df = pd.read_parquet(PARQUET)
    df = fade_filter_df(df, 20)
    df = filter_combined(df)
    df = apply_friction(df)
    df["date"] = pd.to_datetime(df["date"])

    # Daily pnl per instrument
    daily = df.groupby(["date", "instrument"])["pnl_net"].sum().unstack(fill_value=0)
    daily.columns = [f"{c}_pnl" for c in daily.columns]
    daily = daily.sort_index()

    # Prior day shifts
    daily["DAX_prev"] = daily["DAX_pnl"].shift(1)
    daily["US30_prev"] = daily["US30_pnl"].shift(1)
    daily["NIKKEI_prev"] = daily["NIKKEI_pnl"].shift(1)

    # Buckets
    for col in ["DAX_pnl", "US30_pnl", "NIKKEI_pnl", "DAX_prev", "NIKKEI_prev"]:
        daily[f"{col}_bucket"] = daily[col].apply(bucket_pnl)

    # --- Analysis 1: DAX | NIKKEI same-day ---
    print("="*100)
    print("  #1 DAX daily pnl CONDITIONAL on NIKKEI same-day")
    print("  (NIKKEI closes ~06:00 UTC, DAX opens 07:00 UTC — usable ex-ante signal)")
    print("="*100)
    print(f"  {'NIKKEI same-day':<20} {'n days':>8} {'DAX mean':>10} {'DAX median':>12} "
          f"{'DAX win%':>10} {'DAX total':>12}")
    for b in ["1_big_loss", "2_loss", "3_flat", "4_win", "5_big_win"]:
        sub = daily[daily["NIKKEI_pnl_bucket"] == b]
        if len(sub) == 0: continue
        mean_d = sub["DAX_pnl"].mean()
        med_d = sub["DAX_pnl"].median()
        win_pct = (sub["DAX_pnl"] > 0).mean() * 100
        total = sub["DAX_pnl"].sum()
        flag = "⚠️ " if mean_d < 0 else ("  " if mean_d < 20 else "✓ ")
        print(f"  {b:<20} {len(sub):>8} {mean_d:>+10.1f} {med_d:>+12.1f} {win_pct:>9.1f}% "
              f"{total:>+12,.0f} {flag}")

    # --- Analysis 2: DAX | NIKKEI prev-day ---
    print("\n" + "="*100)
    print("  #2 DAX daily pnl CONDITIONAL on NIKKEI prev-day")
    print("="*100)
    print(f"  {'NIKKEI prev-day':<20} {'n days':>8} {'DAX mean':>10} {'DAX median':>12} "
          f"{'DAX win%':>10} {'DAX total':>12}")
    for b in ["1_big_loss", "2_loss", "3_flat", "4_win", "5_big_win"]:
        sub = daily[daily["NIKKEI_prev_bucket"] == b]
        if len(sub) == 0: continue
        mean_d = sub["DAX_pnl"].mean()
        med_d = sub["DAX_pnl"].median()
        win_pct = (sub["DAX_pnl"] > 0).mean() * 100
        total = sub["DAX_pnl"].sum()
        flag = "⚠️ " if mean_d < 0 else ("  " if mean_d < 20 else "✓ ")
        print(f"  {b:<20} {len(sub):>8} {mean_d:>+10.1f} {med_d:>+12.1f} {win_pct:>9.1f}% "
              f"{total:>+12,.0f} {flag}")

    # --- Analysis 3: DAX | DAX prev-day ---
    print("\n" + "="*100)
    print("  #3 DAX daily pnl CONDITIONAL on DAX prev-day (continuation vs reversal)")
    print("="*100)
    print(f"  {'DAX prev-day':<20} {'n days':>8} {'DAX mean':>10} {'DAX median':>12} "
          f"{'DAX win%':>10} {'DAX total':>12}")
    for b in ["1_big_loss", "2_loss", "3_flat", "4_win", "5_big_win"]:
        sub = daily[daily["DAX_prev_bucket"] == b]
        if len(sub) == 0: continue
        mean_d = sub["DAX_pnl"].mean()
        med_d = sub["DAX_pnl"].median()
        win_pct = (sub["DAX_pnl"] > 0).mean() * 100
        total = sub["DAX_pnl"].sum()
        flag = "⚠️ " if mean_d < 0 else ("  " if mean_d < 20 else "✓ ")
        print(f"  {b:<20} {len(sub):>8} {mean_d:>+10.1f} {med_d:>+12.1f} {win_pct:>9.1f}% "
              f"{total:>+12,.0f} {flag}")

    # --- Analysis 4: US30 | NIKKEI + DAX same-day ---
    print("\n" + "="*100)
    print("  #4 US30 daily pnl CONDITIONAL on NIKKEI + DAX same-day combined")
    print("  (US30 opens after both — both NIKKEI and DAX pnl are known ex-ante)")
    print("="*100)
    daily["combined_pnl"] = daily["NIKKEI_pnl"] + daily["DAX_pnl"]
    daily["combined_bucket"] = daily["combined_pnl"].apply(bucket_pnl)
    print(f"  {'NIKKEI+DAX same-day':<20} {'n days':>8} {'US30 mean':>11} {'US30 median':>13} "
          f"{'US30 win%':>11} {'US30 total':>13}")
    for b in ["1_big_loss", "2_loss", "3_flat", "4_win", "5_big_win"]:
        sub = daily[daily["combined_bucket"] == b]
        if len(sub) == 0: continue
        mean_u = sub["US30_pnl"].mean()
        med_u = sub["US30_pnl"].median()
        win_pct = (sub["US30_pnl"] > 0).mean() * 100
        total = sub["US30_pnl"].sum()
        flag = "⚠️ " if mean_u < 0 else ("  " if mean_u < 20 else "✓ ")
        print(f"  {b:<20} {len(sub):>8} {mean_u:>+11.1f} {med_u:>+13.1f} {win_pct:>10.1f}% "
              f"{total:>+13,.0f} {flag}")

    # --- What-if filter: skip DAX when NIKKEI today was big_loss ---
    print("\n" + "="*100)
    print("  What-if filter: SKIP DAX if NIKKEI same-day < -50pt")
    print("="*100)
    skip_days = daily[daily["NIKKEI_pnl"] < -50]
    keep_days = daily[daily["NIKKEI_pnl"] >= -50]
    print(f"  Skipped days: {len(skip_days)} ({len(skip_days)/len(daily)*100:.1f}%)")
    print(f"  DAX pnl on SKIPPED days: total {skip_days['DAX_pnl'].sum():+,.0f}  mean {skip_days['DAX_pnl'].mean():+.1f}")
    print(f"  DAX pnl on KEPT days:    total {keep_days['DAX_pnl'].sum():+,.0f}  mean {keep_days['DAX_pnl'].mean():+.1f}")
    total_saving = -skip_days["DAX_pnl"].sum()
    print(f"  → If skipped: AVOID {total_saving:+,.0f}pt ({'gain' if total_saving > 0 else 'loss — bad filter'})")

    # Same check: skip US30 if NIKKEI+DAX < -100
    print("\n" + "="*100)
    print("  What-if filter: SKIP US30 if NIKKEI+DAX same-day < -100pt")
    print("="*100)
    skip_days_u = daily[daily["combined_pnl"] < -100]
    keep_days_u = daily[daily["combined_pnl"] >= -100]
    print(f"  Skipped days: {len(skip_days_u)} ({len(skip_days_u)/len(daily)*100:.1f}%)")
    print(f"  US30 pnl on SKIPPED days: total {skip_days_u['US30_pnl'].sum():+,.0f}  mean {skip_days_u['US30_pnl'].mean():+.1f}")
    print(f"  US30 pnl on KEPT days:    total {keep_days_u['US30_pnl'].sum():+,.0f}  mean {keep_days_u['US30_pnl'].mean():+.1f}")
    total_saving_u = -skip_days_u["US30_pnl"].sum()
    print(f"  → If skipped: AVOID {total_saving_u:+,.0f}pt")


if __name__ == "__main__":
    main()
