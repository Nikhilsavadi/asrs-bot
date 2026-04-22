"""
Test fade-sub-filters based on segment analysis. Apply each filter combination
and measure combined (base + filtered_fade) PF. TRAIN/TEST split to avoid
curve-fit.
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


def evaluate(name, base, fade, fade_filter=None):
    """Apply fade filter, compute base+filtered_fade PF. Returns stats."""
    if fade_filter:
        kept = fade[fade.apply(fade_filter, axis=1)].copy()
    else:
        kept = fade.copy()
    combined = pd.concat([base, kept], ignore_index=True)
    combined["year"] = pd.to_datetime(combined["date"]).dt.year
    tr = combined[combined.year <= 2017]
    te = combined[combined.year >= 2018]
    print(f"\n  {name}")
    pct = (len(kept)/len(fade)*100) if len(fade) else 0
    print(f"    fade kept: {len(kept):,}/{len(fade):,} ({pct:.1f}%)")
    print(f"    FADE only  PF={pf(kept['pnl_pts_net']):.2f}  "
          f"net={kept['pnl_pts_net'].sum():+,.0f}")
    print(f"    COMBINED  FULL PF={pf(combined['pnl_pts_net']):.2f}  "
          f"net={combined['pnl_pts_net'].sum():+,.0f}")
    print(f"    COMBINED  TRAIN PF={pf(tr['pnl_pts_net']):.2f}  "
          f"TEST PF={pf(te['pnl_pts_net']):.2f}")


def main():
    df = pd.read_parquet(TRADES_CACHE)
    df["is_fade"] = df.get("is_fade", False)
    if df["is_fade"].dtype != bool:
        df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    base = df[~df["is_fade"]].copy()
    fade = df[df["is_fade"]].copy()

    # Apply base filter (COMBINED re-entry filter) to base
    base_f = filter_combined(base)
    base_f = apply_spread(base_f)
    fade = apply_spread(fade)

    print(f"BASE filtered: {len(base_f):,} trades, PF={pf(base_f['pnl_pts_net']):.2f}")
    print(f"FADE raw:      {len(fade):,} trades, PF={pf(fade['pnl_pts_net']):.2f}")

    print(f"\n{'='*90}")
    print(f"  FADE FILTER EXPERIMENTS  (COMBINED = base + filtered_fade, post-spread)")
    print(f"{'='*90}")

    # Baseline
    evaluate("0. No fade (base alone)", base_f, fade.iloc[:0])
    evaluate("1. All fade (baseline candidate)", base_f, fade)

    # Single filters
    evaluate("2. Skip NARROW (bar range <15)", base_f, fade,
             lambda r: r["range_flag"] != "NARROW")
    evaluate("3. Skip NIKKEI (instrument-based)", base_f, fade,
             lambda r: r["instrument"] != "NIKKEI")
    evaluate("4. Only LONG fades (= SHORT winners)", base_f, fade,
             lambda r: r["direction"] == "LONG")
    evaluate("5. Skip bar_range < 15pt", base_f, fade,
             lambda r: r["bar_range"] >= 15)
    evaluate("6. Skip bar_range < 30pt (tighter)", base_f, fade,
             lambda r: r["bar_range"] >= 30)

    # Combinations
    evaluate("7. Skip NARROW + NIKKEI", base_f, fade,
             lambda r: r["range_flag"] != "NARROW" and r["instrument"] != "NIKKEI")
    evaluate("8. Skip NARROW + only LONG", base_f, fade,
             lambda r: r["range_flag"] != "NARROW" and r["direction"] == "LONG")
    evaluate("9. Skip NIKKEI + only LONG", base_f, fade,
             lambda r: r["instrument"] != "NIKKEI" and r["direction"] == "LONG")
    evaluate("10. Triple: skip NARROW + NIKKEI + only LONG", base_f, fade,
             lambda r: (r["range_flag"] != "NARROW"
                         and r["instrument"] != "NIKKEI"
                         and r["direction"] == "LONG"))
    evaluate("11. Keep range ≥30 + only LONG", base_f, fade,
             lambda r: r["bar_range"] >= 30 and r["direction"] == "LONG")
    evaluate("12. Keep ONLY DAX+US30 NORMAL/WIDE bars (both directions)", base_f, fade,
             lambda r: (r["instrument"] != "NIKKEI" and r["range_flag"] != "NARROW"))


if __name__ == "__main__":
    main()
