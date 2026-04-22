"""
analyse_reentry_by_outcome.py — Does the first trade's W/L change re-entry behaviour?

Splits re-entries into 4 buckets:
  - trade_1 WIN  → same-direction re-entry
  - trade_1 WIN  → opposite-direction re-entry
  - trade_1 LOSS → same-direction re-entry
  - trade_1 LOSS → opposite-direction re-entry

Then train (2008-2017) / test (2018-2026) to validate robustness.
"""
import pandas as pd

TRADES = "/root/asrs-bot/data/backtest_firstrate_v2_results.csv"


def stats(d: pd.DataFrame, label: str):
    if len(d) == 0:
        print(f"  {label:<45}  empty"); return
    w = d[d.pnl_pts > 0]; l = d[d.pnl_pts < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    net = d.pnl_pts.sum()
    wr = len(w) / len(d) * 100
    print(f"  {label:<45} n={len(d):>6}  PF={pf:>5.2f}  "
          f"net={net:>+9,.0f}  WR={wr:>4.1f}%  E={net/len(d):>+5.2f}")


def classify_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add trade_num_in_session, first_dir, first_won, is_reentry, same_dir."""
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "first_dir", "pnl_pts": "first_pnl"}
    )
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["first_pnl"] > 0
    df["is_reentry"] = df["tn"] > 1
    df["same_dir"] = df["direction"] == df["first_dir"]
    return df


def dump_bucket(df: pd.DataFrame, period_label: str):
    re = df[df["is_reentry"]]
    print(f"\n--- {period_label} — re-entries split by trade_1 W/L × direction ---")
    stats(re[re["first_won"] & re["same_dir"]],   "first WON → same dir")
    stats(re[re["first_won"] & ~re["same_dir"]],  "first WON → opposite dir")
    stats(re[~re["first_won"] & re["same_dir"]],  "first LOST → same dir")
    stats(re[~re["first_won"] & ~re["same_dir"]], "first LOST → opposite dir")

    print(f"\n--- {period_label} — per instrument: opposite-direction re-entries ---")
    for inst in ["DAX", "US30", "NIKKEI"]:
        sub = re[re["instrument"] == inst]
        print(f"\n  {inst}:")
        stats(sub[sub["first_won"] & ~sub["same_dir"]],  "  first WON → opposite dir")
        stats(sub[~sub["first_won"] & ~sub["same_dir"]], "  first LOST → opposite dir")


def main():
    df = pd.read_csv(TRADES)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = classify_df(df)

    print("=" * 95)
    print("  FULL 18-YEAR (2008-2026)")
    print("=" * 95)
    dump_bucket(df, "FULL")

    print()
    print("=" * 95)
    print("  TRAIN 2008-2017")
    print("=" * 95)
    dump_bucket(df[df["year"] <= 2017], "TRAIN")

    print()
    print("=" * 95)
    print("  TEST 2018-2026")
    print("=" * 95)
    dump_bucket(df[df["year"] >= 2018], "TEST")


if __name__ == "__main__":
    main()
