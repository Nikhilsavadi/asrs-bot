"""
mfe_stress_test.py — apply 60s-timer friction haircut and re-evaluate.

Friction model: each trade's recorded pnl_pts is adjusted for realistic
live fill conditions:
  winning trades: pnl × 0.90  (lose 10% of the win — trail lags peak by
                               ~60s of confirmation + exit slippage)
  losing trades:  pnl × 1.10  (add 10% to loss — stop fires late, fill
                               is further adverse)

Apply to both baseline (candle) and MFE hybrid CSVs, recompute PF / net
/ WR, then run the 4yr MC at median + optimistic scenarios.
"""
import pandas as pd
import numpy as np
import subprocess

BASELINE = "/root/asrs-bot/data/mfe_mc_baseline.csv"
MFE = "/root/asrs-bot/data/mfe_mc_hybrid_50_0.5.csv"

OUT_BASELINE_STRESS = "/root/asrs-bot/data/stress_baseline.csv"
OUT_MFE_STRESS = "/root/asrs-bot/data/stress_mfe.csv"


def stress(in_path, out_path):
    df = pd.read_csv(in_path)
    df = df.copy()
    mask_win = df["pnl_pts"] > 0
    mask_los = df["pnl_pts"] < 0
    df.loc[mask_win, "pnl_pts"] = (df.loc[mask_win, "pnl_pts"] * 0.9).round(1)
    df.loc[mask_los, "pnl_pts"] = (df.loc[mask_los, "pnl_pts"] * 1.1).round(1)
    df.to_csv(out_path, index=False)
    return df


def stats(df, label):
    w = df[df.pnl_pts > 0]
    l = df[df.pnl_pts < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    net = df.pnl_pts.sum()
    wr = len(w) / len(df) * 100
    print(f"  {label:<30} n={len(df):>5}  PF={pf:>5.2f}  net={net:>+10,.0f}  WR={wr:>4.1f}%")


def main():
    print("Applying 10% haircut to wins, 10% amplifier to losses...")
    bl_raw = pd.read_csv(BASELINE)
    mfe_raw = pd.read_csv(MFE)
    bl_stress = stress(BASELINE, OUT_BASELINE_STRESS)
    mfe_stress = stress(MFE, OUT_MFE_STRESS)

    print("\n=== TRADE-LEVEL COMPARISON ===")
    stats(bl_raw, "Baseline RAW")
    stats(bl_stress, "Baseline STRESSED")
    stats(mfe_raw, "MFE hybrid RAW")
    stats(mfe_stress, "MFE hybrid STRESSED")

    print("\n=== TRAIN/TEST robustness under stress ===")
    for label, df in [("Baseline STRESSED", bl_stress), ("MFE hybrid STRESSED", mfe_stress)]:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        print(f"\n  {label}")
        for pn, mask in [("TRAIN 2008-17", df["year"] <= 2017),
                          ("TEST  2018-26", df["year"] >= 2018)]:
            stats(df[mask], f"  {pn}")

    print("\n=== 4-YR MC UNDER STRESS (median scenario, dual account) ===")
    for label, csv in [("Baseline STRESS", OUT_BASELINE_STRESS),
                       ("MFE hybrid STRESS", OUT_MFE_STRESS)]:
        print(f"\n--- {label} ---")
        result = subprocess.run(
            ["python3", "/root/asrs-bot/monte_carlo_4yr.py",
             "--csv", csv, "--scenario", "median", "--dual",
             "--runs", "1500", "--out", f"/tmp/mc_{label.replace(' ','_')}.png"],
            capture_output=True, text=True, timeout=300,
        )
        # Extract the relevant summary lines
        for line in result.stdout.splitlines():
            if any(k in line for k in ["EOY 4:", "YEAR-BY-YEAR", "Year 1:", "Year 2:",
                                        "Year 3:", "Year 4:", "NKD ENABLE", "p50  :",
                                        "WIFE ACCOUNT OPENED"]):
                print(f"    {line.strip()}")


if __name__ == "__main__":
    main()
