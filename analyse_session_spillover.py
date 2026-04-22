"""
analyse_session_spillover.py — Test "continuation guard" on 18yr v2 data.

Rule: after an instrument's session completes with net pnl > THRESHOLD pts,
SKIP subsequent sessions of the same instrument that day.

Post-processes the v2 backtest CSV (COMBINED filter applied first).
No engine re-execution needed — fast.

Variants:
  - Threshold: 50, 75, 100, 150, 200 pts
  - Mode A: only S1 winner blocks S2+S3
  - Mode B: S1 OR S2 winner blocks remaining sessions
"""
import pandas as pd

V2 = "/root/asrs-bot/data/backtest_firstrate_v2_results.csv"


def load_and_filter():
    df = pd.read_csv(V2)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"}
    )
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    drop = df["is_re"] & ~df["same"] & (
        df["instrument"].isin(["DAX", "US30"])
        | ((df["instrument"] == "NIKKEI") & df["first_won"])
    )
    return df[~drop].copy()


def apply_continuation_guard(df, threshold: float, mode: str = "A"):
    """Drop trades in later sessions if earlier session's net > threshold.

    mode="A": only S1 winner blocks S2+S3
    mode="B": S1 or S2 winner blocks remaining sessions
    """
    df = df.copy()
    # Compute per-(date, instrument, session) net
    df["session_num"] = df["signal"].str.extract(r"_S(\d)").astype(int)
    session_net = df.groupby(["date", "instrument", "session_num"])["pnl_pts"].sum().reset_index()

    drop_mask = pd.Series(False, index=df.index)

    for (date_, inst), grp in session_net.groupby(["date", "instrument"]):
        grp = grp.sort_values("session_num")
        s1_net = grp[grp["session_num"] == 1]["pnl_pts"].sum() if (grp["session_num"] == 1).any() else 0
        s2_net = grp[grp["session_num"] == 2]["pnl_pts"].sum() if (grp["session_num"] == 2).any() else 0
        # If S1 big winner, drop S2 + S3
        if s1_net > threshold:
            mask = (df["date"] == date_) & (df["instrument"] == inst) & (df["session_num"].isin([2, 3]))
            drop_mask |= mask
        # Mode B: also drop S3 if S2 big winner
        if mode == "B" and s2_net > threshold:
            mask = (df["date"] == date_) & (df["instrument"] == inst) & (df["session_num"] == 3)
            drop_mask |= mask
    return df[~drop_mask].copy()


def stats(d, label):
    if len(d) == 0:
        print(f"  {label:<45}  empty"); return
    w = d[d.pnl_pts > 0]; l = d[d.pnl_pts < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    net = d.pnl_pts.sum()
    wr = len(w) / len(d) * 100
    print(f"  {label:<45} n={len(d):>6}  PF={pf:>5.2f}  net={net:>+9,.0f}  WR={wr:>4.1f}%")


def main():
    df = load_and_filter()
    print(f"Loaded (COMBINED filtered): {len(df):,} trades")
    print()
    stats(df, "BASELINE (no continuation guard)")
    for period, mask in [("TRAIN 2008-17", df["year"] <= 2017),
                          ("TEST 2018-26", df["year"] >= 2018)]:
        stats(df[mask], f"  {period}")

    for mode in ["A", "B"]:
        print(f"\n=== Mode {mode} — {'only S1 blocks S2+S3' if mode=='A' else 'S1 or S2 blocks next'} ===")
        for thresh in [50, 75, 100, 150, 200]:
            filt = apply_continuation_guard(df, thresh, mode)
            dropped = len(df) - len(filt)
            print(f"\n  threshold={thresh}pts  ({dropped:,} trades dropped = {dropped/len(df)*100:.1f}%)")
            stats(filt, f"    FULL")
            stats(filt[filt["year"] <= 2017], f"    TRAIN")
            stats(filt[filt["year"] >= 2018], f"    TEST")

    # Per-instrument detail for best variant (mode A, threshold 100)
    print("\n\n=== PER-INSTRUMENT: Mode A, threshold=100 ===")
    filt = apply_continuation_guard(df, 100, "A")
    for inst in ["DAX", "US30", "NIKKEI"]:
        base = df[df["instrument"] == inst]
        filt_inst = filt[filt["instrument"] == inst]
        print(f"\n  {inst}:")
        stats(base,      "    baseline")
        stats(filt_inst, "    with guard (>100pt S1 winner blocks S2+S3)")


if __name__ == "__main__":
    main()
