"""
analyse_setup_sizing.py — test per-setup-quality stake sizing.

Post-processes the v2 trade log. For each trade, multiply pnl_pts by a
size-multiplier that depends on (instrument, range_flag). Then recompute
aggregate PF + risk metrics.

COMBINED reverse-reentry filter is applied first for fair comparison.

Sizing schemes to test:
  A) FLAT (baseline)           — all trades at 1.0×
  B) NARROW-favored            — 1.5× narrow, 1.0× normal, 0.5× wide
  C) Per-instrument optimised  — derived from each instrument's PF by flag:
       multiplier ∝ max(0.5, PF / mean_PF_instrument)
  D) Aggressive narrow          — 2× narrow, 1× normal, 0.5× wide
  E) Conservative (defensive)  — 1× narrow, 1× normal, 0.3× wide

Notes:
  - PnL scales linearly with stake, so PF is preserved within any single
    (instrument, flag) bucket. But aggregate PF changes because the
    weighted mix of buckets changes.
  - This can't lift the overall per-trade expectancy by itself — it
    re-weights cohorts. PF improves ONLY if larger-stake trades have
    higher W/L than lower-stake ones.
"""
import pandas as pd
import numpy as np

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


def stats(d: pd.DataFrame, pnl_col="pnl_pts"):
    if len(d) == 0: return None
    w = d[d[pnl_col] > 0]; l = d[d[pnl_col] < 0]
    pf = w[pnl_col].sum() / max(abs(l[pnl_col].sum()), 0.001)
    net = d[pnl_col].sum()
    wr = len(w) / len(d) * 100
    return len(d), pf, net, wr


def print_stats(d, label, pnl_col="pnl_pts"):
    s = stats(d, pnl_col)
    if not s: print(f"  {label:<40}  empty"); return
    n, pf, net, wr = s
    print(f"  {label:<40} n={n:>5}  PF={pf:>5.2f}  net={net:>+9,.0f}  WR={wr:>4.1f}%")


SCHEMES = {
    "A_flat":       {"NARROW": 1.0, "NORMAL": 1.0, "WIDE": 1.0},
    "B_narrow_fav": {"NARROW": 1.5, "NORMAL": 1.0, "WIDE": 0.5},
    "D_aggressive": {"NARROW": 2.0, "NORMAL": 1.0, "WIDE": 0.5},
    "E_defensive":  {"NARROW": 1.0, "NORMAL": 1.0, "WIDE": 0.3},
}


def build_scheme_c(df) -> dict:
    """Optimised: per-(instrument, flag) multiplier from TRAIN-period PF only.

    Rule: multiplier = clip(PF_bucket / PF_inst_mean, 0.5, 2.0).
    Derived from TRAIN only so TEST is honest evaluation.
    """
    train = df[df["year"] <= 2017]
    sch = {}
    for inst in ["DAX", "US30", "NIKKEI"]:
        sub = train[train["instrument"] == inst]
        if len(sub) == 0: continue
        s = stats(sub); _, inst_pf, _, _ = s
        sch[inst] = {}
        for flag in ["NARROW", "NORMAL", "WIDE"]:
            bucket = sub[sub["range_flag"] == flag]
            b_s = stats(bucket)
            if not b_s:
                sch[inst][flag] = 1.0; continue
            _, bucket_pf, _, _ = b_s
            sch[inst][flag] = float(np.clip(bucket_pf / inst_pf, 0.5, 2.0))
    return sch


def apply_scheme(df: pd.DataFrame, scheme, per_inst: bool = False) -> pd.DataFrame:
    df = df.copy()
    if per_inst:
        mults = df.apply(lambda r: scheme.get(r["instrument"], {}).get(r["range_flag"], 1.0), axis=1)
    else:
        mults = df["range_flag"].map(scheme).fillna(1.0)
    df["pnl_sized"] = df["pnl_pts"] * mults
    df["size_mult"] = mults
    return df


def main():
    df = load_and_filter()
    print(f"Loaded: {len(df):,} trades (COMBINED filter applied)")
    print_stats(df, "BASELINE (flat sizing)")
    print()

    # Show current PF by (instrument, range_flag) bucket
    print("PF by (instrument, range_flag) — 18yr full:")
    for inst in ["DAX", "US30", "NIKKEI"]:
        print(f"  {inst}:")
        for flag in ["NARROW", "NORMAL", "WIDE"]:
            b = df[(df["instrument"] == inst) & (df["range_flag"] == flag)]
            print_stats(b, f"    {flag}")

    # Schemes A/B/D/E
    for sname, scheme in SCHEMES.items():
        sized = apply_scheme(df, scheme)
        print(f"\n=== SCHEME {sname} | mults {scheme} ===")
        for pname, years in [("FULL", lambda y: True),
                             ("TRAIN", lambda y: y <= 2017),
                             ("TEST", lambda y: y >= 2018)]:
            sub = sized[sized["year"].apply(years)]
            print_stats(sub, f"  {pname}", pnl_col="pnl_sized")

    # Scheme C: per-instrument, derived from TRAIN
    scheme_c = build_scheme_c(df)
    print(f"\n=== SCHEME C (per-inst optimised, TRAIN-derived) ===")
    print("  Derived multipliers:")
    for inst, flags in scheme_c.items():
        print(f"    {inst}: {flags}")
    sized = apply_scheme(df, scheme_c, per_inst=True)
    for pname, years in [("FULL", lambda y: True),
                         ("TRAIN", lambda y: y <= 2017),
                         ("TEST", lambda y: y >= 2018)]:
        sub = sized[sized["year"].apply(years)]
        print_stats(sub, f"  {pname}", pnl_col="pnl_sized")

    print(f"\n  Per-instrument (TEST) check:")
    for inst in ["DAX", "US30", "NIKKEI"]:
        sub = sized[(sized["instrument"] == inst) & (sized["year"] >= 2018)]
        print_stats(sub, f"    {inst} TEST", pnl_col="pnl_sized")


if __name__ == "__main__":
    main()
