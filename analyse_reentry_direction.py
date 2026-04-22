"""
analyse_reentry_direction.py — Test: should re-entries only be in the OPPOSITE
direction of the first trade? (i.e. if first trade of session was LONG and
stopped, only allow SHORT as second entry, and vice versa.)

Rationale: if the initial breakout failed, retrying the same direction may be
worse than reversing. Test against v2 backtest trade log.
"""
import pandas as pd

TRADES = "/root/asrs-bot/data/backtest_firstrate_v2_results.csv"


def stats(df: pd.DataFrame, label: str):
    if len(df) == 0:
        print(f"  {label:<35}  no trades")
        return
    wins   = df[df["pnl_pts"] > 0]
    losses = df[df["pnl_pts"] < 0]
    w  = wins["pnl_pts"].sum()
    l  = abs(losses["pnl_pts"].sum()) or 0.001
    pf  = w / l
    net = df["pnl_pts"].sum()
    wr  = len(wins) / len(df) * 100
    print(f"  {label:<35} n={len(df):>6}  PF={pf:>5.2f}  "
          f"net={net:>+9,.0f}  WR={wr:.0f}%  avgW={w/max(1,len(wins)):.1f} "
          f"avgL={l/max(1,len(losses)):.1f}")


def run():
    df = pd.read_csv(TRADES)
    # Session key: (date, signal) — within one session's max_entries
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["trade_num_in_session"] = df.groupby(["date", "signal"]).cumcount() + 1

    # First-trade direction per session
    first = df[df["trade_num_in_session"] == 1][["date", "signal", "direction"]]
    first = first.rename(columns={"direction": "first_dir"})
    df = df.merge(first, on=["date", "signal"], how="left")
    df["is_reentry"] = df["trade_num_in_session"] > 1
    df["same_dir_as_first"] = df["direction"] == df["first_dir"]

    n_sessions = df.groupby(["date","signal"]).size().count()
    n_reentry = df["is_reentry"].sum()
    print(f"Total trades: {len(df)}, sessions: {n_sessions}, re-entries: {n_reentry}")
    print()

    print("=== FIRST TRADE ONLY (baseline) ===")
    stats(df[df["trade_num_in_session"] == 1], "trade_1 (all)")

    print()
    print("=== RE-ENTRIES (trade 2+) — split by direction vs first ===")
    re = df[df["is_reentry"]]
    stats(re, "all re-entries")
    stats(re[re["same_dir_as_first"]],  "  same direction")
    stats(re[~re["same_dir_as_first"]], "  opposite direction")

    print()
    print("=== TRADE #2 ONLY ===")
    t2 = df[df["trade_num_in_session"] == 2]
    stats(t2, "trade_2 (all)")
    stats(t2[t2["same_dir_as_first"]],  "  same direction")
    stats(t2[~t2["same_dir_as_first"]], "  opposite direction")

    print()
    print("=== TRADE #3 ONLY ===")
    t3 = df[df["trade_num_in_session"] == 3]
    stats(t3, "trade_3 (all)")
    stats(t3[t3["same_dir_as_first"]],  "  same direction")
    stats(t3[~t3["same_dir_as_first"]], "  opposite direction")

    print()
    print("=== PER INSTRUMENT × RE-ENTRY DIR ===")
    for inst in ["DAX", "US30", "NIKKEI"]:
        sub = df[(df["instrument"] == inst) & df["is_reentry"]]
        print(f"\n  {inst}:")
        stats(sub, "all re-entries")
        stats(sub[sub["same_dir_as_first"]],  "  same direction")
        stats(sub[~sub["same_dir_as_first"]], "  opposite direction")

    # Scenario: what if we BANNED same-direction re-entries?
    print()
    print("=== COUNTERFACTUAL: ban same-dir re-entries ===")
    kept = df[~(df["is_reentry"] & df["same_dir_as_first"])]
    stats(df,   "ORIGINAL (all trades)")
    stats(kept, "FILTERED (no same-dir re-entry)")
    saved = df[df["is_reentry"] & df["same_dir_as_first"]]
    print(f"  trades removed: {len(saved)}  ({saved['pnl_pts'].sum():+.0f}pts)")


if __name__ == "__main__":
    run()
