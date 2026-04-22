"""
backtest_option2_dual_positions.py

Question: does our current sibling-blocking rule (one position per instrument)
cost us meaningful edge vs allowing simultaneous positions across sessions?

Approach: use full 18yr parquet (which models sessions independently = Option 2).
Apply a filter that simulates "current live" behaviour (sibling blocking) and
compare to Option 2.

Blocking rule: for each day/instrument, a later-session trade is BLOCKED if
an earlier-session trade is still active at that trade's ENTRY time.
Entry time approximated from bar_num + session_open; exit time from exit_idx
(absolute day bar index).

Compares:
  - Option 2 (current backtest): each session independent
  - Option 1 (current live): sibling blocking applied
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
UPLIFT = 1.06
PARQUET = "/root/asrs-bot/data/variant_b_fade_trades_s3.parquet"


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


# Day-start bar index 0 corresponds to the first 5-min bar of trading day.
# Session open bar index relative to day start, assuming 5-min bars.
# Day starts at 00:00 local; session open at H:M → bar idx = (H*60 + M) / 5.
def session_open_bar_idx(inst, session):
    cfg = bt.INSTRUMENTS[inst]
    h = cfg[f"s{session}_open_hour"]; m = cfg[f"s{session}_open_minute"]
    return (h * 60 + m) // 5


def pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return w / max(l, 0.001)


def summarise(df, label):
    w = df[df.pnl_net > 0].pnl_net.sum()
    l = abs(df[df.pnl_net < 0].pnl_net.sum())
    p = w / max(l, 0.001)
    wr = (df.pnl_net > 0).mean() * 100
    print(f"  {label:<30} n={len(df):>7,}  PF={p:.2f}  net={df.pnl_net.sum():>+12,.0f}  "
          f"WR={wr:.1f}%")


def main():
    print("Loading parquet ...", flush=True)
    df = pd.read_parquet(PARQUET)
    df["is_fade"] = df.get("is_fade", False).fillna(False).astype(bool)
    df["date"] = pd.to_datetime(df["date"])

    # Keep only BASE trades (fades aren't subject to sibling-block the same way)
    base = df[~df.is_fade].copy().reset_index(drop=True)
    fade = df[df.is_fade].copy().reset_index(drop=True)

    # Derive session number from signal name
    base["session"] = base["signal"].str.extract(r"_S(\d)").astype(int)
    fade["session"] = fade["signal"].str.extract(r"_S(\d)").astype(int)

    # Derive approximate entry bar index: session_open_bar + bar_num
    base["session_open_idx"] = base.apply(
        lambda r: session_open_bar_idx(r["instrument"], r["session"]), axis=1)
    base["entry_idx_approx"] = base["session_open_idx"] + base["bar_num"]
    base["exit_idx"] = base["exit_idx"].astype(int)

    # Apply reverse-reentry filter (same as before)
    base_f = filter_combined(base)

    # Apply blocking rule: for each (date, instrument), sort by session then entry order.
    # A later trade is BLOCKED if any earlier trade has exit_idx >= this trade's entry_idx_approx.
    base_f = base_f.sort_values(["date", "instrument", "session", "entry_idx_approx"]).reset_index(drop=True)

    blocked_mask = np.zeros(len(base_f), dtype=bool)
    # Group by (date, instrument) and process in order
    for (d, inst), grp in base_f.groupby(["date", "instrument"], sort=False):
        idxs = grp.index.to_list()
        active_until = -1
        current_session = None
        for i in idxs:
            row = base_f.iloc[i]
            entry_idx = row["entry_idx_approx"]
            exit_idx = row["exit_idx"]
            session = row["session"]
            # Block if earlier trade is still active at our entry
            if current_session is not None and session != current_session:
                # Cross-session check: blocked if previous session's last active trade
                # still extends past our entry
                if entry_idx < active_until:
                    blocked_mask[i] = True
                    continue
            # Not blocked — update active_until
            if exit_idx > active_until:
                active_until = exit_idx
            current_session = session

    base_f["blocked_by_sibling"] = blocked_mask
    base_f = apply_friction(base_f)

    fade_f = fade.copy()
    fade_f = apply_friction(fade_f)

    # OPTION 2 — no blocking (what our backtest numbers have been)
    opt2 = pd.concat([base_f, fade_f], ignore_index=True)
    # OPTION 1 — live behaviour: drop blocked base trades; fades untouched
    opt1 = pd.concat([base_f[~base_f.blocked_by_sibling], fade_f], ignore_index=True)

    print(f"\n{'='*100}")
    print(f"  OPTION 2 (no blocking) vs OPTION 1 (current live — sibling-blocked)")
    print(f"{'='*100}")
    blocked = base_f[base_f.blocked_by_sibling]
    print(f"  base trades blocked by sibling: {len(blocked):,} / {len(base_f):,} "
          f"({len(blocked)/len(base_f)*100:.1f}%)")
    print(f"  blocked trades net (post-friction): {blocked.pnl_net.sum():+,.0f}  "
          f"PF={pf(blocked.pnl_net):.2f}  WR={(blocked.pnl_net>0).mean()*100:.1f}%")

    summarise(opt2, "OPTION 2 (all trades)")
    summarise(opt1, "OPTION 1 (blocked drop)")

    lift = (opt2.pnl_net.sum() - opt1.pnl_net.sum()) / opt1.pnl_net.sum() * 100
    print(f"\n  Option 2 uplift vs Option 1: {lift:+.1f}%  "
          f"({opt2.pnl_net.sum() - opt1.pnl_net.sum():+,.0f} pts over 18yr)")

    # Per-instrument breakdown
    print(f"\n  {'instrument':<10} {'blocked n':>10} {'blocked net':>14} {'blocked PF':>12}")
    for inst in ["DAX", "US30", "NIKKEI"]:
        b = base_f[(base_f.blocked_by_sibling) & (base_f.instrument == inst)]
        if len(b) == 0:
            print(f"  {inst:<10} {'0':>10} {'—':>14} {'—':>12}")
        else:
            print(f"  {inst:<10} {len(b):>10,} {b.pnl_net.sum():>+14,.0f} {pf(b.pnl_net):>12.2f}")

    # Per-session
    print(f"\n  Blocked trades by (instrument, session):")
    bsum = base_f[base_f.blocked_by_sibling].groupby(["instrument", "session"]).agg(
        n=("pnl_net", "count"),
        net=("pnl_net", "sum"),
    )
    print(bsum.to_string())

    print(f"\n  INTERPRETATION:")
    if lift > 10:
        print(f"  Option 2 adds {lift:.1f}% meaningful — worth shipping after careful review.")
    elif lift > 3:
        print(f"  Option 2 adds {lift:.1f}% marginal — consider but not urgent.")
    else:
        print(f"  Option 2 adds {lift:.1f}% — not worth the complexity/risk.")


if __name__ == "__main__":
    main()
