"""
backtest_reentry_gate.py — rolling 30-day win-rate of reverse-reentry rules.

For each instrument, compare:
  ON  = current config (DAX never, US30 never, NIKKEI after_loss)
  OFF = always allow opposite re-entries

Compute per-day delta = ON - OFF (i.e. how much the RULE contributed).
Roll 30-day windows, count how many windows are positive vs negative.

Goal: show that even bad patches (worst 30-day window) don't invalidate the rule.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

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


def filter_rule(df, allow_all=False):
    """
    If allow_all=False: apply current config (DAX never, US30 never, NIKKEI after_loss).
    If allow_all=True: no filter on opposite re-entries.
    """
    df = df.sort_values(["date", "signal", "entry"]).reset_index(drop=True)
    df["tn"] = df.groupby(["date", "signal"]).cumcount() + 1
    first = df[df["tn"] == 1][["date", "signal", "direction", "pnl_pts"]].rename(
        columns={"direction": "fd", "pnl_pts": "fp"}
    )
    df = df.merge(first, on=["date", "signal"], how="left")
    df["first_won"] = df["fp"] > 0
    df["is_re"] = df["tn"] > 1
    df["same"] = df["direction"] == df["fd"]
    if allow_all:
        return df.copy()
    opp = df["is_re"] & ~df["same"]
    drop = opp & (df["instrument"].isin(["DAX", "US30"])
                  | ((df["instrument"] == "NIKKEI") & df["first_won"]))
    return df[~drop].copy()


def fade_filter(df, fade_min_winner=20):
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


def daily_net(filt_df, inst=None):
    """Sum pnl_net per date. Optionally filter to instrument."""
    if inst:
        filt_df = filt_df[filt_df.instrument == inst]
    d = filt_df.groupby(pd.to_datetime(filt_df["date"]).dt.date)["pnl_net"].sum()
    return d


def rolling_analysis(delta: pd.Series, window: int = 30):
    """Rolling sum + summary."""
    delta = delta.sort_index()
    # Fill missing dates with 0 so rolling windows count actual days
    all_dates = pd.date_range(delta.index.min(), delta.index.max(), freq="D")
    delta = delta.reindex(all_dates, fill_value=0.0)
    roll = delta.rolling(window).sum()
    roll = roll.dropna()
    pos = (roll > 0).sum()
    neg = (roll < 0).sum()
    zero = (roll == 0).sum()
    worst = roll.min()
    best = roll.max()
    median = roll.median()
    mean = roll.mean()
    worst_date = roll.idxmin()
    return {
        "n_windows": len(roll),
        "pos_pct": pos / len(roll) * 100,
        "neg_pct": neg / len(roll) * 100,
        "mean": mean, "median": median, "best": best, "worst": worst,
        "worst_date": worst_date,
    }


def main():
    df = pd.read_parquet(PARQUET)
    df = df.reset_index(drop=True)
    df = fade_filter(df, fade_min_winner=20)

    # Apply friction first (rule doesn't change friction, so we apply once)
    df_on = apply_friction(filter_rule(df, allow_all=False))
    df_off = apply_friction(filter_rule(df, allow_all=True))

    print(f"{'='*100}")
    print(f"  30-day rolling analysis: RULE ON minus RULE OFF (positive = rule HELPS that window)")
    print(f"{'='*100}\n")
    print(f"  {'Instrument':<10} {'ON 18yr':>12} {'OFF 18yr':>12} {'rule net':>12}  "
          f"{'% pos 30d':>10} {'% neg 30d':>10}  {'worst 30d':>12} ({'date':<12}) {'best 30d':>10}  {'median 30d':>12}")

    for inst in ["DAX", "US30", "NIKKEI"]:
        on_daily = daily_net(df_on, inst=inst)
        off_daily = daily_net(df_off, inst=inst)
        # Align indexes
        all_dates = sorted(set(on_daily.index) | set(off_daily.index))
        on_full = on_daily.reindex(all_dates, fill_value=0)
        off_full = off_daily.reindex(all_dates, fill_value=0)
        delta = on_full - off_full   # positive = rule ON adds value

        r = rolling_analysis(delta, window=30)
        on_total = on_full.sum()
        off_total = off_full.sum()
        rule_net = on_total - off_total
        print(f"  {inst:<10} {on_total:>+12,.0f} {off_total:>+12,.0f} {rule_net:>+12,.0f}  "
              f"{r['pos_pct']:>9.1f}% {r['neg_pct']:>9.1f}%  "
              f"{r['worst']:>+12,.0f} ({str(r['worst_date'].date()):<12}) "
              f"{r['best']:>+10,.0f}  {r['median']:>+12,.0f}")

    # Also do total portfolio
    on_total_series = daily_net(df_on)
    off_total_series = daily_net(df_off)
    all_dates = sorted(set(on_total_series.index) | set(off_total_series.index))
    on_full = on_total_series.reindex(all_dates, fill_value=0)
    off_full = off_total_series.reindex(all_dates, fill_value=0)
    delta = on_full - off_full
    r = rolling_analysis(delta, window=30)
    rule_net = on_full.sum() - off_full.sum()
    print(f"\n  {'ALL':<10} {on_full.sum():>+12,.0f} {off_full.sum():>+12,.0f} {rule_net:>+12,.0f}  "
          f"{r['pos_pct']:>9.1f}% {r['neg_pct']:>9.1f}%  "
          f"{r['worst']:>+12,.0f} ({str(r['worst_date'].date()):<12}) "
          f"{r['best']:>+10,.0f}  {r['median']:>+12,.0f}")

    # Interpretation
    print(f"\n{'='*100}")
    print(f"  INTERPRETATION")
    print(f"{'='*100}")
    print(f"  - Positive 30d-window % shows how often the rule HELPED that 30-day period.")
    print(f"  - 'worst 30d' = the period where removing the rule would have paid off most.")
    print(f"  - If worst 30d is small relative to average, rule is robust.")


if __name__ == "__main__":
    main()
