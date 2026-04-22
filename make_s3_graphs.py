"""
Generate 3 graphs from the S3-inclusive parquet:
  - mc_dual_with_s3.png: household equity curves (single + dual, stressed + unstressed)
  - mc_withdraw_150k_s3.png: 15yr £150k/yr withdrawal survival
  - mc_year1_monthly_s3.png: month-by-month Y1 bands
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import backtest as bt
from backtest_v2_atr import filter_combined
for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

SPREAD = {"DAX": 1.5, "US30": 2.75, "NIKKEI": 7.0}
SLIP = {"DAX": 1.0, "US30": 3.0, "NIKKEI": 4.0}
PARQUET = "/root/asrs-bot/data/variant_b_fade_trades_s3.parquet"
UPLIFT = 1.06

START = 2000
WIFE_SEED = 10_000
WIFE_DAY = 252
MIN_STAKE = 0.50
MAX_STAKE = 35.0
DAYS_PER_YEAR = 252
RUNS = 5000

WITHDRAW = 150_000
TRIGGER = 500_000


def fmt(x, _=None):
    if x <= 0: return "£0"
    if x < 1000: return f"£{x:,.0f}"
    if x < 1_000_000: return f"£{x/1000:.0f}k"
    return f"£{x/1_000_000:.1f}M"


def apply_friction(df):
    df = df.copy()
    df["spread_cost"] = df["instrument"].map(SPREAD)
    df["slip_cost"] = np.where(df["pnl_pts"] < 0, df["instrument"].map(SLIP), 0)
    df["pnl_net"] = (df["pnl_pts"] - df["spread_cost"] - df["slip_cost"]) * UPLIFT
    return df


def build_daily():
    df = pd.read_parquet(PARQUET)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False).astype(bool)
    df = df.reset_index(drop=True)
    df["_base_signal"] = df["signal"].str.replace("_FADE", "", regex=False)
    orig_win = {}
    for (date, sig), group in df.groupby(["date", "_base_signal"], sort=False):
        q = group[(~group.is_fade) & (group.reason == "TRAIL_WIN") & (group.pnl_pts > 0)]["pnl_pts"].tolist()
        fades = group[group.is_fade]
        for i, (idx, _) in enumerate(fades.iterrows()):
            if i < len(q): orig_win[idx] = q[i]
    fade = df[df.is_fade].copy()
    fade["orig_win"] = fade.index.map(orig_win)
    fade_f = fade[fade["orig_win"].fillna(0) >= 20].copy()
    base = df[~df.is_fade].copy()
    base_f = filter_combined(base)
    base_f = apply_friction(base_f)
    fade_f = apply_friction(fade_f)
    combined = pd.concat([base_f, fade_f], ignore_index=True)
    daily = combined.groupby(pd.to_datetime(combined["date"]).dt.date)["pnl_net"].sum().values
    return daily


def sim_single(daily, seed, days, stress, start=START):
    rng = np.random.default_rng(seed)
    sample = rng.choice(daily, size=days, replace=True) * stress
    eq = np.empty(days); val = start
    for i in range(days):
        stk = max(MIN_STAKE, min(MAX_STAKE, val * 1e-4))
        val = max(0, val + sample[i] * stk)
        eq[i] = val
    return eq


def sim_dual(daily, seed, days, stress, seed_day=WIFE_DAY, seed_amt=WIFE_SEED,
             withdraw_per_year=0, withdraw_trigger=0):
    rng = np.random.default_rng(seed)
    sm = rng.choice(daily, size=days, replace=True) * stress
    rng2 = np.random.default_rng(seed + 100_000)
    sw = rng2.choice(daily, size=days, replace=True) * stress
    main, wife = START, 0.0
    eq = np.empty(days)
    withdrawing = False
    wd_days = set(y * DAYS_PER_YEAR - 1 for y in range(1, days // DAYS_PER_YEAR + 1))
    for i in range(days):
        if i == seed_day:
            main = max(0, main - seed_amt); wife = seed_amt
        if main > 0:
            stk = max(MIN_STAKE, min(MAX_STAKE, main * 1e-4))
            main = max(0, main + sm[i] * stk)
        if i >= seed_day and wife > 0:
            stk = max(MIN_STAKE, min(MAX_STAKE, wife * 1e-4))
            wife = max(0, wife + sw[i] * stk)
        if withdraw_per_year > 0 and i in wd_days:
            household = main + wife
            if household >= withdraw_trigger: withdrawing = True
            if withdrawing and household > withdraw_per_year * 1.5:
                if household > 0:
                    fm = main / (main + wife) if (main + wife) > 0 else 0.5
                    main -= withdraw_per_year * fm
                    wife -= withdraw_per_year * (1 - fm)
                main = max(0, main); wife = max(0, wife)
        eq[i] = main + wife
    return eq


def pct(eqs, p):
    return np.percentile(eqs, p, axis=0)


def graph_monthly_y1(daily):
    days = 12 * 21
    eqs_u = np.array([sim_single(daily, r, days, 1.0) for r in range(RUNS)])
    eqs_s = np.array([sim_single(daily, r, days, 0.5) for r in range(RUNS)])
    x = np.arange(1, days + 1) / 21  # months

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, eqs, title in [(ax1, eqs_u, "UNSTRESSED (backtest repeats)"),
                            (ax2, eqs_s, "STRESSED 50% (realistic)")]:
        p5, p50, p95 = pct(eqs, 5), pct(eqs, 50), pct(eqs, 95)
        ax.fill_between(x, p5, p95, color="blue", alpha=0.15, label="p5-p95")
        ax.plot(x, p50, color="black", linewidth=2.5, label=f"MEDIAN → {fmt(p50[-1])}")
        ax.plot(x, p5, color="red", linewidth=1.5, label=f"WORST p5 → {fmt(p5[-1])}")
        ax.plot(x, p95, color="blue", linewidth=1.5, label=f"BEST p95 → {fmt(p95[-1])}")
        ax.axhline(5000, color="grey", linestyle=":", label="£5k compound threshold")
        ax.axhline(50000, color="orange", linestyle=":", alpha=0.5, label="£50k cruise")
        ax.set_title(f"{title} — START £{START:,} single account")
        ax.set_xlabel("Month"); ax.set_ylabel("Equity")
        ax.set_xticks(range(0, 13))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("Year 1 Month-by-Month — from £2k, compound 1e-4, £35/pt cap, with S3",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_year1_monthly_s3.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def graph_dual(daily):
    days = 4 * DAYS_PER_YEAR
    x = np.arange(1, days + 1) / DAYS_PER_YEAR

    scenarios = [
        ("SINGLE unstressed",            "single", 1.0, "C0"),
        ("SINGLE 50% stress",            "single", 0.5, "C1"),
        ("DUAL unstressed",              "dual",   1.0, "C2"),
        ("DUAL 50% stress",              "dual",   0.5, "C3"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (lbl, kind, stress, colour) in zip(axes.flat, scenarios):
        if kind == "single":
            eqs = np.array([sim_single(daily, r, days, stress) for r in range(RUNS)])
        else:
            eqs = np.array([sim_dual(daily, r, days, stress) for r in range(RUNS)])
        p5, p50, p95 = pct(eqs, 5), pct(eqs, 50), pct(eqs, 95)
        ax.fill_between(x, p5, p95, color=colour, alpha=0.15)
        ax.plot(x, p5,  color="red",   linewidth=1.3, label=f"WORST p5 → {fmt(p5[-1])}")
        ax.plot(x, p50, color="black", linewidth=2.5, label=f"MEDIAN → {fmt(p50[-1])}")
        ax.plot(x, p95, color="blue",  linewidth=1.3, label=f"BEST p95 → {fmt(p95[-1])}")
        ax.set_title(lbl)
        ax.set_xlabel("Year"); ax.set_ylabel("Equity")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    plt.suptitle("4-Year Projection — from £2k (dual adds wife £10k seed at Y2), with S3",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "/root/asrs-bot/mc_dual_with_s3.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def graph_withdraw(daily):
    days = 15 * DAYS_PER_YEAR
    x = np.arange(1, days + 1) / DAYS_PER_YEAR
    eqs_stressed = np.array([
        sim_dual(daily, r, days, 0.5,
                 withdraw_per_year=WITHDRAW, withdraw_trigger=TRIGGER)
        for r in range(RUNS)
    ])
    p5, p50, p95 = pct(eqs_stressed, 5), pct(eqs_stressed, 50), pct(eqs_stressed, 95)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.fill_between(x, p5, p95, color="green", alpha=0.12, label="p5-p95 range")
    ax.plot(x, p5,  color="red",   linewidth=1.5, label=f"WORST p5 Y15 → {fmt(p5[-1])}")
    ax.plot(x, p50, color="black", linewidth=2.5, label=f"MEDIAN Y15 → {fmt(p50[-1])}")
    ax.plot(x, p95, color="blue",  linewidth=1.5, label=f"BEST p95 Y15 → {fmt(p95[-1])}")
    ax.axhline(TRIGGER, color="orange", linestyle="--",
               label=f"Withdrawal trigger £{TRIGGER:,}")
    # Note first withdrawal year
    for y in range(2, 16):
        ax.axvline(y, color="grey", linestyle=":", alpha=0.25)

    ax.set_title(f"15-Year £{WITHDRAW:,}/yr Withdrawal — Dual £2k + £10k Y2, 50% stress, WITH S3",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Household equity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    out = "/root/asrs-bot/mc_withdraw_150k_s3.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def main():
    print("Loading daily samples from S3 parquet...")
    daily = build_daily()
    print(f"  n={len(daily)} days  mean={daily.mean():+.1f}pt  worst={daily.min():+.0f}  best={daily.max():+.0f}")

    print("\nGenerating graphs...")
    graph_monthly_y1(daily)
    graph_dual(daily)
    graph_withdraw(daily)
    print("\nDone.")


if __name__ == "__main__":
    main()
