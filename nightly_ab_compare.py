"""
nightly_ab_compare.py — A (old: prev_low + tight_threshold) vs B (live: prev_close always).

Runs 18yr firstrate backtest for both variants, posts per-instrument PF summary
to Telegram. Intended for cron at ~22:00 UTC daily.
"""
import os, sys, time, requests, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env (handles special chars like parens in passwords)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

import backtest_v2_atr as bva

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


def send_tg(msg: str):
    if not (TG_TOKEN and TG_CHAT):
        print(msg); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"TG send failed: {e}\n{msg}")


def stats_row(d):
    if len(d) == 0: return "n=0"
    w = d[d["pnl_pts"] > 0]; l = d[d["pnl_pts"] < 0]
    pf = w.pnl_pts.sum() / max(abs(l.pnl_pts.sum()), 0.001)
    return f"n={len(d):,} PF={pf:.2f} net={d.pnl_pts.sum():+,.0f}"


def main():
    t0 = time.time()
    cache = bva.build_cache()

    a = bva.filter_combined(pd.DataFrame(bva.run(cache, trail_close=False, mfe_mode=False)))
    b = bva.filter_combined(pd.DataFrame(bva.run(cache, trail_close=True, mfe_mode=False)))
    a["year"] = pd.to_datetime(a["date"]).dt.year
    b["year"] = pd.to_datetime(b["date"]).dt.year

    lines = [
        "*Nightly A/B Backtest* (18yr firstrate)",
        f"A = OLD: prev_low + tight_threshold",
        f"B = LIVE: prev_close always",
        "",
        f"*FULL*",
        f"  A: {stats_row(a)}",
        f"  B: {stats_row(b)}",
    ]
    for inst in ["DAX", "US30", "NIKKEI"]:
        lines += [
            f"*{inst}*",
            f"  A: {stats_row(a[a['instrument']==inst])}",
            f"  B: {stats_row(b[b['instrument']==inst])}",
        ]

    # MC 4yr £0.50/pt flat — key DD comparison
    rng = np.random.default_rng(42)
    lines += ["", "*MC 4yr £0.50/pt (p50 final £ / max DD)*"]
    for lbl, d in [("A", a), ("B", b)]:
        daily = d.groupby(pd.to_datetime(d["date"]).dt.date)["pnl_pts"].sum()
        eq = bva.mc_sim(daily, rng)
        final = eq[:, -1]
        dd = ((np.maximum.accumulate(eq, axis=1) - eq) / np.maximum.accumulate(eq, axis=1)).max(axis=1) * 100
        lines.append(f"  {lbl}: £{np.percentile(final,50):,.0f} / {np.percentile(dd,50):.2f}%")

    lines.append(f"\n_elapsed: {time.time()-t0:.0f}s_")
    msg = "\n".join(lines)
    print(msg)
    send_tg(msg)


if __name__ == "__main__":
    main()
