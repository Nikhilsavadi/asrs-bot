"""
Rebuild variant_b_fade_trades.parquet with S3 included for US30 + NIKKEI.
Same Variant B + fade logic as before — only difference: extra session.
"""
import os, sys, time, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest as bt
from backtest_v2_fade import build_cache, sim_with_fade

TRADES_CACHE = "/root/asrs-bot/data/variant_b_fade_trades_s3.parquet"

for _c in bt.INSTRUMENTS.values(): _c["max_entries"] = 3

# Add S3 configs from live asrs/config.py
bt.INSTRUMENTS["US30"]["s3_open_hour"] = 13
bt.INSTRUMENTS["US30"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["s3_open_hour"] = 13
bt.INSTRUMENTS["NIKKEI"]["s3_open_minute"] = 0
bt.INSTRUMENTS["NIKKEI"]["session_end_hour"] = 15
bt.INSTRUMENTS["NIKKEI"]["session_end_minute"] = 0


def main():
    t0 = time.time()
    print("Building cache ...", flush=True)
    cache = build_cache()
    print(f"Cache built in {time.time()-t0:.0f}s. Simulating...", flush=True)
    all_trades = []
    for inst, cached in cache.items():
        cfg = bt.INSTRUMENTS[inst]
        print(f"  {inst}: sessions {cached['sessions']}", flush=True)
        for s in cached["sessions"]:
            oh = cfg[f"s{s}_open_hour"]; om = cfg[f"s{s}_open_minute"]
            eh = cfg["session_end_hour"]; em = cfg["session_end_minute"]
            name = f"{inst}_S{s}"
            for d in cached["dates"]:
                d5 = cached["by5"].get(d)
                if d5 is None: continue
                o5, h5, m5 = d5
                d1 = cached["by1"].get(d)
                o1, h1, min1 = d1 if d1 else (None, None, None)
                ts = sim_with_fade(
                    o5, h5, m5, oh, om, eh, em, cfg,
                    day_1min=o1, one_min_hours=h1, one_min_minutes=min1,
                    fade_stop=50, fade_target_mode="extreme",
                )
                for t in ts:
                    t["date"] = str(d); t["instrument"] = inst
                    t["signal"] = f"{name}_FADE" if t.get("is_fade") else name
                    all_trades.append(t)
    df = pd.DataFrame(all_trades)
    if "is_fade" not in df.columns: df["is_fade"] = False
    df["is_fade"] = df["is_fade"].fillna(False)
    df.to_parquet(TRADES_CACHE)
    print(f"\n  Wrote {len(df):,} trades → {TRADES_CACHE}")
    print(f"  Signal distribution:")
    for sig, n in df.groupby("signal").size().items():
        print(f"    {sig}: {n:,}")
    print(f"\n  Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
