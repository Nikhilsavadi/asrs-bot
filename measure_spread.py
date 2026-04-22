"""
measure_spread.py — measure actual IG spread from tick data.
Verifies whether the 15pt NIKKEI assumption is realistic.
"""
import os, pandas as pd
from zoneinfo import ZoneInfo

TICK_DIR = "/root/asrs-bot/data/ticks"

INSTRUMENTS = {
    "DAX":    {"epic": "IX.D.DAX.DAILY.IP",    "tz": "Europe/Berlin",
               "sess_start": (9, 0),  "sess_end": (17, 30)},
    "US30":   {"epic": "IX.D.DOW.DAILY.IP",    "tz": "America/New_York",
               "sess_start": (9, 30), "sess_end": (16, 0)},
    "NIKKEI": {"epic": "IX.D.NIKKEI.DAILY.IP", "tz": "Asia/Tokyo",
               "sess_start": (10, 0), "sess_end": (15, 0)},
}


def main():
    for inst, cfg in INSTRUMENTS.items():
        tz = ZoneInfo(cfg["tz"])
        epic = cfg["epic"]
        files = sorted([f for f in os.listdir(TICK_DIR) if f.startswith(epic)])
        if not files: continue
        spreads = []
        for f in files:
            df = pd.read_csv(f"{TICK_DIR}/{f}")
            df = df.dropna(subset=["utm", "bid", "ofr"])
            df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
            df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
            df = df.dropna(subset=["bid", "ofr"])
            df["spread"] = df["ofr"] - df["bid"]
            df["dt"] = pd.to_datetime(df["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(tz)
            df["hm"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
            ss = cfg["sess_start"][0]*60 + cfg["sess_start"][1]
            se = cfg["sess_end"][0]*60 + cfg["sess_end"][1]
            spreads.append(df[(df["hm"] >= ss) & (df["hm"] < se)]["spread"])
        s = pd.concat(spreads)
        print(f"\n{'='*70}\n  {inst} ({epic}) — {len(files)} days, {len(s):,} session ticks\n{'='*70}")
        print(f"    mean:    {s.mean():>6.2f}pt")
        print(f"    median:  {s.median():>6.2f}pt")
        print(f"    p25:     {s.quantile(0.25):>6.2f}pt")
        print(f"    p75:     {s.quantile(0.75):>6.2f}pt")
        print(f"    p95:     {s.quantile(0.95):>6.2f}pt")
        print(f"    max:     {s.max():>6.2f}pt")
        for t in [1, 2, 3, 5, 10, 20, 50]:
            pct = (s <= t).mean() * 100
            print(f"    ≤{t:>2}pt:  {pct:>5.1f}% of ticks")

    print(f"\n\n{'='*70}\n  NIKKEI by hour (Tokyo time)\n{'='*70}")
    epic = "IX.D.NIKKEI.DAILY.IP"
    tz = ZoneInfo("Asia/Tokyo")
    all_t = []
    for f in sorted([x for x in os.listdir(TICK_DIR) if x.startswith(epic)]):
        df = pd.read_csv(f"{TICK_DIR}/{f}")
        df = df.dropna(subset=["utm", "bid", "ofr"])
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df["ofr"] = pd.to_numeric(df["ofr"], errors="coerce")
        df = df.dropna(subset=["bid", "ofr"])
        df["spread"] = df["ofr"] - df["bid"]
        df["dt"] = pd.to_datetime(df["utm"].astype("int64"), unit="ms", utc=True).dt.tz_convert(tz)
        df["hour"] = df["dt"].dt.hour
        all_t.append(df[["hour", "spread"]])
    ha = pd.concat(all_t)
    print(f"  {'hour':<6} {'n':<10} {'mean':<8} {'median':<8} {'p95':<8}")
    for h in sorted(ha["hour"].unique()):
        d = ha[ha["hour"] == h]
        print(f"  {h:<6} {len(d):<10,} {d['spread'].mean():<6.2f}   "
              f"{d['spread'].median():<6.2f}   {d['spread'].quantile(0.95):<6.2f}")


if __name__ == "__main__":
    main()
