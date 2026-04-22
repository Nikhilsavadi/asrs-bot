"""
extract_ig_bars_from_logs.py — scrape IG bars from bot logs into sidecar CSVs.

Parses lines like:
  2026-04-15 06:25:00 [INFO] shared.ig_stream -- Bar built (IX.D.DAX.DAILY.IP):
    07:15-07:20 CET | O=24003.9 H=24003.9 L=23980.4 C=23981.9

and writes /app/data/ig_bars/<epic>_<date>.csv with columns:
  bar_start_cet, open, high, low, close

This gives the backtest an LTP-based bar source that matches what the
live bot uses (instead of tick-reconstructed mid-price bars).
"""
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

LOG_DIR = Path("/run/asrs/logs")
OUT_DIR = Path("/app/data/ig_bars")

PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*?Bar built \(([^)]+)\): "
    r"(\d{2}:\d{2})-(\d{2}:\d{2}) CET \| O=([\d.]+) H=([\d.]+) L=([\d.]+) C=([\d.]+)"
)

# Tick-bar lines also appear — collect both to dedup by (epic, bar_start).
PATTERN_TICK = re.compile(
    r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*?Tick-bar \(([^)]+)\): "
    r"(\d{2}:\d{2})-(\d{2}:\d{2}) CET \| O=([\d.]+) H=([\d.]+) L=([\d.]+) C=([\d.]+)"
)

CET = ZoneInfo("Europe/Berlin")


def parse_log(path: Path, bars: dict):
    """Add bars from one log file into shared bars dict.
    bars[(epic, bar_start_cet)] = (o, h, l, c, source) — 'ig' preferred over 'tick'.
    """
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            m = PATTERN.match(line)
            source = "ig"
            if not m:
                m = PATTERN_TICK.match(line)
                source = "tick"
            if not m:
                continue
            log_date, log_time, epic, bs, be, o, h, l, c = m.groups()
            # Convert bar_start from log-date + bs to a CET datetime.
            # Log date is UTC/BST server time; the bs HH:MM is CET (per log format).
            # To get the right date for the bar, combine log_date with bs but be
            # careful about day boundary. Simpler: assume bar is in the same CET
            # date as the log line's UTC date rolled forward appropriately.
            # For our instruments (DAX/US30/NIKKEI), sessions don't cross CET midnight
            # so using log_date + CET time works fine.
            bar_start = datetime.strptime(f"{log_date} {bs}", "%Y-%m-%d %H:%M")
            bar_start = bar_start.replace(tzinfo=CET)
            key = (epic, bar_start.isoformat())
            # Prefer 'ig' (candle subscription) over 'tick' (our builder).
            # Both are the SAME bar — 'ig' is IG's official LTP-based OHLC.
            if key in bars and bars[key][4] == "ig":
                continue  # keep existing ig entry
            bars[key] = (float(o), float(h), float(l), float(c), source)


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    bars = {}
    for p in [LOG_DIR / "asrs.log.3", LOG_DIR / "asrs.log.2",
              LOG_DIR / "asrs.log.1", LOG_DIR / "asrs.log"]:
        parse_log(p, bars)
    print(f"Collected {len(bars):,} bars across all logs")

    # Group by (epic, date) → write one CSV per day per epic
    by_file: dict = {}
    for (epic, bar_iso), (o, h, l, c, src) in bars.items():
        bar_start = datetime.fromisoformat(bar_iso)
        date_str = bar_start.astimezone(CET).date().isoformat()
        fname = OUT_DIR / f"{epic}_{date_str}.csv"
        by_file.setdefault(fname, []).append((bar_start, o, h, l, c, src))

    for fname, rows in by_file.items():
        rows.sort(key=lambda r: r[0])
        with open(fname, "w") as f:
            f.write("bar_start_cet,open,high,low,close,source\n")
            for bar_start, o, h, l, c, src in rows:
                f.write(f"{bar_start.isoformat()},{o},{h},{l},{c},{src}\n")
        ig_count = sum(1 for r in rows if r[5] == "ig")
        print(f"  {fname.name}: {len(rows):>4} bars ({ig_count} ig, {len(rows)-ig_count} tick-only)")


if __name__ == "__main__":
    main()
