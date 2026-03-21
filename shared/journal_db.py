"""
journal_db.py — Unified SQLite Trade Journal for DAX + FTSE
═══════════════════════════════════════════════════════════════

Single source of truth for all trades. Replaces per-bot CSV journals.
Tracks cumulative P&L and scaling ladder (£3k banked → +£1/pt).
"""

import csv
import os
import sqlite3
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trade_journal.db")
TZ_UK = ZoneInfo("Europe/London")

_conn: sqlite3.Connection | None = None


# ── Connection ────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument      TEXT NOT NULL,
            date            TEXT NOT NULL,
            trade_num       INTEGER,
            direction       TEXT,
            entry_price     REAL,
            exit_price      REAL,
            entry_intended  REAL DEFAULT 0,
            exit_intended   REAL DEFAULT 0,
            entry_time      TEXT DEFAULT '',
            exit_time       TEXT DEFAULT '',
            pnl_pts         REAL DEFAULT 0,
            pnl_gbp         REAL DEFAULT 0,
            contracts       INTEGER DEFAULT 1,
            contracts_stopped INTEGER DEFAULT 0,
            adds_used       INTEGER DEFAULT 0,
            add_pnl_pts     REAL DEFAULT 0,
            entry_slippage  REAL DEFAULT 0,
            exit_slippage   REAL DEFAULT 0,
            slippage_total  REAL DEFAULT 0,
            tp1_filled      INTEGER DEFAULT 0,
            tp2_filled      INTEGER DEFAULT 0,
            tp1_slippage    REAL DEFAULT 0,
            tp2_slippage    REAL DEFAULT 0,
            mfe             REAL DEFAULT 0,
            bar_range       REAL DEFAULT 0,
            range_flag      TEXT DEFAULT '',
            bar_type        TEXT DEFAULT '',
            signal_bar      INTEGER DEFAULT 4,
            bar5_rule       TEXT DEFAULT '',
            gap_dir         TEXT DEFAULT '',
            overnight_bias  TEXT DEFAULT '',
            exit_reason     TEXT DEFAULT '',
            stake_per_point REAL DEFAULT 1,
            cumulative_pnl  REAL DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scaling_ladder (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument      TEXT NOT NULL,
            threshold_gbp   REAL NOT NULL,
            stake_per_point REAL NOT NULL,
            activated_at    TEXT,
            activated       INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
        CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument);
    """)

    # ── Extended columns for AI analysis (added via ALTER TABLE) ──
    extended_cols = [
        # Market context at entry
        ("signal_type", "TEXT DEFAULT ''"),        # BRACKET, REENTRY, S2
        ("session", "TEXT DEFAULT ''"),            # S1, S2
        ("spx_regime", "TEXT DEFAULT ''"),         # UP, DOWN, FLAT
        ("overnight_high", "REAL DEFAULT 0"),
        ("overnight_low", "REAL DEFAULT 0"),
        ("overnight_range", "REAL DEFAULT 0"),
        ("prev_day_close", "REAL DEFAULT 0"),
        ("gap_pts", "REAL DEFAULT 0"),
        ("gap_pct", "REAL DEFAULT 0"),
        # Bar data at signal time
        ("bar4_open", "REAL DEFAULT 0"),
        ("bar4_high", "REAL DEFAULT 0"),
        ("bar4_low", "REAL DEFAULT 0"),
        ("bar4_close", "REAL DEFAULT 0"),
        ("bar4_body_pct", "REAL DEFAULT 0"),       # abs(close-open)/range
        ("bar4_upper_wick_pct", "REAL DEFAULT 0"),
        ("bar4_lower_wick_pct", "REAL DEFAULT 0"),
        # Trade execution detail
        ("entry_spread", "REAL DEFAULT 0"),         # bid-offer spread at entry
        ("exit_spread", "REAL DEFAULT 0"),
        ("time_in_trade_mins", "REAL DEFAULT 0"),
        ("bars_in_trade", "INTEGER DEFAULT 0"),
        # Price action during trade
        ("mae", "REAL DEFAULT 0"),                  # Max Adverse Excursion
        ("mfe_time_mins", "REAL DEFAULT 0"),        # Time to reach MFE
        ("breakeven_hit", "INTEGER DEFAULT 0"),
        ("breakeven_time_mins", "REAL DEFAULT 0"),
        # Adds detail
        ("add1_price", "REAL DEFAULT 0"),
        ("add1_time", "TEXT DEFAULT ''"),
        ("add2_price", "REAL DEFAULT 0"),
        ("add2_time", "TEXT DEFAULT ''"),
        # Market conditions
        ("day_of_week", "INTEGER DEFAULT 0"),       # 0=Mon, 4=Fri
        ("hour_of_entry", "INTEGER DEFAULT 0"),
        ("ig_spread_at_entry", "REAL DEFAULT 0"),
        # Tick data summary
        ("tick_count_during_trade", "INTEGER DEFAULT 0"),
        ("avg_tick_interval_ms", "REAL DEFAULT 0"),
    ]

    for col_name, col_def in extended_cols:
        try:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # ── Tick log table: raw price data during active trades ──
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tick_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id    INTEGER,
            instrument  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            bid         REAL,
            offer       REAL,
            mid         REAL,
            spread      REAL,
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );
        CREATE INDEX IF NOT EXISTS idx_tick_trade ON tick_log(trade_id);

        CREATE TABLE IF NOT EXISTS bar_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id    INTEGER,
            instrument  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            bar_num     INTEGER,
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );
        CREATE INDEX IF NOT EXISTS idx_bar_trade ON bar_log(trade_id);

        CREATE TABLE IF NOT EXISTS daily_context (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            instrument      TEXT NOT NULL,
            prev_close      REAL DEFAULT 0,
            open_price      REAL DEFAULT 0,
            gap_pts         REAL DEFAULT 0,
            overnight_high  REAL DEFAULT 0,
            overnight_low   REAL DEFAULT 0,
            overnight_range REAL DEFAULT 0,
            spx_prev_close  REAL DEFAULT 0,
            spx_prev_dir    TEXT DEFAULT '',
            vix_level       REAL DEFAULT 0,
            day_of_week     INTEGER DEFAULT 0,
            bar1_range      REAL DEFAULT 0,
            bar2_range      REAL DEFAULT 0,
            bar3_range      REAL DEFAULT 0,
            bar4_range      REAL DEFAULT 0,
            bar5_range      REAL DEFAULT 0,
            morning_trend   TEXT DEFAULT '',
            UNIQUE(date, instrument)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_context(date);
    """)

    conn.commit()
    logger.info(f"Trade journal DB initialized: {DB_PATH}")


def seed_scaling_ladder(ladder_str: str = "", instrument: str = "ALL"):
    """
    Seed scaling ladder from env string like '3000:2,6000:3,9000:4'.
    Only inserts if table is empty for that instrument.
    """
    conn = _get_conn()
    existing = conn.execute(
        "SELECT COUNT(*) FROM scaling_ladder WHERE instrument=?", (instrument,)
    ).fetchone()[0]
    if existing > 0:
        return

    if not ladder_str:
        # Default: +£1/pt every £3,000
        ladder_str = "3000:2,6000:3,9000:4,12000:5,15000:6"

    for entry in ladder_str.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        threshold, stake = entry.split(":", 1)
        conn.execute(
            "INSERT INTO scaling_ladder (instrument, threshold_gbp, stake_per_point) VALUES (?, ?, ?)",
            (instrument, float(threshold), float(stake)),
        )
    conn.commit()
    logger.info(f"Scaling ladder seeded for {instrument}: {ladder_str}")


# ── Insert ────────────────────────────────────────────────────────────────

def insert_trade(instrument: str, trade: dict, state=None) -> int:
    """
    Insert a completed trade. Returns the new row ID.
    Called by dax_bot/journal.py and ftse_bot/journal.py.
    """
    conn = _get_conn()

    # Compute cumulative P&L
    row = conn.execute("SELECT COALESCE(SUM(pnl_gbp), 0) FROM trades").fetchone()
    cum_pnl = row[0] if row else 0
    pnl_gbp = trade.get("pnl_gbp", 0)
    if pnl_gbp == 0 and trade.get("pnl_pts"):
        # Estimate GBP P&L from pts * stake
        stake = trade.get("stake_per_point", 1)
        pnl_gbp = round(float(trade.get("pnl_pts", 0)) * stake, 2)
    cum_pnl += pnl_gbp

    date = ""
    if state and hasattr(state, "date"):
        date = state.date
    if not date:
        date = datetime.now(TZ_UK).strftime("%Y-%m-%d")

    cursor = conn.execute("""
        INSERT INTO trades (
            instrument, date, trade_num, direction,
            entry_price, exit_price, entry_intended, exit_intended,
            entry_time, exit_time,
            pnl_pts, pnl_gbp, contracts, contracts_stopped,
            adds_used, add_pnl_pts,
            entry_slippage, exit_slippage, slippage_total,
            tp1_filled, tp2_filled, tp1_slippage, tp2_slippage,
            mfe, bar_range, range_flag, bar_type,
            signal_bar, bar5_rule, gap_dir, overnight_bias,
            exit_reason, stake_per_point, cumulative_pnl
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?
        )
    """, (
        instrument,
        date,
        trade.get("num", trade.get("trade_num", 0)),
        trade.get("direction", ""),
        trade.get("entry", trade.get("entry_price", 0)),
        trade.get("exit", trade.get("exit_price", 0)),
        trade.get("entry_intended", 0),
        trade.get("exit_intended", 0),
        trade.get("time", trade.get("entry_time", "")),
        trade.get("exit_time", ""),
        trade.get("pnl_pts", 0),
        pnl_gbp,
        trade.get("contracts", 1),
        trade.get("contracts_stopped", 0),
        trade.get("adds_used", 0),
        trade.get("add_pnl_pts", 0),
        trade.get("entry_slippage", 0),
        trade.get("exit_slippage", 0),
        trade.get("slippage_total", 0),
        1 if trade.get("tp1_filled") else 0,
        1 if trade.get("tp2_filled") else 0,
        trade.get("tp1_slippage", 0),
        trade.get("tp2_slippage", 0),
        trade.get("mfe", 0),
        _state_or(state, "bar_range", trade.get("bar_range", 0)),
        _state_or(state, "range_flag", trade.get("range_flag", "")),
        _state_or(state, "bar_type", trade.get("bar_type", "")),
        trade.get("signal_bar", _state_or(state, "bar_number", 4)),
        trade.get("bar5_rule", _state_or(state, "bar5_rule_matched", "")),
        trade.get("gap_dir", _state_or(state, "gap_dir", "")),
        _state_or(state, "overnight_bias", ""),
        trade.get("exit_reason", ""),
        trade.get("stake_per_point", trade.get("stake", 1)),
        cum_pnl,
    ))
    conn.commit()

    trade_id = cursor.lastrowid
    logger.info(f"[{instrument}] Trade #{trade.get('num', '?')} logged to DB "
                f"({trade.get('direction', '?')} {trade.get('pnl_pts', 0)} pts, "
                f"cumulative: £{cum_pnl:.0f})")

    # Check scaling ladder
    _check_scaling_ladder(cum_pnl)

    return trade_id


def _state_or(state, attr, default):
    """Get attribute from state object, or return default."""
    if state and hasattr(state, attr):
        return getattr(state, attr)
    return default


# ── Queries ───────────────────────────────────────────────────────────────

def get_recent_trades(n: int = 10, instrument: str = None) -> list[dict]:
    """Get last N trades, optionally filtered by instrument."""
    conn = _get_conn()
    if instrument:
        rows = conn.execute(
            "SELECT * FROM trades WHERE instrument=? ORDER BY id DESC LIMIT ?",
            (instrument.upper(), n),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_trades_for_date(date: str, instrument: str = None) -> list[dict]:
    """Get all trades for a specific date."""
    conn = _get_conn()
    if instrument:
        rows = conn.execute(
            "SELECT * FROM trades WHERE date=? AND instrument=? ORDER BY id",
            (date, instrument.upper()),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades WHERE date=? ORDER BY id", (date,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_cumulative_pnl() -> float:
    """Get total cumulative P&L in GBP across all instruments."""
    conn = _get_conn()
    row = conn.execute("SELECT COALESCE(SUM(pnl_gbp), 0) FROM trades").fetchone()
    return row[0] if row else 0


def get_cumulative_pnl_by_instrument() -> dict:
    """Get cumulative P&L broken down by instrument."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT instrument, SUM(pnl_gbp) as total, COUNT(*) as trades "
        "FROM trades GROUP BY instrument"
    ).fetchall()
    return {r["instrument"]: {"pnl_gbp": r["total"], "trades": r["trades"]} for r in rows}


def get_weekly_pnl(instrument: str = None) -> dict:
    """Get this week's P&L."""
    conn = _get_conn()
    today = datetime.now(TZ_UK).date()
    from datetime import timedelta
    monday = (today - timedelta(days=today.weekday())).isoformat()
    today_str = today.isoformat()

    if instrument:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl_gbp), 0) as pnl, COUNT(*) as trades "
            "FROM trades WHERE date >= ? AND date <= ? AND instrument=?",
            (monday, today_str, instrument.upper()),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl_gbp), 0) as pnl, COUNT(*) as trades "
            "FROM trades WHERE date >= ? AND date <= ?",
            (monday, today_str),
        ).fetchone()
    return {"pnl_gbp": row["pnl"], "trades": row["trades"]}


def get_stats(instrument: str = None) -> dict:
    """Comprehensive stats for dashboard / Telegram."""
    conn = _get_conn()
    where = "WHERE instrument=?" if instrument else ""
    params = (instrument.upper(),) if instrument else ()

    rows = conn.execute(f"SELECT * FROM trades {where} ORDER BY id", params).fetchall()
    trades = [dict(r) for r in rows]

    if not trades:
        return {"total_trades": 0, "total_pnl_gbp": 0, "win_rate": 0,
                "profit_factor": 0, "max_drawdown_gbp": 0}

    pnls = [t["pnl_gbp"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    # Max drawdown
    running = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        running += p
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    return {
        "total_trades": len(trades),
        "total_pnl_gbp": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.99,
        "max_drawdown_gbp": round(max_dd, 2),
        "expectancy": round(sum(pnls) / len(trades), 2),
    }


# ── Scaling Ladder ────────────────────────────────────────────────────────

def get_current_scaling_tier() -> dict:
    """Get current and next scaling tier."""
    conn = _get_conn()
    cum_pnl = get_cumulative_pnl()

    # Current active tier (highest activated)
    current = conn.execute(
        "SELECT * FROM scaling_ladder WHERE activated=1 ORDER BY threshold_gbp DESC LIMIT 1"
    ).fetchone()

    # Next tier
    next_tier = conn.execute(
        "SELECT * FROM scaling_ladder WHERE activated=0 ORDER BY threshold_gbp ASC LIMIT 1"
    ).fetchone()

    return {
        "cumulative_pnl": round(cum_pnl, 2),
        "current_stake": dict(current)["stake_per_point"] if current else 1,
        "current_threshold": dict(current)["threshold_gbp"] if current else 0,
        "next_stake": dict(next_tier)["stake_per_point"] if next_tier else None,
        "next_threshold": dict(next_tier)["threshold_gbp"] if next_tier else None,
        "progress_to_next": round(cum_pnl - (dict(next_tier)["threshold_gbp"] if next_tier else 0), 2) if next_tier else 0,
    }


def _check_scaling_ladder(cum_pnl: float):
    """Check if cumulative P&L has crossed a scaling threshold."""
    conn = _get_conn()
    unactivated = conn.execute(
        "SELECT * FROM scaling_ladder WHERE activated=0 ORDER BY threshold_gbp ASC"
    ).fetchall()

    for tier in unactivated:
        tier = dict(tier)
        if cum_pnl >= tier["threshold_gbp"]:
            conn.execute(
                "UPDATE scaling_ladder SET activated=1, activated_at=? WHERE id=?",
                (datetime.now(TZ_UK).isoformat(), tier["id"]),
            )
            conn.commit()
            logger.info(
                f"SCALING LADDER: £{cum_pnl:.0f} crossed £{tier['threshold_gbp']:.0f} "
                f"→ new stake £{tier['stake_per_point']}/pt"
            )
        else:
            break  # Ordered by threshold, so no point checking higher ones


# ── CSV Migration ─────────────────────────────────────────────────────────

def migrate_csv():
    """Import existing CSV journals into SQLite (one-time migration)."""
    conn = _get_conn()
    existing = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if existing > 0:
        logger.info(f"DB already has {existing} trades — skipping CSV migration")
        return

    migrated = 0

    # DAX CSV
    dax_csv = os.path.join(os.path.dirname(__file__), "..", "dax_bot", "data", "dax", "trade_journal.csv")
    if os.path.exists(dax_csv):
        with open(dax_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conn.execute("""
                        INSERT INTO trades (
                            instrument, date, trade_num, direction,
                            entry_price, exit_price, entry_intended, exit_intended,
                            entry_time, exit_time,
                            pnl_pts, contracts_stopped,
                            tp1_filled, tp2_filled,
                            entry_slippage, tp1_slippage, tp2_slippage, exit_slippage,
                            slippage_total, mfe,
                            bar_range, range_flag, overnight_bias,
                            signal_bar, bar5_rule, gap_dir
                        ) VALUES (
                            'DAX', ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?,
                            ?, ?,
                            ?, ?,
                            ?, ?, ?, ?,
                            ?, ?,
                            ?, ?, ?,
                            ?, ?, ?
                        )
                    """, (
                        row.get("date", ""),
                        _safe_int(row.get("trade_num", 0)),
                        row.get("direction", ""),
                        _safe_float(row.get("entry", 0)),
                        _safe_float(row.get("exit", 0)),
                        _safe_float(row.get("entry_intended", 0)),
                        _safe_float(row.get("exit_intended", 0)),
                        row.get("entry_time", ""),
                        row.get("exit_time", ""),
                        _safe_float(row.get("pnl_pts", 0)),
                        _safe_int(row.get("contracts_stopped", 0)),
                        1 if row.get("tp1_filled", "").lower() == "true" else 0,
                        1 if row.get("tp2_filled", "").lower() == "true" else 0,
                        _safe_float(row.get("entry_slippage", 0)),
                        _safe_float(row.get("tp1_slippage", 0)),
                        _safe_float(row.get("tp2_slippage", 0)),
                        _safe_float(row.get("exit_slippage", 0)),
                        _safe_float(row.get("slippage_total", 0)),
                        _safe_float(row.get("mfe", 0)),
                        _safe_float(row.get("bar_range", 0)),
                        row.get("range_flag", ""),
                        row.get("overnight_bias", ""),
                        _safe_int(row.get("signal_bar", 4)),
                        row.get("bar5_rule", ""),
                        row.get("gap_dir", ""),
                    ))
                    migrated += 1
                except Exception as e:
                    logger.warning(f"DAX CSV row migration failed: {e}")
        logger.info(f"Migrated {migrated} DAX trades from CSV")

    # FTSE CSV
    ftse_migrated = 0
    ftse_csv = os.path.join(os.path.dirname(__file__), "..", "data", "ftse", "ftse_trades.csv")
    if os.path.exists(ftse_csv):
        with open(ftse_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conn.execute("""
                        INSERT INTO trades (
                            instrument, date, direction,
                            entry_price, exit_price,
                            pnl_pts, pnl_gbp, mfe,
                            bar_range, bar_type, exit_reason,
                            stake_per_point
                        ) VALUES (
                            'FTSE', ?, ?,
                            ?, ?,
                            ?, ?, ?,
                            ?, ?, ?,
                            ?
                        )
                    """, (
                        row.get("date", ""),
                        row.get("direction", ""),
                        _safe_float(row.get("entry", 0)),
                        _safe_float(row.get("exit", 0)),
                        _safe_float(row.get("pnl_pts", 0)),
                        _safe_float(row.get("pnl_gbp", 0)),
                        _safe_float(row.get("mfe", 0)),
                        _safe_float(row.get("bar_width", 0)),
                        row.get("bar_type", ""),
                        row.get("exit_reason", ""),
                        _safe_float(row.get("stake", 1)),
                    ))
                    ftse_migrated += 1
                except Exception as e:
                    logger.warning(f"FTSE CSV row migration failed: {e}")
        logger.info(f"Migrated {ftse_migrated} FTSE trades from CSV")

    conn.commit()
    logger.info(f"CSV migration complete: {migrated + ftse_migrated} total trades")


def _safe_float(val, default=0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        return int(float(val)) if val else default
    except (ValueError, TypeError):
        return default


# ── AI Data Logging ──────────────────────────────────────────────────────

def log_tick(trade_id: int, instrument: str, timestamp: str,
             bid: float, offer: float, mid: float):
    """Log a tick during an active trade for post-analysis."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO tick_log (trade_id, instrument, timestamp, bid, offer, mid, spread) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trade_id, instrument.upper(), timestamp, bid, offer, mid, round(offer - bid, 2)),
        )
        conn.commit()
    except Exception as e:
        pass  # Don't let tick logging break trading


def log_bar(trade_id: int, instrument: str, timestamp: str,
            open_: float, high: float, low: float, close: float, bar_num: int):
    """Log a 5-min bar during an active trade."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO bar_log (trade_id, instrument, timestamp, open, high, low, close, bar_num) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trade_id, instrument.upper(), timestamp, open_, high, low, close, bar_num),
        )
        conn.commit()
    except Exception as e:
        pass


def log_daily_context(date: str, instrument: str, context: dict):
    """Log daily market context for AI analysis."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO daily_context "
            "(date, instrument, prev_close, open_price, gap_pts, "
            "overnight_high, overnight_low, overnight_range, "
            "spx_prev_close, spx_prev_dir, day_of_week, "
            "bar1_range, bar2_range, bar3_range, bar4_range, bar5_range, morning_trend) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (date, instrument.upper(),
             context.get("prev_close", 0), context.get("open_price", 0),
             context.get("gap_pts", 0),
             context.get("overnight_high", 0), context.get("overnight_low", 0),
             context.get("overnight_range", 0),
             context.get("spx_prev_close", 0), context.get("spx_prev_dir", ""),
             context.get("day_of_week", 0),
             context.get("bar1_range", 0), context.get("bar2_range", 0),
             context.get("bar3_range", 0), context.get("bar4_range", 0),
             context.get("bar5_range", 0), context.get("morning_trend", "")),
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Daily context log failed: {e}")


def update_trade_extended(trade_id: int, **kwargs):
    """Update extended columns on an existing trade."""
    try:
        conn = _get_conn()
        sets = ", ".join(f"{k}=?" for k in kwargs.keys())
        vals = list(kwargs.values()) + [trade_id]
        conn.execute(f"UPDATE trades SET {sets} WHERE id=?", vals)
        conn.commit()
    except Exception as e:
        logger.warning(f"Extended trade update failed: {e}")
