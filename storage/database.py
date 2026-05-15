import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

from config.config import DB_PATH


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they do not already exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS apod_entries (
            date            TEXT PRIMARY KEY,
            title           TEXT,
            explanation     TEXT,
            media_type      TEXT,
            url             TEXT,
            hdurl           TEXT,
            copyright       TEXT,
            service_version TEXT,
            -- ML outputs
            classification       TEXT,
            classification_score REAL,
            sentiment_label      TEXT,
            sentiment_score      REAL,
            urgency_score        REAL,
            embedding            BLOB,
            -- housekeeping
            created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            updated_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );

        CREATE TABLE IF NOT EXISTS eonet_events (
            id              TEXT PRIMARY KEY,
            title           TEXT,
            description     TEXT,
            link            TEXT,
            category_id     TEXT,
            category_title  TEXT,
            status          TEXT,
            source_id       TEXT,
            source_url      TEXT,
            event_date      TEXT,
            closed_date     TEXT,
            geometry_type   TEXT,
            coordinates     TEXT,   -- JSON-encoded [lon, lat]
            magnitude_value REAL,
            magnitude_unit  TEXT,
            created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            updated_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );

        CREATE TABLE IF NOT EXISTS anomaly_flags (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            period_start   TEXT,
            period_end     TEXT,
            category_id    TEXT,
            category_title TEXT,
            event_count    INTEGER,
            anomaly_score  REAL,
            confidence     REAL,
            is_anomaly     INTEGER,
            created_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );

        CREATE TABLE IF NOT EXISTS pipeline_status (
            source        TEXT PRIMARY KEY,
            last_updated  TEXT,
            total_records INTEGER,
            status        TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_apod_classification ON apod_entries(classification);
        CREATE INDEX IF NOT EXISTS idx_eonet_category      ON eonet_events(category_id);
        CREATE INDEX IF NOT EXISTS idx_eonet_status        ON eonet_events(status);
        CREATE INDEX IF NOT EXISTS idx_eonet_date          ON eonet_events(event_date);
        CREATE INDEX IF NOT EXISTS idx_anomaly_category    ON anomaly_flags(category_id);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# APOD helpers
# ---------------------------------------------------------------------------

def upsert_apod(records: List[Dict[str, Any]]) -> int:
    """Insert or replace APOD records. Returns number of rows affected."""
    if not records:
        return 0
    conn = get_connection()
    sql = """
        INSERT INTO apod_entries
            (date, title, explanation, media_type, url, hdurl, copyright, service_version)
        VALUES
            (:date, :title, :explanation, :media_type, :url, :hdurl, :copyright, :service_version)
        ON CONFLICT(date) DO UPDATE SET
            title           = excluded.title,
            explanation     = excluded.explanation,
            media_type      = excluded.media_type,
            url             = excluded.url,
            hdurl           = excluded.hdurl,
            copyright       = excluded.copyright,
            updated_at      = strftime('%Y-%m-%dT%H:%M:%SZ','now')
    """
    conn.executemany(sql, records)
    conn.commit()
    count = conn.execute("SELECT changes()").fetchone()[0]
    conn.close()
    return len(records)


def update_apod_classification(date: str, classification: str, score: float) -> None:
    conn = get_connection()
    conn.execute(
        """UPDATE apod_entries
           SET classification=?, classification_score=?,
               updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
           WHERE date=?""",
        (classification, score, date),
    )
    conn.commit()
    conn.close()


def update_apod_sentiment(date: str, label: str, score: float, urgency: float) -> None:
    conn = get_connection()
    conn.execute(
        """UPDATE apod_entries
           SET sentiment_label=?, sentiment_score=?, urgency_score=?,
               updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
           WHERE date=?""",
        (label, score, urgency, date),
    )
    conn.commit()
    conn.close()


def update_apod_embedding(date: str, vector: np.ndarray) -> None:
    conn = get_connection()
    conn.execute(
        """UPDATE apod_entries SET embedding=?,
               updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
           WHERE date=?""",
        (vector.tobytes(), date),
    )
    conn.commit()
    conn.close()


def get_apod_entries(
    classification: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    conn = get_connection()
    clauses, params = [], []
    if classification:
        clauses.append("classification = ?")
        params.append(classification)
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM apod_entries {where} ORDER BY date DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_apod_with_embeddings() -> List[Dict]:
    """Return all APOD rows that have stored embeddings."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT date, title, explanation, classification, embedding "
        "FROM apod_entries WHERE embedding IS NOT NULL ORDER BY date"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d["embedding"]:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        result.append(d)
    return result


def get_unclassified_apod() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT date, explanation FROM apod_entries "
        "WHERE classification IS NULL AND explanation IS NOT NULL ORDER BY date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_apod_for_sentiment() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT date, explanation, classification FROM apod_entries "
        "WHERE classification IN ('Earth Observation','Atmospheric Phenomena') "
        "AND sentiment_label IS NULL AND explanation IS NOT NULL ORDER BY date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_apod_without_embeddings() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT date, explanation FROM apod_entries "
        "WHERE embedding IS NULL AND explanation IS NOT NULL ORDER BY date"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# EONET helpers
# ---------------------------------------------------------------------------

def upsert_eonet(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    conn = get_connection()
    sql = """
        INSERT INTO eonet_events
            (id, title, description, link, category_id, category_title,
             status, source_id, source_url, event_date, closed_date,
             geometry_type, coordinates, magnitude_value, magnitude_unit)
        VALUES
            (:id, :title, :description, :link, :category_id, :category_title,
             :status, :source_id, :source_url, :event_date, :closed_date,
             :geometry_type, :coordinates, :magnitude_value, :magnitude_unit)
        ON CONFLICT(id) DO UPDATE SET
            status         = excluded.status,
            closed_date    = excluded.closed_date,
            coordinates    = excluded.coordinates,
            updated_at     = strftime('%Y-%m-%dT%H:%M:%SZ','now')
    """
    conn.executemany(sql, records)
    conn.commit()
    count = len(records)
    conn.close()
    return count


def get_eonet_events(
    status: Optional[str] = None,
    category_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 2000,
) -> List[Dict]:
    conn = get_connection()
    clauses, params = [], []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if category_id:
        clauses.append("category_id = ?")
        params.append(category_id)
    if start_date:
        clauses.append("event_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("event_date <= ?")
        params.append(end_date)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM eonet_events {where} ORDER BY event_date DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Anomaly helpers
# ---------------------------------------------------------------------------

def upsert_anomaly_flags(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    conn = get_connection()
    conn.execute(
        "DELETE FROM anomaly_flags WHERE period_start >= ? AND period_start <= ?",
        (records[0]["period_start"], records[-1]["period_end"]),
    )
    conn.executemany(
        """INSERT INTO anomaly_flags
               (period_start, period_end, category_id, category_title,
                event_count, anomaly_score, confidence, is_anomaly)
           VALUES
               (:period_start, :period_end, :category_id, :category_title,
                :event_count, :anomaly_score, :confidence, :is_anomaly)""",
        records,
    )
    conn.commit()
    conn.close()


def get_anomaly_flags(only_anomalies: bool = True) -> List[Dict]:
    conn = get_connection()
    where = "WHERE is_anomaly = 1" if only_anomalies else ""
    rows = conn.execute(
        f"SELECT * FROM anomaly_flags {where} ORDER BY period_start DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pipeline status helpers
# ---------------------------------------------------------------------------

def update_pipeline_status(source: str, total_records: int, status: str = "ok") -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO pipeline_status (source, last_updated, total_records, status)
           VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ','now'), ?, ?)
           ON CONFLICT(source) DO UPDATE SET
               last_updated  = strftime('%Y-%m-%dT%H:%M:%SZ','now'),
               total_records = excluded.total_records,
               status        = excluded.status""",
        (source, total_records, status),
    )
    conn.commit()
    conn.close()


def get_pipeline_status() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM pipeline_status ORDER BY source").fetchall()
    conn.close()
    return [dict(r) for r in rows]
