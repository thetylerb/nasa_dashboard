import threading
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import numpy as np
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

from config.config import DATABASE_URL

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 20, dsn=DATABASE_URL)
    return _pool


@contextmanager
def _conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = False
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _rows(conn, sql: str, params=None) -> List[Dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _execute(conn, sql: str, params=None) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def _executemany(conn, sql: str, params_list) -> None:
    with conn.cursor() as cur:
        cur.executemany(sql, params_list)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS apod_entries (
                    date            TEXT PRIMARY KEY,
                    title           TEXT,
                    explanation     TEXT,
                    media_type      TEXT,
                    url             TEXT,
                    hdurl           TEXT,
                    copyright       TEXT,
                    service_version TEXT,
                    classification       TEXT,
                    classification_score REAL,
                    sentiment_label      TEXT,
                    sentiment_score      REAL,
                    urgency_score        REAL,
                    embedding            BYTEA,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
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
                    coordinates     TEXT,
                    magnitude_value REAL,
                    magnitude_unit  TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_flags (
                    id             SERIAL PRIMARY KEY,
                    period_start   TEXT,
                    period_end     TEXT,
                    category_id    TEXT,
                    category_title TEXT,
                    event_count    INTEGER,
                    anomaly_score  REAL,
                    confidence     REAL,
                    is_anomaly     INTEGER,
                    created_at     TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_status (
                    source        TEXT PRIMARY KEY,
                    last_updated  TEXT,
                    total_records INTEGER,
                    status        TEXT
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_apod_classification ON apod_entries(classification)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_eonet_category ON eonet_events(category_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_eonet_status ON eonet_events(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_eonet_date ON eonet_events(event_date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_category ON anomaly_flags(category_id)")


# ---------------------------------------------------------------------------
# APOD helpers
# ---------------------------------------------------------------------------

def upsert_apod(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    sql = """
        INSERT INTO apod_entries
            (date, title, explanation, media_type, url, hdurl, copyright, service_version)
        VALUES
            (%(date)s, %(title)s, %(explanation)s, %(media_type)s, %(url)s, %(hdurl)s,
             %(copyright)s, %(service_version)s)
        ON CONFLICT (date) DO UPDATE SET
            title           = EXCLUDED.title,
            explanation     = EXCLUDED.explanation,
            media_type      = EXCLUDED.media_type,
            url             = EXCLUDED.url,
            hdurl           = EXCLUDED.hdurl,
            copyright       = EXCLUDED.copyright,
            updated_at      = NOW()
    """
    with _conn() as conn:
        _executemany(conn, sql, records)
    return len(records)


def update_apod_classification(date: str, classification: str, score: float) -> None:
    with _conn() as conn:
        _execute(conn,
            "UPDATE apod_entries SET classification=%s, classification_score=%s, updated_at=NOW() WHERE date=%s",
            (classification, score, date))


def update_apod_sentiment(date: str, label: str, score: float, urgency: float) -> None:
    with _conn() as conn:
        _execute(conn,
            "UPDATE apod_entries SET sentiment_label=%s, sentiment_score=%s, urgency_score=%s, updated_at=NOW() WHERE date=%s",
            (label, score, urgency, date))


def update_apod_embedding(date: str, vector: np.ndarray) -> None:
    with _conn() as conn:
        _execute(conn,
            "UPDATE apod_entries SET embedding=%s, updated_at=NOW() WHERE date=%s",
            (psycopg2.Binary(vector.tobytes()), date))


def get_apod_entries(
    classification: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    clauses, params = [], []
    if classification:
        clauses.append("classification = %s")
        params.append(classification)
    if start_date:
        clauses.append("date >= %s")
        params.append(start_date)
    if end_date:
        clauses.append("date <= %s")
        params.append(end_date)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _conn() as conn:
        rows = _rows(conn, f"SELECT * FROM apod_entries {where} ORDER BY date DESC LIMIT %s", params)
    for r in rows:
        r.pop("embedding", None)
    return rows


def get_apod_with_embeddings() -> List[Dict]:
    with _conn() as conn:
        rows = _rows(conn,
            "SELECT date, title, explanation, classification, embedding "
            "FROM apod_entries WHERE embedding IS NOT NULL ORDER BY date")
    result = []
    for r in rows:
        raw = r.get("embedding")
        if raw:
            r["embedding"] = np.frombuffer(bytes(raw), dtype=np.float32)
        result.append(r)
    return result


def get_unclassified_apod() -> List[Dict]:
    with _conn() as conn:
        return _rows(conn,
            "SELECT date, explanation FROM apod_entries "
            "WHERE classification IS NULL AND explanation IS NOT NULL ORDER BY date")


def get_apod_for_sentiment() -> List[Dict]:
    with _conn() as conn:
        return _rows(conn,
            "SELECT date, explanation, classification FROM apod_entries "
            "WHERE classification IN ('Earth Observation','Atmospheric Phenomena') "
            "AND sentiment_label IS NULL AND explanation IS NOT NULL ORDER BY date")


def get_apod_without_embeddings() -> List[Dict]:
    with _conn() as conn:
        return _rows(conn,
            "SELECT date, explanation FROM apod_entries "
            "WHERE embedding IS NULL AND explanation IS NOT NULL ORDER BY date")


# ---------------------------------------------------------------------------
# EONET helpers
# ---------------------------------------------------------------------------

def upsert_eonet(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    sql = """
        INSERT INTO eonet_events
            (id, title, description, link, category_id, category_title,
             status, source_id, source_url, event_date, closed_date,
             geometry_type, coordinates, magnitude_value, magnitude_unit)
        VALUES
            (%(id)s, %(title)s, %(description)s, %(link)s, %(category_id)s, %(category_title)s,
             %(status)s, %(source_id)s, %(source_url)s, %(event_date)s, %(closed_date)s,
             %(geometry_type)s, %(coordinates)s, %(magnitude_value)s, %(magnitude_unit)s)
        ON CONFLICT (id) DO UPDATE SET
            status         = EXCLUDED.status,
            closed_date    = EXCLUDED.closed_date,
            coordinates    = EXCLUDED.coordinates,
            updated_at     = NOW()
    """
    with _conn() as conn:
        _executemany(conn, sql, records)
    return len(records)


def get_eonet_events(
    status: Optional[str] = None,
    category_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 2000,
) -> List[Dict]:
    clauses, params = [], []
    if status:
        clauses.append("status = %s")
        params.append(status)
    if category_id:
        clauses.append("category_id = %s")
        params.append(category_id)
    if start_date:
        clauses.append("event_date >= %s")
        params.append(start_date)
    if end_date:
        clauses.append("event_date <= %s")
        params.append(end_date)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _conn() as conn:
        return _rows(conn, f"SELECT * FROM eonet_events {where} ORDER BY event_date DESC LIMIT %s", params)


# ---------------------------------------------------------------------------
# Anomaly helpers
# ---------------------------------------------------------------------------

def upsert_anomaly_flags(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with _conn() as conn:
        _execute(conn,
            "DELETE FROM anomaly_flags WHERE period_start >= %s AND period_start <= %s",
            (records[0]["period_start"], records[-1]["period_end"]))
        _executemany(conn,
            """INSERT INTO anomaly_flags
                   (period_start, period_end, category_id, category_title,
                    event_count, anomaly_score, confidence, is_anomaly)
               VALUES
                   (%(period_start)s, %(period_end)s, %(category_id)s, %(category_title)s,
                    %(event_count)s, %(anomaly_score)s, %(confidence)s, %(is_anomaly)s)""",
            records)


def get_anomaly_flags(only_anomalies: bool = True) -> List[Dict]:
    where = "WHERE is_anomaly = 1" if only_anomalies else ""
    with _conn() as conn:
        return _rows(conn, f"SELECT * FROM anomaly_flags {where} ORDER BY period_start DESC")


# ---------------------------------------------------------------------------
# Pipeline status helpers
# ---------------------------------------------------------------------------

def update_pipeline_status(source: str, total_records: int, status: str = "ok") -> None:
    with _conn() as conn:
        _execute(conn,
            """INSERT INTO pipeline_status (source, last_updated, total_records, status)
               VALUES (%s, NOW()::TEXT, %s, %s)
               ON CONFLICT (source) DO UPDATE SET
                   last_updated  = NOW()::TEXT,
                   total_records = EXCLUDED.total_records,
                   status        = EXCLUDED.status""",
            (source, total_records, status))


def get_pipeline_status() -> List[Dict]:
    with _conn() as conn:
        return _rows(conn, "SELECT * FROM pipeline_status ORDER BY source")
