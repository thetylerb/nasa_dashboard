"""
pipeline.py — Orchestrates ETL + ML runs.

Usage:
  python pipeline.py --mode backfill    # first-run: full archive + all ML
  python pipeline.py --mode incremental # updates only
  python pipeline.py --mode ml-only     # re-run ML on existing data

Each mode is safe to rerun; all writes use upsert semantics.
"""

import argparse
import logging
import sys
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# ETL steps
# ---------------------------------------------------------------------------

def run_apod_etl(mode: str = "incremental") -> None:
    from ingestion import apod_ingestion as ingest
    from transform import apod_transform as xform
    from storage import database as db

    logger.info("=== APOD ETL [%s] ===", mode)

    if mode == "backfill":
        raw = ingest.fetch_backfill()
    else:
        # Find the latest date already in the DB
        rows = db.get_apod_entries(limit=1)
        last = rows[0]["date"] if rows else None
        if last:
            raw = ingest.fetch_since(last)
        else:
            logger.info("No APOD records found — falling back to backfill.")
            raw = ingest.fetch_backfill()

    records = xform.transform(raw)
    count   = db.upsert_apod(records)
    total   = len(db.get_apod_entries(limit=100_000))
    db.update_pipeline_status("apod", total, "ok")
    logger.info("APOD ETL done: %d upserted, %d total in DB.", count, total)


def run_eonet_etl(mode: str = "incremental") -> None:
    from ingestion import eonet_ingestion as ingest
    from transform import eonet_transform as xform
    from storage import database as db

    logger.info("=== EONET ETL [%s] ===", mode)

    if mode == "backfill":
        raw = ingest.fetch_backfill()
    else:
        rows = db.get_eonet_events(limit=1)
        last = rows[0]["event_date"][:10] if rows else None
        if last:
            raw = ingest.fetch_incremental(since=last)
        else:
            logger.info("No EONET records found — falling back to backfill.")
            raw = ingest.fetch_backfill()

    records = xform.transform(raw)
    count   = db.upsert_eonet(records)
    total   = len(db.get_eonet_events(limit=500_000))
    db.update_pipeline_status("eonet", total, "ok")
    logger.info("EONET ETL done: %d upserted, %d total in DB.", count, total)


# ---------------------------------------------------------------------------
# ML steps
# ---------------------------------------------------------------------------

def run_classification() -> None:
    from models import classifier
    from storage import database as db

    logger.info("=== Zero-shot Classification ===")
    pending = db.get_unclassified_apod()
    if not pending:
        logger.info("No unclassified APOD entries — skipping.")
        return
    logger.info("Classifying %d entries…", len(pending))
    results = classifier.classify_batch(pending)
    for r in results:
        db.update_apod_classification(r["date"], r["classification"], r["classification_score"])
    logger.info("Classification complete: %d entries labelled.", len(results))


def run_sentiment() -> None:
    from models import sentiment
    from storage import database as db

    logger.info("=== Sentiment Scoring ===")
    pending = db.get_apod_for_sentiment()
    if not pending:
        logger.info("No pending sentiment records — skipping.")
        return
    logger.info("Scoring sentiment for %d Earth-related entries…", len(pending))
    results = sentiment.score_batch(pending)
    for r in results:
        db.update_apod_sentiment(r["date"], r["sentiment_label"], r["sentiment_score"], r["urgency_score"])
    logger.info("Sentiment scoring complete: %d entries scored.", len(results))


def run_anomaly_detection() -> None:
    from models import anomaly
    from storage import database as db

    logger.info("=== Anomaly Detection ===")
    events = db.get_eonet_events(limit=500_000)
    if not events:
        logger.info("No EONET events for anomaly detection — skipping.")
        return
    flags = anomaly.detect(events)
    if flags:
        db.upsert_anomaly_flags(flags)
    logger.info("Anomaly detection done: %d flags stored.", len(flags))


def run_embeddings() -> None:
    from models import semantic_search
    from storage import database as db

    logger.info("=== Semantic Embeddings ===")
    pending = db.get_apod_without_embeddings()
    if not pending:
        logger.info("All APOD entries already embedded — skipping.")
        return
    logger.info("Embedding %d APOD entries…", len(pending))
    pairs = semantic_search.embed_batch(pending)
    for record_id, vec in pairs:
        db.update_apod_embedding(record_id, vec)
    logger.info("Embeddings complete: %d entries stored.", len(pairs))


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

def run_backfill() -> None:
    from storage import database as db
    db.init_db()
    run_apod_etl("backfill")
    run_eonet_etl("backfill")
    run_classification()
    run_sentiment()
    run_embeddings()
    run_anomaly_detection()
    logger.info("=== Full backfill complete. ===")


def run_incremental() -> None:
    from storage import database as db
    db.init_db()
    run_apod_etl("incremental")
    run_eonet_etl("incremental")
    run_classification()
    run_sentiment()
    run_embeddings()
    run_anomaly_detection()
    logger.info("=== Incremental update complete. ===")


def run_ml_only() -> None:
    from storage import database as db
    db.init_db()
    run_classification()
    run_sentiment()
    run_embeddings()
    run_anomaly_detection()
    logger.info("=== ML-only run complete. ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA ETL Pipeline")
    parser.add_argument(
        "--mode",
        choices=["backfill", "incremental", "ml-only"],
        default="incremental",
        help="backfill=first run, incremental=daily update, ml-only=rerun models",
    )
    args = parser.parse_args()

    dispatch = {
        "backfill":    run_backfill,
        "incremental": run_incremental,
        "ml-only":     run_ml_only,
    }
    dispatch[args.mode]()
