"""
scheduler.py — Runs the pipeline on a recurring schedule using APScheduler.

  EONET: every 6 hours (new / updated events)
  APOD : once daily  (new picture of the day)

Run with:  python scheduler.py
"""

import logging
import os
import sys
from datetime import datetime, timezone

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scheduler")

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config.config import EONET_UPDATE_HOURS, APOD_UPDATE_HOURS
from pipeline import (
    run_apod_etl,
    run_eonet_etl,
    run_classification,
    run_sentiment,
    run_embeddings,
    run_anomaly_detection,
)
from storage.database import init_db


def eonet_job() -> None:
    logger.info("Scheduled EONET update triggered.")
    try:
        run_eonet_etl("incremental")
        run_anomaly_detection()
    except Exception as exc:
        logger.exception("EONET scheduled job failed: %s", exc)


def apod_job() -> None:
    logger.info("Scheduled APOD update triggered.")
    try:
        run_apod_etl("incremental")
        run_classification()
        run_sentiment()
        run_embeddings()
    except Exception as exc:
        logger.exception("APOD scheduled job failed: %s", exc)


if __name__ == "__main__":
    init_db()
    scheduler = BlockingScheduler(timezone="UTC")

    now = datetime.now(timezone.utc)

    scheduler.add_job(
        eonet_job,
        trigger=IntervalTrigger(hours=EONET_UPDATE_HOURS),
        id="eonet_update",
        name="EONET incremental update",
        replace_existing=True,
        next_run_time=now,
    )
    scheduler.add_job(
        apod_job,
        trigger=IntervalTrigger(hours=APOD_UPDATE_HOURS),
        id="apod_update",
        name="APOD daily update",
        replace_existing=True,
        next_run_time=now,
    )

    logger.info(
        "Scheduler started — EONET every %dh, APOD every %dh. Press Ctrl+C to stop.",
        EONET_UPDATE_HOURS,
        APOD_UPDATE_HOURS,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
