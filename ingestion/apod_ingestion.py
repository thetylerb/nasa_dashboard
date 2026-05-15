"""
Fetch APOD entries from the NASA API.

Supports both full backfill (chunked date ranges) and incremental updates.
The API allows up to 100 dates per request using start_date/end_date.
"""

import time
import logging
from datetime import date, timedelta
from typing import List, Dict, Optional

import requests

from config.config import (
    NASA_API_KEY,
    APOD_BASE_URL,
    APOD_START_DATE,
    APOD_BATCH_SIZE,
    REQUEST_TIMEOUT,
    REQUEST_RETRY_MAX,
    REQUEST_RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)


def _get(params: dict) -> List[Dict]:
    """GET the APOD endpoint with retry logic. Always returns a list of dicts."""
    params["api_key"] = NASA_API_KEY
    for attempt in range(1, REQUEST_RETRY_MAX + 1):
        try:
            resp = requests.get(APOD_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = REQUEST_RETRY_BACKOFF * attempt
                logger.warning("Rate limited — waiting %ss", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else [data]
        except requests.RequestException as exc:
            logger.error("APOD request failed (attempt %d/%d): %s", attempt, REQUEST_RETRY_MAX, exc)
            if attempt < REQUEST_RETRY_MAX:
                time.sleep(REQUEST_RETRY_BACKOFF * attempt)
    return []


def fetch_date_range(start: date, end: date) -> List[Dict]:
    """Fetch all APOD entries between start and end (inclusive), chunked."""
    results = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=APOD_BATCH_SIZE - 1), end)
        logger.info("Fetching APOD %s → %s", cursor, chunk_end)
        batch = _get({
            "start_date": str(cursor),
            "end_date": str(chunk_end),
            "thumbs": "True",
        })
        results.extend(batch)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.25)  # gentle rate-limit buffer
    return results


def fetch_backfill() -> List[Dict]:
    """Fetch the full APOD archive from the first entry to today."""
    start = date.fromisoformat(APOD_START_DATE)
    end = date.today()
    logger.info("Starting APOD full backfill: %s → %s", start, end)
    return fetch_date_range(start, end)


def fetch_since(last_date: str) -> List[Dict]:
    """Fetch entries from the day after last_date up to today."""
    start = date.fromisoformat(last_date) + timedelta(days=1)
    end = date.today()
    if start > end:
        logger.info("APOD is already up to date.")
        return []
    return fetch_date_range(start, end)


def fetch_single(target_date: date) -> Optional[Dict]:
    """Fetch one specific APOD entry."""
    results = _get({"date": str(target_date), "thumbs": "True"})
    return results[0] if results else None
