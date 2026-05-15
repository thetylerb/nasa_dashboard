"""
Fetch EONET natural-event data from NASA's Earth Observatory Natural Event Tracker API v3.

Endpoints used:
  GET /events?status=open          — all currently active events
  GET /events?status=closed        — historical closed events (paginated)
  GET /categories                  — category reference list
"""

import time
import logging
from datetime import date, timedelta
from typing import List, Dict, Optional

import requests

from config.config import (
    EONET_BASE_URL,
    EONET_HISTORY_DAYS,
    REQUEST_TIMEOUT,
    REQUEST_RETRY_MAX,
    REQUEST_RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)


def _get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{EONET_BASE_URL}{path}"
    params = params or {}
    for attempt in range(1, REQUEST_RETRY_MAX + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = REQUEST_RETRY_BACKOFF * attempt
                logger.warning("Rate limited — waiting %ss", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("EONET request failed (attempt %d/%d): %s", attempt, REQUEST_RETRY_MAX, exc)
            if attempt < REQUEST_RETRY_MAX:
                time.sleep(REQUEST_RETRY_BACKOFF * attempt)
    return {}


def fetch_active_events() -> List[Dict]:
    """Return all currently open EONET events."""
    data = _get("/events", {"status": "open", "limit": 1000})
    return data.get("events", [])


def fetch_historical_events(start: date, end: date) -> List[Dict]:
    """
    Return closed events within a date window, paginating at 1000 events/page.
    EONET uses `start` and `end` as ISO strings (YYYY-MM-DD).
    """
    all_events: List[Dict] = []
    page = 1
    while True:
        logger.info("Fetching EONET closed events %s→%s page %d", start, end, page)
        data = _get("/events", {
            "status": "closed",
            "start": str(start),
            "end": str(end),
            "limit": 1000,
            "page": page,
        })
        events = data.get("events", [])
        all_events.extend(events)
        if len(events) < 1000:
            break
        page += 1
        time.sleep(0.5)
    return all_events


def fetch_backfill() -> List[Dict]:
    """Fetch active events + last EONET_HISTORY_DAYS of closed events."""
    end = date.today()
    start = end - timedelta(days=EONET_HISTORY_DAYS)
    logger.info("EONET backfill: %s → %s", start, end)
    active = fetch_active_events()
    historical = fetch_historical_events(start, end)
    return active + historical


def fetch_incremental(since: str) -> List[Dict]:
    """
    Fetch open events (always full refresh) plus closed events from `since` to today.
    `since` is an ISO date string (YYYY-MM-DD).
    """
    active = fetch_active_events()
    start = date.fromisoformat(since)
    end = date.today()
    if start >= end:
        return active
    historical = fetch_historical_events(start, end)
    return active + historical


def fetch_categories() -> List[Dict]:
    """Return EONET category definitions."""
    data = _get("/categories")
    return data.get("categories", [])
