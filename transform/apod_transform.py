"""
Normalize raw APOD API responses into storage-ready dicts.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def transform(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of raw APOD API dicts into clean records matching the
    apod_entries table schema.
    """
    cleaned = []
    for raw in raw_records:
        if not raw.get("date"):
            logger.warning("Skipping APOD record missing date: %s", raw)
            continue
        cleaned.append({
            "date":            raw.get("date", ""),
            "title":           _truncate(raw.get("title", ""), 512),
            "explanation":     raw.get("explanation", ""),
            "media_type":      raw.get("media_type", "image"),
            "url":             raw.get("url", ""),
            "hdurl":           raw.get("hdurl") or raw.get("url", ""),
            "copyright":       _truncate(raw.get("copyright", ""), 256),
            "service_version": raw.get("service_version", "v1"),
        })
    return cleaned


def _truncate(value: str, max_len: int) -> str:
    if value and len(value) > max_len:
        return value[:max_len]
    return value or ""
