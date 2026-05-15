"""
Normalize raw EONET API event records into storage-ready dicts.

EONET v3 event structure (simplified):
{
  "id": "EONET_6254",
  "title": "...",
  "description": "...",
  "link": "...",
  "closed": null | "2024-01-15T00:00:00Z",
  "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
  "sources": [{"id": "GDACS", "url": "..."}],
  "geometry": [
    {
      "magnitudeValue": 20.0,
      "magnitudeUnit": "kts",
      "date": "2024-01-10T18:00:00Z",
      "type": "Point",
      "coordinates": [-75.2, 18.3]
    }
  ]
}

We store one row per event, using the *most recent* geometry point.
"""

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def transform(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for raw in raw_events:
        eid = raw.get("id")
        if not eid:
            continue

        category = _first(raw.get("categories", []))
        source   = _first(raw.get("sources", []))
        geometry = _latest_geometry(raw.get("geometry", []))

        closed_raw = raw.get("closed")
        closed_date = _iso_date(closed_raw) if closed_raw else None

        cleaned.append({
            "id":              eid,
            "title":           raw.get("title", ""),
            "description":     raw.get("description") or "",
            "link":            raw.get("link", ""),
            "category_id":     category.get("id", "") if category else "",
            "category_title":  category.get("title", "") if category else "",
            "status":          "closed" if closed_raw else "open",
            "source_id":       source.get("id", "") if source else "",
            "source_url":      source.get("url", "") if source else "",
            "event_date":      geometry.get("date") if geometry else None,
            "closed_date":     closed_date,
            "geometry_type":   geometry.get("type") if geometry else None,
            "coordinates":     json.dumps(geometry.get("coordinates")) if geometry and geometry.get("coordinates") else None,
            "magnitude_value": geometry.get("magnitudeValue") if geometry else None,
            "magnitude_unit":  geometry.get("magnitudeUnit") if geometry else None,
        })
    return cleaned


def _first(lst: list) -> Optional[dict]:
    return lst[0] if lst else None


def _latest_geometry(geometries: list) -> Optional[dict]:
    """Return the geometry entry with the most recent date."""
    if not geometries:
        return None
    try:
        return max(geometries, key=lambda g: g.get("date") or "")
    except Exception:
        return geometries[-1]


def _iso_date(timestamp: str) -> Optional[str]:
    """Extract YYYY-MM-DD from an ISO timestamp string."""
    if not timestamp:
        return None
    return timestamp[:10]
