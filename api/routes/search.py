"""
Semantic search route.

Embeddings are loaded from the DB once per hour and held in process memory.
The sentence-transformer model (~90 MB) is loaded lazily on first request.
"""

import time
import threading

from fastapi import APIRouter, Query

from models.semantic_search import search
from storage.database import get_apod_with_embeddings

router = APIRouter()

_indexed = None
_indexed_ts = 0.0
_indexed_lock = threading.Lock()
_INDEXED_TTL = 3600  # reload embeddings from DB once per hour


def _get_indexed() -> list:
    global _indexed, _indexed_ts
    now = time.monotonic()
    if _indexed is None or now - _indexed_ts > _INDEXED_TTL:
        with _indexed_lock:
            if _indexed is None or now - _indexed_ts > _INDEXED_TTL:
                _indexed = get_apod_with_embeddings()
                _indexed_ts = time.monotonic()
    return _indexed


@router.get("")
def semantic_search(
    q: str = Query(..., min_length=1, description="Natural-language query"),
    top_k: int = Query(5, ge=1, le=20),
):
    indexed = _get_indexed()
    return search(q, indexed, top_k=top_k)
