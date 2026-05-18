from typing import Optional

from fastapi import APIRouter, Query

from api.cache import ttl_cache
from storage.database import get_apod_entries

router = APIRouter()


@router.get("")
def list_apod(
    classification: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=100_000),
):
    key = f"apod:{classification}:{start_date}:{end_date}:{limit}"
    return ttl_cache(key, 300, lambda: get_apod_entries(
        classification=classification,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    ))
