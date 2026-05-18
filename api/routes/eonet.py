from typing import Optional

from fastapi import APIRouter, Query

from api.cache import ttl_cache
from storage.database import get_eonet_events

router = APIRouter()


@router.get("/events")
def list_events(
    status: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(2000, ge=1, le=500_000),
):
    key = f"eonet:{status}:{category_id}:{start_date}:{end_date}:{limit}"
    return ttl_cache(key, 300, lambda: get_eonet_events(
        status=status,
        category_id=category_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    ))
