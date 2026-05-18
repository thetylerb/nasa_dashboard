from fastapi import APIRouter

from api.cache import ttl_cache
from storage.database import get_pipeline_status

router = APIRouter()


@router.get("")
def pipeline_status():
    return ttl_cache("status", 60, get_pipeline_status)
