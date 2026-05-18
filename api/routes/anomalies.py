from fastapi import APIRouter, Query

from api.cache import ttl_cache
from storage.database import get_anomaly_flags

router = APIRouter()


@router.get("")
def list_anomalies(only_anomalies: bool = Query(True)):
    key = f"anomalies:{only_anomalies}"
    return ttl_cache(key, 600, lambda: get_anomaly_flags(only_anomalies=only_anomalies))
