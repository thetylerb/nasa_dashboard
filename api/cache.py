"""
Simple thread-safe TTL cache for FastAPI route handlers.
"""

import time
import threading
from typing import Any, Callable

_store: dict = {}
_lock = threading.Lock()


def ttl_cache(key: str, ttl: int, fn: Callable) -> Any:
    """Return cached value if fresh, otherwise call fn(), cache, and return."""
    now = time.monotonic()
    with _lock:
        entry = _store.get(key)
        if entry and now - entry["ts"] < ttl:
            return entry["value"]
    value = fn()
    with _lock:
        _store[key] = {"value": value, "ts": time.monotonic()}
    return value


def invalidate(key: str) -> None:
    with _lock:
        _store.pop(key, None)
