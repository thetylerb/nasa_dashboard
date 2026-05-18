"""
dashboard/api_client.py — Shared HTTP client for all Streamlit pages.

All functions call the FastAPI backend and cache results with st.cache_data
so repeated interactions don't hammer the API.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import streamlit as st

from config.config import API_BASE_URL

_TIMEOUT = 30.0


def _get(path: str, params: dict | None = None) -> list | dict:
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    resp = httpx.get(f"{API_BASE_URL}{path}", params=clean, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def get_eonet_events(
    status=None, category_id=None, start_date=None, end_date=None, limit=2000
):
    return _get("/api/eonet/events", {
        "status": status, "category_id": category_id,
        "start_date": start_date, "end_date": end_date, "limit": limit,
    })


@st.cache_data(ttl=300)
def get_apod_entries(
    classification=None, start_date=None, end_date=None, limit=500
):
    return _get("/api/apod", {
        "classification": classification,
        "start_date": start_date, "end_date": end_date, "limit": limit,
    })


@st.cache_data(ttl=600)
def get_anomaly_flags(only_anomalies=True):
    return _get("/api/anomalies", {"only_anomalies": str(only_anomalies).lower()})


@st.cache_data(ttl=60)
def get_pipeline_status():
    return _get("/api/status")


def semantic_search(query: str, top_k: int = 5) -> list:
    return _get("/api/search", {"q": query, "top_k": top_k})
