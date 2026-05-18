"""
api/main.py — FastAPI backend for the NASA Pipeline.

Start with:  uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import apod, eonet, anomalies, search, status

app = FastAPI(
    title="NASA Pipeline API",
    description="Serves APOD and EONET data with ML outputs from the ETL pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(apod.router,      prefix="/api/apod",      tags=["apod"])
app.include_router(eonet.router,     prefix="/api/eonet",     tags=["eonet"])
app.include_router(anomalies.router, prefix="/api/anomalies", tags=["anomalies"])
app.include_router(search.router,    prefix="/api/search",    tags=["search"])
app.include_router(status.router,    prefix="/api/status",    tags=["status"])


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}
