import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"

# PostgreSQL connection string — provided automatically by Railway's PostgreSQL plugin.
# For local dev, run `docker-compose up` and set:
#   DATABASE_URL=postgresql://nasa:nasa_secret@localhost:5432/nasa_pipeline
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Base URL of the FastAPI backend — used by the Streamlit frontend.
# Local dev: http://localhost:8000   Railway: https://<api-service>.up.railway.app
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# NASA API
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
APOD_BASE_URL = "https://api.nasa.gov/planetary/apod"
APOD_START_DATE = "1995-06-16"  # First APOD entry
APOD_BATCH_SIZE = 100           # Max dates per API request

EONET_BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3"
EONET_HISTORY_DAYS = 730        # 2 years of history on first run

# APScheduler
EONET_UPDATE_HOURS = 6
APOD_UPDATE_HOURS = 24

# ML models
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

APOD_CATEGORIES = [
    "Solar System",
    "Deep Space",
    "Earth Observation",
    "Stars and Nebulae",
    "Human Spaceflight",
    "Atmospheric Phenomena",
]

EARTH_RELATED_CATEGORIES = {"Earth Observation", "Atmospheric Phenomena"}

# Anomaly detection
ANOMALY_WINDOW_DAYS = 30
ANOMALY_CONTAMINATION = 0.1     # Expected fraction of outliers

# HTTP
REQUEST_TIMEOUT = 30
REQUEST_RETRY_MAX = 3
REQUEST_RETRY_BACKOFF = 2       # seconds between retries
