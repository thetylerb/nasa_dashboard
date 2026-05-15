"""
dashboard/app.py — Entry point for the Streamlit multi-page app.

Run with:  streamlit run dashboard/app.py
"""

import sys
import os

# Ensure project root is on the path so sub-modules resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from storage.database import init_db, get_pipeline_status

st.set_page_config(
    page_title="NASA Data Pipeline",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise DB on first load (idempotent)
init_db()

# ---------------------------------------------------------------------------
# Sidebar — pipeline status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛰️ NASA Pipeline")
    st.markdown("---")

    st.subheader("Pipeline Status")
    status_rows = get_pipeline_status()
    if status_rows:
        for row in status_rows:
            label = row["source"].upper()
            ts    = (row["last_updated"] or "—")[:16].replace("T", " ")
            total = f"{row['total_records']:,}" if row["total_records"] else "0"
            icon  = "✅" if row["status"] == "ok" else "⚠️"
            st.markdown(
                f"**{icon} {label}**  \n"
                f"Last updated: `{ts}`  \n"
                f"Records: `{total}`"
            )
            st.markdown("---")
    else:
        st.info("Run `python pipeline.py --mode backfill` to populate the database.")

    st.caption("NASA ETL Pipeline · Portfolio Project")

# ---------------------------------------------------------------------------
# Home page content (shown when no page is selected)
# ---------------------------------------------------------------------------

st.title("NASA Data Pipeline & AI Analytics")
st.markdown(
    """
    **A portfolio project** combining ETL engineering, NLP/ML modelling, and interactive data visualisation.

    Use the **sidebar** (or the pages icon at the top left) to navigate between views:

    | Page | What you'll find |
    |------|-----------------|
    | 🗺️ Live Events Map | Active EONET natural events on an interactive world map |
    | 🔭 APOD Explorer | Browse the Astronomy Picture of the Day archive with AI labels |
    | 📈 Trend Analysis | Classification trends, sentiment over time, and EONET frequency |
    | 🚨 Anomaly Report | Isolation Forest–flagged anomalous event periods |
    | 🔍 Semantic Search | Natural-language search over 30 years of APOD descriptions |

    ---
    **Data sources:** NASA APOD API · NASA EONET API
    **Models:** `facebook/bart-large-mnli` · `distilbert-base-uncased-finetuned-sst-2-english` · `all-MiniLM-L6-v2`
    """
)
