"""
Page 1 — Live Events Map
Interactive world map of active and recent EONET events.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px

from storage.database import get_eonet_events

st.set_page_config(page_title="Live Events Map", page_icon="🗺️", layout="wide")

# ---------------------------------------------------------------------------
# Category colour palette
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "severeStorms":     "#E63946",
    "wildfires":        "#F4A261",
    "volcanoes":        "#E76F51",
    "seaLakeIce":       "#457B9D",
    "landslides":       "#A8C5A0",
    "floods":           "#1D3557",
    "earthquakes":      "#6A4C93",
    "drought":          "#C9AE5D",
    "dustHaze":         "#B5C9C3",
    "manmade":          "#999999",
    "snow":             "#A8DADC",
    "temperatureExtremes": "#FF6B6B",
}
DEFAULT_COLOR = "#888888"

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

show_open   = st.sidebar.checkbox("Show open (active) events",   value=True)
show_closed = st.sidebar.checkbox("Show closed (past) events",   value=False)

date_range = st.sidebar.date_input(
    "Event date range",
    value=(date.today() - timedelta(days=90), date.today()),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
)
start_date = str(date_range[0]) if isinstance(date_range, tuple) else str(date_range)
end_date   = str(date_range[1]) if isinstance(date_range, tuple) and len(date_range) > 1 else str(date.today())

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_events(s_open, s_closed, start, end):
    rows = []
    if s_open:
        rows += get_eonet_events(status="open",   start_date=start, end_date=end, limit=5000)
    if s_closed:
        rows += get_eonet_events(status="closed", start_date=start, end_date=end, limit=5000)
    return rows

events = load_events(show_open, show_closed, start_date, end_date)

# ---------------------------------------------------------------------------
# Build map DataFrame
# ---------------------------------------------------------------------------

st.title("🗺️ Live Events Map")

if not events:
    st.info("No events found for the selected filters. Make sure the pipeline has run.")
    st.stop()

records = []
for e in events:
    try:
        coords = json.loads(e["coordinates"]) if e.get("coordinates") else None
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
    except Exception:
        continue
    records.append({
        "id":             e["id"],
        "title":          e["title"],
        "category":       e["category_title"],
        "category_id":    e["category_id"],
        "status":         e["status"],
        "event_date":     e["event_date"],
        "lat":            lat,
        "lon":            lon,
        "magnitude":      e.get("magnitude_value"),
        "magnitude_unit": e.get("magnitude_unit") or "",
        "color":          CATEGORY_COLORS.get(e["category_id"], DEFAULT_COLOR),
    })

df = pd.DataFrame(records)

if df.empty:
    st.warning("Events were found in the DB but none have usable coordinates.")
    st.stop()

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="category",
    hover_name="title",
    hover_data={"event_date": True, "status": True, "magnitude": True,
                "magnitude_unit": True, "lat": False, "lon": False},
    zoom=1,
    height=580,
    mapbox_style="carto-positron",
    color_discrete_map={row["category"]: CATEGORY_COLORS.get(row["category_id"], DEFAULT_COLOR)
                        for row in records},
)
fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, legend_title_text="Category")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("Total Events", len(df))
col2.metric("Open", int((df["status"] == "open").sum()))
col3.metric("Closed", int((df["status"] == "closed").sum()))

st.markdown("---")
st.subheader("Event Details")

display_cols = ["title", "category", "status", "event_date", "magnitude", "magnitude_unit"]
st.dataframe(
    df[display_cols].rename(columns={
        "title": "Title", "category": "Category", "status": "Status",
        "event_date": "Date", "magnitude": "Magnitude", "magnitude_unit": "Unit",
    }).sort_values("Date", ascending=False),
    use_container_width=True,
    hide_index=True,
)
