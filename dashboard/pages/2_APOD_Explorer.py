"""
Page 2 — APOD Explorer
Browse APOD entries by date or category, with AI classification and sentiment labels.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date

import streamlit as st
import pandas as pd

from storage.database import get_apod_entries
from config.config import APOD_CATEGORIES, APOD_START_DATE

st.set_page_config(page_title="APOD Explorer", page_icon="🔭", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

category_options = ["All"] + APOD_CATEGORIES
selected_category = st.sidebar.selectbox("Classification", category_options)

date_range = st.sidebar.date_input(
    "Date range",
    value=(date(2020, 1, 1), date.today()),
    min_value=date.fromisoformat(APOD_START_DATE),
    max_value=date.today(),
)
start_date = str(date_range[0]) if isinstance(date_range, tuple) else str(date_range)
end_date   = str(date_range[1]) if isinstance(date_range, tuple) and len(date_range) > 1 else str(date.today())

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_apod(category, start, end):
    cat = category if category != "All" else None
    return get_apod_entries(classification=cat, start_date=start, end_date=end, limit=1000)

entries = load_apod(selected_category, start_date, end_date)

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("🔭 APOD Explorer")

if not entries:
    st.info("No APOD entries found. Run the pipeline first: `python pipeline.py --mode backfill`")
    st.stop()

df = pd.DataFrame(entries)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Entries shown", len(df))
col2.metric("Classified", int(df["classification"].notna().sum()))
col3.metric("With sentiment", int(df["sentiment_label"].notna().sum()))
col4.metric("Media types", df["media_type"].nunique())

st.markdown("---")

# Pick one entry to display in detail
selected_date = st.selectbox(
    "Select an entry to view",
    options=df["date"].tolist(),
    format_func=lambda d: f"{d}  — {df.loc[df['date']==d, 'title'].values[0]}",
)

entry = df[df["date"] == selected_date].iloc[0]

col_img, col_meta = st.columns([1.4, 1])

with col_img:
    if entry.get("media_type") == "image" and entry.get("url"):
        st.image(entry["url"], use_column_width=True, caption=entry["title"])
    elif entry.get("media_type") == "video" and entry.get("url"):
        st.video(entry["url"])
    else:
        st.info("No media available for this entry.")

with col_meta:
    st.markdown(f"### {entry['title']}")
    st.markdown(f"**Date:** {entry['date']}")
    if entry.get("copyright"):
        st.markdown(f"**Credit:** {entry['copyright']}")

    st.markdown("---")
    st.markdown("**AI Classification**")
    cls   = entry.get("classification") or "—"
    score = entry.get("classification_score")
    st.markdown(f"🏷️ `{cls}`" + (f"  ({score:.1%})" if score else ""))

    st.markdown("**Sentiment**")
    label   = entry.get("sentiment_label") or "—"
    s_score = entry.get("sentiment_score")
    urgency = entry.get("urgency_score")
    sentiment_icon = "😊" if label == "POSITIVE" else "😟" if label == "NEGATIVE" else "😐"
    st.markdown(f"{sentiment_icon} `{label}`" + (f"  ({s_score:.1%})" if s_score else ""))
    if urgency is not None:
        st.progress(float(urgency), text=f"Urgency: {urgency:.1%}")

st.markdown("---")
st.markdown("**Explanation**")
st.write(entry.get("explanation", "—"))

# ---------------------------------------------------------------------------
# Browse table
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Browse All")

display = df[["date", "title", "media_type", "classification", "classification_score",
              "sentiment_label", "urgency_score"]].copy()
display.columns = ["Date", "Title", "Media", "Classification", "Conf", "Sentiment", "Urgency"]
display["Conf"]    = display["Conf"].apply(lambda x: f"{x:.1%}" if x else "—")
display["Urgency"] = display["Urgency"].apply(lambda x: f"{x:.1%}" if x else "—")

st.dataframe(display.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
