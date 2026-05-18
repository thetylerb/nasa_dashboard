"""
Page 3 — Trend Analysis
- APOD classification distribution over time (by year)
- EONET event frequency by category over time
- Sentiment trend for Earth-related APOD, overlaid with storm/wildfire counts
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.api_client import get_apod_entries, get_eonet_events

st.set_page_config(page_title="Trend Analysis", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Load data (cached by api_client)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_apod_all():
    return pd.DataFrame(get_apod_entries(limit=100_000))

@st.cache_data(ttl=600)
def load_eonet_all():
    return pd.DataFrame(get_eonet_events(limit=500_000))

apod_df  = load_apod_all()
eonet_df = load_eonet_all()

st.title("📈 Trend Analysis")

if apod_df.empty and eonet_df.empty:
    st.info("No data available. Run `python pipeline.py --mode backfill` first.")
    st.stop()

# ---------------------------------------------------------------------------
# 1 — APOD topic distribution by year
# ---------------------------------------------------------------------------

st.subheader("APOD Classification Distribution Over Time")

if not apod_df.empty and "classification" in apod_df.columns and apod_df["classification"].notna().any():
    apod_classified = apod_df.dropna(subset=["classification"]).copy()
    apod_classified["year"] = apod_classified["date"].str[:4].astype(int)

    yearly = (
        apod_classified.groupby(["year", "classification"])
        .size()
        .reset_index(name="count")
    )

    fig1 = px.bar(
        yearly,
        x="year",
        y="count",
        color="classification",
        barmode="stack",
        labels={"year": "Year", "count": "# Entries", "classification": "Category"},
        height=420,
    )
    fig1.update_layout(legend_title_text="Category", xaxis=dict(dtick=1))
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Run classification (step 1 of ML pipeline) to see this chart.")

st.markdown("---")

# ---------------------------------------------------------------------------
# 2 — EONET event frequency by category over time (monthly)
# ---------------------------------------------------------------------------

st.subheader("EONET Event Frequency by Category")

if not eonet_df.empty and "event_date" in eonet_df.columns:
    eonet_clean = eonet_df.dropna(subset=["event_date"]).copy()
    eonet_clean["month"] = pd.to_datetime(eonet_clean["event_date"], errors="coerce").dt.to_period("M").astype(str)
    eonet_clean = eonet_clean.dropna(subset=["month"])

    monthly = (
        eonet_clean.groupby(["month", "category_title"])
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )

    fig2 = px.line(
        monthly,
        x="month",
        y="count",
        color="category_title",
        labels={"month": "Month", "count": "Event Count", "category_title": "Category"},
        height=420,
    )
    fig2.update_layout(legend_title_text="Category", xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No EONET event data with dates available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# 3 — Sentiment trend (Earth-related APOD) vs EONET storm + wildfire counts
# ---------------------------------------------------------------------------

st.subheader("Earth-Related APOD Sentiment vs EONET Storm & Wildfire Activity")

earth_cats = {"Earth Observation", "Atmospheric Phenomena"}
eonet_overlay_cats = {"severeStorms", "wildfires"}

if (
    not apod_df.empty
    and "classification" in apod_df.columns
    and apod_df["sentiment_score"].notna().any()
):
    earth_apod = apod_df[apod_df["classification"].isin(earth_cats)].dropna(subset=["sentiment_score"]).copy()
    earth_apod["year"] = earth_apod["date"].str[:4].astype(int)

    sentiment_by_year = (
        earth_apod.groupby("year")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment"})
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sentiment_by_year["year"],
        y=sentiment_by_year["avg_sentiment"],
        mode="lines+markers",
        name="Avg sentiment (Earth APOD)",
        line=dict(color="#2196F3", width=2),
        yaxis="y1",
    ))

    if not eonet_df.empty and "event_date" in eonet_df.columns:
        eonet_overlay = eonet_df[eonet_df["category_id"].isin(eonet_overlay_cats)].copy()
        eonet_overlay["year"] = pd.to_datetime(eonet_overlay["event_date"], errors="coerce").dt.year
        eonet_yearly = eonet_overlay.groupby("year").size().reset_index(name="event_count")

        fig3.add_trace(go.Bar(
            x=eonet_yearly["year"],
            y=eonet_yearly["event_count"],
            name="Storm & wildfire events",
            marker_color="rgba(244, 162, 97, 0.5)",
            yaxis="y2",
        ))

    fig3.update_layout(
        height=440,
        xaxis=dict(title="Year", dtick=1),
        yaxis=dict(title="Avg Sentiment Score", side="left", range=[0, 1]),
        yaxis2=dict(title="EONET Event Count", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99),
        bargap=0.2,
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Run sentiment scoring on Earth-related APOD entries to see this chart.")
