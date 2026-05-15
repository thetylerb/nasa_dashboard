"""
Page 4 — Anomaly Report
Table and timeline of Isolation Forest–flagged anomalous EONET periods.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from storage.database import get_anomaly_flags, get_eonet_events

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_flags():
    return pd.DataFrame(get_anomaly_flags(only_anomalies=True))

@st.cache_data(ttl=600)
def load_all_windows():
    return pd.DataFrame(get_anomaly_flags(only_anomalies=False))

@st.cache_data(ttl=600)
def load_eonet():
    return pd.DataFrame(get_eonet_events(limit=500_000))

flags_df   = load_flags()
all_df     = load_all_windows()
eonet_df   = load_eonet()

st.title("🚨 Anomaly Report")

if flags_df.empty:
    st.info(
        "No anomaly flags found. Run the pipeline with:  \n"
        "`python pipeline.py --mode ml-only`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("Anomalous windows flagged", len(flags_df))
col2.metric("Categories affected", flags_df["category_title"].nunique())
col3.metric("Avg confidence", f"{flags_df['confidence'].mean():.1%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Timeline chart — all windows with anomalies highlighted
# ---------------------------------------------------------------------------

st.subheader("Event Count Timeline with Anomaly Flags")

category_options = ["All"] + sorted(all_df["category_title"].dropna().unique().tolist())
selected_cat = st.selectbox("Category", category_options)

plot_df = all_df.copy() if selected_cat == "All" else all_df[all_df["category_title"] == selected_cat].copy()
plot_df = plot_df.sort_values("period_start")

if not plot_df.empty:
    fig = go.Figure()

    # Background: all window counts
    normal   = plot_df[plot_df["is_anomaly"] == 0]
    anomalous = plot_df[plot_df["is_anomaly"] == 1]

    fig.add_trace(go.Scatter(
        x=normal["period_start"],
        y=normal["event_count"],
        mode="lines",
        name="Normal",
        line=dict(color="#4CAF50", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=anomalous["period_start"],
        y=anomalous["event_count"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="#E63946", size=9, symbol="x"),
        text=anomalous["confidence"].apply(lambda c: f"Confidence: {c:.1%}"),
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Window Start Date",
        yaxis_title="30-Day Event Count",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Anomaly table
# ---------------------------------------------------------------------------

st.subheader("Flagged Anomalous Periods")

display = flags_df[["period_start", "period_end", "category_title", "event_count", "confidence"]].copy()
display.columns = ["Start", "End", "Category", "Event Count", "Confidence"]
display["Confidence"] = display["Confidence"].apply(lambda x: f"{x:.1%}")
display = display.sort_values("Confidence", ascending=False)

st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Drill-down: show actual events for a selected anomalous window
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Drill-down: Events Driving a Spike")

if not flags_df.empty and not eonet_df.empty:
    window_labels = flags_df.apply(
        lambda r: f"{r['period_start']} → {r['period_end']}  ({r['category_title']})", axis=1
    ).tolist()
    selected_window = st.selectbox("Select an anomalous window", options=window_labels)
    idx = window_labels.index(selected_window)
    row = flags_df.iloc[idx]

    mask = (
        (eonet_df["category_id"] == row.get("category_id", ""))
        & (eonet_df["event_date"] >= row["period_start"])
        & (eonet_df["event_date"] <= row["period_end"])
    )
    spike_events = eonet_df[mask][["title", "event_date", "status", "magnitude_value", "magnitude_unit"]]
    spike_events.columns = ["Title", "Date", "Status", "Magnitude", "Unit"]
    st.dataframe(spike_events.sort_values("Date"), use_container_width=True, hide_index=True)
