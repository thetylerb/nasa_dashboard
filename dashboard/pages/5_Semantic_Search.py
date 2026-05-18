"""
Page 5 — Semantic Search
Natural-language search over APOD explanations via the FastAPI backend.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from dashboard.api_client import semantic_search

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Search")
st.markdown(
    "Search 30 years of NASA Astronomy Picture of the Day explanations using natural language. "
    "Results are ranked by semantic similarity, not keyword matching."
)

query = st.text_input(
    "Enter a natural-language query:",
    placeholder="e.g.  'black hole in a distant galaxy'  or  'aurora over frozen landscape'",
)
top_k = st.slider("Results to return", min_value=1, max_value=10, value=5)

if st.button("Search", type="primary") or query:
    if not query.strip():
        st.info("Type a query above and press Search.")
        st.stop()

    with st.spinner("Computing similarity…"):
        results = semantic_search(query.strip(), top_k=top_k)

    if not results:
        st.warning("No results — run `python pipeline.py --mode ml-only` to generate embeddings.")
        st.stop()

    st.markdown(f"### Top {len(results)} results for *\"{query}\"*")
    st.markdown("---")

    for rank, r in enumerate(results, start=1):
        with st.container():
            col_score, col_content = st.columns([0.12, 0.88])

            with col_score:
                pct = int(r["similarity_score"] * 100)
                st.metric(f"#{rank}", f"{pct}%", help="Cosine similarity score")

            with col_content:
                st.markdown(f"**{r['date']}  — {r['title']}**")
                if r.get("classification"):
                    st.markdown(f"🏷️ `{r['classification']}`")
                explanation = r.get("explanation", "")
                snippet     = explanation[:300] + ("…" if len(explanation) > 300 else "")
                st.markdown(snippet)

            st.markdown("---")
