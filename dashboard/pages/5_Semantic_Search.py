"""
Page 5 — Semantic Search
Natural-language search over APOD explanations using sentence-transformers.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from storage.database import get_apod_with_embeddings
from models.semantic_search import search

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Search")
st.markdown(
    "Search 30 years of NASA Astronomy Picture of the Day explanations using natural language. "
    "Results are ranked by semantic similarity, not keyword matching."
)

# ---------------------------------------------------------------------------
# Load indexed records (cached — embeddings are large)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embeddings… (first load may take a moment)")
def load_index():
    return get_apod_with_embeddings()

indexed = load_index()

if not indexed:
    st.warning(
        "No embeddings found in the database.  \n"
        "Run `python pipeline.py --mode ml-only` to generate them."
    )
    st.stop()

st.caption(f"{len(indexed):,} APOD entries indexed.")

# ---------------------------------------------------------------------------
# Search UI
# ---------------------------------------------------------------------------

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
        results = search(query, indexed, top_k=top_k)

    if not results:
        st.warning("No results — the index may be empty.")
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
                classification = r.get("classification")
                if classification:
                    st.markdown(f"🏷️ `{classification}`")

                # Show a 300-character snippet of the explanation
                explanation = r.get("explanation", "")
                snippet     = explanation[:300] + ("…" if len(explanation) > 300 else "")
                st.markdown(snippet)

            st.markdown("---")
