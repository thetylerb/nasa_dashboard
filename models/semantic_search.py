"""
Semantic search over APOD explanations using sentence-transformers.

Workflow:
  1. Embed all APOD explanations with all-MiniLM-L6-v2.
  2. Store vectors (float32 numpy arrays serialised as blobs) in the DB.
  3. At query time, embed the user query and rank stored entries by cosine similarity.
"""

import logging
from typing import List, Dict, Tuple

import numpy as np

from config.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (%s)…", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return a float32 matrix of shape (N, embedding_dim)."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def embed_single(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def embed_batch(
    records: List[Dict],
    text_key: str = "explanation",
    id_key: str = "date",
    batch_size: int = 64,
) -> List[Tuple[str, np.ndarray]]:
    """
    Embed records in batches.
    Returns list of (record_id, vector) tuples.
    """
    results = []
    model = _get_model()
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        texts = [r.get(text_key, "")[:512] for r in chunk]
        ids   = [r.get(id_key) for r in chunk]
        logger.info("Embedding records %d–%d of %d…", i + 1, min(i + batch_size, len(records)), len(records))
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        for record_id, vec in zip(ids, vectors):
            results.append((record_id, vec))
    return results


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Vectorised cosine similarity between query_vec and each row of matrix."""
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def search(
    query: str,
    indexed_records: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Search for the top_k most semantically similar APOD entries.

    `indexed_records` must have an 'embedding' key holding a float32 np.ndarray.
    Returns a list of dicts (same fields as input + 'similarity_score').
    """
    if not indexed_records:
        return []

    query_vec = embed_single(query)
    matrix    = np.vstack([r["embedding"] for r in indexed_records])
    sims      = cosine_similarity(query_vec, matrix)
    top_idx   = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        rec = dict(indexed_records[idx])
        rec["similarity_score"] = round(float(sims[idx]), 4)
        rec.pop("embedding", None)  # don't leak binary blobs to callers
        results.append(rec)
    return results
