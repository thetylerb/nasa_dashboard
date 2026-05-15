"""
Zero-shot classification of APOD explanations using facebook/bart-large-mnli.

Model is loaded once and cached as a module-level singleton to avoid
reloading on every call — loading takes ~10s on CPU.
"""

import logging
from typing import List, Dict, Tuple

from config.config import CLASSIFIER_MODEL, APOD_CATEGORIES

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        logger.info("Loading zero-shot classifier (%s)…", CLASSIFIER_MODEL)
        _pipeline = pipeline(
            "zero-shot-classification",
            model=CLASSIFIER_MODEL,
            device=-1,  # CPU; set to 0 for CUDA GPU
        )
        logger.info("Classifier ready.")
    return _pipeline


def classify_text(text: str) -> Tuple[str, float]:
    """
    Return (best_label, confidence_score) for a single text.
    Score is the softmax probability assigned to the top label.
    """
    if not text or not text.strip():
        return ("Unknown", 0.0)
    clf = _get_pipeline()
    result = clf(text[:1024], candidate_labels=APOD_CATEGORIES, multi_label=False)
    label = result["labels"][0]
    score = round(float(result["scores"][0]), 4)
    return (label, score)


def classify_batch(
    records: List[Dict],
    text_key: str = "explanation",
    id_key: str = "date",
    batch_size: int = 8,
) -> List[Dict]:
    """
    Classify a list of records in batches.

    Each input dict must have `id_key` and `text_key` fields.
    Returns list of dicts: {"date": ..., "classification": ..., "classification_score": ...}
    """
    results = []
    clf = _get_pipeline()
    texts = [r.get(text_key, "")[:1024] for r in records]
    ids   = [r.get(id_key) for r in records]

    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i:i + batch_size]
        chunk_ids   = ids[i:i + batch_size]
        logger.info(
            "Classifying records %d–%d of %d…",
            i + 1, min(i + batch_size, len(texts)), len(texts),
        )
        outputs = clf(chunk_texts, candidate_labels=APOD_CATEGORIES, multi_label=False)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for record_id, out in zip(chunk_ids, outputs):
            results.append({
                id_key:                record_id,
                "classification":      out["labels"][0],
                "classification_score": round(float(out["scores"][0]), 4),
            })

    return results
