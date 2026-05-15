"""
Sentiment and urgency scoring for Earth-related APOD entries.

Uses distilbert-base-uncased-finetuned-sst-2-english for binary
sentiment (POSITIVE / NEGATIVE). Urgency is derived by scanning for
high-frequency climate/disaster keywords and scaling 0–1.
"""

import logging
import re
from typing import List, Dict, Tuple

from config.config import SENTIMENT_MODEL

logger = logging.getLogger(__name__)

_pipeline = None

_URGENCY_KEYWORDS = [
    "catastrophic", "devastating", "severe", "extreme", "crisis",
    "emergency", "unprecedented", "alarming", "threat", "dangerous",
    "record", "wildfire", "flood", "hurricane", "drought", "warming",
    "melting", "collapse", "destruction", "toxic",
]
_URGENCY_PATTERN = re.compile(
    "|".join(_URGENCY_KEYWORDS), flags=re.IGNORECASE
)


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        logger.info("Loading sentiment model (%s)…", SENTIMENT_MODEL)
        _pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=-1,
            truncation=True,
            max_length=512,
        )
        logger.info("Sentiment model ready.")
    return _pipeline


def _urgency_score(text: str) -> float:
    """Keyword-density urgency score in [0, 1]."""
    if not text:
        return 0.0
    matches = len(_URGENCY_PATTERN.findall(text))
    words   = max(len(text.split()), 1)
    # Cap at 5 keyword hits per 100 words → 1.0
    return min(round((matches / words) * 100 / 5, 4), 1.0)


def score_text(text: str) -> Tuple[str, float, float]:
    """Return (label, sentiment_score, urgency_score) for a single text."""
    if not text or not text.strip():
        return ("NEUTRAL", 0.5, 0.0)
    clf = _get_pipeline()
    result = clf(text[:512])[0]
    label = result["label"]          # "POSITIVE" or "NEGATIVE"
    score = round(float(result["score"]), 4)
    urgency = _urgency_score(text)
    return (label, score, urgency)


def score_batch(
    records: List[Dict],
    text_key: str = "explanation",
    id_key: str = "date",
    batch_size: int = 16,
) -> List[Dict]:
    """
    Score a list of records for sentiment and urgency.
    Returns list of dicts with id, sentiment_label, sentiment_score, urgency_score.
    """
    results = []
    clf = _get_pipeline()
    texts = [r.get(text_key, "")[:512] for r in records]
    ids   = [r.get(id_key) for r in records]

    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i:i + batch_size]
        chunk_ids   = ids[i:i + batch_size]
        logger.info("Scoring sentiment %d–%d of %d…", i + 1, min(i + batch_size, len(texts)), len(texts))
        outputs = clf(chunk_texts)
        for record_id, out, raw_text in zip(chunk_ids, outputs, chunk_texts):
            results.append({
                id_key:            record_id,
                "sentiment_label": out["label"],
                "sentiment_score": round(float(out["score"]), 4),
                "urgency_score":   _urgency_score(raw_text),
            })
    return results
