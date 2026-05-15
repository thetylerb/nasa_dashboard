"""
Isolation Forest anomaly detection on EONET event frequency.

Algorithm:
  1. For each EONET category, build a rolling 30-day event-count time series.
  2. Fit an Isolation Forest on the counts.
  3. Flag windows whose anomaly score exceeds the contamination threshold.
  4. Persist flags to the database.
"""

import logging
from datetime import date, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config.config import ANOMALY_WINDOW_DAYS, ANOMALY_CONTAMINATION

logger = logging.getLogger(__name__)


def _rolling_counts(events: List[Dict], window_days: int = ANOMALY_WINDOW_DAYS) -> pd.DataFrame:
    """
    Build a DataFrame of rolling `window_days`-day event counts per category.
    Each row is one window, with columns: period_start, period_end, category_id,
    category_title, event_count.
    """
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])

    all_windows = []
    for (cat_id, cat_title), group in df.groupby(["category_id", "category_title"]):
        dates = group["event_date"].dt.date.sort_values()
        if len(dates) < 2:
            continue
        min_d, max_d = dates.min(), dates.max()
        cursor = min_d
        while cursor + timedelta(days=window_days) <= max_d + timedelta(days=1):
            window_end = cursor + timedelta(days=window_days - 1)
            count = ((dates >= cursor) & (dates <= window_end)).sum()
            all_windows.append({
                "period_start":  str(cursor),
                "period_end":    str(window_end),
                "category_id":   cat_id,
                "category_title": cat_title,
                "event_count":   int(count),
            })
            cursor += timedelta(days=1)

    return pd.DataFrame(all_windows)


def detect(events: List[Dict]) -> List[Dict]:
    """
    Run Isolation Forest on rolling event counts per category.
    Returns a list of anomaly-flag dicts ready for DB insertion.
    """
    windows = _rolling_counts(events)
    if windows.empty:
        logger.warning("No data for anomaly detection.")
        return []

    flag_records = []

    for cat_id, group in windows.groupby("category_id"):
        counts = group["event_count"].values.reshape(-1, 1)
        if len(counts) < 10:
            logger.debug("Skipping %s — too few windows (%d)", cat_id, len(counts))
            continue

        clf = IsolationForest(
            contamination=ANOMALY_CONTAMINATION,
            random_state=42,
            n_estimators=100,
        )
        clf.fit(counts)
        preds  = clf.predict(counts)           # -1 = anomaly, 1 = normal
        scores = clf.decision_function(counts) # negative = more anomalous

        for i, row in enumerate(group.itertuples(index=False)):
            raw_score = float(scores[i])
            # Normalise score to [0, 1] confidence: invert so higher = more anomalous
            confidence = round(max(0.0, min(1.0, 0.5 - raw_score)), 4)
            flag_records.append({
                "period_start":  row.period_start,
                "period_end":    row.period_end,
                "category_id":   row.category_id,
                "category_title": row.category_title,
                "event_count":   row.event_count,
                "anomaly_score": round(raw_score, 4),
                "confidence":    confidence,
                "is_anomaly":    int(preds[i] == -1),
            })

    logger.info(
        "Anomaly detection complete: %d windows, %d flagged.",
        len(flag_records),
        sum(r["is_anomaly"] for r in flag_records),
    )
    return flag_records
