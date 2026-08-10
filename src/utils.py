"""
utils.py
--------
Small shared helpers: query logging to CSV (for the "logging user
queries" enhancement) and a couple of formatting helpers used by the
Streamlit UI.
"""

import csv
import os
from datetime import datetime
from typing import Optional

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "query_log.csv")

_LOG_HEADERS = [
    "timestamp",
    "namespace",
    "question",
    "top_k",
    "similarity_threshold",
    "grounded",
    "confidence",
    "num_sources",
]


def log_query(
    namespace: str,
    question: str,
    top_k: int,
    similarity_threshold: float,
    grounded: bool,
    confidence: float,
    num_sources: int,
) -> None:
    """Append one query record to logs/query_log.csv, creating the file
    with headers on first use."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_LOG_HEADERS)
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                namespace,
                question,
                top_k,
                similarity_threshold,
                grounded,
                confidence,
                num_sources,
            ]
        )


def truncate(text: str, max_chars: int = 220) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def safe_namespace(name: str) -> str:
    """Pinecone namespaces are just strings, but keep them clean/URL-safe
    for readability in the console and across documents."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).lower()
