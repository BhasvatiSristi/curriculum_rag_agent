"""
bm25.py

Sparse retrieval using BM25.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from core.settings import BM25_PATH, TOP_K

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.
    """
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(chunks: List[Dict]) -> None:
    """
    Build and save the BM25 index.
    """
    if not chunks:
        logger.warning("No chunks provided for BM25 indexing.")
        return

    corpus = [tokenize(chunk["text"]) for chunk in chunks]

    bm25 = BM25Okapi(corpus)

    index = {
        "bm25": bm25,
        "chunks": chunks,
    }

    Path(BM25_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(index, f)

    logger.info("BM25 index created with %d chunks.", len(chunks))


def load_bm25_index():
    """
    Load the BM25 index from disk.
    """
    path = Path(BM25_PATH)

    if not path.exists():
        logger.warning("BM25 index not found.")
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def bm25_search(
    query: str,
    top_k: int = TOP_K,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Search documents using BM25.
    """

    index = load_bm25_index()

    if index is None:
        return []

    bm25 = index["bm25"]
    chunks = index["chunks"]

    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []

    for score, chunk in ranked:

        if score <= 0:
            continue

        if source and chunk["source"] != source:
            continue

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": round(float(score), 4),
            }
        )

        if len(results) >= top_k:
            break

    return results


def bm25_size() -> int:
    """
    Return the number of indexed chunks.
    """
    index = load_bm25_index()

    if index is None:
        return 0

    return len(index["chunks"])