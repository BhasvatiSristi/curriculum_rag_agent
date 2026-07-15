"""
hybrid.py

Combine Dense Retrieval and BM25 Retrieval using
Reciprocal Rank Fusion (RRF).
"""

from typing import Dict, List

RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion.
    """

    fused_scores = {}
    chunk_lookup = {}

    # Dense Retrieval
    for rank, chunk in enumerate(dense_results, start=1):

        chunk_id = chunk["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        chunk_lookup[chunk_id] = chunk

    # BM25 Retrieval
    for rank, chunk in enumerate(bm25_results, start=1):

        chunk_id = chunk["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        chunk_lookup[chunk_id] = chunk

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for chunk_id, score in ranked[:top_k]:

        chunk = chunk_lookup[chunk_id].copy()

        chunk["score"] = round(score, 4)

        results.append(chunk)

    return results