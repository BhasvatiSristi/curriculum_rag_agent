"""
retriever.py

Main retrieval interface for the RAG pipeline.
"""

from typing import Dict, List, Optional

from core.settings import TOP_K

from retrieval.vectorstore import dense_search
from retrieval.bm25 import bm25_search
from retrieval.hybrid import reciprocal_rank_fusion


def retrieve(
    query: str,
    mode: str = "hybrid",
    top_k: int = TOP_K,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve relevant chunks using the selected retrieval strategy.

    Modes:
        - dense
        - bm25
        - hybrid
    """

    mode = mode.lower()

    if mode == "dense":
        return dense_search(
            query=query,
            top_k=top_k,
            source=source,
        )

    if mode == "bm25":
        return bm25_search(
            query=query,
            top_k=top_k,
            source=source,
        )

    # Hybrid Retrieval

    dense_results = dense_search(
        query=query,
        top_k=top_k * 2,
        source=source,
    )

    bm25_results = bm25_search(
        query=query,
        top_k=top_k * 2,
        source=source,
    )

    return reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k,
    )