"""
embedder.py

Generate embeddings for documents and user queries using
Hugging Face Sentence Transformers.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from core.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Load the embedding model once when the application starts.
logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
model = SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple documents.

    Args:
        texts: List of document texts.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Generate an embedding for a user query.

    Args:
        query: User question.

    Returns:
        Embedding vector.
    """
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embedding.tolist()