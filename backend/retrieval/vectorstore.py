"""
vectorstore.py

Store and retrieve document embeddings using ChromaDB.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from core.settings import (
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
)

from retrieval.embedder import (
    embed_documents,
    embed_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Initialize ChromaDB
# ---------------------------------------------------------------------

Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# ---------------------------------------------------------------------
# Store Chunks
# ---------------------------------------------------------------------

def add_chunks(chunks: List[Dict]) -> None:
    """
    Embed and store document chunks in ChromaDB.
    """
    if not chunks:
        logger.warning("No chunks to store.")
        return

    existing_ids = set(collection.get()["ids"])

    new_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in existing_ids
    ]

    if not new_chunks:
        logger.info("All chunks already exist.")
        return

    texts = [chunk["text"] for chunk in new_chunks]

    embeddings = embed_documents(texts)

    collection.add(
        ids=[chunk["chunk_id"] for chunk in new_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in new_chunks
        ],
    )

    logger.info("Stored %d chunks.", len(new_chunks))


# ---------------------------------------------------------------------
# Dense Retrieval
# ---------------------------------------------------------------------

def dense_search(
    query: str,
    top_k: int = TOP_K,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Perform dense vector similarity search.
    """

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if source:
        query_args["where"] = {"source": source}

    results = collection.query(**query_args)

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        similarity = max(0.0, 1 - distance)

        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "score": round(similarity, 4),
            }
        )

    return retrieved_chunks


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def collection_size() -> int:
    """
    Return the number of indexed chunks.
    """
    return collection.count()


def list_sources() -> List[str]:
    """
    Return all indexed PDF filenames.
    """
    metadata = collection.get(include=["metadatas"])["metadatas"]

    return sorted(
        {
            item["source"]
            for item in metadata
            if "source" in item
        }
    )


def clear_collection() -> None:
    """
    Remove every indexed chunk.
    """
    global collection

    client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Collection cleared.")