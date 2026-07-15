"""
ingest_pipeline.py

Run the complete ingestion pipeline.
"""

import logging
import time

from backend.core.settings import DATA_DIR
from backend.ingestion.loader import load_all_pdfs
from backend.ingestion.chunker import chunk_documents
from backend.retrieval.vectorstore import add_chunks, collection_size
from backend.retrieval.bm25 import build_bm25_index

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    """
    Execute the complete ingestion pipeline.
    """
    start_time = time.time()

    logger.info("Starting ingestion pipeline...")

    # Load PDFs
    documents = load_all_pdfs(DATA_DIR)
    logger.info("Loaded %d pages.", len(documents))

    # Chunk documents
    chunks = chunk_documents(documents)
    logger.info("Created %d chunks.", len(chunks))

    # Store embeddings
    add_chunks(chunks)
    logger.info("Stored chunks in ChromaDB.")

    # Build BM25 index
    build_bm25_index(chunks)
    logger.info("BM25 index created.")

    elapsed = time.time() - start_time

    logger.info("Ingestion completed in %.2f seconds.", elapsed)
    logger.info("Total chunks in database: %d", collection_size())


if __name__ == "__main__":
    run_ingestion()