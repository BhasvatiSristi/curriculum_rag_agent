"""
loader.py

Load PDF documents and extract page-level text for the ingestion pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pdfplumber

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load a PDF file and extract text from each page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dictionaries containing page text and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    documents = []

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    documents.append(
                        {
                            "text": text.strip(),
                            "source": path.name,
                            "page": page_num,
                        }
                    )

        logger.info("Loaded %d pages from '%s'.", len(documents), path.name)
        return documents

    except Exception as e:
        logger.error("Error while reading '%s': %s", path.name, e)
        raise


def load_all_pdfs(data_dir: str) -> List[Dict]:
    """
    Load all PDF files from a directory.

    Args:
        data_dir: Directory containing PDF files.

    Returns:
        A combined list of page-level documents.
    """
    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in '{directory}'.")

    all_documents = []

    logger.info("Found %d PDF(s).", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Loading '%s'...", pdf_file.name)
        all_documents.extend(load_pdf(str(pdf_file)))

    logger.info("Total pages loaded: %d", len(all_documents))

    return all_documents