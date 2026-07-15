"""
chunker.py

Split documents into overlapping token-aware chunks.
"""

import logging
import re
from typing import Dict, List

import tiktoken

from backend.core.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

tokenizer = tiktoken.get_encoding("cl100k_base")


def split_by_semester(text: str) -> List[str]:
    """
    Split text into semester sections if semester headings exist.
    """
    parts = re.split(r"(Semester\s+\d+)", text)

    if len(parts) < 3:
        return [text]

    sections = []

    for heading, content in zip(parts[1::2], parts[2::2]):
        content = content.strip()
        if content:
            sections.append(f"{heading}\n{content}")

    return sections or [text]


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping token-based chunks.
    """
    chunks = []

    for section in split_by_semester(text):

        lines = [line.strip() for line in section.splitlines() if line.strip()]

        current_lines = []
        current_tokens = 0

        for line in lines:

            token_count = len(tokenizer.encode(line))

            if (
                current_lines
                and current_tokens + token_count > chunk_size
            ):
                chunks.append("\n".join(current_lines))

                overlap_lines = []
                overlap_tokens = 0

                for previous in reversed(current_lines):

                    previous_tokens = len(tokenizer.encode(previous))

                    if overlap_tokens + previous_tokens > overlap:
                        break

                    overlap_lines.insert(0, previous)
                    overlap_tokens += previous_tokens

                current_lines = overlap_lines
                current_tokens = overlap_tokens

            current_lines.append(line)
            current_tokens += token_count

        if current_lines:
            chunks.append("\n".join(current_lines))

    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Convert page-level documents into chunk-level documents.
    """
    all_chunks = []

    for doc in documents:

        chunks = split_text(doc["text"])

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_id": f'{doc["source"]}_p{doc["page"]}_c{index}',
                }
            )

    logger.info("Created %d chunks.", len(all_chunks))

    return all_chunks