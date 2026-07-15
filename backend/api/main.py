"""
main.py

FastAPI application for the Curriculum RAG Assistant.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.generation.generator import generate_answer
from backend.retrieval.retriever import retrieve
from backend.retrieval.vectorstore import (
    collection_size,
    list_sources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Curriculum RAG API",
    description="RAG-powered assistant for IIITDM Curriculum",
    version="1.0.0",
)

# --------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Request / Response Models
# --------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    mode: str = "hybrid"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Curriculum RAG API is running 🚀"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "indexed_chunks": collection_size(),
    }


@app.get("/sources")
def sources():
    return {
        "sources": list_sources()
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    logger.info("Question: %s", request.question)

    try:

        chunks = retrieve(
            query=request.question,
            mode=request.mode,
        )

        answer = generate_answer(
            request.question,
            chunks,
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                }
                for chunk in chunks
            ],
        }

    except Exception as e:

        logger.exception("Chat endpoint failed.")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )