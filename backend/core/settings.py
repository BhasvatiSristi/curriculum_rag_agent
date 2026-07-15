import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Storage
CHROMA_PATH = "backend/storage/chroma_db"
BM25_PATH = "backend/storage/bm25_index.pkl"

# Data
DATA_DIR = "backend/data"

# Collection
COLLECTION_NAME = "curriculum"