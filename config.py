"""Configuration module for AI PDF Assistant.
Handles paths, model selections, hyperparameters, and environment variables.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
EVALUATION_DIR = BASE_DIR / "evaluation"

# Ensure runtime directories exist
PDFS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB Settings
CHROMA_PERSIST_DIRECTORY = str(VECTORSTORE_DIR / "chroma_db")
COLLECTION_NAME = "pdf_rag_collection"

# Embeddings Settings
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Text Splitter Settings
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retriever Settings
DEFAULT_TOP_K = int(os.getenv("TOP_K_RESULTS", "4"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Available LLM Providers
SUPPORTED_PROVIDERS = ["gemini", "ollama", "mock"]
