"""Centralized configuration loaded from .env / environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Application-wide settings, populated from environment variables."""

    # ── Paths ────────────────────────────────────────────────
    PROJECT_ROOT: Path = _PROJECT_ROOT
    PDF_DIR: Path = Path(os.getenv("PDF_DIR", str(_PROJECT_ROOT / "data" / "pdfs")))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db"))

    # ── ChromaDB ─────────────────────────────────────────────
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_documents")

    # ── Embedding ────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Reranker ─────────────────────────────────────────────
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Chunking ─────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "128"))

    # ── Retrieval ────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "50"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "15"))

    # ── Ollama ───────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # ── Server ───────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
