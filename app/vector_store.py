"""ChromaDB vector store — persistence, upsert, and ANN search."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunker import TextChunk
from app.config import settings
from app.embedder import embed_texts

logger = logging.getLogger(__name__)

# Module-level singleton
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB at %s", settings.CHROMA_PERSIST_DIR)
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the default collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # HNSW index with cosine distance
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d existing documents",
            settings.CHROMA_COLLECTION,
            _collection.count(),
        )
    return _collection


def get_collection_stats() -> Dict:
    """Return basic stats about the collection."""
    coll = get_collection()
    return {
        "collection_name": settings.CHROMA_COLLECTION,
        "total_documents": coll.count(),
        "persist_dir": settings.CHROMA_PERSIST_DIR,
    }


# ── Ingest ───────────────────────────────────────────────────────

def upsert_chunks(chunks: List[TextChunk], batch_size: int = 256) -> int:
    """
    Embed and upsert chunks into ChromaDB.

    Returns the number of chunks upserted.
    """
    if not chunks:
        return 0

    coll = get_collection()
    total = len(chunks)
    upserted = 0

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [c.to_metadata_dict() for c in batch]

        # Compute embeddings
        embeddings = embed_texts(texts, batch_size=batch_size, show_progress=False)

        coll.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        upserted += len(batch)
        logger.info("Upserted %d / %d chunks", upserted, total)

    return upserted


# ── Search ───────────────────────────────────────────────────────

def search(
    query_embedding: list | "numpy.ndarray",
    top_k: int = settings.TOP_K_RETRIEVAL,
    where: Optional[Dict] = None,
) -> List[Dict]:
    """
    ANN search over the collection.

    Returns a list of dicts with keys:
        id, text, metadata, distance
    """
    coll = get_collection()

    # Ensure embedding is a plain list
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    kwargs: Dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = coll.query(**kwargs)

    # Unpack ChromaDB's nested lists
    hits: List[Dict] = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return hits


def delete_collection() -> None:
    """Delete the entire collection (for re-ingestion)."""
    global _collection
    client = _get_client()
    try:
        client.delete_collection(settings.CHROMA_COLLECTION)
        _collection = None
        logger.info("Deleted collection '%s'", settings.CHROMA_COLLECTION)
    except Exception as exc:
        logger.warning("Could not delete collection: %s", exc)
