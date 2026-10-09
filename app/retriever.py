"""Retriever — query embedding + ANN search + cross-encoder reranking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import settings
from app.embedder import embed_query
from app.vector_store import search as vector_search

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder reranker
_reranker = None


def _get_reranker():
    """Load the cross-encoder reranker model on first use."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker: %s", settings.RERANKER_MODEL)
            _reranker = CrossEncoder(settings.RERANKER_MODEL)
            logger.info("Reranker loaded successfully")
        except Exception as exc:
            logger.warning("Could not load reranker: %s — skipping reranking", exc)
            _reranker = "DISABLED"
    return _reranker


@dataclass
class RetrievedChunk:
    """A retrieved and (optionally) reranked chunk."""

    chunk_id: str
    text: str
    filename: str
    page_numbers: List[int]
    pdf_id: str
    distance: float  # vector distance (lower = more similar for cosine)
    rerank_score: Optional[float] = None  # cross-encoder score (higher = better)

    @property
    def display_pages(self) -> str:
        """Human-readable page range."""
        pages = self.page_numbers
        if len(pages) == 1:
            return str(pages[0])
        return f"{pages[0]}–{pages[-1]}"

    @property
    def source_label(self) -> str:
        return f"{self.filename} (p. {self.display_pages})"


def _parse_page_numbers(raw: str) -> List[int]:
    """Parse comma-separated page numbers from metadata."""
    try:
        return [int(p.strip()) for p in raw.split(",") if p.strip()]
    except (ValueError, AttributeError):
        return []


def retrieve(
    query: str,
    top_k_retrieval: int = settings.TOP_K_RETRIEVAL,
    top_k_rerank: int = settings.TOP_K_RERANK,
    enable_reranking: bool = True,
    where: Optional[Dict] = None,
) -> List[RetrievedChunk]:
    """
    Full retrieval pipeline:
    1. Embed the query.
    2. ANN search for top_k_retrieval candidates.
    3. Optionally rerank with cross-encoder to top_k_rerank.

    Returns a list of RetrievedChunk sorted by relevance (best first).
    """
    t0 = time.perf_counter()

    # Step 1: embed query
    query_vec = embed_query(query)

    # Step 2: vector search
    raw_hits = vector_search(query_vec, top_k=top_k_retrieval, where=where)
    if not raw_hits:
        logger.info("No results found for query: %s", query[:80])
        return []

    # Convert to RetrievedChunk
    candidates: List[RetrievedChunk] = []
    for hit in raw_hits:
        meta = hit.get("metadata", {})
        candidates.append(
            RetrievedChunk(
                chunk_id=hit["id"],
                text=hit["text"],
                filename=meta.get("filename", "unknown"),
                page_numbers=_parse_page_numbers(meta.get("page_numbers", "")),
                pdf_id=meta.get("pdf_id", ""),
                distance=hit["distance"],
            )
        )

    # Step 3: rerank
    if enable_reranking and len(candidates) > top_k_rerank:
        reranker = _get_reranker()
        if reranker != "DISABLED" and reranker is not None:
            try:
                pairs = [(query, c.text) for c in candidates]
                scores = reranker.predict(pairs)
                for c, score in zip(candidates, scores):
                    c.rerank_score = float(score)
                candidates.sort(key=lambda c: c.rerank_score or 0, reverse=True)
                candidates = candidates[:top_k_rerank]
            except Exception as exc:
                logger.warning("Reranking failed: %s — using vector scores", exc)
                candidates = candidates[:top_k_rerank]
        else:
            candidates = candidates[:top_k_rerank]
    else:
        candidates = candidates[:top_k_rerank]

    elapsed = time.perf_counter() - t0
    logger.info(
        "Retrieved %d chunks in %.2fs (reranking=%s)",
        len(candidates),
        elapsed,
        enable_reranking,
    )

    return candidates
