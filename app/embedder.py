"""Open-source embedding model wrapper using sentence-transformers."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton to avoid reloading the model
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(
            "Embedding model loaded — dimension=%d", _model.get_sentence_embedding_dimension()
        )
    return _model


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors."""
    return _get_model().get_sentence_embedding_dimension()


def embed_texts(texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.

    Parameters
    ----------
    texts : list of str
        Input texts to embed.
    batch_size : int
        Batch size for the model.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a 1-D vector."""
    model = _get_model()
    return model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
