"""Text chunking with configurable size, overlap, and rich metadata."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import settings
from app.pdf_processor import PDFDocument, PageContent

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single text chunk with full provenance metadata."""

    chunk_id: str
    text: str
    pdf_id: str
    filename: str
    page_numbers: List[int]  # may span multiple pages
    chunk_index: int  # position within the document
    token_count: int
    metadata: Dict = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict:
        """Flatten metadata for vector DB storage."""
        return {
            "chunk_id": self.chunk_id,
            "pdf_id": self.pdf_id,
            "filename": self.filename,
            "page_numbers": ",".join(map(str, self.page_numbers)),
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            **self.metadata,
        }


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting (fast heuristic).

    For more accurate counts, swap in tiktoken, but whitespace splitting
    is ~4x faster and close enough for chunking decisions.
    """
    return len(text.split())


def _generate_chunk_id(pdf_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID for deduplication."""
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{pdf_id}_chunk{chunk_index:05d}_{content_hash}"


# ── Sentence-aware splitting ─────────────────────────────────────

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentence-like segments."""
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Core chunker ─────────────────────────────────────────────────

def chunk_document(
    doc: PDFDocument,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[TextChunk]:
    """
    Chunk a PDFDocument into overlapping text passages.

    Strategy:
    1. Concatenate page texts with page-boundary markers.
    2. Split into sentences.
    3. Greedily pack sentences into chunks up to `chunk_size` tokens.
    4. Apply `chunk_overlap` tokens of overlap between consecutive chunks.

    Each chunk records which page(s) it spans.
    """
    if not doc.pages:
        return []

    # Build a list of (sentence, page_number) tuples
    sentence_page_pairs: List[tuple[str, int]] = []
    for page in doc.pages:
        sentences = _split_into_sentences(page.text)
        for sent in sentences:
            sentence_page_pairs.append((sent, page.page_number))

    if not sentence_page_pairs:
        return []

    chunks: List[TextChunk] = []
    current_sentences: List[str] = []
    current_pages: set[int] = set()
    current_tokens = 0
    chunk_idx = 0

    i = 0
    while i < len(sentence_page_pairs):
        sent, page_num = sentence_page_pairs[i]
        sent_tokens = _count_tokens(sent)

        # If adding this sentence exceeds chunk_size, finalize current chunk
        if current_tokens + sent_tokens > chunk_size and current_sentences:
            pages_str = ",".join(map(str, sorted(current_pages)))
            chunk_text = f"Document: {doc.filename}, Pages: {pages_str}\n" + " ".join(current_sentences)
            chunk_id = _generate_chunk_id(doc.pdf_id, chunk_idx, chunk_text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    pdf_id=doc.pdf_id,
                    filename=doc.filename,
                    page_numbers=sorted(current_pages),
                    chunk_index=chunk_idx,
                    token_count=current_tokens,
                )
            )
            chunk_idx += 1

            # Overlap: keep trailing sentences whose total ≤ chunk_overlap
            overlap_sentences: List[str] = []
            overlap_pages: set[int] = set()
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = _count_tokens(s)
                if overlap_tokens + s_tok > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tok
                # find the page for this sentence (approximate)

            # Rebuild current state from overlap
            current_sentences = list(overlap_sentences)
            current_tokens = overlap_tokens
            current_pages = set()
            # Re-derive pages from overlap sentences by scanning back
            overlap_set = set(overlap_sentences)
            for sent_text, pn in sentence_page_pairs[:i]:
                if sent_text in overlap_set:
                    current_pages.add(pn)

            continue  # don't increment i; re-evaluate this sentence

        current_sentences.append(sent)
        current_pages.add(page_num)
        current_tokens += sent_tokens
        i += 1

    # Finalize last chunk
    if current_sentences:
        pages_str = ",".join(map(str, sorted(current_pages)))
        chunk_text = f"Document: {doc.filename}, Pages: {pages_str}\n" + " ".join(current_sentences)
        chunk_id = _generate_chunk_id(doc.pdf_id, chunk_idx, chunk_text)
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                pdf_id=doc.pdf_id,
                filename=doc.filename,
                page_numbers=sorted(current_pages),
                chunk_index=chunk_idx,
                token_count=current_tokens,
            )
        )

    logger.info(
        "Chunked %s → %d chunks (avg %d tokens)",
        doc.filename,
        len(chunks),
        sum(c.token_count for c in chunks) // max(len(chunks), 1),
    )
    return chunks


def chunk_documents(
    docs: List[PDFDocument],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Chunk multiple documents."""
    all_chunks: List[TextChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    logger.info("Total chunks across %d documents: %d", len(docs), len(all_chunks))
    return all_chunks
