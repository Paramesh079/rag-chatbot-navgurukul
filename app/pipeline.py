"""End-to-end ingestion pipeline: PDFs → chunks → embeddings → vector DB."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.chunker import TextChunk, chunk_documents
from app.config import settings
from app.pdf_processor import PDFDocument, extract_all_pdfs, extract_pdf
from app.vector_store import get_collection_stats, upsert_chunks

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""

    total_pdfs: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    ocr_pages: int = 0
    elapsed_seconds: float = 0.0
    per_pdf: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_pdfs": self.total_pdfs,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "ocr_pages": self.ocr_pages,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "per_pdf": self.per_pdf,
            "errors": self.errors,
        }


def ingest_directory(
    pdf_dir: Optional[str | Path] = None,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> IngestionStats:
    """
    Ingest all PDFs in a directory into the vector database.

    Pipeline: Extract → Chunk → Embed → Upsert to ChromaDB.
    """
    pdf_dir = Path(pdf_dir or settings.PDF_DIR)
    stats = IngestionStats()
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE START")
    logger.info("PDF directory: %s", pdf_dir)
    logger.info("Chunk size: %d tokens, overlap: %d tokens", chunk_size, chunk_overlap)
    logger.info("=" * 60)

    # Step 1: Extract
    logger.info("[1/3] Extracting text from PDFs...")
    try:
        documents = extract_all_pdfs(pdf_dir)
    except FileNotFoundError as exc:
        stats.errors.append(str(exc))
        return stats

    stats.total_pdfs = len(documents)

    if not documents:
        logger.warning("No documents extracted.")
        return stats

    # Gather per-PDF stats
    for doc in documents:
        ocr_count = sum(1 for p in doc.pages if p.is_ocr)
        stats.total_pages += doc.total_pages
        stats.ocr_pages += ocr_count
        stats.per_pdf.append(
            {
                "filename": doc.filename,
                "total_pages": doc.total_pages,
                "extracted_pages": len(doc.pages),
                "ocr_pages": ocr_count,
            }
        )

    # Step 2: Chunk
    logger.info("[2/3] Chunking documents...")
    all_chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    stats.total_chunks = len(all_chunks)

    # Update per-pdf chunk counts
    chunk_counts: Dict[str, int] = {}
    for chunk in all_chunks:
        chunk_counts[chunk.filename] = chunk_counts.get(chunk.filename, 0) + 1
    for pdf_stat in stats.per_pdf:
        pdf_stat["chunks"] = chunk_counts.get(pdf_stat["filename"], 0)

    # Step 3: Embed & Upsert
    logger.info("[3/3] Embedding and upserting to vector DB...")
    upserted = upsert_chunks(all_chunks)
    logger.info("Upserted %d chunks to ChromaDB", upserted)

    stats.elapsed_seconds = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE in %.1fs", stats.elapsed_seconds)
    logger.info(
        "PDFs: %d | Pages: %d | Chunks: %d | OCR pages: %d",
        stats.total_pdfs,
        stats.total_pages,
        stats.total_chunks,
        stats.ocr_pages,
    )
    logger.info("=" * 60)

    return stats


def ingest_single_pdf(
    filepath: str | Path,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> Dict:
    """Ingest a single PDF file into the vector database."""
    filepath = Path(filepath)
    t0 = time.perf_counter()

    doc = extract_pdf(filepath)
    chunks = chunk_documents([doc], chunk_size, chunk_overlap)
    upserted = upsert_chunks(chunks)
    elapsed = time.perf_counter() - t0

    return {
        "filename": doc.filename,
        "total_pages": doc.total_pages,
        "extracted_pages": len(doc.pages),
        "ocr_pages": sum(1 for p in doc.pages if p.is_ocr),
        "chunks": len(chunks),
        "upserted": upserted,
        "elapsed_seconds": round(elapsed, 2),
    }
