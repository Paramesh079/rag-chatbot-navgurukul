"""FastAPI application — REST API + static file serving for the chat UI."""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.generator import GenerationResult, check_ollama_health, generate_answer
from app.metrics import QueryMetric, metrics_collector
from app.pipeline import ingest_directory, ingest_single_pdf
from app.retriever import retrieve
from app.vector_store import delete_collection, get_collection_stats

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot over large PDF corpora",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ensure directories ──────────────────────────────────────────
Path(settings.PDF_DIR).mkdir(parents=True, exist_ok=True)

# ── Static files ─────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ── Request / Response models ────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    top_k: int = settings.TOP_K_RERANK
    enable_reranking: bool = True
    model: Optional[str] = None


class SourceInfo(BaseModel):
    filename: str
    page_numbers: list
    chunk_id: str
    distance: float
    rerank_score: Optional[float] = None
    text_preview: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    model: str
    latency: dict
    retrieved_chunks: list[dict]


class IngestRequest(BaseModel):
    pdf_dir: Optional[str] = None
    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap: int = settings.CHUNK_OVERLAP


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the chat UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# Mount static files AFTER the root route
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/api/health")
async def health_check():
    """System health check — Ollama + vector DB."""
    ollama_health = check_ollama_health()
    db_stats = get_collection_stats()
    return {
        "status": "ok",
        "ollama": ollama_health,
        "vector_db": db_stats,
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK,
        },
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Main RAG chat endpoint.

    1. Retrieve relevant chunks.
    2. Generate answer with citations.
    3. Return answer + sources + metrics.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    t_total_start = time.perf_counter()

    # Retrieve
    t_retrieval_start = time.perf_counter()
    chunks = retrieve(
        query=req.query,
        top_k_rerank=req.top_k,
        enable_reranking=req.enable_reranking,
    )
    t_retrieval = time.perf_counter() - t_retrieval_start

    if not chunks:
        return ChatResponse(
            answer="I couldn't find any relevant information in the knowledge base for your question. "
            "Please try rephrasing or check that documents have been ingested.",
            sources=[],
            model=req.model or settings.OLLAMA_MODEL,
            latency={"total_s": round(time.perf_counter() - t_total_start, 3), "retrieval_s": round(t_retrieval, 3), "generation_s": 0},
            retrieved_chunks=[],
        )

    # Generate
    t_gen_start = time.perf_counter()
    result: GenerationResult = generate_answer(
        query=req.query,
        chunks=chunks,
        model=req.model,
    )
    t_gen = time.perf_counter() - t_gen_start

    t_total = time.perf_counter() - t_total_start

    # Record metrics
    metrics_collector.record(
        QueryMetric(
            query=req.query,
            total_latency_s=t_total,
            retrieval_latency_s=t_retrieval,
            generation_latency_s=t_gen,
            num_chunks_retrieved=len(chunks),
            model_used=result.model,
        )
    )

    # Build response
    sources = []
    retrieved_chunks_data = []
    
    # Only return sources to the UI if the LLM actually cited them in its answer.
    # This hides "random documents" for greetings or when no answer could be found.
    if "[Source:" in result.answer:
        for c in chunks:
            sources.append(
                SourceInfo(
                    filename=c.filename,
                    page_numbers=c.page_numbers,
                    chunk_id=c.chunk_id,
                    distance=c.distance,
                    rerank_score=c.rerank_score,
                    text_preview=c.text[:300] + "..." if len(c.text) > 300 else c.text,
                )
            )
            retrieved_chunks_data.append(
                {
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page_numbers": c.page_numbers,
                    "text": c.text,
                    "distance": round(c.distance, 4),
                    "rerank_score": round(c.rerank_score, 4) if c.rerank_score else None,
                }
            )

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        model=result.model,
        latency={
            "total_s": round(t_total, 3),
            "retrieval_s": round(t_retrieval, 3),
            "generation_s": round(t_gen, 3),
        },
        retrieved_chunks=retrieved_chunks_data,
    )


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Trigger document ingestion from the PDF directory."""
    try:
        stats = ingest_directory(
            pdf_dir=req.pdf_dir,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        return {
            "status": "completed",
            "stats": stats.to_dict(),
            "collection": get_collection_stats(),
        }
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a single PDF and ingest it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save to PDF directory
    dest = Path(settings.PDF_DIR) / file.filename
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Ingest
    try:
        result = ingest_single_pdf(dest)
        return {
            "status": "completed",
            "file": result,
            "collection": get_collection_stats(),
        }
    except Exception as exc:
        logger.exception("Ingestion of uploaded file failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/documents")
async def list_documents():
    """List ingested PDFs."""
    pdf_dir = Path(settings.PDF_DIR)
    pdfs = []
    if pdf_dir.exists():
        for fp in sorted(pdf_dir.glob("*.pdf")):
            pdfs.append(
                {
                    "filename": fp.name,
                    "size_mb": round(fp.stat().st_size / (1024 * 1024), 2),
                }
            )
    return {
        "pdf_dir": str(pdf_dir),
        "documents": pdfs,
        "collection": get_collection_stats(),
    }


@app.get("/api/metrics")
async def get_metrics():
    """Return query latency and quality metrics."""
    return {
        "summary": metrics_collector.get_summary(),
        "recent_queries": metrics_collector.get_recent(20),
    }


@app.delete("/api/collection")
async def reset_collection():
    """Delete the vector collection (for re-ingestion)."""
    delete_collection()
    return {"status": "deleted", "collection": settings.CHROMA_COLLECTION}
