# RAG Chatbot — PDF Knowledge Base

A full-stack **Retrieval-Augmented Generation** chatbot that answers questions from a large corpus of PDFs (≥10 PDFs, ≥200 pages each) using a fully open-source stack with 2–5 second response latency.

![Architecture](https://img.shields.io/badge/architecture-RAG-6366f1?style=for-the-badge)
![Stack](https://img.shields.io/badge/stack-open--source-34d399?style=for-the-badge)
![Latency](https://img.shields.io/badge/latency-2--5s-fbbf24?style=for-the-badge)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  Embed Query │───▶│  ANN Search  │───▶│   Reranker    │
│ (MiniLM-L6) │    │  (ChromaDB)  │    │(Cross-Encoder)│
└─────────────┘    └──────────────┘    └───────┬───────┘
                                               │
                                        Top-K chunks
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │  LLM (Ollama)  │
                                      │  + Citations   │
                                      └────────┬───────┘
                                               │
                                         Answer + Sources
```

### Pipeline Components

| Component | Technology | Role |
|-----------|-----------|------|
| **PDF Extraction** | PyMuPDF + Tesseract | Native text + OCR for scanned pages |
| **Chunking** | Custom sentence-aware | 512-token chunks with 128-token overlap |
| **Embeddings** | `all-MiniLM-L6-v2` | Free, open-source, 384-dim vectors |
| **Vector DB** | ChromaDB (HNSW) | Persistent ANN index with cosine similarity |
| **Reranking** | `ms-marco-MiniLM-L-6-v2` | Cross-encoder for precision refinement |
| **LLM** | Ollama (llama3.1 default) | Local, private answer generation |
| **Web UI** | FastAPI + Vanilla JS | Premium dark-theme chat interface |

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** — Install from [ollama.com](https://ollama.com)
3. **Tesseract OCR** (optional, for scanned PDFs) — [Install guide](https://github.com/tesseract-ocr/tesseract)

### Setup

```bash
# 1. Clone / navigate to the project
cd rag_chatbot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull an Ollama model
ollama pull llama3.1
# Or a smaller model for faster responses:
# ollama pull phi3
# ollama pull mistral

# 5. Place your PDFs
# Copy your PDF files into data/pdfs/

# 6. Start the server
python main.py
```

Open **http://localhost:8000** in your browser.

### First-Time Usage

1. **Upload PDFs** via the sidebar upload button, or place them in `data/pdfs/`
2. Click **"Ingest All PDFs"** to run the ingestion pipeline
3. Start asking questions!

---

## 📁 Project Structure

```
rag_chatbot/
├── app/
│   ├── __init__.py          # Package init
│   ├── config.py            # Centralized configuration
│   ├── pdf_processor.py     # PDF extraction (native + OCR)
│   ├── chunker.py           # Sentence-aware text chunking
│   ├── embedder.py          # Embedding model wrapper
│   ├── vector_store.py      # ChromaDB vector store
│   ├── retriever.py         # Retrieval + cross-encoder reranking
│   ├── generator.py         # Ollama LLM answer generation
│   ├── pipeline.py          # End-to-end ingestion orchestrator
│   ├── metrics.py           # Latency & quality metrics
│   └── api.py               # FastAPI REST endpoints
├── static/
│   ├── index.html           # Chat UI
│   ├── style.css            # Premium dark theme
│   └── script.js            # Client-side logic
├── data/
│   └── pdfs/                # Place PDFs here
├── main.py                  # Server entry point
├── requirements.txt         # Python dependencies
├── .env                     # Configuration
└── .env.example             # Configuration template
```

---

## ⚙️ Configuration

All settings are in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1` | Ollama model for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `128` | Token overlap between chunks |
| `TOP_K_RETRIEVAL` | `20` | Candidates from ANN search |
| `TOP_K_RERANK` | `5` | Final results after reranking |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the chat UI |
| `GET` | `/api/health` | System health check |
| `POST` | `/api/chat` | Send a query → get a RAG answer |
| `POST` | `/api/ingest` | Ingest PDFs from directory |
| `POST` | `/api/upload` | Upload & ingest a single PDF |
| `GET` | `/api/documents` | List ingested documents |
| `GET` | `/api/metrics` | Query latency & performance metrics |
| `DELETE` | `/api/collection` | Reset the vector database |

### Chat Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main conclusions?", "top_k": 5}'
```

---

## 📊 Metrics & Evaluation

The system tracks in real-time:
- **Latency**: p50, p95, p99 response times
- **Retrieval breakdown**: Embedding + ANN search + reranking time
- **Generation breakdown**: LLM inference time
- **Chunk utilization**: Average chunks retrieved per query

View metrics at `GET /api/metrics` or in the sidebar dashboard.

---

## 🔧 Performance Tuning

| Goal | Adjustment |
|------|-----------|
| **Faster responses** | Use a smaller Ollama model (`phi3`, `mistral`) |
| **Better accuracy** | Increase `TOP_K_RETRIEVAL`, use larger embedding model |
| **Less memory** | Reduce `CHUNK_SIZE`, use `all-MiniLM-L6-v2` |
| **Handle more PDFs** | ChromaDB scales well; ensure sufficient disk |
| **GPU acceleration** | Ollama auto-uses GPU; sentence-transformers uses CUDA if available |

---

## License

Open-source for educational and research purposes.
