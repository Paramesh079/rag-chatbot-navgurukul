/* ═══════════════════════════════════════════════════════════════
   RAG Chatbot – Client-side Logic
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = "";

// ── State ───────────────────────────────────────────────────────
let isProcessing = false;

// ── DOM Refs ────────────────────────────────────────────────────
const queryInput       = document.getElementById("queryInput");
const sendBtn          = document.getElementById("sendBtn");
const messagesEl       = document.getElementById("messages");
const welcomeScreen    = document.getElementById("welcomeScreen");
const chatContainer    = document.getElementById("chatContainer");
const fileInput        = document.getElementById("fileInput");
const toast            = document.getElementById("toast");
const toastText        = document.getElementById("toastText");
const sidebar          = document.getElementById("sidebar");

// ── Initialization ──────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadDocuments();
    loadMetrics();

    // Auto-resize textarea
    queryInput.addEventListener("input", autoResize);

    // Enter to send (Shift+Enter for newline)
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // File upload handler
    fileInput.addEventListener("change", handleFileUpload);

    // Refresh metrics periodically
    setInterval(loadMetrics, 15000);
});

// ── Auto-resize textarea ────────────────────────────────────────
function autoResize() {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
}

// ── Sidebar Toggle ──────────────────────────────────────────────
function toggleSidebar() {
    sidebar.classList.toggle("collapsed");
}

// ── Toast ───────────────────────────────────────────────────────
function showToast(message) {
    toastText.textContent = message;
    toast.classList.add("active");
}

function hideToast() {
    toast.classList.remove("active");
}

// ── Health Check ────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        // Ollama status
        const ollamaEl = document.getElementById("ollamaStatus");
        const ollamaText = document.getElementById("ollamaStatusText");
        const ollamaIndicator = ollamaEl.querySelector(".status-indicator");

        if (data.ollama?.status === "healthy") {
            ollamaIndicator.className = "status-indicator healthy";
            ollamaText.textContent = data.ollama.configured_model;
            document.getElementById("modelName").textContent = data.ollama.configured_model;
        } else {
            ollamaIndicator.className = "status-indicator unhealthy";
            ollamaText.textContent = "Not connected";
            document.getElementById("modelName").textContent = "Ollama offline";
        }

        // Vector DB status
        const vdbEl = document.getElementById("vectorDbStatus");
        const vdbText = document.getElementById("vectorDbStatusText");
        const vdbIndicator = vdbEl.querySelector(".status-indicator");
        vdbIndicator.className = "status-indicator healthy";
        vdbText.textContent = `${data.vector_db?.total_documents || 0} chunks`;

    } catch (err) {
        console.error("Health check failed:", err);
        document.getElementById("ollamaStatus").querySelector(".status-indicator").className = "status-indicator unhealthy";
        document.getElementById("vectorDbStatus").querySelector(".status-indicator").className = "status-indicator unhealthy";
    }
}

// ── Load Documents ──────────────────────────────────────────────
async function loadDocuments() {
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        const data = await res.json();

        document.getElementById("pdfCount").textContent = data.documents?.length || 0;
        document.getElementById("chunkCount").textContent = data.collection?.total_documents || 0;

        const docList = document.getElementById("docList");
        docList.innerHTML = "";

        if (data.documents && data.documents.length > 0) {
            data.documents.forEach((doc) => {
                const item = document.createElement("div");
                item.className = "doc-item";
                item.innerHTML = `
                    <svg class="doc-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                    <span class="doc-size">${doc.size_mb} MB</span>
                `;
                docList.appendChild(item);
            });
        } else {
            docList.innerHTML = `<div class="doc-item" style="color: var(--text-tertiary); justify-content: center;">No PDFs uploaded yet</div>`;
        }
    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}

// ── Load Metrics ────────────────────────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        const data = await res.json();
        const s = data.summary;

        document.getElementById("metricQueries").textContent = s.total_queries || 0;
        document.getElementById("metricAvgLatency").textContent = s.avg_latency_s ? `${s.avg_latency_s}s` : "—";
        document.getElementById("metricP95").textContent = s.p95_latency_s ? `${s.p95_latency_s}s` : "—";
        document.getElementById("metricAvgChunks").textContent = s.avg_chunks_retrieved || "—";
    } catch (err) {
        // Metrics are non-critical
    }
}

// ── File Upload ─────────────────────────────────────────────────
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    showToast(`Uploading & ingesting ${file.name}…`);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await res.json();
        hideToast();
        loadDocuments();
        checkHealth();

        // Show success in chat
        addSystemMessage(
            `✅ **${file.name}** uploaded and ingested successfully!\n` +
            `Pages: ${data.file?.total_pages || "?"} | Chunks: ${data.file?.chunks || "?"} | ` +
            `Time: ${data.file?.elapsed_seconds || "?"}s`
        );
    } catch (err) {
        hideToast();
        addSystemMessage(`❌ Upload failed: ${err.message}`);
    }

    // Reset input
    fileInput.value = "";
}

// ── Ingest All ──────────────────────────────────────────────────
async function ingestAll() {
    const btn = document.getElementById("ingestBtn");
    btn.disabled = true;
    showToast("Ingesting all PDFs… This may take a while.");

    try {
        const res = await fetch(`${API_BASE}/api/ingest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Ingestion failed");
        }

        const data = await res.json();
        hideToast();
        loadDocuments();
        checkHealth();

        const stats = data.stats;
        addSystemMessage(
            `✅ **Ingestion complete!**\n` +
            `PDFs: ${stats.total_pdfs} | Pages: ${stats.total_pages} | ` +
            `Chunks: ${stats.total_chunks} | OCR pages: ${stats.ocr_pages} | ` +
            `Time: ${stats.elapsed_seconds}s`
        );
    } catch (err) {
        hideToast();
        addSystemMessage(`❌ Ingestion failed: ${err.message}`);
    }

    btn.disabled = false;
}

// ── Send Query ──────────────────────────────────────────────────
async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;

    // Hide welcome screen
    welcomeScreen.style.display = "none";

    // Add user message
    addUserMessage(query);
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Show thinking indicator
    const thinkingId = addThinking();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                top_k: 5,
                enable_reranking: true,
            }),
        });

        // Remove thinking indicator
        removeThinking(thinkingId);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Chat request failed");
        }

        const data = await res.json();
        addAssistantMessage(data);
        loadMetrics();
    } catch (err) {
        removeThinking(thinkingId);
        addSystemMessage(`❌ Error: ${err.message}`);
    }

    isProcessing = false;
    sendBtn.disabled = false;
    queryInput.focus();
}

// Shortcut for welcome hints
function askQuestion(question) {
    queryInput.value = question;
    sendQuery();
}

// ── Message Rendering ───────────────────────────────────────────

function addUserMessage(text) {
    const msg = document.createElement("div");
    msg.className = "message user";
    msg.innerHTML = `
        <div class="message-avatar">You</div>
        <div class="message-body">
            <div class="message-bubble">${escapeHtml(text)}</div>
        </div>
    `;
    messagesEl.appendChild(msg);
    scrollToBottom();
}

function addAssistantMessage(data) {
    const msg = document.createElement("div");
    msg.className = "message assistant";

    // Format answer (simple markdown-like processing)
    const formattedAnswer = formatAnswer(data.answer);

    // Sources
    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
        const sourceTags = data.sources
            .map((s, idx) => {
                const pages = Array.isArray(s.page_numbers) ? s.page_numbers.join(", ") : s.page_numbers;
                return `<span class="source-tag" onclick="showChunkDetail(${idx}, '${data.retrieved_chunks ? "stored" : "none"}')" data-chunk-index="${idx}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    ${escapeHtml(s.filename)} p.${pages}
                </span>`;
            })
            .join("");

        sourcesHtml = `
            <div class="sources-panel">
                <div class="sources-header">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/>
                        <path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/>
                    </svg>
                    Sources
                </div>
                <div class="sources-tags">${sourceTags}</div>
            </div>
        `;
    }

    // Retrieved chunks visualization
    let chunksHtml = "";
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        const chunkCards = data.retrieved_chunks
            .map((c, idx) => {
                const pages = Array.isArray(c.page_numbers) ? c.page_numbers.join(", ") : c.page_numbers;
                const scoreLabel = c.rerank_score != null
                    ? `Score: ${c.rerank_score.toFixed(3)}`
                    : `Dist: ${c.distance.toFixed(3)}`;
                return `
                    <div class="chunk-card" onclick="showChunkModal(${idx})">
                        <div class="chunk-card-header">
                            <span class="chunk-source">${escapeHtml(c.filename)} p.${pages}</span>
                            <span class="chunk-score">${scoreLabel}</span>
                        </div>
                        <div class="chunk-text-preview">${escapeHtml(c.text.substring(0, 200))}…</div>
                    </div>
                `;
            })
            .join("");

        chunksHtml = `
            <div class="chunks-panel">
                <button class="chunks-toggle" onclick="toggleChunks(this)">
                    <span>📊 Retrieved Chunks (${data.retrieved_chunks.length})</span>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </button>
                <div class="chunks-body">${chunkCards}</div>
            </div>
        `;

        // Store chunks data for modal
        msg._chunksData = data.retrieved_chunks;
    }

    // Latency badge
    const totalLatency = data.latency?.total_s || 0;
    const latencyClass = totalLatency < 3 ? "fast" : totalLatency < 5 ? "medium" : "slow";
    const latencyHtml = `
        <div class="latency-badge ${latencyClass}">
            ⚡ ${totalLatency}s total (retrieval: ${data.latency?.retrieval_s || "?"}s, generation: ${data.latency?.generation_s || "?"}s)
        </div>
    `;

    msg.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="message-body">
            <div class="message-bubble">${formattedAnswer}</div>
            ${sourcesHtml}
            ${chunksHtml}
            ${latencyHtml}
        </div>
    `;

    messagesEl.appendChild(msg);
    scrollToBottom();
}

function addSystemMessage(text) {
    const msg = document.createElement("div");
    msg.className = "message assistant";
    msg.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
        <div class="message-body">
            <div class="message-bubble">${formatAnswer(text)}</div>
        </div>
    `;
    messagesEl.appendChild(msg);
    scrollToBottom();
}

let thinkingCounter = 0;

function addThinking() {
    const id = `thinking-${++thinkingCounter}`;
    const msg = document.createElement("div");
    msg.className = "message assistant";
    msg.id = id;
    msg.innerHTML = `
        <div class="message-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="message-body">
            <div class="thinking">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="thinking-text">Searching knowledge base & generating answer…</span>
            </div>
        </div>
    `;
    messagesEl.appendChild(msg);
    scrollToBottom();
    return id;
}

function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Formatting Helpers ──────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    // Convert markdown-like syntax to HTML
    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Source citations [Source: ...]
    html = html.replace(
        /\[Source:\s*(.*?)\]/g,
        '<span class="source-tag" style="display:inline-flex; margin:0 2px; vertical-align:middle;"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/></svg> $1</span>'
    );

    // Line breaks → paragraphs
    html = html
        .split(/\n\n+/)
        .map((p) => `<p>${p.trim()}</p>`)
        .join("");

    // Single line breaks within paragraphs
    html = html.replace(/\n/g, "<br>");

    // Bullet lists (lines starting with - or •)
    html = html.replace(
        /(<p>)((?:\s*[-•]\s+.*(?:<br>)?)+)(<\/p>)/g,
        (match, open, items, close) => {
            const lis = items
                .split(/<br>/)
                .filter((l) => l.trim())
                .map((l) => `<li>${l.replace(/^\s*[-•]\s+/, "")}</li>`)
                .join("");
            return `<ul>${lis}</ul>`;
        }
    );

    // Numbered lists
    html = html.replace(
        /(<p>)((?:\s*\d+\.\s+.*(?:<br>)?)+)(<\/p>)/g,
        (match, open, items, close) => {
            const lis = items
                .split(/<br>/)
                .filter((l) => l.trim())
                .map((l) => `<li>${l.replace(/^\s*\d+\.\s+/, "")}</li>`)
                .join("");
            return `<ol>${lis}</ol>`;
        }
    );

    return html;
}

// ── Chunks Panel Toggle ─────────────────────────────────────────

function toggleChunks(btn) {
    btn.classList.toggle("open");
    const body = btn.nextElementSibling;
    body.classList.toggle("open");
}

// ── Chunk Detail Modal ──────────────────────────────────────────

function showChunkModal(index) {
    // Find the closest message with chunks data
    const messages = document.querySelectorAll(".message.assistant");
    let chunksData = null;

    // Walk backwards to find the message
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i]._chunksData) {
            chunksData = messages[i]._chunksData;
            break;
        }
    }

    if (!chunksData || !chunksData[index]) return;

    const chunk = chunksData[index];
    const pages = Array.isArray(chunk.page_numbers) ? chunk.page_numbers.join(", ") : chunk.page_numbers;

    document.getElementById("modalTitle").textContent = `Chunk: ${chunk.filename}`;
    document.getElementById("modalBody").innerHTML = `
        <div class="meta-row">
            <span class="meta-label">File:</span>
            <span class="meta-value">${escapeHtml(chunk.filename)}</span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Pages:</span>
            <span class="meta-value">${pages}</span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Chunk ID:</span>
            <span class="meta-value">${escapeHtml(chunk.chunk_id)}</span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Distance:</span>
            <span class="meta-value">${chunk.distance?.toFixed(4) || "N/A"}</span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Rerank Score:</span>
            <span class="meta-value">${chunk.rerank_score?.toFixed(4) || "N/A"}</span>
        </div>
        <div class="chunk-full-text">${escapeHtml(chunk.text)}</div>
    `;

    document.getElementById("chunkModal").classList.add("active");
}

// Store chunk data on the message element
const originalAddAssistantMessage = addAssistantMessage;
addAssistantMessage = function (data) {
    originalAddAssistantMessage(data);
    // Attach data to last message
    const lastMsg = messagesEl.lastElementChild;
    if (lastMsg && data.retrieved_chunks) {
        lastMsg._chunksData = data.retrieved_chunks;
    }
};

function showChunkDetail(index) {
    showChunkModal(index);
}

function closeModal(event) {
    if (event.target === event.currentTarget) {
        event.target.classList.remove("active");
    }
}

// ── Scroll ──────────────────────────────────────────────────────

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}
