"""PDF text extraction — native text via PyMuPDF, OCR via Tesseract for scanned pages."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Try importing pytesseract; graceful degradation if unavailable
try:
    import pytesseract
    import sys
    import os

    # On Windows, set explicit path if tesseract is installed but not in PATH
    if sys.platform == "win32":
        tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path

    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False
    logger.warning("pytesseract not installed – OCR will be skipped for scanned pages.")


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""

    pdf_id: str
    filename: str
    page_number: int  # 1-indexed
    text: str
    is_ocr: bool = False
    language: Optional[str] = None


@dataclass
class PDFDocument:
    """All extracted pages from a single PDF."""

    pdf_id: str
    filename: str
    filepath: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)


# ── Cleaning helpers ─────────────────────────────────────────────

_HEADER_FOOTER_RE = re.compile(
    r"^(?:\s*page\s+\d+\s*$"              # "Page 123"
    r"|\s*\d+\s*$"                          # bare page numbers
    r"|\s*[-–—]\s*\d+\s*[-–—]\s*$"         # "- 12 -"
    r"|\s*©.*$"                             # copyright lines
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_text(text: str) -> str:
    """Normalize whitespace, strip headers/footers, and collapse blank lines."""
    # Remove common header/footer patterns
    text = _HEADER_FOOTER_RE.sub("", text)
    # Normalize unicode whitespace
    text = re.sub(r"\u00a0", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


# ── Core extraction ──────────────────────────────────────────────

def _extract_text_native(page: pymupdf.Page) -> str:
    """Extract embedded text from a PDF page using PyMuPDF."""
    return page.get_text("text") or ""


def _extract_text_ocr(page: pymupdf.Page, dpi: int = 150) -> str:
    """Render page to image and run Tesseract OCR."""
    if not _HAS_TESSERACT:
        return ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception as exc:
        logger.warning("OCR failed on page: %s", exc)
        return ""


def _is_scanned_page(page: pymupdf.Page, min_text_len: int = 50) -> bool:
    """Heuristic: if native text is very short the page is likely scanned."""
    native = _extract_text_native(page)
    return len(native.strip()) < min_text_len


def _detect_language(text: str) -> Optional[str]:
    """Best-effort language detection."""
    try:
        from langdetect import detect

        if len(text.strip()) > 20:
            return detect(text)
    except Exception:
        pass
    return None


# ── Public API ───────────────────────────────────────────────────

def extract_pdf(filepath: str | Path, pdf_id: Optional[str] = None) -> PDFDocument:
    """
    Extract all text from a PDF file.

    Uses native text extraction where possible, falling back to OCR for
    scanned pages.  Returns a `PDFDocument` with per-page content.
    """
    filepath = Path(filepath)
    if pdf_id is None:
        pdf_id = filepath.stem

    import concurrent.futures
    import os

    with pymupdf.open(str(filepath)) as doc:
        total_pages = len(doc)
        
    logger.info("Processing %s (%d pages)", filepath.name, total_pages)

    def process_single_page(page_idx: int) -> Optional[PageContent]:
        with pymupdf.open(str(filepath)) as local_doc:
            page = local_doc[page_idx]
            is_ocr_flag = False

            if _is_scanned_page(page):
                text_content = _extract_text_ocr(page)
                is_ocr_flag = True
            else:
                text_content = _extract_text_native(page)

            text_content = _clean_text(text_content)

            if not text_content.strip():
                return None

            lang = _detect_language(text_content)

            return PageContent(
                pdf_id=pdf_id,
                filename=filepath.name,
                page_number=page_idx + 1,
                text=text_content,
                is_ocr=is_ocr_flag,
                language=lang,
            )

    pages: List[PageContent] = []
    max_workers = min(16, (os.cpu_count() or 4) + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_single_page, range(total_pages)):
            if result is not None:
                pages.append(result)
    logger.info(
        "Extracted %d non-empty pages from %s (OCR: %d)",
        len(pages),
        filepath.name,
        sum(1 for p in pages if p.is_ocr),
    )

    return PDFDocument(
        pdf_id=pdf_id,
        filename=filepath.name,
        filepath=str(filepath),
        total_pages=total_pages,
        pages=pages,
    )


def extract_all_pdfs(pdf_dir: str | Path) -> List[PDFDocument]:
    """Extract text from every PDF in a directory."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return []

    logger.info("Found %d PDFs in %s", len(pdf_files), pdf_dir)
    documents: List[PDFDocument] = []
    
    import concurrent.futures
    import os

    def safe_extract(fp: Path) -> Optional[PDFDocument]:
        try:
            return extract_pdf(fp)
        except Exception as exc:
            logger.error("Failed to process %s: %s", fp.name, exc)
            return None

    # Limit document-level concurrency to avoid overwhelming the system, 
    # since page-level concurrency is already happening inside extract_pdf.
    max_doc_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_doc_workers) as executor:
        for result in executor.map(safe_extract, pdf_files):
            if result is not None:
                documents.append(result)

    return documents
