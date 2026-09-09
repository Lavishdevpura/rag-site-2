"""
PDF -> Markdown conversion, preserving real section-heading structure.

Ported from rag_site_1 (backend/app/pdf_to_markdown.py, itself ported from
Layla's RAG_InsureAI/app/pdf_to_markdown.py on 2026-09-04) — uses
pymupdf4llm, which derives heading LEVELS from the PDF's own font-size/bold
metadata rather than guessing from already-flattened plain text, so headings
come from the document's actual visual structure. This is a genuinely
different signal than document_loader.py's existing pypdf/pdfplumber
extraction path, which pulls plain text only with no heading information
at all.
"""
import logging
import re

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF file to Markdown text with real heading structure.

    Pages with no extractable text layer (scanned/image pages) fall back
    to OCR automatically — pymupdf4llm handles this internally.

    Raises whatever pymupdf4llm raises on a genuinely unreadable/corrupt
    file — callers should treat this the same as any other document-load
    failure, not silently swallow it here.
    """
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(pdf_path)
    logger.info(
        "[pdf_to_markdown] converted %s -> %d chars, %d header lines",
        pdf_path, len(md), len(re.findall(r"^#{1,6}\s+.+$", md, re.MULTILINE)),
    )
    return md


def load_pdf_pages_as_markdown(pdf_path: str) -> dict[int, str]:
    """
    Convert a PDF to markdown PER PAGE, returning {0-indexed page_num: text}
    — the same shape _load_pdf()'s pypdf path already builds, so it can be
    dropped straight into the existing per-page repair pipeline
    (_strip_repeating_page_furniture) with no changes to that function.

    Uses pymupdf4llm's own page_chunks=True mode rather than splitting the
    combined-document convert_pdf_to_markdown() output ourselves — page
    boundaries are already resolved correctly by the library (each chunk
    carries its own metadata["page_number"], 1-indexed).

    Raises whatever pymupdf4llm raises — same contract as
    convert_pdf_to_markdown(), callers handle fallback.
    """
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    pages: dict[int, str] = {}
    for chunk in chunks:
        page_num_1indexed = chunk.get("metadata", {}).get("page_number")
        text = (chunk.get("text") or "").strip()
        if page_num_1indexed and text:
            pages[page_num_1indexed - 1] = text
    logger.info(
        "[pdf_to_markdown] converted %s -> %d/%d page(s) with text",
        pdf_path, len(pages), len(chunks),
    )
    return pages
