"""
sentence_semantic_chunker.py — Sentence-level semantic chunker.

Forked from Layla (InsureHub-RAG) 2026-08-17 as this fork's replacement for
semantic_chunker.py's paragraph-based SemanticChunker and rag.py's
SectionChunker wrapper. The user identified the paragraph-level grouping
as a mistake worth not repeating here, and asked for a different strategy:

  1. Split each document into individual sentences.
  2. Embed every sentence with the shared BGE model (same singleton
     TurboVec already loads — no second model in memory).
  3. Compute cosine similarity between consecutive sentence embeddings,
     converted to distance (1 - similarity).
  4. Per-document breakpoint = the 95th percentile of THAT document's own
     distance distribution (a document whose sentences are naturally more
     varied gets a proportionally looser split threshold than a uniform
     one) — split wherever a pair's distance exceeds it.
  5. Enforce a 500-token ceiling per chunk and a 50-token overlap between
     consecutive chunks, both counted with tiktoken (cl100k_base) rather
     than word/char counts.

Deliberately does NOT do any of SectionChunker's heading-detection or
per-section policy_type classification (verify_and_enrich_section_
metadata, classify_chunk_intent, the cross-type merge veto, page-header
topic mining) — none of that is needed with metadata filtering off for
this fork (see ENABLE_METADATA_FILTERING in metadata_tagger.py). Chunks
simply don't carry a "section"/"policy_type"/"keywords" key; every
downstream reader already falls back safely (e.g.
chunk.metadata.get("policy_type", "general")) when the key is absent.

Keeps the same split_documents(docs, doc_type=..., llm=..., filename=...)
-> list[Document] call signature as the old SectionChunker purely for
drop-in compatibility with rag.py's RAGPipeline — doc_type/llm/filename
are accepted but unused.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

import numpy as np
import tiktoken
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Tuning ────────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
BREAKPOINT_PERCENTILE = 95.0
_MIN_SENTENCE_CHARS = 2
_WORD_WINDOW_SIZE = 20  # fallback granularity for punctuation-free text (raw transcripts)

# ── Embedding model ──────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
_default_model: Any = None
_tokenizer: Any = None


def _get_default_embed_model() -> Any:
    global _default_model
    if _default_model is None:
        logger.warning(
            "[SentenceSemanticChunker] No embed_model passed — falling back to the "
            "shared TurboVec model for '%s'. Pass embed_model explicitly to avoid this path.",
            EMBED_MODEL_NAME,
        )
        # Same reasoning as Layla's chunker: go through TurboVec's shared
        # getter rather than constructing a second SentenceTransformer —
        # avoids a second full model copy in memory and CPU-only device
        # resolution on GPU hosts.
        from turbovec_store import _get_shared_embed_model
        _default_model = _get_shared_embed_model(EMBED_MODEL_NAME)
    return _default_model


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ── Step 1: sentence splitting ───────────────────────────────────────────
# Lightweight regex splitter, no NLP dependency. Splits on sentence-ending
# punctuation followed by whitespace and a capital/digit/quote — the usual
# heuristic false-split (an abbreviation like "Dr." or "U.S.") gets glued
# back onto the following piece via the abbreviation check below.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "e.g", "i.e", "u.s", "u.k", "no", "fig", "approx", "inc", "ltd", "co", "st",
})
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences. Falls back to fixed word windows when
    no real sentence punctuation is found at all (raw transcripts)."""
    text = text.strip()
    if not text:
        return []

    raw_pieces = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    sentences: List[str] = []
    buf = ""
    for piece in raw_pieces:
        buf = f"{buf} {piece}".strip() if buf else piece
        last_word = re.split(r"\s+", buf)[-1].rstrip(".").lower()
        if last_word in _ABBREVIATIONS:
            continue  # false split after an abbreviation — keep accumulating
        sentences.append(buf)
        buf = ""
    if buf:
        sentences.append(buf)

    if len(sentences) >= 2:
        return [s for s in sentences if len(s) >= _MIN_SENTENCE_CHARS]

    # No real sentence punctuation (e.g. an auto-generated transcript with
    # no periods) — fixed word windows give the embedding step something
    # more granular than one giant blob to work with.
    words = text.split()
    if len(words) >= _WORD_WINDOW_SIZE:
        windows = [
            " ".join(words[i:i + _WORD_WINDOW_SIZE])
            for i in range(0, len(words), _WORD_WINDOW_SIZE)
        ]
        return [w for w in windows if len(w) >= _MIN_SENTENCE_CHARS]

    return [text] if len(text) >= _MIN_SENTENCE_CHARS else []


def _page_sort_key(doc: Document) -> int:
    p = doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("page_num") or 0
    try:
        return int(p)
    except (TypeError, ValueError):
        return 0


def _sentences_with_pages(docs: List[Document]) -> tuple:
    """Flatten *docs* (already sorted by page, all from one source) into one
    sentence list, remembering which page each sentence came from. Sentences
    from consecutive pages sit next to each other in the same list, so the
    similarity/breakpoint logic below naturally spans page breaks instead of
    forcing a chunk boundary at every page edge — no marker-embedding trick
    needed since we never re-merge into one blob of raw text."""
    sentences: List[str] = []
    pages: List[Any] = []
    for doc in docs:
        page_value = doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("page_num") or 0
        for sent in _split_sentences(doc.page_content):
            sentences.append(sent)
            pages.append(page_value)
    return sentences, pages


def _group_docs_by_source(docs: List[Document]) -> tuple:
    groups: dict = {}
    order: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("filename") or str(id(doc))
        if src not in groups:
            groups[src] = []
            order.append(src)
        groups[src].append(doc)
    return groups, order


class SentenceSemanticChunker:
    """Sentence-level semantic chunker — see module docstring for the
    algorithm. Parameters mirror the fork plan's locked-in values but are
    overridable for testing."""

    def __init__(
        self,
        embed_model: Any = None,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
        breakpoint_percentile: float = BREAKPOINT_PERCENTILE,
    ):
        self._model = embed_model
        self._max_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._breakpoint_percentile = breakpoint_percentile

    def _model_or_default(self) -> Any:
        return self._model if self._model is not None else _get_default_embed_model()

    # ── Steps 2-4: embed, measure distance, group at the percentile breakpoint ──
    def _group_sentences(
        self, sentences: List[str], pages: List[Any], model: Any,
    ) -> tuple:
        if len(sentences) == 1:
            return sentences, pages

        embeddings = model.encode(
            sentences, normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        )
        distances = [
            float(1.0 - np.dot(embeddings[i - 1], embeddings[i]))
            for i in range(1, len(embeddings))
        ]
        breakpoint_distance = float(np.percentile(distances, self._breakpoint_percentile)) if distances else 0.0

        tokenizer = _get_tokenizer()
        chunks: List[str] = []
        chunk_pages: List[Any] = []
        current: List[str] = [sentences[0]]
        current_page = pages[0]
        current_tokens = len(tokenizer.encode(sentences[0]))

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_tokens = len(tokenizer.encode(sent))
            semantic_break = distances[i - 1] > breakpoint_distance
            # Step 5a: 500-token ceiling — force a split even without a
            # semantic breakpoint if the group would exceed it.
            would_overflow = current_tokens + sent_tokens > self._max_tokens

            if semantic_break or would_overflow:
                chunks.append(" ".join(current))
                chunk_pages.append(current_page)
                current = [sent]
                current_page = pages[i]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
            chunk_pages.append(current_page)

        logger.info(
            "[SentenceSemanticChunker] %d sentences -> %d chunks (breakpoint=%.3f @ p%.0f, max_tokens=%d)",
            len(sentences), len(chunks), breakpoint_distance, self._breakpoint_percentile, self._max_tokens,
        )
        return chunks, chunk_pages

    # ── Step 5b: 50-token overlap between consecutive chunks ────────────────
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self._overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks
        tokenizer = _get_tokenizer()
        result: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = tokenizer.encode(chunks[i - 1])
            tail_tokens = prev_tokens[-self._overlap_tokens:]
            tail = tokenizer.decode(tail_tokens).strip()
            current = chunks[i]
            # Skip overlap if the next chunk already starts with the same
            # content (consecutive pages sometimes repeat boundary text).
            if tail and not current.startswith(tail[:60]):
                result.append(f"{tail} {current}")
            else:
                result.append(current)
        return result

    def split_documents(
        self,
        docs: List[Document],
        doc_type: str = "document",  # accepted for signature compat, unused
        llm: Any = None,             # accepted for signature compat, unused
        filename: str = "",          # accepted for signature compat, unused
    ) -> List[Document]:
        if not docs:
            return []

        groups, order = _group_docs_by_source(docs)
        model = self._model_or_default()
        result: List[Document] = []

        for src in order:
            group = groups[src]
            has_pages = any("page" in d.metadata for d in group)
            group_sorted = sorted(group, key=_page_sort_key) if has_pages else group

            sentences, pages = _sentences_with_pages(group_sorted)
            if not sentences:
                continue

            chunk_texts, chunk_pages = self._group_sentences(sentences, pages, model)
            chunk_texts = self._apply_overlap(chunk_texts)

            base_meta = dict(group_sorted[0].metadata)
            for idx, (text, page) in enumerate(zip(chunk_texts, chunk_pages)):
                result.append(Document(
                    page_content=text,
                    metadata={
                        **base_meta,
                        "page": page,
                        "chunk_index": idx,
                        "chunking_method": "sentence_semantic",
                    },
                ))

        return result
