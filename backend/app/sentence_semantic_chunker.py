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

Originally shipped without any of SectionChunker's heading-detection or
per-section policy_type classification, on the reasoning that none of it
was needed with metadata filtering off for this fork. ENABLE_METADATA_
FILTERING is now on (see metadata_tagger.py), and retrieval genuinely
depends on chunk.metadata["policy_type"] being accurate — added back a
lightweight, heading-aware version below (Step 6):

  6. After sentence-grouping, re-scan the document's own raw text for
     heading-shaped lines (ported from semantic_chunker.py's
     _is_heading_candidate/_extract_sections — same heuristic, proven on
     this KB's documents already). Each chunk inherits the heading of
     whichever detected section its first sentence falls under. Chunks
     sharing a section_id (source::heading::occurrence) get their
     policy_type decided ONCE from the combined section text via
     metadata_tagger.regex_first_pass_policy_type() — a section-wide
     regex vote is far more reliable than guessing from one ~500-token
     chunk in isolation, same reasoning Layla's SectionChunker documents.
     Deliberately regex-only, no LLM call: ingestion here is designed to
     never block on a slow LLM response (see api.py's _ingest_file,
     which already tags documents with llm=None for the same reason) —
     accuracy is capped at what the fast first-pass regex gets right,
     but that beats one uniform document-level tag for a document that
     spans more than one policy type under different headings.

     The occurrence index in section_id is a deliberate difference from
     Layla's version, not an oversight: Layla's section_id is bare
     f"{source}::{heading}" with no positional disambiguator, so if the
     exact same heading text appears twice in one document (e.g.
     "General Exclusions" under both a Health chapter and a Motor
     chapter of one combined-lines PDF), both occurrences collapse into
     one group and get classified together as if they were one
     contiguous section — confirmed by reading Layla's actual grouping
     code (rag.py's `sections.setdefault(sid, []).append(chunk)`), not
     yet hit in practice there but real. Appending an occurrence count
     here closes that gap from the start rather than porting it.

Chunks from a document with no detectable heading structure at all
(plain prose, webpages, YouTube transcripts) simply don't get a
section_heading — policy_type falls through to api.py's existing
doc-level tag_document() fallback exactly as before, no regression.

Keeps the same split_documents(docs, doc_type=..., llm=..., filename=...)
-> list[Document] call signature as the old SectionChunker purely for
drop-in compatibility with rag.py's RAGPipeline — doc_type/llm/filename
are accepted but unused (section_id uses the same per-document grouping
key _group_docs_by_source already computes from each doc's own
metadata, not the filename argument, since api.py's real call site
doesn't pass one).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

import numpy as np
import tiktoken
from langchain_core.documents import Document

from metadata_tagger import regex_first_pass_policy_type

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
    sentence list, remembering which page each sentence came from.

    Splits sentences on the FULL concatenated text across all pages, not
    per-page. A real sentence can straddle a PDF page break (its subject on
    one page, the predicate carrying the actual defining fact on the next)
    — calling _split_sentences() separately on each doc.page_content tears
    that one sentence into two fragments before the chunker ever sees it as
    a single unit, and each fragment then competes independently for a
    chunk boundary (including Step 5a's 500-token forced split, which has
    no way to know the two fragments were ever one sentence). Confirmed
    live: an HDFC Ergo group-health clause — "Expenses related to the
    treatment of a pre-existing disease (PED) and its direct complications
    shall be excluded until the expiry of 36 months..." — split exactly at
    its page break into "...complications" (page 58) and "shall be
    excluded until..." (page 59) as two separate sentence units. The
    forced-split landed on that same seam, so the retrieved "36 months"
    chunk had lost the words identifying what the 36 months was FOR, and
    generation grabbed an unrelated clause's "no waiting period" line
    instead — a wrong-clause misattribution downstream of a chunk that was
    already missing its own subject. Page numbers are tracked via a
    word-count offset instead of per-page pre-splitting, so each sentence
    still gets attributed to the page its first word actually appears on.
    """
    text_parts: List[str] = []
    page_word_boundaries: List[tuple] = []  # (cumulative_word_count_through_this_page, page_value)
    word_count = 0
    for doc in docs:
        page_value = doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("page_num") or 0
        content = doc.page_content
        if not content:
            continue
        text_parts.append(content)
        word_count += len(content.split())
        page_word_boundaries.append((word_count, page_value))

    full_text = " ".join(text_parts)
    sentences = _split_sentences(full_text)
    if not sentences:
        return [], [], ""

    pages: List[Any] = []
    word_cursor = 0
    boundary_idx = 0
    for sent in sentences:
        while (
            boundary_idx < len(page_word_boundaries) - 1
            and word_cursor >= page_word_boundaries[boundary_idx][0]
        ):
            boundary_idx += 1
        pages.append(page_word_boundaries[boundary_idx][1] if page_word_boundaries else 0)
        word_cursor += len(sent.split())
    return sentences, pages, full_text


# ── Step 6: heading detection (ported from semantic_chunker.py) ─────────
# Same heuristic Layla's chunker already validated on real policy-wording
# PDFs — a heading is a short, Title-Case-or-ALLCAPS line, not a sentence.
_HEADING_BOILERPLATE = {
    "learning objectives", "lesson outline", "lesson round-up", "lesson round up",
    "self-test questions", "self test questions", "professional programme",
    "study material", "list of recommended books", "arrangement of study lessons",
    "practice test paper",
}
_HEADING_PAGE_MARKER_RE = re.compile(r"^[A-Z][A-Z0-9&.\-]{1,12}\s+\d+$")
_HEADING_TOC_RE = re.compile(r"(\.{2,}|…)|\s\d+$")
_TITLE_CASE_SKIP_WORDS = {
    "a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "is",
    "are", "with", "by", "at", "from", "as", "vs", "vs.",
}


def _is_title_case(line: str) -> bool:
    if line[-1:] in ".?!,;:":
        return False
    words = line.split()
    significant = [
        w for w in words
        if any(c.isalpha() for c in w) and w.lower().strip(".,()") not in _TITLE_CASE_SKIP_WORDS
    ]
    if not significant:
        return False
    capitalized = sum(1 for w in significant if w[:1].isupper())
    return capitalized / len(significant) >= 0.8


def _is_heading_candidate(line: str) -> bool:
    line = line.strip()
    if not (3 <= len(line) < 70):
        return False
    if not (line.isupper() or _is_title_case(line)):
        return False
    if len(line.split()) < 2:
        return False
    if line.lower() in _HEADING_BOILERPLATE:
        return False
    if _HEADING_PAGE_MARKER_RE.match(line):
        return False
    if _HEADING_TOC_RE.search(line):
        return False
    return True


def _heading_boundaries(full_text: str) -> List[tuple]:
    """Return [(word_offset, heading_text), ...] for every genuine heading
    line found in *full_text*, in document order. word_offset is the
    cumulative whitespace-split word count up to (not including) that
    heading line — deliberately word-count, not character-count: a
    sentence's text gets whitespace-normalized when _split_sentences
    rejoins pieces (multiple newlines/spaces collapse to one space), so
    a later attempt to re-find a sentence's raw substring inside
    full_text is fragile and was confirmed live to misattribute headings
    (a chunk containing "17.24 Tobacco"'s own body text got attributed to
    the NEXT heading, "17.25 Additional Exclusions", instead). Word count
    survives that whitespace collapsing exactly, since str.split() is
    itself whitespace-run-insensitive — the same reasoning
    _sentences_with_pages already relies on for page-boundary tracking.

    A heading recurring more than twice is almost certainly a running
    page header/footer, not a real structural boundary — excluded the
    same way Layla's _extract_sections already does."""
    lines = full_text.split("\n")
    candidates = [l.strip() for l in lines if _is_heading_candidate(l)]
    freq: dict[str, int] = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1
    genuine = {h for h, c in freq.items() if c <= 2}

    boundaries: List[tuple] = []
    word_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped in genuine and _is_heading_candidate(stripped):
            boundaries.append((word_count, stripped))
        word_count += len(line.split())
    return boundaries


def _assign_section_ids(
    sentences: List[str], full_text: str, source: str,
) -> tuple:
    """For each sentence, find which heading-bounded section (if any) it
    falls under, by tracking each sentence's own cumulative word-count
    position against the heading boundary list (see _heading_boundaries
    for why word count, not character offset). Returns (section_headings,
    section_ids) parallel to *sentences*.

    section_id includes a running occurrence index per heading text
    (source::heading::N) — see module docstring for why: without it, the
    same heading text recurring in two unrelated parts of one document
    would silently merge their policy_type classification together.
    """
    boundaries = _heading_boundaries(full_text)
    if not boundaries:
        return [""] * len(sentences), [f"{source}::chunk" for _ in sentences]

    occurrence: dict[str, int] = {}
    numbered_boundaries: List[tuple] = []  # (word_offset, heading, occurrence_index)
    for offset, heading in boundaries:
        occurrence[heading] = occurrence.get(heading, 0) + 1
        numbered_boundaries.append((offset, heading, occurrence[heading]))

    headings: List[str] = []
    ids: List[str] = []
    word_cursor = 0
    boundary_idx = 0
    for sent in sentences:
        while (
            boundary_idx < len(numbered_boundaries) - 1
            and numbered_boundaries[boundary_idx + 1][0] <= word_cursor
        ):
            boundary_idx += 1
        if word_cursor < numbered_boundaries[0][0]:
            headings.append("")
            ids.append(f"{source}::chunk")
        else:
            _, heading, occ = numbered_boundaries[boundary_idx]
            headings.append(heading)
            ids.append(f"{source}::{heading}::{occ}")
        word_cursor += len(sent.split())
    return headings, ids


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
        self,
        sentences: List[str],
        pages: List[Any],
        section_headings: List[str],
        section_ids: List[str],
        model: Any,
    ) -> tuple:
        if len(sentences) == 1:
            return sentences, pages, section_headings, section_ids

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
        # A chunk inherits the section of its FIRST sentence — a semantic
        # breakpoint occasionally lands mid-section rather than exactly on
        # a heading boundary, same approximation Layla's chunker makes.
        chunk_headings: List[str] = []
        chunk_ids: List[str] = []
        current: List[str] = [sentences[0]]
        current_page = pages[0]
        current_heading = section_headings[0]
        current_id = section_ids[0]
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
                chunk_headings.append(current_heading)
                chunk_ids.append(current_id)
                current = [sent]
                current_page = pages[i]
                current_heading = section_headings[i]
                current_id = section_ids[i]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
            chunk_pages.append(current_page)
            chunk_headings.append(current_heading)
            chunk_ids.append(current_id)

        logger.info(
            "[SentenceSemanticChunker] %d sentences -> %d chunks (breakpoint=%.3f @ p%.0f, max_tokens=%d)",
            len(sentences), len(chunks), breakpoint_distance, self._breakpoint_percentile, self._max_tokens,
        )
        return chunks, chunk_pages, chunk_headings, chunk_ids

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

            sentences, pages, full_text = _sentences_with_pages(group_sorted)
            if not sentences:
                continue

            sent_headings, sent_ids = _assign_section_ids(sentences, full_text, src)
            chunk_texts, chunk_pages, chunk_headings, chunk_ids = self._group_sentences(
                sentences, pages, sent_headings, sent_ids, model,
            )
            pre_overlap_texts = chunk_texts
            chunk_texts = self._apply_overlap(chunk_texts)

            # ── Step 6b: decide policy_type once per section, not once per
            # chunk — group this document's own chunks by section_id and
            # vote from their COMBINED (pre-overlap, to avoid double-
            # counting) text plus the section's heading. See module
            # docstring for why this beats guessing from one isolated
            # ~500-token chunk.
            section_texts: dict[str, list[str]] = {}
            section_heading_by_id: dict[str, str] = {}
            for sid, heading, text in zip(chunk_ids, chunk_headings, pre_overlap_texts):
                section_texts.setdefault(sid, []).append(text)
                section_heading_by_id.setdefault(sid, heading)
            section_policy_type: dict[str, str] = {}
            for sid, texts in section_texts.items():
                try:
                    ptype = regex_first_pass_policy_type(
                        section_heading_by_id.get(sid, ""), "\n\n".join(texts),
                    )
                except Exception as exc:
                    logger.debug("[SentenceSemanticChunker] section policy_type skipped: %s", exc)
                    ptype = "general"
                section_policy_type[sid] = ptype

            base_meta = dict(group_sorted[0].metadata)
            for idx, (text, page) in enumerate(zip(chunk_texts, chunk_pages)):
                sid = chunk_ids[idx]
                heading = chunk_headings[idx]
                meta = {
                    **base_meta,
                    "page": page,
                    "chunk_index": idx,
                    "chunking_method": "sentence_semantic",
                    "section_id": sid,
                }
                if heading:
                    meta["section_heading"] = heading
                # Only set policy_type when the section-level vote actually
                # found something specific — leaving it unset for a
                # "general" section lets api.py's existing chunk-wins/
                # doc-level-fallback guard apply the document-level tag
                # instead, which may carry more signal than one ambiguous
                # section in isolation.
                if section_policy_type.get(sid, "general") != "general":
                    meta["policy_type"] = section_policy_type[sid]
                result.append(Document(page_content=text, metadata=meta))

        return result
