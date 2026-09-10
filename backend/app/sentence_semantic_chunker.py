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
  5. Enforce a 500-token ceiling per chunk (tiktoken/cl100k_base), and
     carry the LAST FULL SENTENCE of each chunk forward as a one-sentence
     overlap prefix on the next chunk (2026-09-10 — replaced a fixed
     50-token tail, which sliced raw tokens with no regard for sentence
     boundaries and could hand the next chunk half of a sentence, or a
     meaningless token fragment, instead of real leading context. Reuses
     _split_sentences, the same sentence-splitter Step 1 already applies
     to the whole document, so the overlap is always a complete,
     grammatical sentence rather than an arbitrary token slice).

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

Real markdown structure added 2026-09-08 (ported from rag_site_1, itself
ported from Layla 2026-09-04) — document_loader.py's PDF loader now tries
pymupdf4llm first (see pdf_to_markdown.py), which derives heading LEVELS
from the PDF's own font-size/bold metadata rather than guessing from
plain text. _md_structure_boundaries() parses that markdown with a real
parser (markdown-it-py) into a genuine token tree — real heading_open
tokens with correct level info — instead of guessing at line shapes, and
is tried FIRST in _assign_section_ids(); the plain-text ALLCAPS/Title-Case
heuristic (_heading_boundaries, Step 6 above) remains the fallback for
content with no real markdown (webpages, transcripts, or a PDF that fell
back to plain pypdf/pdfplumber extraction).

Sub-heading detection rebuilt 2026-09-09 — the original version tried to
manufacture a sub-heading from list-item text (the first ~7 words of a
numbered/lettered item, gated by a spaCy clause check). Confirmed live
that approach was broken at the root, not just mistuned: a list item's
own text is body content under whatever real heading it falls in, never
a heading itself, so truncating it can't produce a real label — a
third-party test PDF and rag_site_1's own real corpus both showed the
same failure shape (labels cut off mid-sentence, e.g. "Policy Exceptions
to cover applying to the"). Now uses ONLY real heading tokens for both
tiers: the document's own shallowest heading level is section_heading,
the next level down is sub_heading — genuine nested structure (e.g. an
H3 under an H2), the same reliable detection section_heading already
used, just one tier deeper. No spaCy dependency needed for this anymore.
A chunk's heading/sub_heading still forces a hard chunk-boundary break
during sentence-grouping (Step 2-4) — without it, sub-headings essentially
never survive to final chunk metadata (confirmed in rag_site_1: 0/83 on a
real document) because the semantic grouper otherwise merges straight
across a sub-heading transition.

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
OVERLAP_SENTENCES = 1
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
    # Currency/regulatory abbreviations common in Indian insurance
    # documents (2026-09-10, found via live testing): "Rs." immediately
    # followed by a figure — "(Rs. 25,00,000)" — matches the split
    # regex's own trigger (period, space, then a digit) just as reliably
    # as any sentence boundary does, incorrectly cutting a monetary
    # figure away from the clause introducing it. Confirmed live: this
    # silently split "...a sub-limit of twenty-five lakh rupees (Rs." from
    # "25,00,000) per Policy Period, which sub-limit forms part of..." —
    # two fake "sentences" with the actual number orphaned at the start of
    # the second one, undermining even the one-sentence overlap fix (an
    # overlap can only carry forward a REAL last sentence; it can't fix a
    # sentence that was never really one boundary to begin with).
    "rs",
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
        # Strip leading punctuation too, not just trailing (2026-09-10,
        # found alongside the "Rs." gap above) — a parenthesized
        # abbreviation like "(Rs." left the leading "(" attached, so the
        # lookup compared "(rs" against the abbreviations set instead of
        # "rs" and never matched, even once "rs" was added to the set.
        last_word = re.sub(r"[^a-z.]", "", re.split(r"\s+", buf)[-1].lower()).rstrip(".")
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
#
# Markdown "#"/"##" lines checked FIRST (2026-09-08, ported from
# rag_site_1) — document_loader.py's _load_pdf() now tries a pymupdf4llm
# markdown conversion before falling back to plain pypdf/pdfplumber text,
# and pymupdf4llm derives heading levels from the PDF's own font-size/bold
# metadata rather than guessing from plain text. A markdown heading line
# is trusted directly rather than run through the Title-Case/ALLCAPS
# heuristic built for plain-text extraction.
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")
_HEADING_DECORATION_RE = re.compile(r"\*\*|__|<u>|</u>|<b>|</b>|<sup>|</sup>|<sub>|</sub>")


def _clean_heading_text(text: str) -> str:
    return _HEADING_DECORATION_RE.sub("", text).strip()


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
    _md_match = _MARKDOWN_HEADING_RE.match(line)
    if _md_match:
        heading_text = line[_md_match.end():].strip()
        if not (1 <= len(heading_text) < 200):
            return False
        if heading_text.lower() in _HEADING_BOILERPLATE:
            return False
        return True
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
    """Return [(word_offset, heading_text), ...] for every genuine ALLCAPS/
    Title-Case heading LINE found in plain (non-markdown) *full_text* —
    the fallback path for content with no real markdown structure at all
    (webpages, video transcripts, or a PDF that fell back to the plain
    pypdf/pdfplumber extraction path). See _md_structure_boundaries for
    the PRIMARY path used on real markdown text (from pymupdf4llm), which
    _assign_section_ids tries first — this function only ever runs when
    that one finds nothing.

    word_offset is the cumulative whitespace-split word count up to (not
    including) that heading line — deliberately word-count, not
    character-count: a sentence's text gets whitespace-normalized when
    _split_sentences rejoins pieces (multiple newlines/spaces collapse to
    one space), so a later attempt to re-find a sentence's raw substring
    inside full_text is fragile and was confirmed live to misattribute
    headings (a chunk containing "17.24 Tobacco"'s own body text got
    attributed to the NEXT heading, "17.25 Additional Exclusions",
    instead). Word count survives that whitespace collapsing exactly,
    since str.split() is itself whitespace-run-insensitive — the same
    reasoning _sentences_with_pages already relies on for page-boundary
    tracking.

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
            _md_match = _MARKDOWN_HEADING_RE.match(stripped)
            heading_text = _clean_heading_text(
                stripped[_md_match.end():] if _md_match else stripped
            )
            boundaries.append((word_count, heading_text))
        word_count += len(line.split())
    return boundaries


# ── Structure via a real Markdown parser (2026-09-08, ported from
# rag_site_1) ─────────────────────────────────────────────────────────────
# Earlier versions of heading detection here (and in Layla's own
# semantic_chunker.py, ported from) scanned *full_text* LINE BY LINE with
# regex, guessing at heading/list-item shapes. Two real problems with that
# approach, confirmed live in rag_site_1 against a real, independently-
# downloaded IRDAI regulatory PDF (not a self-constructed test case):
#
#   1. Line-by-line regex has no concept of NESTING. A real clause list like
#      "b) In case of LC; - i. the LC shall be issued... - ii. the cedant
#      may choose..." has "i."/"ii." genuinely NESTED inside "b)", not flat
#      siblings of "a)"/"b)" — regex matching each line in isolation cannot
#      represent that at all, it only ever produces a flat stream of
#      "found a marker" events.
#   2. Guessing at markdown SHAPE (is this bold? does it start with a
#      bullet glyph? a clause marker?) is an ever-growing pile of special
#      cases that still misses whatever shape wasn't anticipated yet.
#
# markdown-it-py (already an installed dependency of pymupdf4llm — no new
# library needed) parses *full_text* into a REAL token tree: genuine
# heading_open/inline pairs for actual headings (however the source's own
# markdown encoded them), and genuine list_item_open tokens for list items,
# WITH CORRECT NESTING understood natively via the parser's own `level`
# and `map` (line-range) attributes. This replaces line-by-line regex
# shape-guessing with a parser that actually understands document
# structure.
try:
    from markdown_it import MarkdownIt
    _MD_PARSER: Optional["MarkdownIt"] = MarkdownIt("commonmark")
except Exception as _md_import_exc:  # pragma: no cover - defensive only
    logger.warning(
        "[SentenceSemanticChunker] markdown-it-py unavailable (%s) — "
        "structure detection falls back to the plain-text heuristic only",
        _md_import_exc,
    )
    _MD_PARSER = None


def _md_structure_boundaries(full_text: str) -> tuple:
    """Parse *full_text* as real Markdown and return (heading_boundaries,
    subheading_boundaries) — both [(word_offset, label), ...] in document
    order, the same shape the plain-text fallback (_heading_boundaries)
    produces, so callers don't need to know which path found them.

    Rebuilt 2026-09-09 (real bug, not a tuning tweak — a live, independent
    test against a genuine third-party PDF, plus rag_site_1's own real
    corpus, both showed the same failure): the previous version treated a
    numbered/lettered LIST ITEM's own body text as a sub-heading, taking
    its first ~7 words as the "label". A list item's text is body content
    under whatever real heading it falls in — it was never a heading to
    begin with, so no amount of smarter truncation could fix labels like
    "Policy Exceptions to cover applying to the" (confirmed live, cut off
    mid-sentence on both the third-party test PDF and rag_site_1's real
    IRDAI corpus) — the input being truncated genuinely isn't a title.

    A genuine sub-heading is what it sounds like: a real, separate heading
    ONE LEVEL DEEPER than the section it sits under (an H3 under an H2, or
    an H2 under an H1) — the exact same kind of structure section_heading
    itself already detects correctly (real heading_open/inline token
    pairs, whatever level the source's own markdown used), just one tier
    down. So this function now uses ONLY real heading tokens for both —
    list items are never consulted at all. Every heading actually found
    is grouped by its own level (h1=1 .. h6=6); the document's OWN
    shallowest level becomes section_heading, and the next level down
    becomes sub_heading. A document with 3+ real levels still collapses
    everything deeper than the top tier into that one sub_heading field
    (a single string, not a hierarchy) — matching the field's existing
    shape rather than losing the top-level distinction, which is the one
    that actually matters. A document with only one heading level (no
    real nested structure at all) correctly returns an empty subheading
    list — sub_heading should be genuinely absent then, not invented.
    """
    if _MD_PARSER is None or not full_text.strip():
        return [], []
    tokens = _MD_PARSER.parse(full_text)
    lines = full_text.split("\n")
    word_offsets: List[int] = []
    cursor = 0
    for line in lines:
        word_offsets.append(cursor)
        cursor += len(line.split())
    word_offsets.append(cursor)

    all_headings: List[tuple] = []  # (word_offset, label, level)
    pending_map: Optional[list] = None
    pending_level: Optional[int] = None

    for tok in tokens:
        if tok.type == "heading_open":
            pending_map = tok.map
            pending_level = (
                int(tok.tag[1]) if tok.tag and len(tok.tag) == 2 and tok.tag[1:].isdigit() else 1
            )
        elif tok.type == "inline" and pending_map is not None:
            offset = word_offsets[pending_map[0]]
            label = _clean_heading_text(tok.content).rstrip(":;,.")
            if label:
                all_headings.append((offset, label, pending_level))
            pending_map, pending_level = None, None

    if not all_headings:
        return [], []

    # The real "section" tier is the shallowest level that actually
    # REPEATS (appears 2+ times) — not just the shallowest level present.
    # Confirmed live this distinction matters: a real third-party PDF has
    # exactly one H1 (a cover-page document title, "Arch Insurance") sitting
    # above dozens of real H2 sections ("Making a Claim", "Policy",
    # "Cancellation", ...). Treating that lone H1 as the section tier
    # demoted every genuinely useful section title down into sub_heading
    # instead, while section_heading — the field retrieval and citations
    # actually read — ended up with nothing but the same document title on
    # every single chunk. A one-off top-level heading is a title, not a
    # section boundary; the level that actually organizes the document
    # into multiple real parts is the one that recurs. Falls back to the
    # plain shallowest level if NO level repeats at all (e.g. a short
    # document with 2-3 uniquely-occurring headings) — there's no
    # "recurring tier" signal to prefer over that case.
    level_counts: dict = {}
    for _, _, lvl in all_headings:
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    _repeating_levels = [lvl for lvl, cnt in level_counts.items() if cnt >= 2]
    section_level = min(_repeating_levels) if _repeating_levels else min(level_counts)

    headings = [(off, label) for off, label, lvl in all_headings if lvl == section_level]
    subheadings = [(off, label) for off, label, lvl in all_headings if lvl > section_level]

    # Cover-title exclusion (2026-09-09, ported from rag_site_1 — identical
    # bug, shared logic): a document/circular's own cover title can sit at
    # the SAME pymupdf4llm-assigned level as its genuine, later chapter
    # headings — different from the one-off-title case the repeating-level
    # rule above already handles (a lone heading with NOTHING else at that
    # level, correctly demoted out of section_level entirely), this shape
    # has real, repeating chapter headings at the same level too, so the
    # level legitimately becomes section_level — but the cover title still
    # gets wrongly included as if it were one of the chapters. Confirmed
    # live on rag_site_1's real IRDAI reinsurance circular test document:
    # its own cover title (occurring only on page 1) sat at H1 alongside
    # genuine chapter headings, and became the "section" for 8 chunks of
    # unrelated cover-page/letterhead content (address, email). Loops
    # rather than a single strip — pymupdf4llm can render the same cover
    # title TWICE near the document start (once as plain bold text, once
    # as a formal H1 token a few words later), and a one-shot removal only
    # catches the first occurrence. Keeps removing from the front only
    # while it's both near the document's start AND enough OTHER real
    # headings remain at this level afterward (>=2) to still be meaningful
    # as the section tier — so a document whose genuine first section
    # legitimately opens on page 1 doesn't lose it, and this can't strip
    # real content no matter how many near-start duplicates a title
    # happens to render as.
    _COVER_TITLE_MAX_OFFSET = 50
    while (
        len(headings) >= 3
        and headings[0][0] <= _COVER_TITLE_MAX_OFFSET
    ):
        headings = headings[1:]

    return headings, subheadings


def _assign_section_ids(
    sentences: List[str], full_text: str, source: str,
) -> tuple:
    """For each sentence, find which heading-bounded section (if any) it
    falls under, and which bold numbered/lettered sub-item (if any) within
    that section, by tracking each sentence's own cumulative word-count
    position against the heading and sub-heading boundary lists (see
    _heading_boundaries for why word count, not character offset). Returns
    (section_headings, section_sub_headings, section_ids) parallel to
    *sentences*.

    A sub-heading is scoped to its OWN parent heading section only — it
    never carries over past the next heading boundary, even if no further
    sub-heading boundary resets it first (a section with one sub-item list
    shouldn't leak that sub-item's label onto the next unrelated section's
    sentences).

    section_id includes the sub_heading plus a running occurrence index
    per (heading, sub_heading) pair (source::heading::sub_heading::N) —
    without the occurrence index, the same heading (or heading+sub_heading
    pair) recurring in two unrelated parts of one document would silently
    merge their sentences into one section, and its policy_type
    classification with it.
    """
    # Real markdown structure (from pymupdf4llm's PDF conversion) tried
    # first (2026-09-08, ported from rag_site_1) — it understands actual
    # document structure (heading levels, list nesting), not guessed text
    # shapes. Falls back to the plain-text ALLCAPS/Title-Case heuristic
    # only when there's no real markdown at all (webpages, video
    # transcripts, or a PDF that used the plain-text extraction fallback)
    # — that fallback never finds sub-headings (plain text has no
    # bold/list markup to signal one), which matches this file's own
    # behavior before this port.
    boundaries, sub_boundaries = _md_structure_boundaries(full_text)
    if not boundaries:
        boundaries = _heading_boundaries(full_text)
        sub_boundaries = []
    if not boundaries:
        return [""] * len(sentences), [""] * len(sentences), [f"{source}::chunk" for _ in sentences]

    numbered_boundaries: List[tuple] = []  # (word_offset, heading)
    for offset, heading in boundaries:
        numbered_boundaries.append((offset, heading))

    headings: List[str] = []
    sub_headings: List[str] = []
    ids: List[str] = []
    occurrence: dict[tuple, int] = {}
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
            sub_headings.append("")
            ids.append(f"{source}::chunk")
            word_cursor += len(sent.split())
            continue

        _, heading = numbered_boundaries[boundary_idx]
        _section_start = numbered_boundaries[boundary_idx][0]
        _section_end = (
            numbered_boundaries[boundary_idx + 1][0]
            if boundary_idx + 1 < len(numbered_boundaries) else float("inf")
        )
        current_sub = ""
        for _sub_offset, _sub_text in sub_boundaries:
            if _sub_offset > word_cursor:
                break
            if _section_start <= _sub_offset < _section_end:
                current_sub = _sub_text

        # Occurrence increments once per NEW (heading, sub_heading) span
        # entered, not once per sentence — detected by a change from the
        # immediately preceding sentence's own (heading, sub_heading).
        key = (heading, current_sub)
        if not headings or headings[-1] != heading or sub_headings[-1] != current_sub:
            occurrence[key] = occurrence.get(key, 0) + 1
        occ = occurrence[key]

        headings.append(heading)
        sub_headings.append(current_sub)
        _sub_part = f"::{current_sub}" if current_sub else ""
        ids.append(f"{source}::{heading}{_sub_part}::{occ}")
        word_cursor += len(sent.split())
    return headings, sub_headings, ids


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
        overlap_sentences: int = OVERLAP_SENTENCES,
        breakpoint_percentile: float = BREAKPOINT_PERCENTILE,
    ):
        self._model = embed_model
        self._max_tokens = max_chunk_tokens
        self._overlap_sentences = overlap_sentences
        self._breakpoint_percentile = breakpoint_percentile

    def _model_or_default(self) -> Any:
        return self._model if self._model is not None else _get_default_embed_model()

    # ── Steps 2-4: embed, measure distance, group at the percentile breakpoint ──
    def _group_sentences(
        self,
        sentences: List[str],
        pages: List[Any],
        section_headings: List[str],
        section_sub_headings: List[str],
        section_ids: List[str],
        model: Any,
    ) -> tuple:
        if len(sentences) == 1:
            return sentences, pages, section_headings, section_sub_headings, section_ids

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
        # A heading/sub_heading transition forces a hard chunk break
        # (2026-09-08, ported from rag_site_1). This chunker groups by
        # semantic similarity across the WHOLE document — without this
        # explicit force-break, nothing stops the semantic grouper from
        # merging sentences across a sub-heading transition, and a chunk
        # only ever reports whichever heading/sub_heading its OWN first
        # sentence happened to carry. Confirmed live in rag_site_1
        # (2026-09-04): without this, sub_heading survived to the final
        # chunk metadata 0/83 times on a real re-ingested document despite
        # the per-sentence detection correctly finding real sub-headings
        # throughout — sub-headings are fine-grained enough that a
        # semantically-grouped chunk almost always spans past one, unlike
        # coarser top-level headings.
        chunk_headings: List[str] = []
        chunk_sub_headings: List[str] = []
        chunk_ids: List[str] = []
        current: List[str] = [sentences[0]]
        current_page = pages[0]
        current_heading = section_headings[0]
        current_sub_heading = section_sub_headings[0]
        current_id = section_ids[0]
        current_tokens = len(tokenizer.encode(sentences[0]))

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_tokens = len(tokenizer.encode(sent))
            semantic_break = distances[i - 1] > breakpoint_distance
            # Step 5a: 500-token ceiling — force a split even without a
            # semantic breakpoint if the group would exceed it.
            would_overflow = current_tokens + sent_tokens > self._max_tokens
            # Force a break on any heading/sub_heading transition — see
            # the comment above chunk_headings' declaration.
            section_break = (
                section_headings[i] != current_heading
                or section_sub_headings[i] != current_sub_heading
            )

            if semantic_break or would_overflow or section_break:
                chunks.append(" ".join(current))
                chunk_pages.append(current_page)
                chunk_headings.append(current_heading)
                chunk_sub_headings.append(current_sub_heading)
                chunk_ids.append(current_id)
                current = [sent]
                current_page = pages[i]
                current_heading = section_headings[i]
                current_sub_heading = section_sub_headings[i]
                current_id = section_ids[i]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
            chunk_pages.append(current_page)
            chunk_headings.append(current_heading)
            chunk_sub_headings.append(current_sub_heading)
            chunk_ids.append(current_id)

        logger.info(
            "[SentenceSemanticChunker] %d sentences -> %d chunks (breakpoint=%.3f @ p%.0f, max_tokens=%d)",
            len(sentences), len(chunks), breakpoint_distance, self._breakpoint_percentile, self._max_tokens,
        )
        return chunks, chunk_pages, chunk_headings, chunk_sub_headings, chunk_ids

    # ── Step 5b: N-sentence overlap between consecutive chunks ──────────────
    # 2026-09-10, user's explicit direction: replaced a fixed 50-TOKEN tail
    # (a raw tiktoken slice with no regard for sentence boundaries) with a
    # fixed number of COMPLETE trailing sentences instead. Confirmed live
    # this matters, not just theoretical: a real ingested test document's
    # own dense, multi-item paragraph got its "Cyber Extortion" clause
    # split exactly at a chunk boundary — the chunk ending there carried
    # the concept ("...third party threatening to release, damage, or deny
    # access to the Insured's") but the token-based overlap tail, sliced
    # with no sentence awareness, didn't reliably carry that whole
    # sentence forward, so the NEXT chunk (which had the actual number)
    # started mid-sentence with no self-contained context of its own.
    # Reusing _split_sentences (the same splitter Step 1 already applies to
    # the whole document) means the carried-forward text is always a
    # complete, grammatical sentence — never a half-sentence or a
    # meaningless token fragment — giving the next chunk genuine standalone
    # context for whatever fact its own primary content continues.
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self._overlap_sentences <= 0 or len(chunks) <= 1:
            return chunks
        result: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_sentences = _split_sentences(chunks[i - 1])
            tail = " ".join(prev_sentences[-self._overlap_sentences:]).strip()
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

            sent_headings, sent_sub_headings, sent_ids = _assign_section_ids(sentences, full_text, src)
            chunk_texts, chunk_pages, chunk_headings, chunk_sub_headings, chunk_ids = self._group_sentences(
                sentences, pages, sent_headings, sent_sub_headings, sent_ids, model,
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
                sub_heading = chunk_sub_headings[idx]
                meta = {
                    **base_meta,
                    "page": page,
                    "chunk_index": idx,
                    "chunking_method": "sentence_semantic",
                    "section_id": sid,
                }
                if heading:
                    meta["section_heading"] = heading
                if sub_heading:
                    meta["sub_heading"] = sub_heading
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
