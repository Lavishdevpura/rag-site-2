"""
sql_rag/schema_index.py — Hybrid (semantic + BM25) retrieval over schema
chunks, fused with Reciprocal Rank Fusion (RRF).

Adapted from the standalone Sql Query Generator project (2026-08-17),
simplified for this codebase:
  - No FAISS, no second embedding model. The standalone project's FAISS
    IndexFlatIP over normalized vectors is exact brute-force cosine
    similarity, not approximate search — for a schema with a realistic
    number of tables (dozens, not millions), a plain numpy dot-product
    over an in-memory array does the identical thing without a new
    dependency. Reuses the SAME shared BGE embedding model the document
    RAG already has loaded (turbovec_store._get_shared_embed_model) rather
    than loading a second, separate model (the original project's
    all-MiniLM-L6-v2) into memory.
  - Index is rebuilt on demand (call reset_schema_index() after a schema
    change) rather than only at process startup — a table can be added at
    any time relative to this process's lifetime.
"""
import re
import threading
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from sql_rag.config import SCHEMA_HYBRID_ALPHA, SCHEMA_RETRIEVAL_TOP_K
from sql_rag.schema_loader import load_schema_chunks

RRF_K = 60  # standard RRF constant


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class _SchemaIndex:
    """Process-wide singleton holding both the embedding matrix and the
    BM25 index over the current schema chunks. Rebuilding is cheap (schema
    chunk counts are small — one chunk per table) so a full rebuild on
    every reset is simpler and safer than trying to diff/patch in place."""

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None  # (N, dim), L2-normalized
        self._bm25: BM25Okapi | None = None
        self._built = False

    def _build(self) -> None:
        chunks = load_schema_chunks()
        self._chunks = chunks
        if not chunks:
            self._embeddings = None
            self._bm25 = None
            self._built = True
            return

        from turbovec_store import _get_shared_embed_model, EMBED_MODEL_NAME
        model = _get_shared_embed_model(EMBED_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        self._embeddings = model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False,
        ).astype(np.float32)

        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        self._built = True

    def ensure_built(self) -> None:
        if not self._built:
            with self._lock:
                if not self._built:
                    self._build()

    def rebuild(self) -> None:
        with self._lock:
            self._build()

    def semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self.ensure_built()
        if self._embeddings is None or not self._chunks:
            return []
        from turbovec_store import _get_shared_embed_model, EMBED_MODEL_NAME
        model = _get_shared_embed_model(EMBED_MODEL_NAME)
        q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._embeddings @ q_vec  # cosine similarity, both sides normalized
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    def bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self.ensure_built()
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            chunk = dict(self._chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
        return results


_index = _SchemaIndex()


def reset_schema_index() -> None:
    """Force a rebuild on next use — call after a schema change (a table
    added/removed) so retrieval sees it without a process restart."""
    _index.rebuild()


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def hybrid_retrieve(
    query: str,
    top_k: int | None = None,
    alpha: float | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant schema chunks via RRF-fused hybrid search."""
    top_k = top_k if top_k is not None else SCHEMA_RETRIEVAL_TOP_K
    alpha = alpha if alpha is not None else SCHEMA_HYBRID_ALPHA
    fetch_k = max(top_k * 2, 10)

    semantic_results = _index.semantic_search(query, fetch_k)
    bm25_results = _index.bm25_search(query, fetch_k)

    scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    for rank, chunk in enumerate(semantic_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + alpha * _rrf_score(rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) * _rrf_score(rank)
        chunk_map[cid] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in sorted_ids[:top_k]:
        chunk = dict(chunk_map[cid])
        chunk["hybrid_score"] = round(scores[cid], 6)
        results.append(chunk)
    return results


def build_context_string(chunks: list[dict[str, Any]]) -> str:
    parts = [f"[Context {i}]\n{c['text']}" for i, c in enumerate(chunks, 1)]
    return "\n\n".join(parts)
