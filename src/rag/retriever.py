"""Embedding + retrieval pipeline with relevance scoring, category filters and attribution.

The embedder is pluggable (``rag.embedder``): TF-IDF by default (no key, builds in milliseconds), or a neural
model for true semantic search. The index is FAISS ``flat`` (exact) or ``hnsw`` (approximate), set by
``rag.index_type``; see ``src/rag/embedders.py`` and ``src/rag/vector_store.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.config import Settings
from src.core.models import Source
from src.rag.documents import Chunk, build_chunks, load_articles, load_glossary
from src.rag.embedders import Embedder, TfidfEmbedder, create_embedder
from src.rag.vector_store import VectorStore
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# With an approximate index, fetch this many candidates per wanted result before filtering (min FETCH_FLOOR).
OVERSAMPLE = 10
FETCH_FLOOR = 50


@dataclass
class RetrievedChunk:
    text: str
    title: str
    category: str
    reference: str
    score: float

    def to_source(self) -> Source:
        return Source(self.title, self.category, self.reference, round(self.score, 3))


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        top_k: int = 4,
        min_score: float = 0.05,
        embedder: Embedder | None = None,
        index_type: str = "flat",
        index_options: dict | None = None,
    ):
        self.chunks = chunks
        self.top_k = top_k
        self.min_score = min_score
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.index_type = index_type
        self._index_options = index_options or {}
        self._store: VectorStore | None = None
        if chunks:
            self._build()

    @classmethod
    def from_settings(cls, settings: Settings) -> "Retriever":
        kb_dir = settings.resolve_path(settings.get("rag.knowledge_base_dir", "src/data/knowledge_base"))
        glossary = settings.resolve_path(settings.get("rag.glossary_file", "src/data/glossary.json"))
        docs = load_articles(kb_dir) + load_glossary(glossary)
        chunks = build_chunks(
            docs,
            int(settings.get("rag.chunk_words", 140)),
            int(settings.get("rag.chunk_overlap_words", 30)),
            str(settings.get("rag.chunk_strategy", "recursive")),
        )
        log.info("knowledge base: %d documents -> %d chunks", len(docs), len(chunks))
        return cls(
            chunks,
            int(settings.get("rag.top_k", 4)),
            float(settings.get("rag.min_score", 0.05)),
            embedder=create_embedder(settings),
            index_type=str(settings.get("rag.index_type", "flat")).lower(),
            index_options={
                "hnsw_m": int(settings.get("rag.hnsw_m", 32)),
                "ef_construction": int(settings.get("rag.hnsw_ef_construction", 200)),
                "ef_search": int(settings.get("rag.hnsw_ef_search", 64)),
            },
        )

    def _build(self) -> None:
        # Titles are repeated so they weigh more than body text.
        corpus = [f"{c.title}. {c.title}. {c.text}" for c in self.chunks]
        matrix = self.embedder.fit_transform(corpus)
        try:
            self._store = VectorStore(matrix.shape[1], index_type=self.index_type, **self._index_options)
        except ValueError as exc:
            log.warning("%s; using flat index", exc)
            self._store = VectorStore(matrix.shape[1])
        self._store.add(matrix)

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self.chunks})

    @property
    def document_count(self) -> int:
        return len({c.doc_id for c in self.chunks})

    @property
    def backend(self) -> str:
        return self._store.backend if self._store else "none"

    def search(
        self,
        query: str,
        k: int | None = None,
        category: str | list[str] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k chunks (one per article) above ``min_score``, optionally within categories."""
        if not query.strip() or self._store is None:
            return []
        k = k or self.top_k
        floor = self.min_score if min_score is None else min_score
        allowed = {category} if isinstance(category, str) else set(category or [])
        qvec = self.embedder.encode_query(query)
        if not np.any(qvec):
            return []
        total = len(self.chunks)
        # Exact index: the corpus is small, so rank everything then filter (avoids missing in-category hits).
        # Approximate index: fetch a generous candidate pool, and widen to everything if filtering leaves too few.
        fetch = min(total, max(k * OVERSAMPLE, FETCH_FLOOR)) if self._store.approximate else total
        while True:
            results = self._collect(self._store.search(qvec, fetch), k, floor, allowed)
            if len(results) >= k or fetch >= total:
                return results
            fetch = total

    def _collect(self, ranked: list[tuple[int, float]], k: int, floor: float, allowed: set[str]) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if score < floor:
                break
            if (allowed and chunk.category not in allowed) or chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            results.append(RetrievedChunk(chunk.text, chunk.title, chunk.category, chunk.reference, score))
            if len(results) >= k:
                break
        return results
