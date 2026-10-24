"""Embedding + retrieval pipeline with relevance scoring, category filters and attribution.

Embeddings are TF-IDF vectors (unigrams + bigrams, sublinear tf) so the system needs
no embedding API key and builds in milliseconds; the vector store interface is
embedder-agnostic if you want to swap in a neural embedding model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.config import Settings
from src.core.models import Source
from src.rag.documents import Chunk, build_chunks, load_articles, load_glossary
from src.rag.vector_store import VectorStore
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


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
    def __init__(self, chunks: list[Chunk], top_k: int = 4, min_score: float = 0.05):
        self.chunks = chunks
        self.top_k = top_k
        self.min_score = min_score
        self._vectorizer: TfidfVectorizer | None = None
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
        )
        log.info("knowledge base: %d documents -> %d chunks", len(docs), len(chunks))
        return cls(chunks, int(settings.get("rag.top_k", 4)), float(settings.get("rag.min_score", 0.05)))

    def _build(self) -> None:
        # Titles are repeated so they weigh more than body text.
        corpus = [f"{c.title}. {c.title}. {c.text}" for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = self._vectorizer.fit_transform(corpus).toarray().astype("float32")
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
        if not query.strip() or self._store is None or self._vectorizer is None:
            return []
        k = k or self.top_k
        floor = self.min_score if min_score is None else min_score
        allowed = {category} if isinstance(category, str) else set(category or [])
        qvec = self._vectorizer.transform([query]).toarray().astype("float32")
        if not np.any(qvec):
            return []
        # Corpus is small, so rank everything then filter; avoids missing in-category hits.
        ranked = self._store.search(qvec[0], len(self.chunks))
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
