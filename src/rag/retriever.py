"""Embedding + retrieval pipeline with relevance scoring, category filters and attribution.

Three retrieval modes (``rag.embedder``):

* ``hybrid`` (default) - **meaning first, exact words second.** A neural embedder (``rag.hybrid_semantic``) carries most of
  the score so paraphrases match by meaning; a TF-IDF channel adds ``rag.hybrid_lexical_weight`` (default 0.3) so rare
  exact terms ("529", "HSA", "Form 8606", "wash-sale") still count. If the neural model is unavailable it falls back to
  TF-IDF alone with a warning.
* ``sentence_transformers`` / ``openai`` - neural only.
* ``tfidf`` - lexical only (no key, no download).

Each channel has its own FAISS index (``rag.index_type``: exact ``flat`` or approximate ``hnsw``); a chunk's final score is
the weighted sum of its channel scores. See ``src/rag/embedders.py`` and ``src/rag/vector_store.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.core.config import Settings
from src.core.models import Source
from src.rag.documents import Chunk, build_chunks, load_articles, load_glossary
from src.rag.embedders import Embedder, TfidfEmbedder, create_channels
from src.rag.pdf_loader import load_pdfs
from src.rag.vector_store import VectorStore
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# With an approximate index, fetch this many candidates per wanted result before filtering (min FETCH_FLOOR).
OVERSAMPLE = 10
FETCH_FLOOR = 50

# Codes such as "8880", "1099-R" or "590-A": 3+ characters containing a digit. Plain years are too common to mean anything.
_IDENTIFIER = re.compile(r"(?<![\w-])(?=[\w-]*\d)[\w-]{3,}(?![\w-])")
_YEAR = re.compile(r"(19|20|21)\d\d")


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
        lexical: Embedder | None = None,
        lexical_weight: float = 0.3,
    ):
        """``embedder`` is the primary channel; pass ``lexical`` too for hybrid search (scores are blended
        ``(1 - lexical_weight) * primary + lexical_weight * lexical``)."""
        self.chunks = chunks
        self.top_k = top_k
        self.min_score = min_score
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.lexical = lexical
        self.lexical_weight = lexical_weight if lexical is not None else 0.0
        self.index_type = index_type
        self._index_options = index_options or {}
        self._channels: list[tuple[Embedder, VectorStore, float]] = []
        self._haystack: str | None = None
        if chunks:
            self._build()

    @property
    def mode(self) -> str:
        return "hybrid" if self.lexical is not None else self.embedder.name

    @classmethod
    def from_settings(cls, settings: Settings) -> "Retriever":
        kb_dir = settings.resolve_path(settings.get("rag.knowledge_base_dir", "src/data/knowledge_base"))
        glossary = settings.resolve_path(settings.get("rag.glossary_file", "src/data/glossary.json"))
        docs = load_articles(kb_dir) + load_glossary(glossary)
        pdf_dir = settings.get("rag.pdf_dir", "src/data/pdfs")
        if pdf_dir:  # one document per PDF page; set rag.pdf_dir to an empty value to disable
            docs += load_pdfs(settings.resolve_path(pdf_dir), settings.resolve_path(settings.get("rag.pdf_cache_dir", ".cache/pdf_text")))
        chunks = build_chunks(
            docs,
            int(settings.get("rag.chunk_words", 140)),
            int(settings.get("rag.chunk_overlap_words", 30)),
            str(settings.get("rag.chunk_strategy", "recursive")),
        )
        log.info("knowledge base: %d documents -> %d chunks", len(docs), len(chunks))
        primary, lexical, weight = create_channels(settings)
        # Score scales differ by mode: TF-IDF cosine is low and sparse, neural cosine is rarely near zero, and a hybrid
        # score is a blend, so each has its own floor (calibrate with scripts/eval_retrieval.py --calibrate).
        if lexical is not None:
            floor = float(settings.get("rag.hybrid_min_score", 0.30))
        elif primary.name != "tfidf":
            floor = float(settings.get("rag.neural_min_score", 0.30))
        else:
            floor = float(settings.get("rag.min_score", 0.05))
        return cls(
            chunks,
            int(settings.get("rag.top_k", 4)),
            floor,
            embedder=primary,
            index_type=str(settings.get("rag.index_type", "flat")).lower(),
            index_options={
                "hnsw_m": int(settings.get("rag.hnsw_m", 32)),
                "ef_construction": int(settings.get("rag.hnsw_ef_construction", 200)),
                "ef_search": int(settings.get("rag.hnsw_ef_search", 64)),
            },
            lexical=lexical,
            lexical_weight=weight,
        )

    def _build(self) -> None:
        # Titles are repeated so they weigh more than body text.
        corpus = [f"{c.title}. {c.title}. {c.text}" for c in self.chunks]
        channels = [(self.embedder, 1.0 - self.lexical_weight)]
        if self.lexical is not None:
            channels.append((self.lexical, self.lexical_weight))
        for embedder, weight in channels:
            matrix = embedder.fit_transform(corpus)
            try:
                store = VectorStore(matrix.shape[1], index_type=self.index_type, **self._index_options)
            except ValueError as exc:
                log.warning("%s; using flat index", exc)
                store = VectorStore(matrix.shape[1])
            store.add(matrix)
            self._channels.append((embedder, store, weight))

    @property
    def _store(self) -> VectorStore | None:  # primary channel's index
        return self._channels[0][1] if self._channels else None

    def has_identifier(self, query: str) -> bool:
        """True if the query contains a code (form number, publication, rule number...) that appears verbatim in the knowledge base.

        Lets exact-term lookups such as "what is form 8880" count as on-topic even though they contain no finance keyword.
        """
        if self._haystack is None:
            self._haystack = "\n".join(c.text.lower() for c in self.chunks)
        for token in _IDENTIFIER.findall(query.lower()):
            if _YEAR.fullmatch(token):
                continue
            if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", self._haystack):
                return True
        return False

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
        if not query.strip() or not self._channels:
            return []
        k = k or self.top_k
        floor = self.min_score if min_score is None else min_score
        allowed = {category} if isinstance(category, str) else set(category or [])
        queries = [(embedder.encode_query(query), store, weight) for embedder, store, weight in self._channels]
        queries = [(q, store, weight) for q, store, weight in queries if np.any(q)]
        if not queries:
            return []
        total = len(self.chunks)
        approximate = any(store.approximate for _, store, _ in queries)
        # Exact index: the corpus is small, so score everything then filter (avoids missing in-category hits).
        # Approximate index: fetch a generous candidate pool, and widen to everything if filtering leaves too few.
        fetch = min(total, max(k * OVERSAMPLE, FETCH_FLOOR)) if approximate else total
        while True:
            results = self._collect(self._fuse(queries, fetch), k, floor, allowed)
            if len(results) >= k or fetch >= total:
                return results
            fetch = total

    def _fuse(self, queries, fetch: int) -> list[tuple[int, float]]:
        """Weighted sum of per-channel scores, best first. A channel that did not return a chunk contributes 0."""
        fused: dict[int, float] = {}
        for qvec, store, weight in queries:
            for idx, score in store.search(qvec, fetch):
                fused[idx] = fused.get(idx, 0.0) + weight * score
        return sorted(fused.items(), key=lambda item: -item[1])

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
