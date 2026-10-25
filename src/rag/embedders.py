"""Embedding back ends for the retriever. All return L2-normalised float32 vectors, so inner product = cosine.

* ``tfidf``                 - default; lexical (unigrams + bigrams), no key, no download, builds in milliseconds.
* ``sentence_transformers`` - local neural model (``pip install sentence-transformers``); true semantic matching.
* ``openai``                - OpenAI embeddings API (needs ``OPENAI_API_KEY``); token usage goes to ``llm_usage.jsonl``.

``create_embedder`` never raises: if the chosen back end is unavailable it logs a warning and uses TF-IDF.
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.core.config import Settings
from src.utils.logging_utils import get_logger
from src.utils.usage_log import caller_scope, record_usage

log = get_logger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
OPENAI_BATCH = 96


def cached_matrix(cache_dir: Path | None, model: str, texts: list[str], compute) -> np.ndarray:
    """Corpus embeddings are the slow (or paid) part of startup, so reuse them while the model and texts are unchanged."""
    if cache_dir is None:
        return compute(texts)
    key = hashlib.sha1((model + "\0" + "\0".join(texts)).encode("utf-8")).hexdigest()[:24]
    path = Path(cache_dir) / f"{key}.npy"
    try:
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    except (OSError, ValueError):
        pass
    matrix = compute(texts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    except OSError:
        pass
    return matrix


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class Embedder(ABC):
    name = "base"

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Embed the corpus (fitting any vocabulary first). Returns shape (len(texts), dim)."""

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """Embed one query. Returns shape (dim,); an all-zero vector means 'nothing to match on'."""


class TfidfEmbedder(Embedder):
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.fit_transform(texts).toarray().astype("float32")  # sklearn already L2-normalises

    def encode_query(self, text: str) -> np.ndarray:
        return self._vectorizer.transform([text]).toarray().astype("float32")[0]


class SentenceTransformerEmbedder(Embedder):
    name = "sentence_transformers"

    def __init__(self, model: str = DEFAULT_ST_MODEL, cache_dir: Path | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = model
        self.cache_dir = cache_dir
        self._model = SentenceTransformer(model)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return cached_matrix(self.cache_dir, self.model, texts, lambda t: _normalise(self._model.encode(t, show_progress_bar=False)))

    def encode_query(self, text: str) -> np.ndarray:
        return _normalise(self._model.encode([text], show_progress_bar=False))[0]


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, cache_dir: Path | None = None) -> None:
        from openai import OpenAI  # optional dependency

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self.cache_dir = cache_dir
        self._client = OpenAI()

    def _embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), OPENAI_BATCH):
            resp = self._client.embeddings.create(model=self.model, input=texts[start : start + OPENAI_BATCH])
            rows.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
            usage = getattr(resp, "usage", None)
            with caller_scope("embeddings"):
                record_usage("openai", self.model, getattr(usage, "prompt_tokens", 0) or 0, 0)
        return _normalise(np.array(rows))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return cached_matrix(self.cache_dir, self.model, texts, self._embed)  # also avoids paying for the corpus again

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def _cache_dir(settings: Settings) -> Path | None:
    folder = settings.get("rag.embedding_cache_dir", ".cache/embeddings")
    return settings.resolve_path(folder) if folder else None


def create_channels(settings: Settings) -> tuple[Embedder, Embedder | None, float]:
    """(primary embedder, lexical embedder or None, lexical weight) for ``rag.embedder``.

    ``hybrid`` = the neural embedder named by ``rag.hybrid_semantic`` (default sentence_transformers) blended with TF-IDF at
    ``rag.hybrid_lexical_weight``. If the neural half cannot load, retrieval degrades to TF-IDF only (with a warning).
    """
    choice = str(settings.get("rag.embedder", "tfidf")).lower()
    if choice != "hybrid":
        return create_embedder(settings), None, 0.0
    cache_dir = _cache_dir(settings)
    semantic_name = str(settings.get("rag.hybrid_semantic", "sentence_transformers")).lower()
    try:
        if semantic_name == "openai":
            semantic: Embedder = OpenAIEmbedder(settings.get("rag.openai_embedding_model", DEFAULT_OPENAI_MODEL), cache_dir=cache_dir)
        elif semantic_name == "sentence_transformers":
            semantic = SentenceTransformerEmbedder(settings.get("rag.sentence_transformers_model", DEFAULT_ST_MODEL), cache_dir=cache_dir)
        else:
            raise ValueError(f"unknown rag.hybrid_semantic {semantic_name!r}")
    except Exception as exc:
        log.warning("hybrid search: semantic half %r unavailable (%s); using TF-IDF only", semantic_name, exc)
        return TfidfEmbedder(), None, 0.0
    weight = min(max(float(settings.get("rag.hybrid_lexical_weight", 0.3)), 0.0), 1.0)
    return semantic, TfidfEmbedder(), weight


def create_embedder(settings: Settings) -> Embedder:
    choice = str(settings.get("rag.embedder", "tfidf")).lower()
    if choice == "hybrid":  # single-embedder callers get the semantic half's fallback behaviour
        return create_channels(settings)[0]
    try:
        if choice == "sentence_transformers":
            return SentenceTransformerEmbedder(settings.get("rag.sentence_transformers_model", DEFAULT_ST_MODEL), cache_dir=_cache_dir(settings))
        if choice == "openai":
            return OpenAIEmbedder(settings.get("rag.openai_embedding_model", DEFAULT_OPENAI_MODEL), cache_dir=_cache_dir(settings))
        if choice != "tfidf":
            log.warning("unknown embedder %r; using tfidf", choice)
    except Exception as exc:  # missing package, no key, model download failed...
        log.warning("embedder %r unavailable (%s); falling back to tfidf", choice, exc)
    return TfidfEmbedder()
