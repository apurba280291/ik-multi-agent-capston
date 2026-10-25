"""Embedding back ends for the retriever. All return L2-normalised float32 vectors, so inner product = cosine.

* ``tfidf``                 - default; lexical (unigrams + bigrams), no key, no download, builds in milliseconds.
* ``sentence_transformers`` - local neural model (``pip install sentence-transformers``); true semantic matching.
* ``openai``                - OpenAI embeddings API (needs ``OPENAI_API_KEY``); token usage goes to ``llm_usage.jsonl``.

``create_embedder`` never raises: if the chosen back end is unavailable it logs a warning and uses TF-IDF.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from src.core.config import Settings
from src.utils.logging_utils import get_logger
from src.utils.usage_log import caller_scope, record_usage

log = get_logger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
OPENAI_BATCH = 96


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

    def __init__(self, model: str = DEFAULT_ST_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = model
        self._model = SentenceTransformer(model)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._model.encode(texts, show_progress_bar=False))

    def encode_query(self, text: str) -> np.ndarray:
        return _normalise(self._model.encode([text], show_progress_bar=False))[0]


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        from openai import OpenAI  # optional dependency

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
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
        return self._embed(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def create_embedder(settings: Settings) -> Embedder:
    choice = str(settings.get("rag.embedder", "tfidf")).lower()
    try:
        if choice == "sentence_transformers":
            return SentenceTransformerEmbedder(settings.get("rag.sentence_transformers_model", DEFAULT_ST_MODEL))
        if choice == "openai":
            return OpenAIEmbedder(settings.get("rag.openai_embedding_model", DEFAULT_OPENAI_MODEL))
        if choice != "tfidf":
            log.warning("unknown embedder %r; using tfidf", choice)
    except Exception as exc:  # missing package, no key, model download failed...
        log.warning("embedder %r unavailable (%s); falling back to tfidf", choice, exc)
    return TfidfEmbedder()
