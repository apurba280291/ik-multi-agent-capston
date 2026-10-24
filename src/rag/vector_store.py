"""Vector index: FAISS inner-product index with a NumPy fallback."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None
    HAS_FAISS = False


class VectorStore:
    """Cosine-similarity search over L2-normalised vectors."""

    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self._use_faiss = use_faiss and HAS_FAISS
        self._vectors = np.zeros((0, dim), dtype="float32")
        self._index = faiss.IndexFlatIP(dim) if self._use_faiss else None

    @property
    def backend(self) -> str:
        return "faiss" if self._use_faiss else "numpy"

    def __len__(self) -> int:
        return self._vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        self._vectors = np.vstack([self._vectors, vectors])
        if self._index is not None:
            self._index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` (row_index, score) pairs, best first."""
        if len(self) == 0 or k <= 0:
            return []
        k = min(k, len(self))
        q = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = (self._vectors @ q.T).ravel()
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]
