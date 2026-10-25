"""Vector index: FAISS inner-product index (exact ``flat`` or approximate ``hnsw``) with a NumPy fallback."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None
    HAS_FAISS = False

INDEX_TYPES = ("flat", "hnsw")


class VectorStore:
    """Cosine-similarity search over L2-normalised vectors.

    ``flat`` is exact brute force (best for a few thousand vectors). ``hnsw`` is a graph-based approximate index
    (Hierarchical Navigable Small World): ``hnsw_m`` = links per node, ``ef_construction`` = build-time search
    breadth, ``ef_search`` = query-time breadth (higher = better recall, slower). Without FAISS both fall back to NumPy.
    """

    def __init__(
        self,
        dim: int,
        use_faiss: bool = True,
        index_type: str = "flat",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}")
        self.dim = dim
        self.index_type = index_type
        self.ef_search = ef_search
        self._use_faiss = use_faiss and HAS_FAISS
        self._vectors = np.zeros((0, dim), dtype="float32")
        self._index = None
        if self._use_faiss:
            if index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self._index.hnsw.efConstruction = ef_construction
                self._index.hnsw.efSearch = ef_search
            else:
                self._index = faiss.IndexFlatIP(dim)

    @property
    def backend(self) -> str:
        if not self._use_faiss:
            return "numpy"
        return "faiss-hnsw" if self.index_type == "hnsw" else "faiss"

    @property
    def approximate(self) -> bool:
        return self._use_faiss and self.index_type == "hnsw"

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
            if self.approximate:
                self._index.hnsw.efSearch = max(self.ef_search, k)  # efSearch < k would silently return fewer hits
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = (self._vectors @ q.T).ravel()
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]
