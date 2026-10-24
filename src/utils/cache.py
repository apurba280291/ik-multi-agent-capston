"""Thread-safe TTL cache that can serve stale entries as a fallback."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable


class TTLCache:
    def __init__(self, ttl_seconds: float = 1800, clock: Callable[[], float] = time.time):
        self.ttl = ttl_seconds
        self._clock = clock
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, allow_stale: bool = False) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if allow_stale or self._clock() - stored_at <= self.ttl:
            return value
        return None

    def age(self, key: str) -> float | None:
        with self._lock:
            entry = self._data.get(key)
        return None if entry is None else self._clock() - entry[0]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (self._clock(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
