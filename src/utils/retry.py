"""Exponential-backoff retry helper and a minimal rate limiter."""
from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

from src.utils.logging_utils import get_logger

T = TypeVar("T")
log = get_logger(__name__)


MAX_ATTEMPTS = 5  # hard ceiling on attempts, whatever the caller or config asks for
MAX_DELAY = 30.0  # seconds; caps a single backoff sleep so total wait stays bounded


def retry_with_backoff(
    fn: Callable[[], T],
    retries: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` up to ``retries`` times (never more than ``MAX_ATTEMPTS``), doubling the delay after each failure."""
    retries = min(max(1, retries), MAX_ATTEMPTS)
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn()
        except exceptions as exc:  # noqa: PERF203
            last = exc
            if attempt < retries - 1:
                delay = min(base_delay * (2**attempt), MAX_DELAY)
                log.warning("attempt %d/%d failed (%s); retrying in %.2fs", attempt + 1, retries, exc, delay)
                sleep(delay)
    assert last is not None
    raise last


class RateLimiter:
    """Enforces a minimum spacing between calls (``per_minute`` calls/min)."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next = max(now, self._next) + self.interval
