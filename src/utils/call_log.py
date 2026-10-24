"""Append-only JSONL record of every agent call, so intermittent failures can be counted and traced.

Disable with ``FINANCE_CALL_LOG=off``; change the folder with ``FINANCE_LOG_DIR`` (default ``logs/``).
Queries are stored truncated (they are already PII-redacted by the workflow guardrail).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
QUERY_CHARS = 120


def log_dir() -> Path:
    return Path(os.environ.get("FINANCE_LOG_DIR") or Path(__file__).resolve().parents[2] / "logs")


def call_log_path() -> Path:
    return log_dir() / "agent_calls.jsonl"


def record_call(agent: str, query: str, ok: bool, duration_ms: float, error: str | None = None) -> None:
    """Never raises: monitoring must not break a chat turn."""
    if os.environ.get("FINANCE_CALL_LOG", "").lower() == "off":
        return
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent": agent,
        "ok": ok,
        "ms": round(duration_ms, 1),
        "error": error,
        "query": query[:QUERY_CHARS],
    }
    try:
        path = call_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
