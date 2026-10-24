"""Append-only JSONL record of LLM token usage (``logs/llm_usage.jsonl``), one row per LLM call.

``caller`` says which part of the app made the call (an agent name, or ``router``); agents set it
through ``caller_scope``. Disable with ``FINANCE_CALL_LOG=off``.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from pathlib import Path

from src.utils.call_log import log_dir

_LOCK = threading.Lock()
_CALLER: contextvars.ContextVar[str] = contextvars.ContextVar("llm_caller", default="router")


def usage_log_path() -> Path:
    return log_dir() / "llm_usage.jsonl"


@contextlib.contextmanager
def caller_scope(name: str):
    token = _CALLER.set(name)
    try:
        yield
    finally:
        _CALLER.reset(token)


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 characters per token) for providers that report no usage."""
    return max(1, len(text) // 4)


def record_usage(provider: str, model: str, input_tokens: int, output_tokens: int, estimated: bool = False) -> None:
    """Never raises: monitoring must not break a chat turn."""
    if os.environ.get("FINANCE_CALL_LOG", "").lower() == "off":
        return
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caller": _CALLER.get(),
        "provider": provider,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "estimated": estimated,
    }
    try:
        path = usage_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass
