"""Provider-agnostic LLM clients with an offline fallback.

The agents compute facts deterministically (calculations, retrieved passages,
market data) and use the LLM only to rewrite those facts conversationally. If no
LLM is configured or a call fails, the deterministic draft is returned unchanged,
so the product degrades gracefully instead of failing.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from src.core.config import Settings
from src.utils.logging_utils import get_logger
from src.utils.retry import RateLimiter, retry_with_backoff

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    name = "base"
    available = True

    @abstractmethod
    def generate(self, system: str, prompt: str) -> str: ...


class OfflineLLM(LLMClient):
    """Placeholder used when no API key is configured."""

    name = "offline"
    available = False

    def generate(self, system: str, prompt: str) -> str:
        raise LLMError("no LLM provider configured")


class _RemoteLLM(LLMClient):
    def __init__(self, settings: Settings):
        self.temperature = float(settings.get("llm.temperature", 0.2))
        self.max_tokens = int(settings.get("llm.max_tokens", 900))
        self._limiter = RateLimiter(int(settings.get("llm.requests_per_minute", 50)))

    @abstractmethod
    def _call(self, system: str, prompt: str) -> str: ...

    def generate(self, system: str, prompt: str) -> str:
        self._limiter.wait()
        try:
            text = retry_with_backoff(lambda: self._call(system, prompt), retries=2, base_delay=1.0)
        except Exception as exc:
            raise LLMError(f"{self.name} call failed: {exc}") from exc
        if not text or not text.strip():
            raise LLMError(f"{self.name} returned an empty response")
        return text.strip()


class AnthropicLLM(_RemoteLLM):
    name = "anthropic"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import anthropic

        self.model = settings.get("llm.anthropic_model", "claude-sonnet-5-5")
        self._client = anthropic.Anthropic()

    def _call(self, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class GeminiLLM(_RemoteLLM):
    name = "gemini"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from google import genai

        self.model = settings.get("llm.gemini_model", "gemini-2.0-flash")
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def _call(self, system: str, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": system,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )
        return resp.text or ""


class OpenAILLM(_RemoteLLM):
    name = "openai"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from openai import OpenAI

        self.model = settings.get("llm.openai_model", "gpt-4o-mini")
        self._client = OpenAI()

    def _call(self, system: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", AnthropicLLM),
    "gemini": ("GOOGLE_API_KEY", GeminiLLM),
    "openai": ("OPENAI_API_KEY", OpenAILLM),
}


def create_llm(settings: Settings) -> LLMClient:
    """Build the configured client; never raises - falls back to ``OfflineLLM``."""
    choice = str(settings.get("llm.provider", "auto")).lower()
    if choice == "offline":
        return OfflineLLM()
    candidates = list(_PROVIDERS) if choice == "auto" else [choice]
    for name in candidates:
        if name not in _PROVIDERS:
            log.warning("unknown llm provider %r", name)
            continue
        env_key, cls = _PROVIDERS[name]
        if not os.environ.get(env_key):
            continue
        try:
            client = cls(settings)
            log.info("using %s LLM", name)
            return client
        except Exception as exc:  # missing SDK, bad config, ...
            log.warning("could not initialise %s LLM: %s", name, exc)
    log.info("no LLM available - running in offline mode")
    return OfflineLLM()
