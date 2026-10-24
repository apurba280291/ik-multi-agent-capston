"""Base class shared by all six specialised agents."""
from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.guardrails import rewrite_is_faithful
from src.core.llm import LLMClient, LLMError, OfflineLLM
from src.core.models import AgentContext, AgentResponse
from src.utils.cache import TTLCache
from src.utils.call_log import record_call
from src.utils.logging_utils import get_logger
from src.utils.usage_log import caller_scope

if TYPE_CHECKING:
    from src.data.market_data import MarketDataService
    from src.rag.retriever import Retriever

POLISH_CACHE_MAX = 300

SYSTEM_PROMPT = (
    "You are Finnie, a friendly financial-education assistant for beginners. "
    "Rewrite the DRAFT so it answers the user's question conversationally in plain language. "
    "Use ONLY facts, numbers and citations present in the draft; never invent figures, tickers, "
    "returns or sources. Keep any [n] citation markers and any markdown tables. Do not give "
    "personalised buy/sell advice. Keep it concise."
)


class BaseAgent(ABC):
    name = "base"
    description = ""

    def __init__(
        self,
        llm: LLMClient | None = None,
        retriever: "Retriever | None" = None,
        market: "MarketDataService | None" = None,
        polish_max_chars: int = 6000,
        polish_cache_ttl: float = 3600,
    ):
        self.polish_max_chars = polish_max_chars  # 0 = no limit
        self._polish_cache = TTLCache(polish_cache_ttl)
        self.llm = llm or OfflineLLM()
        self.retriever = retriever
        self.market = market
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        """Produce a response; may raise - ``handle`` converts errors to fallbacks."""

    def handle(self, query: str, ctx: AgentContext) -> AgentResponse:
        """Error-safe entry point used by the workflow."""
        start = time.perf_counter()
        try:
            with caller_scope(self.name):  # tags any LLM call made by this agent in the token-usage log
                resp = self.run(query, ctx)
            record_call(self.name, query, resp.ok, (time.perf_counter() - start) * 1000, resp.error)
            return resp
        except Exception as exc:
            self.log.exception("agent %s failed", self.name)
            record_call(self.name, query, False, (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}")
            return AgentResponse(
                agent=self.name,
                content=(
                    f"I ran into a problem in the {self.description or self.name} step, so I can't "
                    "give a complete answer to that part right now. Please try rephrasing or try again shortly."
                ),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def polish(self, draft: str, query: str) -> str:
        """Let the LLM rewrite a grounded draft; fall back to the draft on any failure."""
        if not self.llm.available:
            return draft
        # Token savers: very long drafts (big tables) are returned as-is, and identical requests reuse a cached rewrite.
        if self.polish_max_chars and len(draft) > self.polish_max_chars:
            return draft
        key = hashlib.sha256(f"{self.name}\0{query}\0{draft}".encode()).hexdigest()
        cached = self._polish_cache.get(key)
        if cached is not None:
            return cached
        try:
            rewrite = self.llm.generate(SYSTEM_PROMPT, f"USER QUESTION:\n{query}\n\nDRAFT:\n{draft}")
        except LLMError as exc:
            self.log.warning("LLM polish skipped: %s", exc)
            return draft
        # Guardrail: reject rewrites that add figures not in the draft or drop citations.
        if not rewrite_is_faithful(draft, rewrite, query):
            self.log.warning("LLM rewrite rejected by guardrail; using grounded draft")
            result = draft
        else:
            result = rewrite
        if len(self._polish_cache) >= POLISH_CACHE_MAX:
            self._polish_cache.clear()
        self._polish_cache.set(key, result)  # also caches rejections, so we don't pay again for the same bad rewrite
        return result
