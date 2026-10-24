"""Base class shared by all six specialised agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.llm import LLMClient, LLMError, OfflineLLM
from src.core.models import AgentContext, AgentResponse
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.data.market_data import MarketDataService
    from src.rag.retriever import Retriever

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
    ):
        self.llm = llm or OfflineLLM()
        self.retriever = retriever
        self.market = market
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        """Produce a response; may raise - ``handle`` converts errors to fallbacks."""

    def handle(self, query: str, ctx: AgentContext) -> AgentResponse:
        """Error-safe entry point used by the workflow."""
        try:
            return self.run(query, ctx)
        except Exception as exc:
            self.log.exception("agent %s failed", self.name)
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
        try:
            return self.llm.generate(SYSTEM_PROMPT, f"USER QUESTION:\n{query}\n\nDRAFT:\n{draft}")
        except LLMError as exc:
            self.log.warning("LLM polish skipped: %s", exc)
            return draft
