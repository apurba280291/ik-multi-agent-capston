"""Facade that wires settings, LLM, RAG, market data, agents and the graph together."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_agent import GoalPlanningAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.tax_education import TaxEducationAgent
from src.core.config import Settings, load_settings
from src.core.llm import LLMClient, create_llm
from src.data.market_data import MarketDataService
from src.rag.retriever import Retriever
from src.utils.logging_utils import get_logger
from src.workflow.graph import FALLBACK, build_graph
from src.workflow.router import Router

log = get_logger(__name__)


@dataclass
class ChatResult:
    """What one chat turn returns to the caller (Streamlit app, MCP server or tests)."""

    answer: str
    agents: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    route_method: str = ""
    ok: bool = True


class FinanceAssistant:
    """Single entry point used by the Streamlit app, the MCP server and the tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        retriever: Retriever | None = None,
        market: MarketDataService | None = None,
    ):
        # Each dependency can be injected (tests pass mocks); otherwise it is built from config.yaml / env vars.
        self.settings = settings or load_settings()
        self.llm = llm or create_llm(self.settings)  # falls back to OfflineLLM when no API key is set
        self.retriever = retriever or Retriever.from_settings(self.settings)  # RAG over the knowledge base
        self.market = market or MarketDataService(self.settings)  # live quotes with cache/fallbacks
        kwargs = dict(
            llm=self.llm, retriever=self.retriever, market=self.market,
            # Token savers (config.yaml `llm.*`): skip rewriting oversized drafts, reuse rewrites of identical requests.
            polish_max_chars=int(self.settings.get("llm.polish_max_chars", 6000)),
            polish_cache_ttl=float(self.settings.get("llm.polish_cache_ttl_seconds", 3600)),
        )
        # Six specialised agents, keyed by agent name so the graph and router can look them up.
        self.agents = {
            a.name: a
            for a in (
                FinanceQAAgent(**kwargs),
                PortfolioAnalysisAgent(**kwargs),
                MarketAnalysisAgent(**kwargs),
                GoalPlanningAgent(**kwargs),
                NewsSynthesizerAgent(**kwargs),
                TaxEducationAgent(**kwargs),
            )
        }
        # Router decides which agent(s) handle a query (LLM classification, or keyword rules when offline).
        # `llm.routing_confident_score: 0` disables the "skip the LLM when rules are sure" shortcut.
        self.router = Router(self.llm, int(self.settings.get("llm.routing_confident_score", 5)) or None)
        # Compile the LangGraph workflow: understand -> route -> agent(s) -> compose,
        # with conversation memory checkpointed per session_id. Last arg caps remembered chat turns.
        self.graph = build_graph(self.agents, self.router, int(self.settings.get("app.max_history_turns", 20)))

    def chat(
        self,
        query: str,
        session_id: str = "default",
        profile: dict[str, Any] | None = None,
        portfolio: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Run one conversational turn. Never raises; failures become a friendly fallback message."""
        # LangGraph uses thread_id to load/save this session's memory (history, profile, portfolio).
        config = {"configurable": {"thread_id": session_id}}
        inputs: dict[str, Any] = {"query": query or ""}
        if profile:
            # Merge new profile fields over what the session already knows rather than replacing it.
            inputs["profile"] = {**self.get_state(session_id).get("profile", {}), **profile}
        if portfolio is not None:
            inputs["portfolio"] = portfolio
        try:
            out = self.graph.invoke(inputs, config)  # runs the whole graph and returns the final state
        except Exception:
            log.exception("workflow failed")
            return ChatResult(FALLBACK, ok=False)
        # Pick out only what the UI / MCP server needs from the graph's final state.
        return ChatResult(
            answer=out.get("final_answer", FALLBACK),
            agents=out.get("agents_used", []),
            sources=out.get("sources", []),
            trace=out.get("trace", []),
            route_method=out.get("route_method", ""),
        )

    def get_state(self, session_id: str = "default") -> dict[str, Any]:
        """Return the saved state (history, profile, portfolio, ...) for a session; empty dict if none."""
        try:
            return dict(self.graph.get_state({"configurable": {"thread_id": session_id}}).values or {})
        except Exception:
            return {}
