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
        self.settings = settings or load_settings()
        self.llm = llm or create_llm(self.settings)
        self.retriever = retriever or Retriever.from_settings(self.settings)
        self.market = market or MarketDataService(self.settings)
        kwargs = dict(llm=self.llm, retriever=self.retriever, market=self.market)
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
        self.router = Router(self.llm)
        self.graph = build_graph(self.agents, self.router, int(self.settings.get("app.max_history_turns", 20)))

    def chat(
        self,
        query: str,
        session_id: str = "default",
        profile: dict[str, Any] | None = None,
        portfolio: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Run one conversational turn. Never raises; failures become a friendly fallback message."""
        config = {"configurable": {"thread_id": session_id}}
        inputs: dict[str, Any] = {"query": query or ""}
        if profile:
            inputs["profile"] = {**self.get_state(session_id).get("profile", {}), **profile}
        if portfolio is not None:
            inputs["portfolio"] = portfolio
        try:
            out = self.graph.invoke(inputs, config)
        except Exception:
            log.exception("workflow failed")
            return ChatResult(FALLBACK, ok=False)
        return ChatResult(
            answer=out.get("final_answer", FALLBACK),
            agents=out.get("agents_used", []),
            sources=out.get("sources", []),
            trace=out.get("trace", []),
            route_method=out.get("route_method", ""),
        )

    def get_state(self, session_id: str = "default") -> dict[str, Any]:
        try:
            return dict(self.graph.get_state({"configurable": {"thread_id": session_id}}).values or {})
        except Exception:
            return {}
