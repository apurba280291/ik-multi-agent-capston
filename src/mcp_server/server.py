"""MCP server exposing Finnie's finance tools to Claude Desktop (stdio transport).

Run:  python -m src.mcp_server.server
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP  # noqa: E402
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.core.goal_planner import plan_goal  # noqa: E402
from src.core.portfolio_analysis import analyze_portfolio  # noqa: E402
from src.utils.formatting import with_disclaimer  # noqa: E402
from src.utils.parsing import parse_holdings  # noqa: E402
from src.workflow.assistant import FinanceAssistant  # noqa: E402

mcp = FastMCP("finnie-finance")
_assistant: FinanceAssistant | None = None


def assistant() -> FinanceAssistant:
    """Lazily build the assistant so the server starts instantly and reports errors per tool call."""
    global _assistant
    if _assistant is None:
        _assistant = FinanceAssistant()
    return _assistant


@mcp.tool()
def ask_finance_assistant(question: str, session_id: str = "mcp") -> str:
    """Ask Finnie any personal-finance question; routes to the right specialised agent(s)."""
    return assistant().chat(question, session_id).answer


@mcp.tool()
def search_knowledge_base(query: str, category: str = "", limit: int = 4) -> str:
    """Semantic search over the curated financial-education knowledge base (returns cited passages)."""
    hits = assistant().retriever.search(query, k=max(1, min(limit, 10)), category=category or None)
    if not hits:
        return "No matching articles found."
    return "\n\n".join(f"[{h.title} | {h.category} | score {h.score:.2f}]\n{h.text}\nSource: {h.reference}" for h in hits)


@mcp.tool()
def get_stock_quote(symbol: str) -> str:
    """Latest price, daily change and trend indicators for a ticker (with data-freshness label)."""
    a = assistant()
    try:
        q, trend = a.market.get_quote(symbol), a.market.analyze_trend(symbol)
    except ValueError as exc:
        return f"Invalid symbol: {exc}"
    return json.dumps({"quote": q.to_dict(), "trend": trend, "freshness": a.market.freshness(q)}, indent=2, default=str)


@mcp.tool()
def analyze_portfolio_holdings(holdings: str, risk_tolerance: str = "moderate") -> str:
    """Analyse holdings written like '10 shares of VTI, 5 AAPL, $2000 in BND' (allocation, risk, diversification)."""
    parsed = parse_holdings(holdings)
    if not parsed:
        return "Couldn't parse any holdings. Example: '10 VTI, 5 AAPL, $2000 in BND'."
    a = assistant()
    quotes = a.market.get_quotes([h["symbol"] for h in parsed if "shares" in h])
    result = analyze_portfolio(parsed, {s: q.price for s, q in quotes.items()}, risk_tolerance)
    result.pop("positions", None)
    return with_disclaimer(json.dumps(result, indent=2, default=str))


@mcp.tool()
def plan_financial_goal(target: float, years: float, current_savings: float = 0, monthly_contribution: float = 0, risk: str = "moderate") -> str:
    """Project a savings goal: expected value, required monthly saving and Monte Carlo success probability."""
    try:
        plan = plan_goal(target, years, current_savings, monthly_contribution, risk, simulations=1000)
    except ValueError as exc:
        return str(exc)
    for key in ("monte_carlo", "deterministic_path"):
        plan.pop(key)
    return with_disclaimer(json.dumps(plan, indent=2))


@mcp.tool()
def explain_term(term: str) -> str:
    """Plain-English definition of a financial term from the glossary."""
    hits = assistant().retriever.search(term, k=1, category="glossary")
    return hits[0].text if hits else f"No glossary entry found for '{term}'."


if __name__ == "__main__":
    mcp.run()
