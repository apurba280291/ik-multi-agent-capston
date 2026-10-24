import json

import pytest

pytest.importorskip("mcp")

from src.mcp_server import server  # noqa: E402


@pytest.fixture(autouse=True)
def _assistant(assistant, monkeypatch):
    monkeypatch.setattr(server, "_assistant", assistant)


def test_tools_registered():
    import asyncio

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert names == {
        "ask_finance_assistant", "search_knowledge_base", "get_stock_quote",
        "analyze_portfolio_holdings", "plan_financial_goal", "explain_term",
    }


def test_ask_and_search_and_explain():
    assert "ETF" in server.ask_finance_assistant("What is an ETF?")
    assert "Source:" in server.search_knowledge_base("expense ratio", limit=2)
    assert server.search_knowledge_base("zxqv wlkj") == "No matching articles found."
    assert "stock" in server.explain_term("Stock").lower()
    assert "No glossary entry" in server.explain_term("zxqv")


def test_quote_tool():
    data = json.loads(server.get_stock_quote("AAPL"))
    assert data["quote"]["symbol"] == "AAPL" and "freshness" in data
    assert "Invalid symbol" in server.get_stock_quote("not a symbol")


def test_portfolio_and_goal_tools():
    out = server.analyze_portfolio_holdings("10 VTI, $1000 in BND")
    assert "total_value" in out and "Educational information" in out
    assert "Couldn't parse" in server.analyze_portfolio_holdings("nothing here")
    plan = server.plan_financial_goal(50000, 8, 5000, 300)
    assert "success_probability" in plan and "monte_carlo" not in plan
    assert "positive" in server.plan_financial_goal(0, 5)


def test_lazy_assistant_construction(monkeypatch):
    monkeypatch.setattr(server, "_assistant", None)
    assert server.assistant() is server.assistant()
