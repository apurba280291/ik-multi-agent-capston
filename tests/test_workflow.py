import pytest

from src.workflow.graph import ADVICE_NOTE, FALLBACK, INTRO, OFF_TOPIC
from src.utils.formatting import DISCLAIMER


def test_greeting_and_off_topic(assistant):
    assert assistant.chat("hello", "g").answer == INTRO
    r = assistant.chat("tell me a joke about cats", "g")
    assert OFF_TOPIC in r.answer and DISCLAIMER in r.answer and r.agents == []


def test_single_agent_flow_with_trace(assistant):
    r = assistant.chat("What is compound interest?", "t1")
    assert r.agents == ["finance_qa"] and r.sources and DISCLAIMER in r.answer
    assert any(t.startswith("route[rules]") for t in r.trace) and r.trace[-1] == "compose: done"


def test_all_six_agents_reachable(assistant):
    prompts = {
        "finance_qa": "What is diversification?",
        "portfolio_analysis": "I own 10 VTI and 5 BND, analyze my portfolio",
        "market_analysis": "How is AAPL doing today?",
        "goal_planning": "I want $40,000 in 6 years, have $2,000, can save $400 a month",
        "news_synthesizer": "latest headlines on microsoft",
        "tax_education": "Explain a Roth IRA",
    }
    for expected, prompt in prompts.items():
        assert expected in assistant.chat(prompt, f"s-{expected}").agents, prompt


def test_multi_agent_orchestration_and_shared_state(assistant):
    q = "I own 20 AAPL and 10 BND. Analyze my portfolio and plan my goal of $100,000 in 10 years with $500 a month"
    r = assistant.chat(q, "multi")
    assert r.agents == ["portfolio_analysis", "goal_planning"]  # pipeline order preserved
    assert "### Your portfolio" in r.answer and "### Goal plan" in r.answer
    assert "current portfolio holds" in r.answer  # goal agent consumed portfolio agent's output


def test_multi_agent_sources_consolidated(assistant):
    r = assistant.chat("explain what is a Roth IRA and also latest news on tesla", "src")
    assert len(r.agents) >= 2
    assert r.answer.count("**Sources**") == 1


def test_conversation_memory_tickers_and_portfolio(assistant):
    assistant.chat("How is NVDA doing?", "mem")
    follow = assistant.chat("and what's the latest news?", "mem")  # no ticker: uses remembered NVDA
    assert "NVDA" in follow.answer
    assistant.chat("I own 5 VTI and 3 BND", "mem")
    state = assistant.get_state("mem")
    assert {h["symbol"] for h in state["portfolio"]} == {"VTI", "BND"}
    assert len(state["history"]) == 6 and state["history"][0][0] == "user"
    again = assistant.chat("review my portfolio", "mem")  # portfolio persisted from earlier turn
    assert "Portfolio value" in again.answer


def test_sessions_are_isolated(assistant):
    assistant.chat("I own 5 VTI", "A")
    assert assistant.get_state("B") == {}
    assert "Portfolio** tab" in assistant.chat("analyze my portfolio", "B").answer


def test_profile_extraction_and_override(assistant):
    assistant.chat("I'm an aggressive investor and I'm 28", "prof")
    assert assistant.get_state("prof")["profile"] == {"risk_tolerance": "aggressive", "age": 28}
    assistant.chat("hello", "prof", profile={"risk_tolerance": "conservative"})
    assert assistant.get_state("prof")["profile"]["risk_tolerance"] == "conservative"
    assert assistant.get_state("prof")["profile"]["age"] == 28


def test_portfolio_passed_in_from_ui(assistant):
    r = assistant.chat("analyze my portfolio", "ui", portfolio=[{"symbol": "VOO", "shares": 3}])
    assert "VOO" in r.answer


def test_advice_guardrail(assistant):
    r = assistant.chat("Should I buy TSLA?", "adv")
    assert ADVICE_NOTE in r.answer and DISCLAIMER in r.answer


def test_history_is_trimmed(settings, retriever, market):
    from src.core.config import Settings
    from src.workflow.assistant import FinanceAssistant

    s = Settings({**settings.raw, "app": {"max_history_turns": 2}})
    bot = FinanceAssistant(settings=s, retriever=retriever, market=market)
    for i in range(5):
        bot.chat(f"what is a stock {i}", "trim")
    assert len(bot.get_state("trim")["history"]) == 4


def test_agent_failure_is_contained(assistant, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(assistant.market, "get_quote", boom)
    monkeypatch.setattr(assistant.market, "market_overview", boom)
    r = assistant.chat("how is the market today", "fail")
    assert "problem" in r.answer and any("FAILED" in t for t in r.trace)


def test_partial_failure_still_returns_other_agents(assistant, monkeypatch):
    monkeypatch.setattr(assistant.market, "get_news", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    r = assistant.chat("explain what is a Roth IRA and also latest news on tesla", "partial")
    assert "tax_education" in r.agents and "Tax education" in r.answer and "problem" in r.answer


def test_graph_level_failure_returns_fallback(assistant, monkeypatch):
    monkeypatch.setattr(assistant.graph, "invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("graph down")))
    r = assistant.chat("hi", "x")
    assert r.answer == FALLBACK and not r.ok


def test_llm_polish_in_full_flow(settings, retriever, market, fake_llm):
    from src.workflow.assistant import FinanceAssistant

    bot = FinanceAssistant(settings=settings, llm=fake_llm, retriever=retriever, market=market)
    r = bot.chat("What is an ETF?", "llm")
    assert "POLISHED" in r.answer and r.sources
    assert fake_llm.calls
    bad_router = bot.chat("What is an ETF?", "llm2")
    assert bad_router.route_method in {"llm", "rules"}


def test_get_state_unknown_session(assistant):
    assert assistant.get_state("never-used") == {}


def test_malformed_inputs(assistant):
    for q in ["", "   ", "💸" * 50, "A" * 5000, "'; DROP TABLE users; --", "$$$ ??? !!!"]:
        r = assistant.chat(q, "weird")
        assert isinstance(r.answer, str) and r.answer
