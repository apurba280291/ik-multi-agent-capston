import pytest

from src.workflow.router import AGENT_ORDER, Router, rule_route
from tests.conftest import FakeLLM


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is an ETF?", ["finance_qa"]),
        ("Explain diversification", ["finance_qa"]),
        ("Analyze my portfolio", ["portfolio_analysis"]),
        ("I own 10 VTI and 5 AAPL", ["portfolio_analysis"]),
        ("How is NVDA doing today?", ["market_analysis"]),
        ("what's the price of Tesla", ["market_analysis"]),
        ("I want $50,000 in 8 years", ["goal_planning"]),
        ("How much do I need to save for retirement?", ["goal_planning"]),
        ("Roth vs traditional IRA", ["tax_education"]),
        ("What is the wash sale rule?", ["tax_education"]),
        ("Latest news on Apple", ["news_synthesizer"]),
    ],
)
def test_rule_routing(query, expected):
    assert rule_route(query).intents[0] == expected[0]


def test_multi_intent_ordering_follows_pipeline():
    d = rule_route("Analyze my portfolio of 10 VTI and show the market trend, then explain what is a Roth IRA")
    assert len(d.intents) >= 2
    assert d.intents == sorted(d.intents, key=AGENT_ORDER.index)
    assert len(d.intents) <= 3


def test_explain_plus_specific_agent_adds_qa():
    d = rule_route("Explain what is a Roth IRA and plan my goal of $100k in 10 years")
    assert "finance_qa" in d.intents and "goal_planning" in d.intents


def test_greeting_offtopic_and_empty():
    r = Router()
    assert r.route("hi").greeting and r.route("").greeting and r.route("What can you do").greeting
    assert r.route("what's the weather like").off_topic
    assert r.route("write me a poem about cats").off_topic


def test_followup_inherits_previous_route():
    r = Router()
    hist = [("user", "How is NVDA doing?"), ("assistant", "...")]
    d = r.route("and why is that?", hist)
    assert d.method == "context" and d.intents == ["market_analysis"]
    assert r.route("what's the weather like", hist).off_topic  # not a follow-up


def test_followup_without_previous_user_message():
    assert Router()._follow_up([]) == []


def test_advice_flag():
    d = Router().route("Should I buy TSLA right now?")
    assert d.advice_request and "market_analysis" in d.intents
    assert not Router().route("What is a stock?").advice_request


def test_llm_routing_validated():
    r = Router(FakeLLM(reply='Sure: ["goal_planning", "bogus", "market_analysis"]'))
    d = r.route("plan my savings and market")
    assert d.method == "llm" and d.intents == ["market_analysis", "goal_planning"]


def test_llm_routing_falls_back_on_garbage_or_error():
    d = Router(FakeLLM(reply="not json")).route("What is an ETF?")
    assert d.method == "rules" and d.intents == ["finance_qa"]
    d = Router(FakeLLM(reply='["bogus"]')).route("What is an ETF?")
    assert d.method == "rules"

    def raise_error(s, p):
        from src.core.llm import LLMError

        raise LLMError("x")

    assert Router(FakeLLM(reply=raise_error)).route("What is an ETF?").method == "rules"
