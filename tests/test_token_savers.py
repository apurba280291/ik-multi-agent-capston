from src.core.base_agent import BaseAgent
from src.core.models import AgentContext, AgentResponse
from src.workflow.router import Router, rule_route
from tests.conftest import FakeLLM


class Talker(BaseAgent):
    name = "talker"

    def run(self, query, ctx):
        return AgentResponse(self.name, self.polish("draft 1", query))


def test_identical_request_reuses_cached_rewrite():
    llm = FakeLLM()
    agent = Talker(llm=llm)
    first = agent.handle("same question", AgentContext()).content
    second = agent.handle("same question", AgentContext()).content
    assert first == second == "POLISHED: draft 1"
    assert len(llm.calls) == 1
    agent.handle("different question", AgentContext())
    assert len(llm.calls) == 2


def test_oversized_draft_skips_llm():
    llm = FakeLLM()
    agent = Talker(llm=llm, polish_max_chars=5)
    assert agent.handle("q", AgentContext()).content == "draft 1"
    assert llm.calls == []


def test_confident_rules_skip_llm_routing():
    query = "I own 10 VTI and 5 AAPL, analyze my portfolio"
    rules = rule_route(query)
    assert min(rules.scores[a] for a in rules.intents) >= 5
    llm = FakeLLM(reply='["tax_education"]')
    decision = Router(llm).route(query)
    assert llm.calls == [] and decision.method == "rules"
    # shortcut disabled -> the LLM is asked and wins
    assert Router(llm, confident_score=None).route(query).method == "llm"


def test_unsure_rules_still_ask_llm():
    llm = FakeLLM(reply='["tax_education"]')
    decision = Router(llm).route("What is an ETF?")
    assert len(llm.calls) == 1 and decision.method == "llm"


def test_retry_never_exceeds_five_attempts_and_caps_delay():
    import pytest

    from src.utils.retry import MAX_ATTEMPTS, MAX_DELAY, retry_with_backoff

    calls, sleeps = [], []

    def always_fails():
        calls.append(1)
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        retry_with_backoff(always_fails, retries=1000, base_delay=100, sleep=sleeps.append)
    assert len(calls) == MAX_ATTEMPTS == 5
    assert sleeps and max(sleeps) <= MAX_DELAY
