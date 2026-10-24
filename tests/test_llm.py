import pytest

from src.core import llm as llm_mod
from src.core.base_agent import BaseAgent
from src.core.config import Settings
from src.core.llm import LLMError, OfflineLLM, create_llm
from src.core.models import AgentContext, AgentResponse
from tests.conftest import FakeLLM


def test_offline_llm():
    o = OfflineLLM()
    assert not o.available
    with pytest.raises(LLMError):
        o.generate("s", "p")


def test_create_llm_offline_and_auto_without_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert isinstance(create_llm(Settings({"llm": {"provider": "offline"}})), OfflineLLM)
    assert isinstance(create_llm(Settings({"llm": {"provider": "auto"}})), OfflineLLM)
    assert isinstance(create_llm(Settings({"llm": {"provider": "bogus"}})), OfflineLLM)
    assert isinstance(create_llm(Settings({"llm": {"provider": "openai"}})), OfflineLLM)  # key missing


def test_create_llm_survives_init_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    class Boom:
        def __init__(self, settings):
            raise ImportError("sdk missing")

    monkeypatch.setitem(llm_mod._PROVIDERS, "openai", ("OPENAI_API_KEY", Boom))
    assert isinstance(create_llm(Settings({"llm": {"provider": "openai"}})), OfflineLLM)


def test_create_llm_picks_provider_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")

    class Ok(FakeLLM):
        def __init__(self, settings):
            super().__init__()

    monkeypatch.setitem(llm_mod._PROVIDERS, "anthropic", ("ANTHROPIC_API_KEY", Ok))
    assert create_llm(Settings({"llm": {"provider": "auto"}})).name == "fake"


class Remote(llm_mod._RemoteLLM):
    name = "remote"

    def __init__(self, behaviour):
        super().__init__(Settings({"llm": {"requests_per_minute": 0}}))
        self.behaviour, self.n = behaviour, 0

    def _call(self, system, prompt):
        self.n += 1
        return self.behaviour(self.n)


def test_remote_llm_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_mod, "retry_with_backoff", lambda fn, **kw: fn())

    def behave(n):
        return "  hello  "

    assert Remote(behave).generate("s", "p") == "hello"


def test_remote_llm_wraps_errors_and_empty(monkeypatch):
    monkeypatch.setattr(llm_mod, "retry_with_backoff", lambda fn, **kw: fn())

    def boom(n):
        raise RuntimeError("api down")

    with pytest.raises(LLMError, match="api down"):
        Remote(boom).generate("s", "p")
    with pytest.raises(LLMError, match="empty"):
        Remote(lambda n: "   ").generate("s", "p")


class Dummy(BaseAgent):
    name = "dummy"
    description = "dummy thing"

    def run(self, query, ctx):
        if query == "boom":
            raise RuntimeError("kaboom")
        return AgentResponse(self.name, self.polish("draft", query))


def test_base_agent_error_isolation():
    r = Dummy().handle("boom", AgentContext())
    assert not r.ok and "kaboom" in r.error and "dummy thing" in r.content


def test_polish_uses_llm_and_falls_back():
    assert Dummy().handle("q", AgentContext()).content == "draft"  # offline
    llm = FakeLLM()
    assert Dummy(llm=llm).handle("q", AgentContext()).content == "POLISHED: draft"
    assert "ONLY facts" in llm.calls[0][0]

    class Failing(FakeLLM):
        def generate(self, system, prompt):
            raise LLMError("nope")

    assert Dummy(llm=Failing()).handle("q", AgentContext()).content == "draft"
