import json

from src.core import llm as llm_mod
from src.core.base_agent import BaseAgent
from src.core.config import Settings
from src.core.models import AgentContext, AgentResponse
from src.utils.usage_log import usage_log_path


class Reporting(llm_mod._RemoteLLM):
    name, model = "remote", "m1"

    def __init__(self, counts):
        super().__init__(Settings({"llm": {"requests_per_minute": 0}}))
        self.counts = counts

    def _call(self, system, prompt):
        if self.counts:
            self._note_usage(*self.counts)
        return "a reply of some length"


def _rows():
    return [json.loads(line) for line in usage_log_path().read_text(encoding="utf-8").splitlines()]


def test_reported_usage_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_LOG_DIR", str(tmp_path))
    Reporting((120, 45)).generate("sys", "prompt")
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (120, 45, False)
    assert row["provider"] == "remote" and row["model"] == "m1" and row["caller"] == "router"


def test_missing_usage_is_estimated(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_LOG_DIR", str(tmp_path))
    Reporting(None).generate("s" * 40, "p" * 40)
    (row,) = _rows()
    assert row["estimated"] and row["input_tokens"] == 20 and row["output_tokens"] > 0


def test_agent_calls_are_tagged_with_agent_name(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_LOG_DIR", str(tmp_path))

    class Talker(BaseAgent):
        name = "talker"

        def run(self, query, ctx):
            return AgentResponse(self.name, self.polish("draft 1", query))

    Talker(llm=Reporting((10, 5))).handle("q", AgentContext())
    assert _rows()[0]["caller"] == "talker"
