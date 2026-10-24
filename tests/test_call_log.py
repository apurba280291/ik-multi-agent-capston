import json

from src.core.base_agent import BaseAgent
from src.core.models import AgentContext, AgentResponse
from src.utils.call_log import call_log_path


class Flaky(BaseAgent):
    name = "flaky"

    def run(self, query, ctx):
        if "boom" in query:
            raise TimeoutError("upstream timed out")
        return AgentResponse(agent=self.name, content="fine")


def _rows():
    return [json.loads(line) for line in call_log_path().read_text(encoding="utf-8").splitlines()]


def test_success_and_failure_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_LOG_DIR", str(tmp_path))
    agent = Flaky()
    assert agent.handle("hello", AgentContext()).ok
    assert not agent.handle("boom", AgentContext()).ok
    ok_row, fail_row = _rows()
    assert ok_row["ok"] and ok_row["agent"] == "flaky" and ok_row["error"] is None
    assert not fail_row["ok"] and fail_row["error"].startswith("TimeoutError") and fail_row["query"] == "boom"


def test_call_log_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCE_CALL_LOG", "off")
    Flaky().handle("hello", AgentContext())
    assert not call_log_path().exists()
