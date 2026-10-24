import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["FINANCE_LOG_DIR"] = tempfile.mkdtemp(prefix="finnie-logs-")  # keep test runs out of the real logs/
os.environ["FINANCE_LLM_PROVIDER"] = "offline"
os.environ["FINANCE_MARKET_PROVIDER"] = "mock"
for key in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY"):
    os.environ.pop(key, None)

from src.core.config import load_settings  # noqa: E402
from src.core.llm import LLMClient  # noqa: E402
from src.data.market_data import MarketDataService  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402
from src.workflow.assistant import FinanceAssistant  # noqa: E402


class FakeLLM(LLMClient):
    """Echoes a tagged draft so tests can see that polishing happened."""

    name = "fake"
    available = True

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        if callable(self.reply):
            return self.reply(system, prompt)
        return self.reply or "POLISHED: " + prompt.split("DRAFT:\n")[-1]


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def retriever(settings):
    return Retriever.from_settings(settings)


@pytest.fixture
def market(settings):
    return MarketDataService(settings)  # provider=mock via env


@pytest.fixture
def assistant(settings, retriever, market):
    return FinanceAssistant(settings=settings, retriever=retriever, market=market)


@pytest.fixture
def fake_llm():
    return FakeLLM()
