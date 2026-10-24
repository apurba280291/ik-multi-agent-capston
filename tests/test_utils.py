import pytest

from src.core.config import Settings, load_settings
from src.utils.cache import TTLCache
from src.utils.formatting import DISCLAIMER, citation_block, money, pct, with_disclaimer
from src.core.models import Source
from src.utils.retry import RateLimiter, retry_with_backoff


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_cache_ttl_and_stale():
    clock = Clock()
    c = TTLCache(10, clock=clock)
    assert c.get("a") is None
    c.set("a", 1)
    assert c.get("a") == 1 and len(c) == 1
    clock.t = 11
    assert c.get("a") is None
    assert c.get("a", allow_stale=True) == 1
    assert c.age("a") == 11 and c.age("zzz") is None
    c.clear()
    assert len(c) == 0


def test_retry_succeeds_after_failures_with_backoff():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("boom")
        return "ok"

    assert retry_with_backoff(flaky, retries=3, base_delay=1, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]


def test_retry_raises_last_error():
    with pytest.raises(ValueError):
        retry_with_backoff(lambda: (_ for _ in ()).throw(ValueError("x")), retries=2, base_delay=0, sleep=lambda s: None)


def test_retry_does_not_catch_other_exceptions():
    with pytest.raises(KeyError):
        retry_with_backoff(lambda: {}["x"], retries=3, exceptions=(ValueError,), sleep=lambda s: None)


def test_rate_limiter_spaces_calls():
    clock, sleeps = Clock(), []

    def sleep(d):
        sleeps.append(d)
        clock.t += d

    rl = RateLimiter(60, clock=clock, sleep=sleep)
    rl.wait()
    rl.wait()
    rl.wait()
    assert sleeps == [1.0, 1.0]
    assert RateLimiter(0).interval == 0


def test_formatting():
    assert money(1234.6) == "$1,235"
    assert money(-5.256, 2) == "-$5.26"
    assert pct(12.345) == "12.3%"
    assert with_disclaimer("hi").endswith(DISCLAIMER)
    assert with_disclaimer(with_disclaimer("hi")).count(DISCLAIMER) == 1
    assert citation_block([]) == ""
    block = citation_block([Source("T", "cat", "http://x")])
    assert "[1] T (cat) - http://x" in block


def test_settings_lookup_and_set():
    s = Settings({"a": {"b": 1}})
    assert s.get("a.b") == 1 and s.get("a.c", 5) == 5 and s.get("z.y") is None
    s.set("x.y.z", 3)
    assert s.get("x.y.z") == 3


def test_load_settings_missing_file_gives_defaults(tmp_path):
    s = load_settings(tmp_path / "nope.yaml")
    assert s.get("rag.top_k") is None and s.get("llm.provider") == "offline"


def test_load_settings_bad_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("a: [unclosed", encoding="utf-8")
    assert load_settings(p).get("a") is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_PROVIDER", "gemini")
    assert load_settings().get("llm.provider") == "gemini"


def test_resolve_path(settings):
    assert settings.resolve_path("src/data").exists()
    assert settings.resolve_path(str(settings.resolve_path("src"))).exists()
