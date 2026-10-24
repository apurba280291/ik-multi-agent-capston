import pandas as pd
import pytest

from src.core.config import Settings
from src.data.market_data import (
    AlphaVantageProvider,
    MarketDataService,
    MockProvider,
    PriceHistory,
    ProviderError,
    Provider,
    Quote,
    RateLimitError,
    YFinanceProvider,
    NewsItem,
    normalize_symbol,
    trend_metrics,
)
from src.utils.cache import TTLCache

SETTINGS = Settings({"market": {"cache_ttl_seconds": 1800, "retries": 3, "backoff_base_seconds": 0, "indices": {"SPY": "S&P 500", "QQQ": "Nasdaq"}}})


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, fail_times=0, error=ProviderError("down"), price=100.0):
        self.fail_times, self.error, self.price, self.calls = fail_times, error, price, 0

    def fetch_quote(self, symbol):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return Quote(symbol, self.price, self.price - 1, "fake", "now")

    def fetch_history(self, symbol, period):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return PriceHistory(symbol, ["d1", "d2"], [1.0, 2.0], "fake", "now")

    def fetch_news(self, symbol):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return [NewsItem("Headline", symbol=symbol)]


def svc(primary=None, secondary=None, cache=None):
    return MarketDataService(SETTINGS, primary=primary or FakeProvider(), secondary=secondary, cache=cache, sleep=lambda s: None)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_symbol_validation():
    assert normalize_symbol(" brk-b ") == "BRK-B"
    for bad in ("", "DROP TABLE", "A" * 20, "AA PL"):
        with pytest.raises(ValueError):
            normalize_symbol(bad)


def test_quote_math():
    q = Quote("X", 110, 100, "s", "t")
    assert q.change == 10 and q.change_pct == 10
    assert Quote("X", 1, 0, "s", "t").change_pct == 0
    assert q.to_dict()["change_pct"] == 10


def test_quote_cached_within_ttl():
    p = FakeProvider()
    s = svc(p)
    s.get_quote("AAPL")
    s.get_quote("aapl")
    assert p.calls == 1


def test_retries_then_succeeds():
    p = FakeProvider(fail_times=2)
    q = svc(p).get_quote("AAPL")
    assert q.source == "fake" and p.calls == 3 and not q.is_mock


def test_falls_back_to_secondary_provider():
    primary, secondary = FakeProvider(fail_times=99), FakeProvider(price=50)
    q = svc(primary, secondary).get_quote("AAPL")
    assert q.price == 50 and primary.calls == 3


def test_rate_limit_not_retried():
    primary = FakeProvider(fail_times=99, error=RateLimitError("quota"))
    q = svc(primary).get_quote("AAPL")
    assert primary.calls == 1 and q.is_mock  # skipped retries, fell through to mock


def test_stale_cache_used_when_providers_fail():
    clock = Clock()
    cache = TTLCache(10, clock=clock)
    p = FakeProvider()
    s = svc(p, cache=cache)
    s.get_quote("AAPL")
    clock.t = 100
    p.fail_times = 10**6
    q = s.get_quote("AAPL")
    assert q.is_stale and q.price == 100
    assert "Cached" in MarketDataService.freshness(q)


def test_mock_fallback_when_everything_fails_and_flagged():
    s = svc(FakeProvider(fail_times=99))
    q = s.get_quote("AAPL")
    assert q.is_mock and q.source == "mock"
    assert "Sample data" in MarketDataService.freshness(q)
    assert "Live via" in MarketDataService.freshness(Quote("X", 1, 1, "yfinance", "t"))


def test_unknown_symbol_everywhere_raises_provider_error():
    with pytest.raises(ProviderError):
        svc(FakeProvider(fail_times=99)).get_quote("NOSUCHTICKER")


def test_get_quotes_skips_bad_symbols():
    s = svc(FakeProvider())
    out = s.get_quotes(["AAPL", "bad symbol", "MSFT"])
    assert set(out) == {"AAPL", "MSFT"}


def test_cash_is_always_one_dollar():
    assert svc(FakeProvider(fail_times=99)).get_quote("CASH").price == 1.0


def test_history_and_trend(market):
    h = market.get_history("VTI", "6mo")
    assert h.is_mock and len(h.dates) == len(h.closes) == 126 and h.closes[-1] == 300.0
    t = market.analyze_trend("VTI")
    assert t["symbol"] == "VTI" and t["trend"] in {"uptrend", "downtrend", "sideways"} and t["is_mock"]


def test_mock_history_is_deterministic():
    p = MockProvider()
    assert p.fetch_history("AAPL", "1mo").closes == p.fetch_history("AAPL", "1mo").closes
    assert set(p.fetch_history("CASH", "1mo").closes) == {1.0}


def test_news_fallbacks():
    live = svc(FakeProvider()).get_news("SPY")
    assert live[0].title == "Headline"
    sample = svc(FakeProvider(fail_times=99)).get_news("SPY")
    assert sample and all(n.is_sample for n in sample)
    s = svc(FakeProvider())
    s.get_news("SPY")
    s.get_news("SPY")  # second call served from cache
    assert s.primary.calls == 1


def test_market_overview_skips_failures(market):
    rows = market.market_overview()
    assert {r["symbol"] for r in rows} >= {"SPY", "QQQ"} and all("name" in r for r in rows)


def test_provider_selection_from_settings(monkeypatch):
    s = MarketDataService(Settings({"market": {"provider": "mock"}}))
    assert s.primary is None
    assert isinstance(MarketDataService(Settings({"market": {"provider": "yfinance"}})).primary, YFinanceProvider)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k")
    both = MarketDataService(Settings({"market": {"provider": "alpha_vantage"}}))
    assert isinstance(both.primary, AlphaVantageProvider) and isinstance(both.secondary, YFinanceProvider)
    assert isinstance(MarketDataService(Settings({"market": {"provider": "yfinance"}})).secondary, AlphaVantageProvider)


# --- Alpha Vantage with an injected HTTP client
class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def av(payload):
    return AlphaVantageProvider("key", http_get=lambda url, params, timeout: Resp(payload))


def test_alpha_vantage_quote_and_history():
    q = av({"Global Quote": {"05. price": "101.5", "08. previous close": "100"}}).fetch_quote("X")
    assert q.price == 101.5 and q.source == "alpha_vantage"
    series = {f"2025-01-{d:02d}": {"4. close": str(d)} for d in range(1, 11)}
    h = av({"Time Series (Daily)": series}).fetch_history("X", "1mo")
    assert h.closes == [float(d) for d in range(1, 11)]


def test_alpha_vantage_errors():
    with pytest.raises(RateLimitError):
        av({"Note": "limit reached"}).fetch_quote("X")
    with pytest.raises(ProviderError):
        av({"Error Message": "bad"}).fetch_quote("X")
    with pytest.raises(ProviderError):
        av({"Global Quote": {}}).fetch_quote("X")
    with pytest.raises(ProviderError):
        av({}).fetch_history("X", "6mo")


# --- yfinance provider with a faked ticker (no network)
class FakeTicker:
    def __init__(self, df, news=None):
        self.df, self.news = df, news or []

    def history(self, **kwargs):
        return self.df


def yf_provider(df, news=None):
    p = YFinanceProvider()
    p._ticker = lambda symbol: FakeTicker(df, news)
    return p


def test_yfinance_quote_history_news():
    idx = pd.date_range("2025-01-01", periods=5)
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    news = [
        {"content": {"title": "New format", "provider": {"displayName": "Wire"}, "canonicalUrl": {"url": "http://u"}, "summary": "s"}},
        {"title": "Old format", "publisher": "Old", "link": "http://o"},
        {"content": {}},
    ]
    p = yf_provider(df, news)
    q = p.fetch_quote("X")
    assert (q.price, q.prev_close) == (5.0, 4.0)
    assert len(p.fetch_history("X", "5d").closes) == 5
    items = p.fetch_news("X")
    assert [i.title for i in items] == ["New format", "Old format"] and items[0].publisher == "Wire"


def test_yfinance_empty_data_raises():
    with pytest.raises(ProviderError):
        yf_provider(pd.DataFrame()).fetch_history("X", "1mo")
    one = pd.DataFrame({"Close": [1.0]}, index=pd.date_range("2025-01-01", periods=1))
    with pytest.raises(ProviderError):
        yf_provider(one).fetch_quote("X")


def test_trend_metrics_labels():
    up = trend_metrics([float(i) for i in range(1, 80)])
    assert up["trend"] == "uptrend" and up["return_period_pct"] > 0 and up["max_drawdown_pct"] == 0
    down = trend_metrics([float(i) for i in range(80, 1, -1)])
    assert down["trend"] == "downtrend" and down["max_drawdown_pct"] < 0
    assert trend_metrics([1.0, 1.0, 1.0] * 10)["trend"] == "sideways"
    assert trend_metrics([5.0])["trend"] == "insufficient data"
    short = trend_metrics([1, 2, 3, 4, 5] * 5)  # >=20 but <50 points
    assert short["sma50"] is None and short["trend"] in {"uptrend", "downtrend"}
