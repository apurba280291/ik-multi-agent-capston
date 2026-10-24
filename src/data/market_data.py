"""Real-time market data with caching, retries/backoff and graceful fallbacks.

Fallback chain for every request:
    fresh cache -> primary provider (with retries) -> secondary provider -> stale cache -> mock data

Every result carries ``source``, ``as_of`` and ``is_stale``/``is_mock`` flags so the UI and
agents can show data-freshness indicators and never present fallback data as live.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.config import PROJECT_ROOT, Settings
from src.utils.cache import TTLCache
from src.utils.logging_utils import get_logger
from src.utils.retry import retry_with_backoff

log = get_logger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")
DATA_DIR = PROJECT_ROOT / "src" / "data"


class ProviderError(RuntimeError):
    """A provider failed to return usable data."""


class RateLimitError(ProviderError):
    """Provider quota exhausted; retrying immediately is pointless."""


class _NoRetry(Exception):
    """Wraps RateLimitError so the retry helper (which retries ProviderError etc.) lets it through."""


@dataclass
class Quote:
    symbol: str
    price: float
    prev_close: float
    source: str
    as_of: str
    is_stale: bool = False
    is_mock: bool = False

    @property
    def change(self) -> float:
        return self.price - self.prev_close

    @property
    def change_pct(self) -> float:
        return (self.change / self.prev_close * 100) if self.prev_close else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(change=self.change, change_pct=self.change_pct)
        return d


@dataclass
class PriceHistory:
    symbol: str
    dates: list[str]
    closes: list[float]
    source: str
    as_of: str
    is_stale: bool = False
    is_mock: bool = False


@dataclass
class NewsItem:
    title: str
    publisher: str = ""
    summary: str = ""
    url: str = ""
    symbol: str = ""
    is_sample: bool = False


def normalize_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if not _SYMBOL_RE.match(sym):
        raise ValueError(f"invalid ticker symbol: {symbol!r}")
    return sym


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------- providers
class Provider:
    name = "base"

    def fetch_quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    def fetch_history(self, symbol: str, period: str) -> PriceHistory:
        raise NotImplementedError

    def fetch_news(self, symbol: str) -> list[NewsItem]:
        return []


class YFinanceProvider(Provider):
    name = "yfinance"

    def __init__(self, timeout: float = 10):
        self.timeout = timeout

    def _ticker(self, symbol: str):
        import yfinance as yf

        return yf.Ticker(symbol)

    def fetch_history(self, symbol: str, period: str = "6mo") -> PriceHistory:
        df = self._ticker(symbol).history(period=period, auto_adjust=True, timeout=self.timeout)
        if df is None or df.empty or "Close" not in df:
            raise ProviderError(f"yfinance returned no data for {symbol}")
        closes = [float(x) for x in df["Close"].dropna().tolist()]
        dates = [d.strftime("%Y-%m-%d") for d in df["Close"].dropna().index]
        return PriceHistory(symbol, dates, closes, self.name, _now())

    def fetch_quote(self, symbol: str) -> Quote:
        hist = self.fetch_history(symbol, "5d")
        if len(hist.closes) < 2:
            raise ProviderError(f"not enough history for {symbol}")
        return Quote(symbol, hist.closes[-1], hist.closes[-2], self.name, _now())

    def fetch_news(self, symbol: str) -> list[NewsItem]:
        items = []
        for raw in (self._ticker(symbol).news or [])[:8]:
            content = raw.get("content", raw)
            title = content.get("title")
            if not title:
                continue
            provider = content.get("provider") or {}
            url = (content.get("canonicalUrl") or {}).get("url") or content.get("link", "")
            items.append(
                NewsItem(
                    title=title,
                    publisher=provider.get("displayName") or content.get("publisher", ""),
                    summary=content.get("summary", "") or "",
                    url=url,
                    symbol=symbol,
                )
            )
        return items


class AlphaVantageProvider(Provider):
    name = "alpha_vantage"
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: float = 10, http_get: Callable | None = None):
        self.api_key = api_key
        self.timeout = timeout
        self._get = http_get

    def _request(self, params: dict[str, str]) -> dict:
        get = self._get
        if get is None:
            import requests

            get = requests.get
        resp = get(self.URL, params={**params, "apikey": self.api_key}, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        if "Note" in payload or "Information" in payload:
            raise RateLimitError(payload.get("Note") or payload.get("Information"))
        if "Error Message" in payload:
            raise ProviderError(payload["Error Message"])
        return payload

    def fetch_quote(self, symbol: str) -> Quote:
        data = self._request({"function": "GLOBAL_QUOTE", "symbol": symbol}).get("Global Quote") or {}
        try:
            price, prev = float(data["05. price"]), float(data["08. previous close"])
        except (KeyError, ValueError) as exc:
            raise ProviderError(f"Alpha Vantage returned no quote for {symbol}") from exc
        return Quote(symbol, price, prev, self.name, _now())

    def fetch_history(self, symbol: str, period: str = "6mo") -> PriceHistory:
        size = "full" if period in {"2y", "5y", "max"} else "compact"
        series = self._request(
            {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": size}
        ).get("Time Series (Daily)")
        if not series:
            raise ProviderError(f"Alpha Vantage returned no history for {symbol}")
        days = sorted(series)
        keep = _PERIOD_DAYS.get(period, 126)
        days = days[-keep:]
        return PriceHistory(symbol, days, [float(series[d]["4. close"]) for d in days], self.name, _now())


_PERIOD_DAYS = {"5d": 5, "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504}


class MockProvider(Provider):
    """Deterministic illustrative data (never presented as live)."""

    name = "mock"

    def __init__(self, path: Path | None = None):
        try:
            raw = json.loads((path or DATA_DIR / "mock_market.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        self._prices = {k: v for k, v in raw.items() if not k.startswith("_")}
        news_path = DATA_DIR / "sample_news.json"
        try:
            self._news = json.loads(news_path.read_text(encoding="utf-8")).get("general", [])
        except (OSError, json.JSONDecodeError):
            self._news = []

    def _entry(self, symbol: str) -> dict:
        if symbol == "CASH":
            return {"price": 1.0, "prev_close": 1.0}
        if symbol in self._prices:
            return self._prices[symbol]
        raise ProviderError(f"no sample data for {symbol}")

    def fetch_quote(self, symbol: str) -> Quote:
        e = self._entry(symbol)
        return Quote(symbol, e["price"], e["prev_close"], self.name, _now(), is_mock=True)

    def fetch_history(self, symbol: str, period: str = "6mo") -> PriceHistory:
        e = self._entry(symbol)
        n = _PERIOD_DAYS.get(period, 126)
        rng = np.random.default_rng(sum(map(ord, symbol)))
        vol = 0.0 if symbol == "CASH" else 0.012
        rets = rng.normal(0.0 if symbol == "CASH" else 0.0004, vol, n)
        path = e["price"] * np.exp(-np.cumsum(rets[::-1]))[::-1]  # ends exactly at the sample price
        path[-1] = e["price"]
        if n >= 2:
            path[-2] = e["prev_close"]
        dates = [d.strftime("%Y-%m-%d") for d in _business_days(n)]
        return PriceHistory(symbol, dates, [float(x) for x in path], self.name, _now(), is_mock=True)

    def fetch_news(self, symbol: str) -> list[NewsItem]:
        return [
            NewsItem(n["title"], n.get("publisher", ""), n.get("summary", ""), "", symbol, is_sample=True)
            for n in self._news
        ]


def _business_days(n: int):
    import pandas as pd

    return list(pd.bdate_range(end=datetime.now().date(), periods=n))


# --------------------------------------------------------------------------- service
class MarketDataService:
    def __init__(
        self,
        settings: Settings,
        primary: Provider | None = None,
        secondary: Provider | None = None,
        mock: Provider | None = None,
        cache: TTLCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        ttl = float(settings.get("market.cache_ttl_seconds", 1800))
        self.cache = cache if cache is not None else TTLCache(ttl)
        self.retries = int(settings.get("market.retries", 3))
        self.backoff = float(settings.get("market.backoff_base_seconds", 0.5))
        self._sleep = sleep
        self.mock = mock or MockProvider()
        self.indices: dict[str, str] = settings.get("market.indices", {}) or {}
        self.primary, self.secondary = primary, secondary
        if primary is None:
            self.primary, self.secondary = self._providers_from_settings()

    def _providers_from_settings(self) -> tuple[Provider | None, Provider | None]:
        choice = str(self.settings.get("market.provider", "yfinance")).lower()
        timeout = float(self.settings.get("market.timeout_seconds", 10))
        av_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        av = AlphaVantageProvider(av_key, timeout) if av_key else None
        if choice == "mock":
            return None, None
        if choice == "alpha_vantage":
            return av, YFinanceProvider(timeout)
        return YFinanceProvider(timeout), av

    @staticmethod
    def _call_once(call: Callable[[Provider], Any], provider: Provider):
        try:
            return call(provider)
        except RateLimitError as exc:
            raise _NoRetry(str(exc)) from exc

    # ---- generic fetch-with-fallback
    def _fetch(self, kind: str, key: str, call: Callable[[Provider], Any]):
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        for provider in (self.primary, self.secondary):
            if provider is None:
                continue
            try:
                # Rate-limit errors are not retried: backing off for seconds won't reset a daily quota.
                result = retry_with_backoff(
                    lambda p=provider: self._call_once(call, p),
                    self.retries,
                    self.backoff,
                    exceptions=(ProviderError, OSError, ValueError, KeyError, RuntimeError),
                    sleep=self._sleep,
                )
                self.cache.set(key, result)
                return result
            except Exception as exc:
                log.warning("%s %s via %s failed: %s", kind, key, provider.name, exc)
        stale = self.cache.get(key, allow_stale=True)
        if stale is not None:
            stale.is_stale = True
            return stale
        return call(self.mock)

    def get_quote(self, symbol: str) -> Quote:
        sym = normalize_symbol(symbol)
        if sym == "CASH":
            return self.mock.fetch_quote(sym)
        return self._fetch("quote", f"quote:{sym}", lambda p: p.fetch_quote(sym))

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for s in symbols:
            try:
                out[normalize_symbol(s)] = self.get_quote(s)
            except (ValueError, ProviderError) as exc:
                log.warning("skipping %s: %s", s, exc)
        return out

    def get_history(self, symbol: str, period: str = "6mo") -> PriceHistory:
        sym = normalize_symbol(symbol)
        return self._fetch("history", f"hist:{sym}:{period}", lambda p: p.fetch_history(sym, period))

    def get_news(self, symbol: str = "SPY", limit: int = 6) -> list[NewsItem]:
        sym = normalize_symbol(symbol)
        key = f"news:{sym}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached[:limit]
        for provider in (self.primary, self.secondary):
            if provider is None:
                continue
            try:
                items = provider.fetch_news(sym)
                if items:
                    self.cache.set(key, items)
                    return items[:limit]
            except Exception as exc:
                log.warning("news %s via %s failed: %s", sym, provider.name, exc)
        return self.mock.fetch_news(sym)[:limit]

    def market_overview(self) -> list[dict[str, Any]]:
        rows = []
        for sym, label in self.indices.items():
            try:
                q = self.get_quote(sym)
            except Exception as exc:  # never break the dashboard for one symbol
                log.warning("overview skip %s: %s", sym, exc)
                continue
            rows.append({"name": label, **q.to_dict()})
        return rows

    @staticmethod
    def freshness(item: Quote | PriceHistory) -> str:
        if item.is_mock:
            return f"Sample data (live feed unavailable) - illustrative only, {item.as_of}"
        if item.is_stale:
            return f"Cached data (may be outdated) from {item.source}, {item.as_of}"
        return f"Live via {item.source}, as of {item.as_of}"

    def analyze_trend(self, symbol: str, period: str = "6mo") -> dict[str, Any]:
        h = self.get_history(symbol, period)
        return trend_metrics(h.closes) | {
            "symbol": h.symbol,
            "source": h.source,
            "is_mock": h.is_mock,
            "is_stale": h.is_stale,
            "as_of": h.as_of,
        }


def trend_metrics(closes: list[float]) -> dict[str, Any]:
    """SMA20/50, 1-month and period return, annualised volatility, max drawdown, trend label."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < 2:
        return {"last": float(arr[-1]) if arr.size else math.nan, "trend": "insufficient data"}
    rets = np.diff(arr) / arr[:-1]
    sma20 = float(arr[-20:].mean()) if arr.size >= 20 else None
    sma50 = float(arr[-50:].mean()) if arr.size >= 50 else None
    peak = np.maximum.accumulate(arr)
    max_dd = float(((arr - peak) / peak).min() * 100)
    last = float(arr[-1])
    if sma20 and sma50 and last > sma50 and sma20 > sma50:
        trend = "uptrend"
    elif sma20 and sma50 and last < sma50 and sma20 < sma50:
        trend = "downtrend"
    elif sma20 and not sma50 and last != sma20:
        trend = "uptrend" if last > sma20 else "downtrend"
    else:
        trend = "sideways"
    month = arr[-22] if arr.size >= 22 else arr[0]
    return {
        "last": last,
        "sma20": sma20,
        "sma50": sma50,
        "return_1m_pct": float((last / month - 1) * 100),
        "return_period_pct": float((last / arr[0] - 1) * 100),
        "volatility_annual_pct": float(rets.std(ddof=1) * math.sqrt(252) * 100) if rets.size > 1 else 0.0,
        "max_drawdown_pct": max_dd,
        "trend": trend,
    }
