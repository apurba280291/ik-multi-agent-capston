"""Light-weight extraction of tickers, holdings, money, durations and risk words from free text."""
from __future__ import annotations

import re
from datetime import date

from src.data.assets import load_metadata

# Company names/aliases -> ticker.
ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
    "nvidia": "NVDA", "tesla": "TSLA", "meta": "META", "facebook": "META", "netflix": "NFLX",
    "disney": "DIS", "walmart": "WMT", "exxon": "XOM", "coca-cola": "KO", "coca cola": "KO",
    "berkshire": "BRK-B", "jpmorgan": "JPM", "visa": "V", "bitcoin": "BTC-USD", "gold": "GLD",
    "s&p 500": "SPY", "s&p": "SPY", "nasdaq": "QQQ", "dow": "DIA", "russell 2000": "IWM",
}
_NOT_TICKERS = {"A", "I", "IRA", "ETF", "ETFS", "HSA", "CPI", "GDP", "VIX", "RMD", "USA", "FED"}


def extract_tickers(text: str) -> list[str]:
    """Known tickers (uppercase tokens), ``$XYZ`` cashtags and company aliases, in order of appearance."""
    known = set(load_metadata())
    found: list[tuple[int, str]] = []
    for m in re.finditer(r"\$([A-Za-z]{1,5}(?:-[A-Za-z])?)\b(?!\d)", text):
        sym = m.group(1).upper()
        if sym not in _NOT_TICKERS:
            found.append((m.start(), sym))
    for m in re.finditer(r"\b[A-Z]{1,5}(?:-[A-Z]{1,3})?\b", text):
        sym = m.group(0)
        if sym in known and sym not in _NOT_TICKERS and len(sym) > 1 and sym != "CASH":
            found.append((m.start(), sym))
    lower = text.lower()
    for alias, sym in ALIASES.items():
        for m in re.finditer(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lower):
            found.append((m.start(), sym))
    out: list[str] = []
    for _, sym in sorted(found):
        if sym not in out:
            out.append(sym)
    return out


_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
_MONEY = re.compile(r"(\$)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|million|mm|m|billion|b)?\b(?!\s*(?:%|years?|yrs?|months?))", re.I)


def parse_money_mentions(text: str) -> list[dict]:
    """Dollar-like amounts with position and surrounding context."""
    out = []
    for m in _MONEY.finditer(text):
        has_dollar, num, mult = m.group(1), m.group(2), (m.group(3) or "").lower()
        if not has_dollar and not mult and float(num.replace(",", "")) < 100:
            continue  # bare small numbers are probably not money
        value = float(num.replace(",", "")) * _MULT.get(mult, 1)
        # Keep the trailing context within this clause so it can't borrow words that describe the next amount.
        after = re.split(r"[.;,$]|\band\b", text[m.end() : m.end() + 30].lower())[0]
        out.append(
            {
                "value": value,
                "start": m.start(),
                "before": text[max(0, m.start() - 40) : m.start()].lower(),
                "after": after,
            }
        )
    return out


def parse_years(text: str, today: date | None = None) -> float | None:
    today = today or date.today()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"\bby\s+(?:the\s+year\s+)?(20\d\d)\b", text, re.I)
    if m and int(m.group(1)) > today.year:
        return float(int(m.group(1)) - today.year)
    m = re.search(r"(\d+)\s*months?\b", text, re.I)
    return float(m.group(1)) / 12 if m else None


def parse_risk(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(aggressive|high[- ]risk|risk[- ]tolerant|growth[- ]oriented)\b", t):
        return "aggressive"
    if re.search(r"\b(conservative|low[- ]risk|risk[- ]averse|cautious|safe)\b", t):
        return "conservative"
    if re.search(r"\b(moderate|medium[- ]risk|balanced)\b", t):
        return "moderate"
    return None


_HOLD_SHARES = re.compile(
    r"(?<![\w$])(\d[\d,]*(?:\.\d+)?)\s*(?:shares?\s+(?:of\s+)?)?\$?([A-Za-z]{1,5}(?:-[A-Za-z]{1,3})?)\b"
)
_HOLD_VALUE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\s*(?:in|of|worth of)\s+\$?([A-Za-z]{1,5}(?:-[A-Za-z]{1,3})?)\b", re.I)


def parse_holdings(text: str) -> list[dict]:
    """Parse "10 shares of AAPL, 5 MSFT, $2,000 in VTI" into holdings.

    Only symbols that are known tickers (or written as ``$XYZ``) are accepted, which keeps
    ordinary words ("5 years", "20 percent") from being mistaken for positions.
    """
    known = set(load_metadata())
    holdings: dict[str, dict] = {}
    spans: list[tuple[int, int]] = []
    for m in _HOLD_VALUE.finditer(text):
        sym = m.group(3).upper()
        if sym in known:
            value = float(m.group(1).replace(",", "")) * _MULT.get((m.group(2) or "").lower(), 1)
            holdings.setdefault(sym, {"symbol": sym})["value"] = holdings.get(sym, {}).get("value", 0) + value
            spans.append(m.span())
    for m in _HOLD_SHARES.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        raw = m.group(2)
        sym = raw.upper()
        cashtag = text[m.start(2) - 1 : m.start(2)] == "$"
        if sym in known or (cashtag and 1 <= len(sym) <= 5):
            if sym == "CASH":
                continue
            shares = float(m.group(1).replace(",", ""))
            h = holdings.setdefault(sym, {"symbol": sym})
            h["shares"] = h.get("shares", 0) + shares
    return list(holdings.values())
