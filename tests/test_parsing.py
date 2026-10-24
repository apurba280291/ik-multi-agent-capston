from datetime import date

import pytest

from src.utils.parsing import extract_tickers, parse_holdings, parse_money_mentions, parse_risk, parse_years


def test_extract_tickers_variants():
    assert extract_tickers("How is NVDA vs AAPL doing?") == ["NVDA", "AAPL"]
    assert extract_tickers("what about $tsla?") == ["TSLA"]
    assert extract_tickers("price of Apple and bitcoin") == ["AAPL", "BTC-USD"]
    assert extract_tickers("Explain ETF and IRA") == []
    assert extract_tickers("S&P 500 today") == ["SPY"]


def test_single_letter_requires_cashtag():
    assert "V" not in extract_tickers("a V shaped recovery")
    assert extract_tickers("buy $V") == ["V"]


def test_parse_holdings_shares_and_values():
    h = parse_holdings("I own 10 shares of VTI, 5 AAPL and $2,000 in BND")
    by = {x["symbol"]: x for x in h}
    assert by["VTI"]["shares"] == 10 and by["AAPL"]["shares"] == 5 and by["BND"]["value"] == 2000


def test_parse_holdings_ignores_words_and_merges_duplicates():
    assert parse_holdings("I want it in 5 years for 20 percent") == []
    h = parse_holdings("3 AAPL and 2 AAPL")
    assert h == [{"symbol": "AAPL", "shares": 5}]
    assert parse_holdings("$5k in vti")[0]["value"] == 5000


def test_parse_money():
    vals = [m["value"] for m in parse_money_mentions("save $50,000, or 2k, maybe 1.5 million, also 5 years")]
    assert vals == [50000, 2000, 1_500_000]


def test_parse_years():
    today = date(2026, 1, 1)
    assert parse_years("in 10 years") == 10
    assert parse_years("by 2040", today) == 14
    assert parse_years("in 18 months") == 1.5
    assert parse_years("by 2020", today) is None
    assert parse_years("whenever") is None


@pytest.mark.parametrize(
    "text,risk",
    [("I'm an aggressive investor", "aggressive"), ("very cautious", "conservative"), ("a balanced approach", "moderate"), ("hello", None)],
)
def test_parse_risk(text, risk):
    assert parse_risk(text) == risk
