import pytest

from src.core.portfolio_analysis import analyze_portfolio, risk_label
from src.data.assets import asset_info, load_sample_portfolios

PRICES = {"VTI": 300.0, "BND": 75.0, "AAPL": 200.0, "TSLA": 300.0, "VXUS": 60.0}


def test_basic_totals_and_weights():
    a = analyze_portfolio([{"symbol": "VTI", "shares": 10}, {"symbol": "BND", "shares": 20}], PRICES)
    assert a["total_value"] == 4500
    assert sum(p["weight"] for p in a["positions"]) == pytest.approx(1)
    assert a["positions"][0]["symbol"] == "VTI"  # sorted by value
    assert a["allocation_by_class"]["stock"] == pytest.approx(3000 / 4500)
    assert a["weighted_expense_ratio"] == pytest.approx(0.03)


def test_value_and_cash_holdings_need_no_price():
    a = analyze_portfolio([{"symbol": "VTI", "value": 1000}, {"symbol": "CASH", "shares": 500}], {})
    assert a["total_value"] == 1500
    assert a["allocation_by_class"]["cash"] == pytest.approx(1 / 3)


def test_unpriced_and_invalid_holdings_excluded():
    a = analyze_portfolio(
        [{"symbol": "VTI", "shares": 1}, {"symbol": "ZZZZ", "shares": 3}, {"symbol": "", "shares": 1}, {"symbol": "BND", "value": 0}],
        PRICES,
    )
    assert a["unpriced"] == ["ZZZZ"] and a["total_value"] == 300


def test_empty_portfolio():
    a = analyze_portfolio([], {})
    assert a["empty"] and a["total_value"] == 0
    assert analyze_portfolio([{"symbol": "ZZZZ", "shares": 1}], {})["empty"]


def test_single_stock_is_flagged_risky():
    a = analyze_portfolio([{"symbol": "TSLA", "shares": 100}], PRICES)
    text = " ".join(a["observations"])
    assert a["diversification_score"] < 15
    assert a["risk_label"] in {"Moderate-High", "High"}
    assert "TSLA is 100%" in text and "Only 1 holding" in text
    assert "No bonds or cash" in text


def test_balanced_portfolio_scores_better_than_single_stock():
    balanced = analyze_portfolio(
        [{"symbol": "VTI", "shares": 40}, {"symbol": "VXUS", "shares": 100}, {"symbol": "BND", "shares": 100}], PRICES
    )
    single = analyze_portfolio([{"symbol": "TSLA", "shares": 100}], PRICES)
    assert balanced["diversification_score"] > single["diversification_score"]
    assert balanced["risk_score"] < single["risk_score"]
    assert abs(balanced["stress_loss_pct"]) < abs(single["stress_loss_pct"])


def test_sector_concentration_and_unknown_symbol():
    a = analyze_portfolio([{"symbol": "AAPL", "shares": 10}, {"symbol": "XYZQ", "value": 100}], {**PRICES})
    assert any("Technology" in o for o in a["observations"])
    unknown = next(p for p in a["positions"] if p["symbol"] == "XYZQ")
    assert unknown["known"] is False and unknown["sector"] == "Unclassified"


def test_crypto_flagged():
    a = analyze_portfolio([{"symbol": "BTC-USD", "value": 500}, {"symbol": "GLD", "value": 500}], {})
    text = " ".join(a["observations"])
    assert "Crypto is 50%" in text and "single stock" not in text


def test_high_fees_flagged(monkeypatch):
    from src.core import portfolio_analysis as pa

    monkeypatch.setattr(pa, "asset_info", lambda s: {**asset_info("VTI"), "expense_ratio": 1.2})
    a = analyze_portfolio([{"symbol": "VTI", "value": 1000}], {})
    assert any("fund fees" in o for o in a["observations"])


def test_risk_tolerance_changes_allocation_comment():
    holdings = [{"symbol": "BND", "value": 900}, {"symbol": "VTI", "value": 100}]
    cons = " ".join(analyze_portfolio(holdings, PRICES, "aggressive")["observations"])
    assert "fewer" not in cons and "less than" in cons  # far less equity than an aggressive illustration


@pytest.mark.parametrize("score,label", [(1, "Low"), (3.5, "Moderate-Low"), (6, "Moderate"), (7, "Moderate-High"), (9, "High")])
def test_risk_label(score, label):
    assert risk_label(score) == label


def test_asset_info_and_samples():
    assert asset_info("vti")["asset_class"] == "stock"
    assert asset_info("nope")["known"] is False
    samples = load_sample_portfolios()
    assert len(samples) >= 3
    for holdings in samples.values():
        assert analyze_portfolio(holdings, {h["symbol"]: 100.0 for h in holdings})["total_value"] > 0
