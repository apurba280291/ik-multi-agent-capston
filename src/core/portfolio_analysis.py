"""Portfolio metrics: values, allocation, fees, diversification, risk, stress test, observations.

Pure functions - prices are passed in, so the logic is deterministic and easily tested.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.finance_math import (
    ASSET_CLASS_RISK,
    diversification_score,
    hhi,
    normalize_risk,
    suggested_allocation,
)
from src.data.assets import asset_info

# Rough peak-to-trough shocks in a severe bear market (illustrative, not a forecast).
STRESS_SHOCK = {
    "stock": -0.35, "international_stock": -0.38, "bond": -0.05, "cash": 0.0,
    "real_estate": -0.30, "commodity": -0.15, "crypto": -0.60,
}
FUND_SECTORS = {"Broad Market", "Global", "International", "Bonds", "Dividend", "Small Cap"}


def risk_label(score: float) -> str:
    if score < 3:
        return "Low"
    if score < 5:
        return "Moderate-Low"
    if score < 6.5:
        return "Moderate"
    if score < 8:
        return "Moderate-High"
    return "High"


def analyze_portfolio(
    holdings: list[dict[str, Any]], prices: dict[str, float], risk_tolerance: str = "moderate"
) -> dict[str, Any]:
    """Analyse holdings of the form ``{"symbol", "shares"}`` or ``{"symbol", "value"}``.

    ``prices`` maps symbol -> latest price; holdings without a price (and no explicit value)
    are reported under ``unpriced`` and excluded from the totals.
    """
    positions: list[dict[str, Any]] = []
    unpriced: list[str] = []
    for h in holdings:
        sym = str(h.get("symbol", "")).upper().strip()
        if not sym:
            continue
        shares = h.get("shares")
        price = 1.0 if sym == "CASH" else prices.get(sym)
        if h.get("value") is not None:
            value = float(h["value"])
        elif shares is not None and price:
            value = float(shares) * float(price)
        else:
            unpriced.append(sym)
            continue
        if value <= 0:
            continue
        info = asset_info(sym)
        positions.append(
            {
                "symbol": sym, "name": info["name"] or sym, "shares": shares, "price": price,
                "value": value, "asset_class": info["asset_class"], "sector": info["sector"],
                "expense_ratio": info["expense_ratio"], "known": info.get("known", False),
            }
        )
    total = sum(p["value"] for p in positions)
    if total <= 0:
        return {"empty": True, "total_value": 0.0, "positions": [], "unpriced": unpriced, "observations": []}

    for p in positions:
        p["weight"] = p["value"] / total
    positions.sort(key=lambda p: p["value"], reverse=True)

    by_class: dict[str, float] = defaultdict(float)
    by_sector: dict[str, float] = defaultdict(float)
    for p in positions:
        by_class[p["asset_class"]] += p["weight"]
        by_sector[p["sector"]] += p["weight"]

    weights = [p["weight"] for p in positions]
    weighted_er = sum(p["weight"] * p["expense_ratio"] for p in positions)  # in percent
    risk_score = sum(p["weight"] * ASSET_CLASS_RISK.get(p["asset_class"], 8.0) for p in positions)
    stress = sum(p["weight"] * STRESS_SHOCK.get(p["asset_class"], -0.35) for p in positions)
    equity = by_class.get("stock", 0) + by_class.get("international_stock", 0)

    result = {
        "empty": False,
        "total_value": total,
        "positions": positions,
        "unpriced": unpriced,
        "allocation_by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
        "allocation_by_sector": dict(sorted(by_sector.items(), key=lambda kv: -kv[1])),
        "weighted_expense_ratio": weighted_er,
        "annual_fees": total * weighted_er / 100,
        "diversification_score": diversification_score(weights, list(by_class.values())),
        "hhi": hhi(weights),
        "top_holding_weight": weights[0],
        "risk_score": round(risk_score, 1),
        "risk_label": risk_label(risk_score),
        "stress_loss_pct": stress * 100,
        "equity_weight": equity,
    }
    result["observations"] = build_observations(result, risk_tolerance)
    return result


def build_observations(a: dict[str, Any], risk_tolerance: str) -> list[str]:
    """Educational talking points (not instructions to trade)."""
    obs: list[str] = []
    top = a["positions"][0]
    positions = a["positions"]
    if len(positions) < 5 and a["diversification_score"] < 60:
        obs.append(f"Only {len(positions)} holding(s): a broad index fund spreads risk across hundreds of companies.")
    if top["weight"] > 0.25 and top["sector"] not in FUND_SECTORS and top["asset_class"] == "stock":
        obs.append(
            f"{top['symbol']} is {top['weight']:.0%} of the portfolio. Many educators suggest keeping any single "
            "stock around 5-10% to limit concentration risk."
        )
    big_sectors = [s for s, w in a["allocation_by_sector"].items() if w > 0.5 and s not in FUND_SECTORS | {"Cash"}]
    for s in big_sectors:
        obs.append(f"{a['allocation_by_sector'][s]:.0%} sits in the {s} sector, so one industry's downturn would hit hard.")
    cls = a["allocation_by_class"]
    if cls.get("bond", 0) + cls.get("cash", 0) < 0.05:
        obs.append("No bonds or cash: nothing cushions a stock-market decline. The right mix depends on your time horizon.")
    if cls.get("international_stock", 0) == 0 and a["equity_weight"] > 0.3:
        obs.append("No international exposure; some investors add it for geographic diversification.")
    if cls.get("crypto", 0) > 0.10:
        obs.append(f"Crypto is {cls['crypto']:.0%} of the portfolio - highly volatile; speculation is often capped at a small share.")
    if a["weighted_expense_ratio"] > 0.5:
        obs.append(f"Weighted fund fees are {a['weighted_expense_ratio']:.2f}%; broad index ETFs often cost under 0.10%.")
    target = suggested_allocation(normalize_risk(risk_tolerance), 15)["stocks"]
    gap = a["equity_weight"] * 100 - target
    if abs(gap) >= 15:
        direction = "more" if gap > 0 else "less"
        obs.append(
            f"Stocks are {a['equity_weight']:.0%} of the portfolio, about {abs(gap):.0f} points {direction} than the ~{target}% "
            f"often used as an illustration for a {normalize_risk(risk_tolerance)} investor with a long horizon."
        )
    if a["diversification_score"] >= 75 and not obs:
        obs.append("The mix looks broadly diversified across holdings and asset classes.")
    if not obs:
        obs.append("No major red flags in these simple checks, but they cannot capture your goals or full finances.")
    return obs
