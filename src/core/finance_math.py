"""Pure financial calculations (no I/O) used by the portfolio and goal agents."""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Illustrative long-run nominal assumptions - NOT forecasts. (expected return, volatility)
RISK_ASSUMPTIONS: dict[str, tuple[float, float]] = {
    "conservative": (0.045, 0.06),
    "moderate": (0.065, 0.11),
    "aggressive": (0.08, 0.16),
}

# Rough relative risk score (0-10) per asset class, used for the simple portfolio risk rating.
ASSET_CLASS_RISK: dict[str, float] = {
    "cash": 0.5,
    "bond": 3.0,
    "real_estate": 6.0,
    "commodity": 6.5,
    "stock": 8.0,
    "international_stock": 8.5,
    "crypto": 10.0,
}


def normalize_risk(label: str | None) -> str:
    label = (label or "moderate").lower()
    if label in RISK_ASSUMPTIONS:
        return label
    if label in {"low", "cautious", "safe"}:
        return "conservative"
    if label in {"high", "growth", "risky"}:
        return "aggressive"
    return "moderate"


def future_value(present: float, monthly: float, annual_rate: float, years: float) -> float:
    """Future value with monthly contributions (end of month) and monthly compounding."""
    n = int(round(years * 12))
    if n <= 0:
        return float(present)
    r = (1 + annual_rate) ** (1 / 12) - 1
    if abs(r) < 1e-12:
        return present + monthly * n
    growth = (1 + r) ** n
    return present * growth + monthly * (growth - 1) / r


def required_monthly(target: float, present: float, annual_rate: float, years: float) -> float:
    """Monthly contribution needed to reach ``target`` (0 if already on track)."""
    n = int(round(years * 12))
    if n <= 0:
        return max(0.0, target - present)
    r = (1 + annual_rate) ** (1 / 12) - 1
    growth = (1 + r) ** n
    gap = target - present * growth
    if gap <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return gap / n
    return gap * r / (growth - 1)


def yearly_projection(present: float, monthly: float, annual_rate: float, years: int) -> list[float]:
    return [future_value(present, monthly, annual_rate, y) for y in range(0, int(years) + 1)]


def monte_carlo(
    present: float,
    monthly: float,
    annual_return: float,
    volatility: float,
    years: int,
    target: float | None = None,
    simulations: int = 2000,
    seed: int | None = 42,
) -> dict:
    """Simulate yearly returns ~ Normal(mu, sigma); returns percentile paths and success odds."""
    rng = np.random.default_rng(seed)
    years = max(1, int(years))
    returns = np.clip(rng.normal(annual_return, volatility, size=(simulations, years)), -0.9, None)
    balances = np.zeros((simulations, years + 1))
    balances[:, 0] = present
    contrib = monthly * 12
    for y in range(years):
        balances[:, y + 1] = balances[:, y] * (1 + returns[:, y]) + contrib * (1 + returns[:, y] / 2)
    result = {
        "years": list(range(years + 1)),
        "p10": np.percentile(balances, 10, axis=0).tolist(),
        "p50": np.percentile(balances, 50, axis=0).tolist(),
        "p90": np.percentile(balances, 90, axis=0).tolist(),
        "simulations": simulations,
    }
    if target is not None:
        result["success_probability"] = float((balances[:, -1] >= target).mean())
    return result


def hhi(weights: Sequence[float]) -> float:
    """Herfindahl-Hirschman index of weights that sum to 1 (1/n = perfectly even, 1 = one holding)."""
    return float(sum(w * w for w in weights))


def diversification_score(holding_weights: Sequence[float], asset_class_weights: Sequence[float]) -> float:
    """0-100 score combining holding-level spread and asset-class breadth.

    70% weight: "effective number of holdings" (1/HHI), reaching full marks at 10;
    30% weight: number of asset classes held (full marks at 4).
    A single-holding portfolio scores ~7; a broad multi-asset one scores 90+.
    """
    w = [x for x in holding_weights if x > 0]
    if not w:
        return 0.0
    effective_n = 1 / hhi(w)
    spread = min(1.0, (effective_n - 1) / 9)
    classes = len([x for x in asset_class_weights if x > 0.01])
    breadth = min(classes, 4) / 4
    return round(100 * (0.7 * spread + 0.3 * breadth), 1)


def suggested_allocation(risk: str, years: float) -> dict[str, int]:
    """Textbook-style stock/bond/cash split by risk tolerance, shifted by time horizon."""
    base = {"conservative": 35, "moderate": 60, "aggressive": 80}[normalize_risk(risk)]
    if years < 3:
        base = min(base, 20)
    elif years < 5:
        base = min(base, 40)
    elif years >= 20:
        base = min(base + 10, 90)
    cash = 10 if years < 3 else 5
    bonds = 100 - base - cash
    return {"stocks": base, "bonds": bonds, "cash": cash}
