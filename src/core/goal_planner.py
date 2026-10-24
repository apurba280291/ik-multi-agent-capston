"""Goal projection: required savings, deterministic growth and Monte Carlo odds."""
from __future__ import annotations

from typing import Any

from src.core.finance_math import (
    RISK_ASSUMPTIONS,
    future_value,
    monte_carlo,
    normalize_risk,
    required_monthly,
    suggested_allocation,
    yearly_projection,
)


def plan_goal(
    target: float,
    years: float,
    current: float = 0.0,
    monthly: float = 0.0,
    risk: str = "moderate",
    simulations: int = 2000,
) -> dict[str, Any]:
    """Return projections for reaching ``target`` in ``years`` under illustrative return assumptions."""
    if target <= 0 or years <= 0:
        raise ValueError("target and years must be positive")
    risk = normalize_risk(risk)
    mu, sigma = RISK_ASSUMPTIONS[risk]
    whole_years = max(1, int(round(years)))

    scenarios = {
        "cautious": max(mu - 0.02, 0.0),
        "expected": mu,
        "optimistic": mu + 0.02,
    }
    needed = {name: required_monthly(target, current, rate, years) for name, rate in scenarios.items()}
    projected = {name: future_value(current, monthly, rate, years) for name, rate in scenarios.items()}
    mc = monte_carlo(current, monthly, mu, sigma, whole_years, target=target, simulations=simulations)
    success = mc["success_probability"]
    on_track = projected["expected"] >= target

    return {
        "target": target,
        "years": years,
        "current": current,
        "monthly": monthly,
        "risk": risk,
        "expected_return": mu,
        "volatility": sigma,
        "scenarios": scenarios,
        "required_monthly": needed,
        "projected_value": projected,
        "total_contributed": current + monthly * 12 * years,
        "success_probability": success,
        "on_track": on_track,
        "allocation": suggested_allocation(risk, years),
        "deterministic_path": yearly_projection(current, monthly, mu, whole_years),
        "monte_carlo": mc,
        "shortfall": max(0.0, target - projected["expected"]),
        "status": status_label(success),
    }


def status_label(success: float) -> str:
    if success >= 0.75:
        return "On track"
    if success >= 0.5:
        return "Possible, but with little margin"
    return "Off track at current savings"
