import pytest

from src.core import finance_math as fm
from src.core.goal_planner import plan_goal, status_label


def test_future_value_known_values():
    # $1,000 at 7% for 10 years, no contributions -> 1000 * 1.07^10
    assert fm.future_value(1000, 0, 0.07, 10) == pytest.approx(1967.15, rel=1e-3)
    assert fm.future_value(1000, 100, 0, 2) == 3400
    assert fm.future_value(500, 50, 0.05, 0) == 500


def test_required_monthly_roundtrip():
    need = fm.required_monthly(50_000, 0, 0.06, 10)
    assert fm.future_value(0, need, 0.06, 10) == pytest.approx(50_000, rel=1e-6)
    assert 290 < need < 320  # matches the knowledge-base example (~$300)


def test_required_monthly_edge_cases():
    assert fm.required_monthly(1000, 5000, 0.05, 5) == 0.0
    assert fm.required_monthly(1200, 0, 0.0, 1) == 100
    assert fm.required_monthly(1000, 400, 0.05, 0) == 600


def test_projection_length_and_growth():
    path = fm.yearly_projection(1000, 100, 0.05, 5)
    assert len(path) == 6 and path[0] == 1000
    assert all(b > a for a, b in zip(path, path[1:]))


def test_monte_carlo_is_deterministic_and_ordered():
    a = fm.monte_carlo(1000, 100, 0.07, 0.15, 10, target=5000, simulations=500, seed=1)
    b = fm.monte_carlo(1000, 100, 0.07, 0.15, 10, target=5000, simulations=500, seed=1)
    assert a == b
    assert all(lo <= mid <= hi for lo, mid, hi in zip(a["p10"], a["p50"], a["p90"]))
    assert 0 <= a["success_probability"] <= 1
    assert fm.monte_carlo(0, 0, 0.05, 0.1, 1)["p50"][0] == 0


def test_hhi_and_diversification():
    assert fm.hhi([0.5, 0.5]) == 0.5
    assert fm.diversification_score([], []) == 0.0
    single = fm.diversification_score([1.0], [1.0])
    broad = fm.diversification_score([0.1] * 10, [0.6, 0.3, 0.1, 0.0, 0.0][:4] + [0.0])
    assert single < 10 and broad > 85


@pytest.mark.parametrize(
    "label,expected",
    [("low", "conservative"), ("HIGH", "aggressive"), ("moderate", "moderate"), (None, "moderate"), ("weird", "moderate")],
)
def test_normalize_risk(label, expected):
    assert fm.normalize_risk(label) == expected


def test_suggested_allocation_sums_and_horizon_shift():
    for risk in ("conservative", "moderate", "aggressive"):
        for years in (1, 4, 10, 30):
            alloc = fm.suggested_allocation(risk, years)
            assert sum(alloc.values()) == 100 and min(alloc.values()) >= 0
    assert fm.suggested_allocation("aggressive", 2)["stocks"] < fm.suggested_allocation("aggressive", 25)["stocks"]
    assert fm.suggested_allocation("aggressive", 4)["stocks"] == 40


def test_plan_goal_structure_and_risk_ordering():
    cons = plan_goal(50_000, 8, 5000, 300, "conservative", simulations=300)
    aggr = plan_goal(50_000, 8, 5000, 300, "aggressive", simulations=300)
    assert aggr["expected_return"] > cons["expected_return"]
    assert aggr["required_monthly"]["expected"] < cons["required_monthly"]["expected"]
    assert set(cons["scenarios"]) == {"cautious", "expected", "optimistic"}
    assert cons["total_contributed"] == 5000 + 300 * 96


def test_plan_goal_on_track_vs_off_track():
    assert plan_goal(10_000, 10, 20_000, 0, "moderate", simulations=200)["on_track"]
    off = plan_goal(1_000_000, 5, 0, 100, "moderate", simulations=200)
    assert not off["on_track"] and off["shortfall"] > 0 and off["status"].startswith("Off")


def test_plan_goal_validation():
    with pytest.raises(ValueError):
        plan_goal(0, 5)
    with pytest.raises(ValueError):
        plan_goal(1000, 0)


def test_status_labels():
    assert status_label(0.9) == "On track"
    assert "little margin" in status_label(0.6)
    assert status_label(0.1).startswith("Off")
