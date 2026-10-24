from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.core.goal_planner import plan_goal  # noqa: E402
from src.web_app import charts  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "web_app" / "app.py")


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_app_renders_all_tabs_without_errors(app):
    assert not app.exception
    assert len(app.tabs) == 5
    labels = {m.label for m in app.metric}
    assert {"Expected value", "Chance of reaching goal", "Trend"} <= labels


def test_chat_flow_and_portfolio_sync(app):
    app.chat_input[0].set_value("I own 10 VTI and 5 BND. Analyze my portfolio").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert {h["symbol"] for h in app.session_state["portfolio"]} == {"VTI", "BND"}


def test_example_prompt_button_runs(app):
    app.button(key="ex0").click().run()
    assert not app.exception and len(app.chat_message) == 2


def test_new_conversation_resets(app):
    app.chat_input[0].set_value("What is an ETF?").run()
    next(b for b in app.sidebar.button if b.label == "New conversation").click().run()
    assert len(app.chat_message) == 0


def test_charts_build():
    plan = plan_goal(50000, 8, 5000, 300, simulations=200)
    assert len(charts.goal_fan(plan).data) >= 3
    assert charts.allocation_donut({"stock": 0.6, "bond": 0.4}, "t").data[0].hole == 0.55
    assert charts.weight_bars(["A", "B"], [0.7, 0.3], "t").data[0].y == (70.0, 30.0)
    assert len(charts.price_chart([f"2025-01-{d:02d}" for d in range(1, 29)], [float(d) for d in range(1, 29)], "X").data) == 2
    assert len(charts.day_change_bars([{"name": "A", "change_pct": 1.0}, {"name": "B", "change_pct": -1.0}]).data) == 1
