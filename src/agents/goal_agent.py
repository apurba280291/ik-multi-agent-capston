"""Goal Planning agent: turns a savings goal into projections that respect risk tolerance."""
from __future__ import annotations

import re

from src.core.base_agent import BaseAgent
from src.core.goal_planner import plan_goal
from src.core.models import AgentContext, AgentResponse
from src.utils.formatting import money, pct
from src.utils.parsing import parse_money_mentions, parse_risk, parse_years

GOAL_TYPES = {
    "retirement": ("retire", "retirement"),
    "home down payment": ("house", "home", "down payment", "mortgage"),
    "education": ("college", "tuition", "school", "education"),
    "emergency fund": ("emergency",),
    "car": ("car", "vehicle"),
    "vacation": ("vacation", "trip", "travel", "wedding"),
}
_MONTHLY = re.compile(r"(/\s*(?:mo|month)|per month|a month|each month|monthly|every month)")
_CURRENT = re.compile(r"(have|saved|current|already|starting|balance|invested|got)\s*(?:about|around|of)?\s*$|(?:saved|invested)")

ASK = (
    "To build a plan I need a bit more: {missing}. For example: "
    "*\"I want $60,000 for a house in 8 years. I have $5,000 saved and can add $400 a month. I'm a moderate investor.\"*"
)


def parse_goal(query: str, ctx: AgentContext) -> dict:
    """Extract target/years/current/monthly/risk from free text, falling back to the user profile."""
    target = current = monthly = None
    candidates = []
    for m in parse_money_mentions(query):
        if _MONTHLY.search(m["after"]) or _MONTHLY.search(m["before"][-15:]):
            monthly = monthly if monthly is not None else m["value"]
        elif _CURRENT.search(m["before"]) or re.match(r"\s*(?:saved|invested|already)", m["after"]):
            current = current if current is not None else m["value"]
        else:
            candidates.append(m["value"])
    if candidates:
        target = max(candidates)
    q = query.lower()
    goal_type = next((g for g, words in GOAL_TYPES.items() if any(w in q for w in words)), "savings goal")
    years = parse_years(query)
    if years is None and ctx.profile.get("age") and goal_type == "retirement":
        years = max(1, int(ctx.profile.get("retirement_age", 65)) - int(ctx.profile["age"]))
    return {
        "target": target,
        "years": years,
        "current": current if current is not None else float(ctx.profile.get("current_savings", 0) or 0),
        "monthly": monthly if monthly is not None else float(ctx.profile.get("monthly_contribution", 0) or 0),
        "risk": parse_risk(query) or ctx.risk_tolerance,
        "goal_type": goal_type,
    }


class GoalPlanningAgent(BaseAgent):
    name = "goal_planning"
    description = "goal planning"

    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        g = parse_goal(query, ctx)
        missing = [label for key, label in (("target", "the target amount"), ("years", "your time horizon (years)")) if not g[key]]
        if missing:
            return AgentResponse(agent=self.name, content=ASK.format(missing=" and ".join(missing)), data={"needs_input": True, "parsed": g})

        plan = plan_goal(g["target"], g["years"], g["current"], g["monthly"], g["risk"])
        ctx.shared["goal_plan"] = plan
        # Inter-agent hand-off: reference the user's portfolio analysis if the portfolio agent ran first.
        port = ctx.shared.get("portfolio_analysis")
        draft = self._draft(g, plan, port)
        return AgentResponse(agent=self.name, content=self.polish(draft, query), data={"plan": plan, "parsed": g})

    @staticmethod
    def _draft(g: dict, p: dict, portfolio: dict | None) -> str:
        yrs = f"{p['years']:g}"
        a = p["allocation"]
        lines = [
            f"**Goal: {money(p['target'])} in {yrs} years ({g['goal_type']})** - {p['risk']} risk profile",
            "",
            f"- Starting from {money(p['current'])} and adding {money(p['monthly'])}/month, the **expected-case** value is "
            f"**{money(p['projected_value']['expected'])}** (assumes {pct(p['expected_return'] * 100)} average yearly return).",
            f"- Range of assumptions ({pct(p['scenarios']['cautious'] * 100)} to {pct(p['scenarios']['optimistic'] * 100)}): "
            f"{money(p['projected_value']['cautious'])} to {money(p['projected_value']['optimistic'])}.",
            f"- **Monte Carlo odds of reaching the target: {p['success_probability']:.0%}** "
            f"({p['monte_carlo']['simulations']:,} simulated market paths, volatility {pct(p['volatility'] * 100, 0)}). Status: **{p['status']}**.",
            "",
            "**Monthly saving needed to hit the target**",
            f"- Expected case: **{money(p['required_monthly']['expected'])}/month**",
            f"- Cautious case: {money(p['required_monthly']['cautious'])}/month; optimistic: {money(p['required_monthly']['optimistic'])}/month",
            "",
            f"**Illustrative mix for this horizon and risk level:** {a['stocks']}% stocks / {a['bonds']}% bonds / {a['cash']}% cash "
            "(typically via low-cost diversified funds).",
        ]
        if not p["on_track"] and p["monthly"] > 0:
            lines.append(
                f"\nTo close the {money(p['shortfall'])} expected-case gap you could save more, extend the timeline, "
                "or lower the target."
            )
        if p["years"] < 5 and p["risk"] != "conservative":
            lines.append("\nWith under 5 years to go, a market drop could hurt more than you can recover, so the mix above is shifted toward safety.")
        if portfolio and not portfolio.get("empty"):
            lines.append(
                f"\nYour current portfolio holds {portfolio['equity_weight']:.0%} stocks versus roughly {a['stocks']}% in the illustration above."
            )
        lines.append("\n*Projections use simple illustrative assumptions - actual returns will vary and can be negative.*")
        return "\n".join(lines)
