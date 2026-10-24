"""Portfolio Analysis agent: values holdings with live prices and reports key metrics."""
from __future__ import annotations

from src.core.base_agent import BaseAgent
from src.core.models import AgentContext, AgentResponse
from src.core.portfolio_analysis import analyze_portfolio
from src.utils.formatting import money, pct
from src.utils.parsing import parse_holdings

NO_PORTFOLIO = (
    "I don't have a portfolio to analyse yet. Tell me your holdings, for example "
    "*\"I own 10 shares of VTI, 5 shares of AAPL and $2,000 in BND\"*, or enter them in the **Portfolio** tab."
)


class PortfolioAnalysisAgent(BaseAgent):
    name = "portfolio_analysis"
    description = "portfolio analysis"

    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        from_query = parse_holdings(query)
        holdings = from_query or ctx.portfolio
        if not holdings:
            return AgentResponse(agent=self.name, content=NO_PORTFOLIO, data={"needs_input": True})

        symbols = [h["symbol"] for h in holdings if h.get("shares") is not None]
        quotes = self.market.get_quotes(symbols) if self.market else {}
        prices = {s: q.price for s, q in quotes.items()}
        analysis = analyze_portfolio(holdings, prices, ctx.risk_tolerance)
        if analysis["empty"]:
            return AgentResponse(
                agent=self.name,
                content="I couldn't price any of those holdings. Check the ticker symbols or give dollar values.",
                ok=False,
                data={"unpriced": analysis["unpriced"]},
            )

        ctx.shared["portfolio_analysis"] = analysis
        draft = self._draft(analysis, quotes)
        return AgentResponse(
            agent=self.name,
            content=self.polish(draft, query),
            data={"analysis": analysis, "holdings": holdings, "quotes": {s: q.to_dict() for s, q in quotes.items()}},
        )

    @staticmethod
    def _draft(a: dict, quotes: dict) -> str:
        lines = [f"**Portfolio value: {money(a['total_value'])}** across {len(a['positions'])} holdings", ""]
        lines += ["| Holding | Value | Weight | Asset class |", "|---|---:|---:|---|"]
        for p in a["positions"][:12]:
            lines.append(f"| {p['symbol']} | {money(p['value'])} | {pct(p['weight'] * 100)} | {p['asset_class'].replace('_', ' ')} |")
        alloc = ", ".join(f"{k.replace('_', ' ')} {pct(v * 100, 0)}" for k, v in a["allocation_by_class"].items())
        lines += [
            "",
            f"- **Allocation:** {alloc}",
            f"- **Diversification score:** {a['diversification_score']:.0f}/100",
            f"- **Risk rating:** {a['risk_label']} ({a['risk_score']}/10)",
            f"- **Stress test:** a severe bear market could cut this portfolio roughly {abs(a['stress_loss_pct']):.0f}% "
            f"(about {money(a['total_value'] * abs(a['stress_loss_pct']) / 100)}) - illustrative only",
            f"- **Weighted expense ratio:** {a['weighted_expense_ratio']:.2f}% (~{money(a['annual_fees'])}/year in fund fees)",
            "",
            "**Things to think about**",
        ]
        lines += [f"- {o}" for o in a["observations"]]
        if a["unpriced"]:
            lines.append(f"\n*Could not price: {', '.join(a['unpriced'])} (excluded from totals).*")
        mock = [s for s, q in quotes.items() if q.is_mock or q.is_stale]
        if mock:
            lines.append(f"\n*Prices for {', '.join(mock)} are sample/cached data, not live quotes.*")
        return "\n".join(lines)
