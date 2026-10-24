"""Market Analysis agent: live quotes, trend indicators and a market snapshot."""
from __future__ import annotations

from src.core.base_agent import BaseAgent
from src.core.models import AgentContext, AgentResponse
from src.data.market_data import MarketDataService
from src.utils.formatting import money, pct
from src.utils.parsing import extract_tickers


class MarketAnalysisAgent(BaseAgent):
    name = "market_analysis"
    description = "market analysis"

    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        if self.market is None:
            raise RuntimeError("market data service not configured")
        tickers = extract_tickers(query) or ctx.last_tickers
        if tickers:
            return self._tickers(query, tickers[:4], ctx)
        return self._overview(query, ctx)

    def _tickers(self, query: str, tickers: list[str], ctx: AgentContext) -> AgentResponse:
        sections, data = [], {}
        for sym in tickers:
            try:
                q = self.market.get_quote(sym)
                trend = self.market.analyze_trend(sym)
            except Exception as exc:
                sections.append(f"**{sym}** - no data available ({exc}).")
                continue
            arrow = "up" if q.change >= 0 else "down"
            lines = [
                f"**{sym}: {money(q.price, 2)}** ({arrow} {pct(abs(q.change_pct), 2)} vs previous close)",
                f"- 1-month return: {pct(trend.get('return_1m_pct', 0))}; 6-month: {pct(trend.get('return_period_pct', 0))}",
                f"- Trend: **{trend['trend']}**"
                + (f" (20-day avg {money(trend['sma20'], 2)}" if trend.get("sma20") else "")
                + (f", 50-day avg {money(trend['sma50'], 2)})" if trend.get("sma50") else (")" if trend.get("sma20") else "")),
                f"- Volatility (annualised): {pct(trend.get('volatility_annual_pct', 0))}; worst drawdown in period: {pct(trend.get('max_drawdown_pct', 0))}",
                f"- *{MarketDataService.freshness(q)}*",
            ]
            sections.append("\n".join(lines))
            data[sym] = {"quote": q.to_dict(), "trend": trend}
        ctx.shared["tickers"] = list(data)
        ctx.shared["market_data"] = data
        draft = "\n\n".join(sections) + (
            "\n\nTrend indicators describe past prices; they don't predict future moves."
        )
        return AgentResponse(agent=self.name, content=self.polish(draft, query), data={"tickers": list(data), "details": data})

    def _overview(self, query: str, ctx: AgentContext) -> AgentResponse:
        rows = self.market.market_overview()
        if not rows:
            raise RuntimeError("no market data available")
        lines = ["**Market snapshot**", "", "| Index / ETF | Price | Day change |", "|---|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['name']} ({r['symbol']}) | {money(r['price'], 2)} | {pct(r['change_pct'], 2)} |")
        up = [r for r in rows if r["change_pct"] >= 0]
        best = max(rows, key=lambda r: r["change_pct"])
        worst = min(rows, key=lambda r: r["change_pct"])
        mood = "broadly positive" if len(up) >= 0.7 * len(rows) else "broadly negative" if len(up) <= 0.3 * len(rows) else "mixed"
        lines += [
            "",
            f"{len(up)} of {len(rows)} tracked markets are up - the mood is **{mood}**. "
            f"Strongest: {best['name']} ({pct(best['change_pct'], 2)}); weakest: {worst['name']} ({pct(worst['change_pct'], 2)}).",
            "Daily moves are normal noise for long-term investors.",
            f"*{MarketDataService.freshness(self.market.get_quote(rows[0]['symbol']))}*",
        ]
        ctx.shared["market_overview"] = rows
        return AgentResponse(agent=self.name, content=self.polish("\n".join(lines), query), data={"overview": rows})
