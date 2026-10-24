"""News Synthesizer agent: summarises headlines and links them to educational context."""
from __future__ import annotations

import re

from src.core.base_agent import BaseAgent
from src.core.models import AgentContext, AgentResponse, Source
from src.utils.formatting import citation_block
from src.utils.parsing import extract_tickers

POSITIVE = {"rise", "rises", "gain", "gains", "beat", "beats", "surge", "rally", "record", "strong", "growth", "up", "jumps", "boost", "boosts", "boosting", "lifts", "lifting", "advance", "advanced", "ease", "eases", "easing", "cools"}
NEGATIVE = {"fall", "falls", "drop", "drops", "miss", "misses", "slump", "plunge", "weak", "weaker", "cut", "cuts", "loss", "losses", "down", "worries", "concern", "concerns", "slip", "slipped", "decline", "declines", "lawsuit", "probe", "recession"}
THEMES = {
    "interest rates": (r"\b(rates?|fed|central bank|rate cut|borrowing)\b", "interest rates and the stock market"),
    "inflation": (r"\b(inflation|cpi|prices)\b", "inflation and purchasing power"),
    "earnings": (r"\b(earnings|profit|revenue|guidance)\b", "price-to-earnings ratio earnings"),
    "bond yields": (r"\b(yields?|treasur(?:y|ies)|bonds?)\b", "bond prices and interest rates"),
    "volatility": (r"\b(volatil\w*|selloff|sell-off|turmoil|vix)\b", "volatility standard deviation"),
}


def sentiment(text: str) -> int:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & POSITIVE) - len(words & NEGATIVE)


class NewsSynthesizerAgent(BaseAgent):
    name = "news_synthesizer"
    description = "news summary"

    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        if self.market is None:
            raise RuntimeError("market data service not configured")
        tickers = extract_tickers(query) or ctx.last_tickers or ["SPY"]
        items = []
        for sym in tickers[:2]:
            items += self.market.get_news(sym, limit=5)
        seen, unique = set(), []
        for it in items:
            if it.title not in seen:
                seen.add(it.title)
                unique.append(it)
        unique = unique[:6]
        if not unique:
            return AgentResponse(agent=self.name, content="I couldn't retrieve any recent headlines right now.", ok=False)

        scores = [sentiment(f"{i.title} {i.summary}") for i in unique]
        pos, neg = sum(s > 0 for s in scores), sum(s < 0 for s in scores)
        tone = "mostly positive" if pos > neg else "mostly negative" if neg > pos else "mixed"
        is_sample = any(i.is_sample for i in unique)

        lines = [f"**News digest for {', '.join(tickers[:2])}** - overall tone: **{tone}** ({pos} positive / {neg} negative / {len(unique) - pos - neg} neutral)", ""]
        for it in unique:
            who = f" - {it.publisher}" if it.publisher else ""
            lines.append(f"- {it.title}{who}" + (f" ({it.url})" if it.url else ""))
        if is_sample:
            lines.append("\n*Live news unavailable - these are sample headlines for demonstration.*")

        blob = " ".join(f"{i.title} {i.summary}" for i in unique).lower()
        sources: list[Source] = []
        context_lines = []
        if self.retriever:
            for theme, (pattern, topic) in THEMES.items():
                if re.search(pattern, blob) and len(sources) < 2:
                    hit = next(iter(self.retriever.search(topic, k=1)), None)
                    if hit and hit.title not in {s.title for s in sources}:
                        sources.append(hit.to_source())
                        context_lines.append(f"- **{theme.title()}** [{len(sources)}]: {hit.text.split('. ')[0].rstrip('.')}.")
        if context_lines:
            lines += ["", "**Why it matters (background)**", *context_lines]
        lines.append("\nHeadlines are noise for most long-term investors; they are context, not a signal to trade.")
        draft = "\n".join(lines) + ("\n\n" + citation_block(sources) if sources else "")
        content = self.polish(draft, query)
        if sources and self.llm.available and "Sources" not in content:
            content += "\n\n" + citation_block(sources)
        return AgentResponse(
            agent=self.name, content=content, sources=sources,
            data={"tone": tone, "headlines": [i.title for i in unique], "tickers": tickers[:2]},
        )
