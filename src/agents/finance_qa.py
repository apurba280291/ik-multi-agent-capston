"""Finance Q&A agent: RAG-grounded answers to general financial-education questions."""
from __future__ import annotations

from src.core.base_agent import BaseAgent
from src.core.models import AgentContext, AgentResponse
from src.utils.formatting import citation_block

_FOLLOW_UP = ("it", "that", "this", "those", "them", "why", "how", "more", "what about", "and")


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    description = "financial education Q&A"
    categories: list[str] | None = None  # subclasses (e.g. tax) can restrict retrieval

    def _expand_follow_up(self, query: str, ctx: AgentContext) -> str:
        """Short follow-ups ("why is that?") borrow the previous user question for retrieval."""
        words = query.lower().split()
        if len(words) <= 5 and ctx.history and any(w in _FOLLOW_UP for w in words):
            prev = next((m for role, m in reversed(ctx.history) if role == "user"), "")
            return f"{prev} {query}".strip()
        return query

    def retrieve(self, query: str, ctx: AgentContext):
        if self.retriever is None:
            return []
        category = ctx.shared.get("category_hint") or self.categories
        return self.retriever.search(self._expand_follow_up(query, ctx), category=category)

    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        hits = self.retrieve(query, ctx)
        if not hits:
            return AgentResponse(
                agent=self.name,
                content=(
                    "I couldn't find a verified article in my knowledge base that answers that. "
                    "Try asking about a specific topic such as ETFs, diversification, bonds, compound "
                    "interest, retirement accounts or taxes, or check Investor.gov for authoritative guidance."
                ),
                data={"hits": 0},
            )
        top = hits[0]
        if top.category == "glossary":
            definition = top.text.split(": ", 1)[-1]
            parts = [f"**Quick definition - {top.title}** [1]\n{definition}"]
        else:
            parts = [f"**{top.title}** [1]\n{top.text}"]
        extra = [h for h in hits[1:3] if h.score >= max(0.5 * top.score, self.retriever.min_score)]
        for i, h in enumerate(extra, 2):
            parts.append(f"**Also relevant - {h.title}** [{i}]\n{_first_sentences(h.text, 2)}")
        sources = [h.to_source() for h in [top, *extra]]
        draft = "\n\n".join(parts) + "\n\n" + citation_block(sources)
        content = self.polish(draft, query)
        if self.llm.available and "Sources" not in content:
            content += "\n\n" + citation_block(sources)
        return AgentResponse(
            agent=self.name,
            content=content,
            sources=sources,
            data={"hits": len(hits), "top_score": top.score},
        )


def _first_sentences(text: str, n: int) -> str:
    sentences = text.replace("\n", " ").split(". ")
    out = ". ".join(sentences[:n]).strip()
    return out if out.endswith(".") else out + "."
