"""LangGraph workflow: guardrail/understand -> route -> agent(s) -> compose.

    START -> understand -> route --(conditional)--> <agent> --(conditional)--> <agent> ... -> compose -> END

* ``understand`` resets per-turn fields, updates user profile and remembered tickers.
* ``route`` picks one to three agents; the first conditional edge sends control to the first of them.
* each agent node pops itself from ``queue`` and may leave data in ``shared`` for later agents.
* ``compose`` merges responses, adds citations, notes and the disclaimer, and records history.
Conversation memory persists per ``thread_id`` through LangGraph's checkpointer.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.core.base_agent import BaseAgent
from src.core.guardrails import BLOCKED_REPLY, is_injection, redact_pii
from src.core.models import AgentContext, AgentResponse, Source
from src.utils.formatting import citation_block, with_disclaimer
from src.utils.logging_utils import get_logger
from src.utils.parsing import extract_tickers, parse_risk
from src.workflow.router import Router

import re

log = get_logger(__name__)

SECTION_TITLES = {
    "market_analysis": "Market data",
    "news_synthesizer": "News",
    "portfolio_analysis": "Your portfolio",
    "goal_planning": "Goal plan",
    "tax_education": "Tax education",
    "finance_qa": "Explanation",
}
INTRO = (
    "Hi, I'm **Finnie**, your financial-education assistant. I can:\n"
    "- explain concepts (*\"What is an ETF?\"*)\n"
    "- analyse a portfolio (*\"I own 10 VTI, 5 AAPL and $2,000 in BND\"*)\n"
    "- look up markets (*\"How is NVDA doing?\"*)\n"
    "- plan goals (*\"I want $50,000 in 8 years, I have $5,000 and can save $300/month\"*)\n"
    "- summarise news and explain tax accounts (*\"Roth vs traditional IRA\"*)"
)
OFF_TOPIC = "I'm focused on personal finance and investing education, so I can't help with that one. Try asking about investing, markets, saving goals or taxes!"
ADVICE_NOTE = (
    "**A note on advice:** I can't tell you what to buy or sell, or promise returns - I'm an educational tool, not a licensed "
    "advisor. Above are the facts and concepts that can help you decide, ideally with a qualified professional."
)
FALLBACK = "Sorry - something went wrong on my side while handling that. Please try again in a moment."


class FinanceState(TypedDict, total=False):
    query: str
    profile: dict[str, Any]
    portfolio: list[dict[str, Any]]
    history: list[tuple[str, str]]
    last_tickers: list[str]
    intents: list[str]
    queue: list[str]
    responses: list[dict[str, Any]]
    shared: dict[str, Any]
    flags: dict[str, bool]
    route_method: str
    trace: list[str]
    final_answer: str
    sources: list[dict[str, Any]]
    agents_used: list[str]


def _update_profile(profile: dict[str, Any], query: str) -> dict[str, Any]:
    profile = dict(profile or {})
    risk = parse_risk(query)
    if risk and re.search(r"\b(i am|i'm|my risk|im|as an?)\b", query.lower()):
        profile["risk_tolerance"] = risk
    age = re.search(r"\b(?:i am|i'm|im|age)\s+(\d{2})\b(?!\s*%)", query.lower())
    if age and 16 <= int(age.group(1)) <= 90:
        profile["age"] = int(age.group(1))
    return profile


def build_graph(agents: dict[str, BaseAgent], router: Router, max_history: int = 20, checkpointer=None):
    def understand(state: FinanceState) -> dict[str, Any]:
        # Input guardrails: mask personal data before it reaches agents/LLM/history, and flag prompt injection.
        query, redacted = redact_pii(state.get("query", ""))
        blocked = is_injection(query)
        tickers = extract_tickers(query)
        last = tickers + [t for t in state.get("last_tickers", []) if t not in tickers] if tickers else state.get("last_tickers", [])
        trace = [f"understand: tickers={tickers or '-'}"]
        if redacted:
            trace.append("guardrail: PII redacted")
        if blocked:
            trace.append("guardrail: prompt injection blocked")
        return {
            "query": query,
            "profile": _update_profile(state.get("profile", {}), query),
            "last_tickers": last[:5],
            "responses": [], "queue": [], "intents": [], "shared": {}, "flags": {"blocked": blocked},
            "final_answer": "", "sources": [], "agents_used": [],
            "trace": trace,
        }

    def route(state: FinanceState) -> dict[str, Any]:
        if state.get("flags", {}).get("blocked"):
            return {}  # nothing to route; compose sends the refusal
        d = router.route(state["query"], state.get("history", []))
        flags = {"greeting": d.greeting, "off_topic": d.off_topic, "advice": d.advice_request}
        return {
            "intents": d.intents, "queue": list(d.intents), "flags": flags, "route_method": d.method,
            "trace": state.get("trace", []) + [f"route[{d.method}]: {d.intents or ('greeting' if d.greeting else 'off-topic' if d.off_topic else '-')}"],
        }

    def next_step(state: FinanceState) -> str:
        queue = state.get("queue", [])
        return queue[0] if queue else "compose"

    def make_agent_node(name: str):
        agent = agents[name]

        def node(state: FinanceState) -> dict[str, Any]:
            ctx = AgentContext(
                profile=state.get("profile", {}), portfolio=state.get("portfolio", []),
                history=state.get("history", []), last_tickers=state.get("last_tickers", []),
                shared=dict(state.get("shared", {})),
            )
            resp = agent.handle(state["query"], ctx)
            return {
                "responses": state.get("responses", []) + [resp.to_dict()],
                "queue": state.get("queue", [])[1:],
                "shared": ctx.shared,
                "trace": state.get("trace", []) + [f"{name}: {'ok' if resp.ok else 'FAILED ' + str(resp.error)}"],
            }

        return node

    def compose(state: FinanceState) -> dict[str, Any]:
        flags, responses = state.get("flags", {}), state.get("responses", [])
        update: dict[str, Any] = {}
        sources: list[Source] = []
        if flags.get("blocked"):
            answer = BLOCKED_REPLY
        elif flags.get("greeting"):
            answer = INTRO
        elif flags.get("off_topic"):
            answer = OFF_TOPIC
        elif not responses:
            answer = FALLBACK
        else:
            parts = []
            for r in responses:
                body = r["content"]
                for s in r.get("sources", []):
                    if all(s["title"] != x.title for x in sources):
                        sources.append(Source(**s))
                if len(responses) > 1:
                    body = f"### {SECTION_TITLES.get(r['agent'], r['agent'])}\n{body}"
                parts.append(body)
                holdings = r.get("data", {}).get("holdings")
                if holdings:
                    update["portfolio"] = holdings
            answer = "\n\n".join(parts)
            if len(responses) > 1 and sources:
                # Per-agent citation blocks use local numbering, so consolidate by dropping them.
                answer = re.sub(r"\n\n\*\*Sources\*\*\n(?:\[\d+\].*\n?)+", "\n", answer).rstrip()
                answer += "\n\n" + citation_block(sources)
            if flags.get("advice"):
                answer += "\n\n" + ADVICE_NOTE
        if not flags.get("greeting"):
            answer = with_disclaimer(answer)
        history = (state.get("history", []) + [("user", state["query"]), ("assistant", answer)])[-2 * max_history :]
        update.update(
            final_answer=answer, history=history, sources=[s.__dict__ for s in sources],
            agents_used=[r["agent"] for r in responses],
            trace=state.get("trace", []) + ["compose: done"],
        )
        return update

    g = StateGraph(FinanceState)
    g.add_node("understand", understand)
    g.add_node("route", route)
    g.add_node("compose", compose)
    targets = {n: n for n in agents} | {"compose": "compose"}
    for name in agents:
        g.add_node(name, make_agent_node(name))
        g.add_conditional_edges(name, next_step, targets)
    g.add_edge(START, "understand")
    g.add_edge("understand", "route")
    g.add_conditional_edges("route", next_step, targets)
    g.add_edge("compose", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
