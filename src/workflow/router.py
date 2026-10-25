"""Query classification: decides which agent(s) handle a message.

Strategy: if an LLM is available it proposes intents as JSON, which are validated against the
known agent names; otherwise (or on any failure) weighted keyword rules decide. The rules are
always the safety net, so routing never fails.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.core.guardrails import is_advice_request
from src.core.llm import LLMClient, LLMError, OfflineLLM
from src.utils.logging_utils import get_logger
from src.utils.parsing import extract_tickers, parse_holdings

log = get_logger(__name__)

# Execution order matters: later agents can consume earlier agents' shared data.
AGENT_ORDER = ["market_analysis", "news_synthesizer", "portfolio_analysis", "goal_planning", "tax_education", "finance_qa"]
MAX_AGENTS = 3

RULES: dict[str, list[tuple[str, int]]] = {
    "tax_education": [
        (r"\b(tax(es|able|ed)?|401\s*\(?k\)?|403\s*\(?b\)?|ira|roth|hsa|529|capital gains?|deduct\w*|wash[- ]sale|rmd|irs|tax[- ]loss|bracket)\b", 3),
    ],
    "portfolio_analysis": [
        (r"\b(my|our)\s+(portfolio|holdings|investments|positions|allocation)\b", 4),
        (r"\b(analy[sz]e|review|check|evaluate|assess|rebalanc\w*)\b.*\b(portfolio|holdings|allocation)\b", 4),
        (r"\bi\s+(own|hold|have)\b.*\b(shares?|etf|stocks?)\b", 3),
        (r"\bportfolio\b", 1),
    ],
    "market_analysis": [
        (r"\b(price|quote|trading|trend|chart|moving average|volatility of|market (today|overview|snapshot|update)|how is the market|indexes|indices|s&p|nasdaq|dow|vix)\b", 3),
        (r"\b(how (is|are|did)|what'?s)\b.*\b(doing|performing|trading|up|down)\b", 2),
        (r"\b(stock market|markets?)\s+(today|now|right now)\b", 3),
    ],
    "goal_planning": [
        (r"\b(goal|plan(ning)? (for|to)|save (up )?for|saving for|how much (do i need|should i save)|reach \$?\d|target of|retire(ment)? (in|at|by)|down payment|nest egg|build wealth)\b", 3),
        (r"\$\s?\d[\d,]*\s*(k|m|million)?\b.*\b(in|within|by)\s+\d+\s*(years?|yrs?)", 3),
        (r"\b(monthly|per month|a month)\b.*\b(invest|save|contribut\w*)", 2),
        (r"\b(invest|save|contribut\w*)\b.*\b(monthly|per month|a month)", 2),
    ],
    "news_synthesizer": [
        (r"\b(news|headlines?|latest on|what happened|why (is|did|are)\b.*\b(down|up|falling|rising|drop\w*|surg\w*)|summari[sz]e.*(news|market))\b", 3),
    ],
    "finance_qa": [
        (r"\b(what (is|are|does)|explain|define|meaning of|how (does|do|is)|difference between|why (do|does|is)|tell me about|teach me|learn|eli5|beginner)\b", 2),
    ],
}
FINANCE_WORDS = re.compile(
    r"\b(invest\w*|stock\w*|bonds?|funds?|etfs?|money|financ\w*|sav(e|ing|ings)|retire\w*|tax\w*|market\w*|portfolio|loans?|debt|budget\w*|"
    r"inflation|interest|dividends?|crypto\w*|bitcoin|mortgage|credit|income|wealth|asset\w*|broker\w*|risk|diversif\w*|compound|"
    r"401k|ira|roth|hsa|529|equity|shares?|yield|price|rates?|economy|fed)\b|401\s?\(\s?k\s?\)|403\s?\(\s?b\s?\)",
    re.I,
)
FOLLOW_UP = re.compile(r"\b(it|that|this|them|those|these|why|how come|more|and|also|then|what about|how about|really|explain)\b", re.I)
GREETING =re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|help|what can you do|who are you)\b[\s!.?]*$", re.I)


@dataclass
class RouteDecision:
    intents: list[str] = field(default_factory=list)
    method: str = "rules"
    scores: dict[str, int] = field(default_factory=dict)
    greeting: bool = False
    off_topic: bool = False
    advice_request: bool = False


def score_query(query: str) -> dict[str, int]:
    q = query.lower()
    scores = {agent: sum(w for pat, w in rules if re.search(pat, q)) for agent, rules in RULES.items()}
    # Structured hints: holdings listed in the message, tickers with a price-like question.
    if len(parse_holdings(query)) >= 1 and re.search(r"\b(own|hold|have|portfolio|shares?)\b", q):
        scores["portfolio_analysis"] += 3
    if extract_tickers(query) and re.search(r"\b(price|trading|worth|cost|quote|doing|up|down|trend)\b", q):
        scores["market_analysis"] += 2
    return scores


def rule_route(query: str) -> RouteDecision:
    scores = score_query(query)
    chosen = [a for a, s in scores.items() if s >= 3 and a != "finance_qa"]
    # Specific agents outrank the generic educational one; keep Q&A only for explicit "explain" asks
    # alongside another intent, or when nothing else matched.
    if scores["finance_qa"] >= 2 and chosen and re.search(r"\b(explain|what is|what are|define)\b", query.lower()):
        chosen.append("finance_qa")
    if not chosen:
        chosen = ["finance_qa"]
    chosen = sorted(chosen, key=lambda a: -scores[a])[:MAX_AGENTS]
    return RouteDecision(sorted(chosen, key=AGENT_ORDER.index), "rules", scores)


class Router:
    def __init__(self, llm: LLMClient | None = None, confident_score: int | None = 5):
        self.llm = llm or OfflineLLM()
        # Token saver: when every agent the rules picked scores at least this, skip the LLM routing call.
        # ``None`` disables the shortcut (always ask the LLM when one is available).
        self.confident_score = confident_score

    def route(self, query: str, history: list[tuple[str, str]] | None = None) -> RouteDecision:
        text = query.strip()
        if not text:
            return RouteDecision(greeting=True)
        if GREETING.match(text):
            return RouteDecision(greeting=True)
        advice = is_advice_request(text)  # guardrail: see src/core/guardrails.py
        decision = rule_route(text)
        if not FINANCE_WORDS.search(text) and not extract_tickers(text) and not parse_holdings(text):
            # Only a short message with a follow-up cue ("why is that?", "and for tesla?") inherits context.
            if not (history and FOLLOW_UP.search(text) and len(text.split()) <= 7):
                return RouteDecision(off_topic=True, advice_request=advice)
            decision.method = "context"
            decision.intents = self._follow_up(history) or ["finance_qa"]
        elif self.llm.available and not self._rules_confident(decision):
            llm_intents = self._llm_route(text, history or [])
            if llm_intents:
                decision.intents, decision.method = sorted(llm_intents, key=AGENT_ORDER.index), "llm"
        decision.advice_request = advice
        if advice and extract_tickers(text) and "market_analysis" not in decision.intents:
            # "Should I buy X?" can't be answered with advice, but current data on X is useful context.
            decision.intents = sorted([*decision.intents[: MAX_AGENTS - 1], "market_analysis"], key=AGENT_ORDER.index)
        return decision

    def _rules_confident(self, decision: RouteDecision) -> bool:
        if self.confident_score is None or not decision.intents:
            return False
        return all(decision.scores.get(a, 0) >= self.confident_score for a in decision.intents)

    @staticmethod
    def _follow_up(history: list[tuple[str, str]]) -> list[str]:
        """Re-use the previous turn's routing for terse follow-ups."""
        prev = next((m for role, m in reversed(history) if role == "user"), "")
        return rule_route(prev).intents if prev else []

    def _llm_route(self, query: str, history: list[tuple[str, str]]) -> list[str]:
        system = (
            "Classify the user's finance message. Reply with ONLY a JSON list (max 3) chosen from: "
            f"{AGENT_ORDER}. finance_qa=general education, portfolio_analysis=analyse holdings, "
            "market_analysis=prices/trends, goal_planning=saving goals, news_synthesizer=headlines, tax_education=taxes/accounts."
        )
        recent = "\n".join(f"{r}: {m[:200]}" for r, m in history[-4:])
        try:
            raw = self.llm.generate(system, f"Recent conversation:\n{recent}\n\nMessage: {query}")
            match = re.search(r"\[.*?\]", raw, re.S)
            parsed = json.loads(match.group(0)) if match else []
        except (LLMError, json.JSONDecodeError, TypeError) as exc:
            log.warning("LLM routing failed (%s); using rules", exc)
            return []
        valid = [a for a in dict.fromkeys(parsed) if a in AGENT_ORDER]
        return valid[:MAX_AGENTS]
