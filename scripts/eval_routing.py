"""Evaluate agent routing on evals/routing_eval.json.

    python scripts/eval_routing.py                 # keyword rules only (offline, deterministic)
    python scripts/eval_routing.py --misses        # also list wrongly routed queries
    python scripts/eval_routing.py --min-exact 0.8
    python scripts/eval_routing.py --orchestrator  # rules vs the LLM orchestrator (needs an LLM API key; costs a few calls)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402
from src.core.llm import create_llm  # noqa: E402
from src.rag.evaluation import evaluate_routing, load_queries  # noqa: E402
from src.workflow.orchestrator import AGENT_CATALOG, Orchestrator  # noqa: E402
from src.workflow.router import Router  # noqa: E402


def show(label: str, report, misses: bool) -> None:
    print(
        f"{label:14} {report.n} queries   exact match {report.exact_match:.0%}   "
        f"agent recall {report.agent_recall:.0%}   agent precision {report.agent_precision:.0%}"
    )
    if misses:
        for m in report.misses:
            print(f"  MISS {m['q']!r}\n       expected {m['expected']}  got {m['got']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--min-exact", type=float, default=0.0, help="fail (exit 1) if exact-match accuracy is below this")
    ap.add_argument("--orchestrator", action="store_true", help="also evaluate the LLM orchestrator's plans (needs an API key)")
    args = ap.parse_args()

    queries = load_queries("routing_eval.json")
    router = Router()  # no LLM: the deterministic rules that act as the orchestrator's safety net
    rules = evaluate_routing(lambda q: router.route(q).intents, queries)
    show("rules", rules, args.misses)
    best = rules.exact_match

    if args.orchestrator:
        settings = load_settings()
        llm = create_llm(settings)
        if not llm.available:
            sys.exit("No LLM configured: set ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY (and install its SDK).")
        orchestrator = Orchestrator(llm, list(AGENT_CATALOG), max_agents=int(settings.get("orchestrator.max_agents", 3)))
        fell_back = 0

        def plan(q: str) -> list[str]:
            nonlocal fell_back
            decision = router.route(q)
            if not decision.intents:
                return decision.intents
            result = orchestrator.plan(q, [], {}, decision.intents)
            fell_back += result.source != "llm"
            return result.agents

        llm_report = evaluate_routing(plan, queries)
        show(f"orchestrator", llm_report, args.misses)
        print(f"  LLM plans used: {llm_report.n - fell_back}/{llm_report.n} (the rest fell back to the rules); model: {llm.name}")
        best = llm_report.exact_match
    if best < args.min_exact:
        sys.exit(f"exact match {best:.0%} is below the required {args.min_exact:.0%}")


if __name__ == "__main__":
    main()
