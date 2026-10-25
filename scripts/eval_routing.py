"""Evaluate agent routing (keyword rules, offline) on evals/routing_eval.json.

    python scripts/eval_routing.py            # accuracy summary
    python scripts/eval_routing.py --misses   # also list wrongly routed queries
    python scripts/eval_routing.py --min-exact 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.evaluation import evaluate_routing, load_queries  # noqa: E402
from src.workflow.router import Router  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--min-exact", type=float, default=0.0, help="fail (exit 1) if exact-match accuracy is below this")
    args = ap.parse_args()

    router = Router()  # no LLM: measures the deterministic rules that act as the safety net
    report = evaluate_routing(lambda q: router.route(q).intents, load_queries("routing_eval.json"))
    print(
        f"{report.n} queries   exact match {report.exact_match:.0%}   "
        f"agent recall {report.agent_recall:.0%}   agent precision {report.agent_precision:.0%}"
    )
    if args.misses:
        for m in report.misses:
            print(f"  MISS {m['q']!r}\n       expected {m['expected']}  got {m['got']}")
    if report.exact_match < args.min_exact:
        sys.exit(f"exact match {report.exact_match:.0%} is below the required {args.min_exact:.0%}")


if __name__ == "__main__":
    main()
