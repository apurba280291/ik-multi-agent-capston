"""Performance benchmarks (offline LLM + mock market data, so numbers reflect our own code).

Usage: python scripts/benchmark.py
"""
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("FINANCE_LLM_PROVIDER", "offline")
os.environ.setdefault("FINANCE_MARKET_PROVIDER", "mock")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

from src.core.config import load_settings  # noqa: E402
from src.core.goal_planner import plan_goal  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402
from src.workflow.assistant import FinanceAssistant  # noqa: E402

QUERIES = [
    "What is an ETF?",
    "I own 10 VTI, 20 AAPL and $2,000 in BND. Analyze my portfolio",
    "How is NVDA doing?",
    "I want $50,000 in 8 years, have $5,000, save $300 a month",
    "Roth vs traditional IRA",
    "latest news on tesla",
]


def timed(fn, n=1):
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    return statistics.median(samples), max(samples)


def main():
    settings = load_settings()
    t = time.perf_counter()
    retriever = Retriever.from_settings(settings)
    print(f"Knowledge base build ({retriever.document_count} docs, {retriever.backend}): {(time.perf_counter() - t) * 1000:.0f} ms")
    med, worst = timed(lambda: retriever.search("how does diversification reduce risk"), 200)
    print(f"Retrieval latency: median {med:.2f} ms, worst {worst:.2f} ms (200 queries)")
    med, worst = timed(lambda: plan_goal(50_000, 8, 5000, 300, "moderate"), 20)
    print(f"Goal plan + 2,000-path Monte Carlo: median {med:.1f} ms, worst {worst:.1f} ms")

    bot = FinanceAssistant(settings=settings, retriever=retriever)
    print("End-to-end chat turn (router -> agent -> compose), median / worst over 5 runs:")
    for i, q in enumerate(QUERIES):
        med, worst = timed(lambda: bot.chat(q, f"bench-{i}-{time.time()}"), 5)
        print(f"  {med:7.1f} ms / {worst:7.1f} ms  {q[:60]}")


if __name__ == "__main__":
    main()
