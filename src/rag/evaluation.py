"""Retrieval and routing evaluation: metrics over labelled query sets (see ``evals/*.json``).

Retrieval metrics (queries that have at least one relevant article):
  hit@k      - share of queries with a relevant article in the top k
  MRR        - mean of 1 / rank of the first relevant article (0 if absent from the top k)
  precision@k- share of the top-k results that are relevant, averaged over queries
Out-of-scope queries (no relevant article exists) are scored separately:
  no-answer rate - share that correctly return no hits (the rest are false positives)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rag.retriever import Retriever

EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"


def load_queries(name: str) -> list[dict[str, Any]]:
    return json.loads((EVALS_DIR / name).read_text(encoding="utf-8"))["queries"]


@dataclass
class RetrievalReport:
    k: int
    counts: dict[str, int] = field(default_factory=dict)  # per kind
    hit_at_1: float = 0.0
    hit_at_k: float = 0.0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    no_answer_rate: float | None = None
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    misses: list[dict[str, Any]] = field(default_factory=list)


def _first_relevant_rank(titles: list[str], expected: list[str]) -> int | None:
    wanted = {t.lower() for t in expected}
    for rank, title in enumerate(titles, start=1):
        if title.lower() in wanted:
            return rank
    return None


def evaluate_retrieval(retriever: Retriever, queries: list[dict[str, Any]], k: int = 3) -> RetrievalReport:
    report = RetrievalReport(k=k)
    answerable: list[tuple[str, int | None, float]] = []  # (kind, first_rank, precision)
    out_of_scope_ok: list[bool] = []
    for item in queries:
        hits = retriever.search(item["q"], k=k)
        titles = [h.title for h in hits]
        report.counts[item["kind"]] = report.counts.get(item["kind"], 0) + 1
        if not item["expected"]:
            ok = not hits
            out_of_scope_ok.append(ok)
            if not ok:
                report.misses.append({"q": item["q"], "kind": item["kind"], "expected": [], "got": titles})
            continue
        rank = _first_relevant_rank(titles, item["expected"])
        relevant = sum(1 for t in titles if t.lower() in {e.lower() for e in item["expected"]})
        answerable.append((item["kind"], rank, relevant / k))
        if rank is None:
            report.misses.append({"q": item["q"], "kind": item["kind"], "expected": item["expected"], "got": titles})

    def summarise(rows: list[tuple[str, int | None, float]]) -> dict[str, float]:
        n = len(rows) or 1
        return {
            "n": len(rows),
            "hit@1": sum(1 for _, r, _ in rows if r == 1) / n,
            f"hit@{k}": sum(1 for _, r, _ in rows if r is not None) / n,
            "mrr": sum(1 / r for _, r, _ in rows if r) / n,
            f"precision@{k}": sum(p for _, _, p in rows) / n,
        }

    overall = summarise(answerable)
    report.hit_at_1, report.hit_at_k = overall["hit@1"], overall[f"hit@{k}"]
    report.mrr, report.precision_at_k = overall["mrr"], overall[f"precision@{k}"]
    for kind in sorted({kind for kind, _, _ in answerable}):
        report.by_kind[kind] = summarise([row for row in answerable if row[0] == kind])
    if out_of_scope_ok:
        report.no_answer_rate = sum(out_of_scope_ok) / len(out_of_scope_ok)
    return report


@dataclass
class ScoreCalibration:
    off_topic_top: list[float]   # best score returned for each out-of-scope query (ignoring the floor)
    relevant: list[float]        # score of the first correct hit for each answerable query that found one
    suggested_floor: float | None  # midpoint of the gap, or None if off-topic scores overlap relevant ones

    @property
    def separable(self) -> bool:
        return self.suggested_floor is not None

    def keep_all_floor(self, margin: float = 0.05) -> float | None:
        """Highest floor (a ``margin`` below the weakest correct hit) that still keeps every correct hit."""
        return round(max(0.0, min(self.relevant) - margin), 3) if self.relevant else None

    def off_topic_rejected(self, floor: float) -> int:
        return sum(1 for score in self.off_topic_top if score < floor)


def calibrate_floor(retriever: Retriever, queries: list[dict[str, Any]], k: int = 3) -> ScoreCalibration:
    """Where could ``min_score`` sit? Compare off-topic top scores with the scores of correct hits."""
    off_topic, relevant = [], []
    for item in queries:
        hits = retriever.search(item["q"], k=k, min_score=0.0)
        if not hits:
            continue
        if not item["expected"]:
            off_topic.append(hits[0].score)
            continue
        wanted = {e.lower() for e in item["expected"]}
        found = next((h.score for h in hits if h.title.lower() in wanted), None)
        if found is not None:
            relevant.append(found)
    suggested = None
    if off_topic and relevant and max(off_topic) < min(relevant):
        suggested = round((max(off_topic) + min(relevant)) / 2, 3)
    return ScoreCalibration(sorted(off_topic), sorted(relevant), suggested)


@dataclass
class RoutingReport:
    n: int
    exact_match: float
    agent_recall: float     # of the agents that should run, share that did
    agent_precision: float  # of the agents that ran, share that should have
    misses: list[dict[str, Any]] = field(default_factory=list)


def evaluate_routing(route, queries: list[dict[str, Any]]) -> RoutingReport:
    """``route(query) -> list[str]`` of agent names."""
    exact = tp = want = got_total = 0
    misses = []
    for item in queries:
        got, expected = set(route(item["q"])), set(item["expected"])
        exact += got == expected
        tp += len(got & expected)
        want += len(expected)
        got_total += len(got)
        if got != expected:
            misses.append({"q": item["q"], "expected": sorted(expected), "got": sorted(got)})
    n = len(queries)
    return RoutingReport(n, exact / n, tp / want if want else 0.0, tp / got_total if got_total else 0.0, misses)
