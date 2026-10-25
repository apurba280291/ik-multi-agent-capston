"""Evaluate retrieval quality on evals/retrieval_eval.json.

    python scripts/eval_retrieval.py                          # configured retriever (config.yaml)
    python scripts/eval_retrieval.py --misses                 # also list queries that failed
    python scripts/eval_retrieval.py --compare                # tfidf vs neural (if installed) x flat vs hnsw
    python scripts/eval_retrieval.py --embedder openai --index hnsw --chunking fixed -k 5
    python scripts/eval_retrieval.py --min-hit 0.8            # exit 1 if hit@k is below 0.8 (for CI)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402
from src.rag.evaluation import evaluate_retrieval, load_queries  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402


def build(embedder: str | None, index: str | None, chunking: str | None) -> Retriever:
    settings = load_settings()
    for key, value in (("rag.embedder", embedder), ("rag.index_type", index), ("rag.chunk_strategy", chunking)):
        if value:
            settings.set(key, value)
    return Retriever.from_settings(settings)


def run(label: str, retriever: Retriever, queries: list[dict], k: int, show_misses: bool) -> float:
    start = time.perf_counter()
    report = evaluate_retrieval(retriever, queries, k)
    ms = (time.perf_counter() - start) * 1000 / len(queries)
    oos = "n/a" if report.no_answer_rate is None else f"{report.no_answer_rate:.0%}"
    print(
        f"{label:34} hit@1 {report.hit_at_1:5.0%}  hit@{k} {report.hit_at_k:5.0%}  MRR {report.mrr:.3f}  "
        f"P@{k} {report.precision_at_k:.2f}  no-answer {oos:>4}  {ms:5.1f} ms/query  [{retriever.embedder.name}/{retriever.backend}]"
    )
    for kind, m in report.by_kind.items():
        print(f"    {kind:12} n={int(m['n']):<3} hit@1 {m['hit@1']:5.0%}  hit@{k} {m[f'hit@{k}']:5.0%}  MRR {m['mrr']:.3f}")
    if show_misses:
        for miss in report.misses:
            print(f"    MISS [{miss['kind']}] {miss['q']!r}\n         expected {miss['expected'] or 'no hits'}  got {miss['got']}")
    return report.hit_at_k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--embedder", choices=["tfidf", "sentence_transformers", "openai"])
    ap.add_argument("--index", choices=["flat", "hnsw"])
    ap.add_argument("--chunking", choices=["fixed", "recursive"])
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--min-hit", type=float, default=0.0, help="fail (exit 1) if hit@k is below this")
    args = ap.parse_args()
    queries = load_queries("retrieval_eval.json")
    print(f"{len(queries)} labelled queries, k={args.k}\n")

    if args.compare:
        configs = [("tfidf", "flat"), ("tfidf", "hnsw"), ("sentence_transformers", "flat"), ("openai", "flat")]
        for embedder, index in configs:
            r = build(embedder, index, args.chunking)
            if r.embedder.name != embedder:  # factory fell back (package/key missing)
                print(f"{embedder + ' + ' + index:34} skipped (unavailable here)")
                continue
            run(f"{embedder} + {index}", r, queries, args.k, args.misses)
        return
    hit = run("configured", build(args.embedder, args.index, args.chunking), queries, args.k, args.misses)
    if hit < args.min_hit:
        sys.exit(f"hit@{args.k} {hit:.0%} is below the required {args.min_hit:.0%}")


if __name__ == "__main__":
    main()
