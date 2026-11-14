"""Evaluate retrieval quality on evals/retrieval_eval.json.

    python scripts/eval_retrieval.py                          # configured retriever (config.yaml)
    python scripts/eval_retrieval.py --misses                 # also list queries that failed
    python scripts/eval_retrieval.py --compare                # tfidf vs neural (if installed) x flat vs hnsw
    python scripts/eval_retrieval.py --embedder openai --index hnsw --chunking fixed -k 5
    python scripts/eval_retrieval.py --min-hit 0.8            # exit 1 if hit@k is below 0.8 (for CI)
    python scripts/eval_retrieval.py --set pdf_eval.json      # questions answered by the indexed PDF (pages)
    python scripts/eval_retrieval.py --no-pdf                 # index only the curated articles (ignore PDFs)
    python scripts/eval_retrieval.py --embedder sentence_transformers --calibrate   # suggest a min_score floor
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402
from src.rag.evaluation import calibrate_floor, evaluate_retrieval, load_queries  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402


def build(embedder: str | None, index: str | None, chunking: str | None, no_pdf: bool = False,
          lexical_weight: float | None = None, min_score: float | None = None) -> Retriever:
    settings = load_settings()
    if lexical_weight is not None:
        settings.set("rag.hybrid_lexical_weight", lexical_weight)
    if no_pdf:
        settings.set("rag.pdf_dir", "")
    for key, value in (("rag.embedder", embedder), ("rag.index_type", index), ("rag.chunk_strategy", chunking)):
        if value:
            settings.set(key, value)
    retriever = Retriever.from_settings(settings)
    if min_score is not None:
        retriever.min_score = min_score
    return retriever


def run(label: str, retriever: Retriever, queries: list[dict], k: int, show_misses: bool) -> float:
    start = time.perf_counter()
    report = evaluate_retrieval(retriever, queries, k)
    ms = (time.perf_counter() - start) * 1000 / len(queries)
    oos = "n/a" if report.no_answer_rate is None else f"{report.no_answer_rate:.0%}"
    print(
        f"{label:34} hit@1 {report.hit_at_1:5.0%}  hit@{k} {report.hit_at_k:5.0%}  MRR {report.mrr:.3f}  "
        f"P@{k} {report.precision_at_k:.2f}  no-answer {oos:>4}  {ms:5.1f} ms/query  [{retriever.mode}/{retriever.backend}]"
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
    ap.add_argument("--embedder", choices=["tfidf", "sentence_transformers", "openai", "hybrid"])
    ap.add_argument("--index", choices=["flat", "hnsw"])
    ap.add_argument("--chunking", choices=["fixed", "recursive"])
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--set", default="retrieval_eval.json", help="file in evals/ (default retrieval_eval.json)")
    ap.add_argument("--no-pdf", action="store_true", help="ignore PDFs under rag.pdf_dir")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--lexical-weight", type=float, help="hybrid: share of the score from exact-word matching (default from config)")
    ap.add_argument("--min-score", type=float, help="override the score floor for this run")
    ap.add_argument("--calibrate", action="store_true", help="compare off-topic vs correct-hit scores and suggest a min_score floor")
    ap.add_argument("--min-hit", type=float, default=0.0, help="fail (exit 1) if hit@k is below this")
    args = ap.parse_args()
    queries = load_queries(args.set)
    print(f"{len(queries)} labelled queries, k={args.k}\n")

    if args.calibrate:
        r = build(args.embedder, args.index, args.chunking, args.no_pdf, args.lexical_weight, args.min_score)
        cal = calibrate_floor(r, queries, args.k)
        rel = cal.relevant
        print(f"mode {r.mode}, current floor {r.min_score}")
        print(f"off-topic top scores ({len(cal.off_topic_top)}): {[round(x, 3) for x in cal.off_topic_top]}")
        if rel:
            print(f"correct-hit scores ({len(rel)}): min {rel[0]:.3f}  median {rel[len(rel) // 2]:.3f}  max {rel[-1]:.3f}")
        keep = cal.keep_all_floor()
        if keep is not None:
            print(f"floor that keeps every correct hit (0.05 margin): {keep} -> rejects {cal.off_topic_rejected(keep)} of {len(cal.off_topic_top)} off-topic queries")
        if cal.separable:
            print(f"suggested rag.min_score floor: {cal.suggested_floor} (midpoint of the gap; the sets are small, so keep a margin)")
        else:
            print("off-topic scores overlap correct hits: no floor separates them perfectly; pick the trade-off above (the router also filters off-topic questions before search)")
        return
    if args.compare:
        configs = [("tfidf", "flat"), ("tfidf", "hnsw"), ("sentence_transformers", "flat"), ("hybrid", "flat"), ("hybrid", "hnsw"), ("openai", "flat")]
        for embedder, index in configs:
            r = build(embedder, index, args.chunking, args.no_pdf, args.lexical_weight, args.min_score)
            if r.mode != embedder:  # factory fell back (package/key missing)
                print(f"{embedder + ' + ' + index:34} skipped (unavailable here)")
                continue
            run(f"{embedder} + {index}", r, queries, args.k, args.misses)
        return
    hit = run("configured", build(args.embedder, args.index, args.chunking, args.no_pdf, args.lexical_weight, args.min_score), queries, args.k, args.misses)
    if hit < args.min_hit:
        sys.exit(f"hit@{args.k} {hit:.0%} is below the required {args.min_hit:.0%}")


if __name__ == "__main__":
    main()
