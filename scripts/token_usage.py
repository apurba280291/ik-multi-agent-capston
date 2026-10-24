"""Summarise logs/llm_usage.jsonl: LLM calls and input/output tokens per caller (agent or router).

    python scripts/token_usage.py                   # totals per caller
    python scripts/token_usage.py --since 2026-10-25
    python scripts/token_usage.py --calls           # every call, one per line
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.usage_log import usage_log_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="", help="ISO date/time prefix, e.g. 2026-10-25")
    ap.add_argument("--calls", action="store_true", help="list every call")
    args = ap.parse_args()

    path = usage_log_path()
    if not path.exists():
        sys.exit(f"No usage log yet at {path}. It is only written when an LLM API key is configured and used.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ts", "") >= args.since:
            rows.append(row)

    if args.calls:
        for r in rows:
            mark = "~" if r["estimated"] else " "
            print(f"{r['ts']} {r['caller']:20} {r['model']:24} in={mark}{r['input_tokens']:>6} out={mark}{r['output_tokens']:>6}")
        print()

    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        t = totals[r["caller"]]
        t[0] += 1
        t[1] += r["input_tokens"]
        t[2] += r["output_tokens"]
    print(f"{'caller':20} {'calls':>6} {'input':>9} {'output':>9} {'avg in':>8} {'avg out':>8}")
    for caller, (n, tin, tout) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        print(f"{caller:20} {n:>6} {tin:>9} {tout:>9} {tin // n:>8} {tout // n:>8}")
    n, tin, tout = (sum(v[i] for v in totals.values()) for i in range(3))
    print(f"{'TOTAL':20} {n:>6} {tin:>9} {tout:>9}")
    if any(r["estimated"] for r in rows):
        print("\n~ = estimated (~4 chars/token) because the provider reported no usage")


if __name__ == "__main__":
    main()
