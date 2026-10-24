"""Summarise logs/agent_calls.jsonl: failure rate, latency and top errors per agent.

    python scripts/agent_stats.py                 # all history
    python scripts/agent_stats.py --since 2026-10-24
    python scripts/agent_stats.py --failures      # also list each failed call with its query
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.call_log import call_log_path  # noqa: E402


def load(since: str) -> list[dict]:
    path = call_log_path()
    if not path.exists():
        sys.exit(f"No call log yet at {path}. Chat with the app first.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ts", "") >= since:
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="", help="ISO date/time prefix, e.g. 2026-10-24")
    ap.add_argument("--failures", action="store_true", help="list individual failed calls")
    args = ap.parse_args()

    rows = load(args.since)
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    print(f"{'agent':20} {'calls':>6} {'fail':>5} {'fail%':>6} {'p50 ms':>8} {'max ms':>8}  top error")
    for agent, calls in sorted(by_agent.items()):
        failed = [c for c in calls if not c["ok"]]
        ms = [c["ms"] for c in calls]
        top = Counter((c["error"] or "?").split(":")[0] for c in failed).most_common(1)
        print(
            f"{agent:20} {len(calls):>6} {len(failed):>5} {100 * len(failed) / len(calls):>5.1f}% "
            f"{statistics.median(ms):>8.0f} {max(ms):>8.0f}  {top[0][0] + ' x' + str(top[0][1]) if top else '-'}"
        )
    if not rows:
        print("(no calls in range)")
    if args.failures:
        print("\nFailed calls:")
        for r in rows:
            if not r["ok"]:
                print(f"  {r['ts']} {r['agent']:18} {r['error']}\n      query: {r['query']!r}")


if __name__ == "__main__":
    main()
