"""Display helpers and the standard educational disclaimer."""
from __future__ import annotations

DISCLAIMER = (
    "*Educational information only - not personalized financial, tax or legal advice. "
    "Investing involves risk, including loss of principal. Consider speaking with a "
    "licensed professional before making decisions.*"
)


def money(value: float, decimals: int = 0) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def with_disclaimer(text: str) -> str:
    return text if DISCLAIMER in text else f"{text}\n\n{DISCLAIMER}"


def citation_block(sources) -> str:
    """Numbered source list matching the [n] markers used in agent drafts."""
    if not sources:
        return ""
    lines = []
    for i, s in enumerate(sources, 1):
        ref = f" - {s.reference}" if s.reference else ""
        lines.append(f"[{i}] {s.label()}{ref}")
    return "**Sources**\n" + "\n".join(lines)
