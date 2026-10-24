"""Guardrails: input checks (advice, prompt injection, PII) and output validation of LLM rewrites.

Everything here is pure and deterministic so it works in offline mode and is easy to test.
Topic filtering stays in the router (it needs the finance vocabulary and conversation history).
"""
from __future__ import annotations

import re

# --- advice requests -------------------------------------------------------------------------------
ADVICE = re.compile(
    r"\b(should i (buy|sell|invest|put|move|hold|short)|what (stock|etf|fund)s? should i|tell me what to (buy|sell)|"
    r"guarantee\w*|sure thing|best (stock|etf|fund|crypto) to (buy|invest)|can i get rich|"
    r"(is|are) \w+( \w+)? (a )?(good|safe|smart) (buy|pick|bet|investment) (for me)?|will \w+ (go up|double|moon|crash))\b",
    re.I,
)

# --- prompt injection / jailbreak ------------------------------------------------------------------
INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above|earlier|system)?\s*"
    r"(instructions?|prompts?|rules?|guidelines?)|"
    r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)|"
    r"\b(jailbreak|developer mode|DAN mode)\b|"
    r"you are now\b.{0,40}\b(unrestricted|unfiltered|no rules)|"
    r"pretend (you|to) (have no|are not bound)",
    re.I,
)
BLOCKED_REPLY = (
    "I can't follow instructions that try to change how I work, but I'm happy to help with investing, markets, "
    "saving goals or taxes. What would you like to learn?"
)

# --- PII ---------------------------------------------------------------------------------------------
_PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[email]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card]"),
    (re.compile(r"(?<!\d)(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}(?!\d)"), "[phone]"),
]


def is_advice_request(text: str) -> bool:
    return bool(ADVICE.search(text))


def is_injection(text: str) -> bool:
    return bool(INJECTION.search(text))


def redact_pii(text: str) -> tuple[str, bool]:
    """Mask emails, SSNs, card and phone numbers. Returns (clean_text, anything_was_redacted)."""
    found = False
    for pattern, label in _PII_PATTERNS:
        text, n = pattern.subn(label, text)
        found = found or n > 0
    return text, found


# --- output validation -------------------------------------------------------------------------------
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CITATION = re.compile(r"\[\d+\]")


def _numbers(text: str) -> set[float]:
    """Numeric values in ``text``; small whole numbers (list items, months) are ignored as noise."""
    out: set[float] = set()
    for m in _NUMBER.findall(text):
        value = float(m.replace(",", ""))
        if value > 12 or value != int(value):
            out.add(round(value, 2))
    return out


def rewrite_is_faithful(draft: str, rewrite: str, query: str = "") -> bool:
    """True if an LLM rewrite introduced no new figures and kept every [n] citation marker."""
    if not rewrite.strip():
        return False
    allowed = _numbers(draft) | _numbers(query)
    if _numbers(rewrite) - allowed:
        return False
    return set(_CITATION.findall(draft)) <= set(_CITATION.findall(rewrite))
