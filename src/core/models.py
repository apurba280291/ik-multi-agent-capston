"""Shared data models passed between agents and the workflow."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Source:
    title: str
    category: str = ""
    reference: str = ""
    score: float = 0.0

    def label(self) -> str:
        return f"{self.title} ({self.category})" if self.category else self.title


@dataclass
class AgentResponse:
    agent: str
    content: str
    sources: list[Source] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentContext:
    """Everything an agent may read; ``shared`` lets agents hand data to later agents."""

    profile: dict[str, Any] = field(default_factory=dict)
    portfolio: list[dict[str, Any]] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    last_tickers: list[str] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_tolerance(self) -> str:
        return str(self.profile.get("risk_tolerance", "moderate")).lower()
