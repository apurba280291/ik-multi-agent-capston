"""Configuration loading: config.yaml + environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    """Thin wrapper over the parsed YAML supporting dotted-path lookups."""

    def __init__(self, raw: dict[str, Any] | None = None):
        self.raw = raw or {}

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve_path(self, path: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p


_ENV_OVERRIDES = {
    "FINANCE_LLM_PROVIDER": "llm.provider",
    "FINANCE_MARKET_PROVIDER": "market.provider",
}


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load settings from ``path`` / $FINANCE_CONFIG / ./config.yaml.

    A missing or malformed file yields defaults rather than crashing the app.
    """
    cfg_path = Path(path or os.environ.get("FINANCE_CONFIG", PROJECT_ROOT / "config.yaml"))
    raw: dict[str, Any] = {}
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    settings = Settings(raw)
    for env_name, dotted in _ENV_OVERRIDES.items():
        if os.environ.get(env_name):
            settings.set(dotted, os.environ[env_name])
    return settings
