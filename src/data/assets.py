"""Asset reference data (class, sector, expense ratio) and sample portfolios."""
from __future__ import annotations

import json
from functools import lru_cache

from src.core.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "src" / "data"
UNKNOWN = {"name": "", "asset_class": "stock", "sector": "Unclassified", "expense_ratio": 0.0, "known": False}


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, dict]:
    try:
        raw = json.loads((DATA_DIR / "asset_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: {**v, "known": True} for k, v in raw.items() if not k.startswith("_")}


def asset_info(symbol: str) -> dict:
    info = load_metadata().get(symbol.upper())
    return info if info else {**UNKNOWN, "name": symbol.upper()}


@lru_cache(maxsize=1)
def load_sample_portfolios() -> dict[str, list[dict]]:
    try:
        return json.loads((DATA_DIR / "sample_portfolios.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
