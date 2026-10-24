"""Central logging setup."""
from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
