"""Central logging setup: console plus a rotating ``logs/app.log`` so failures can be searched later."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.utils.call_log import log_dir

_CONFIGURED = False
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        try:
            folder = log_dir()
            folder.mkdir(parents=True, exist_ok=True)
            handlers.append(RotatingFileHandler(folder / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"))
        except OSError:  # read-only filesystem etc.: console logging is enough
            pass
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=FORMAT, handlers=handlers)
        _CONFIGURED = True
    return logging.getLogger(name)
