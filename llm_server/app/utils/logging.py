# app/utils/logging.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — logging utilities
-----------------------------------------
Central logging configuration for the LLM server.

We try to:
- Use a consistent format across all modules.
- Honour settings.debug when available (more verbose in dev).
- Play nice with Uvicorn/FastAPI logs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(
    *,
    debug: bool = False,
    level: Optional[int] = None,
) -> None:
    """
    Configure root logging for the process.

    Parameters
    ----------
    debug:
        If True, default log level becomes DEBUG, otherwise INFO.
        This is typically wired from settings.debug.
    level:
        Optional explicit logging level (overrides debug flag) such as
        logging.DEBUG or logging.INFO.

    This function is idempotent: calling it multiple times is safe.
    """
    # Decide base level
    if level is not None:
        base_level = level
    else:
        base_level = logging.DEBUG if debug else logging.INFO

    # Basic formatter: timestamp, level, logger name, message
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # If logging is already configured (handlers exist), just adjust levels.
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(base_level)
        for h in root.handlers:
            h.setLevel(base_level)
        return

    logging.basicConfig(
        level=base_level,
        format=fmt,
        datefmt=datefmt,
    )

    # Tweak noisy loggers if needed
    for noisy in ("uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(noisy).setLevel(os.getenv("SAVO_NOISY_LOG_LEVEL", "WARNING"))


def get_logger(name: str) -> logging.Logger:
    """
    Small convenience wrapper around logging.getLogger.

    Usage:
        from app.utils import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
