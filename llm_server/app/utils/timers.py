# app/utils/timers.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — timing utilities
----------------------------------------
Lightweight helpers for measuring execution time and logging it.

These are especially useful for:
- Measuring Tier1 vs Tier2 latency.
- Profiling parts of the pipeline without cluttering code.
"""

from __future__ import annotations

import logging
import time
from contextlib import ContextDecorator
from typing import Callable, Optional


class Stopwatch(ContextDecorator):
    """
    Simple stopwatch context manager.

    Example:
        from app.utils import Stopwatch, get_logger

        logger = get_logger(__name__)

        with Stopwatch("Tier1 call", logger):
            call_tier1_model(...)

    This will log something like:
        Tier1 call took 0.237 s
    """

    def __init__(
        self,
        label: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
    ) -> None:
        self.label = label
        self.logger = logger or logging.getLogger(__name__)
        self.level = level
        self._start: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        elapsed = time.perf_counter() - self._start
        self.logger.log(self.level, "%s took %.3f s", self.label, elapsed)


def log_duration(
    label: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Callable[..., Stopwatch]:
    """
    Decorator factory to measure and log duration of a function.

    Example:

        from app.utils import log_duration, get_logger
        logger = get_logger(__name__)

        @log_duration("generate_reply_text", logger)
        def generate_reply_text(...):
            ...

    On each call it will log:
        generate_reply_text took 0.145 s
    """
    log = logger or logging.getLogger(__name__)

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log.log(level, "%s took %.3f s", label, elapsed)

        return wrapper

    return decorator
