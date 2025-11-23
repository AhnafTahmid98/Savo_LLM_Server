# app/utils/__init__.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Utility toolbox
---------------------------------------
Shared helper functions that are used across the LLM server:

- file_io   : safe JSON/text read/write helpers
- logging   : central logging configuration
- timers    : small timing/profiling helpers

Import from here when it makes sense, for a clean public API, e.g.:

    from app.utils import setup_logging, read_json_safely
"""

from __future__ import annotations

from .file_io import (  # noqa: F401
    read_json_safely,
    write_json_atomic,
    read_text_safely,
    write_text_atomic,
)

from .logging import (  # noqa: F401
    setup_logging,
    get_logger,
)

from .timers import (  # noqa: F401
    Stopwatch,
    log_duration,
)
