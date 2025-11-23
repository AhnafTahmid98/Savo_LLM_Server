# app/utils/file_io.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — file_io utilities
-----------------------------------------
Safe helpers for reading/writing small JSON or text files.

Goals:
- Avoid duplicated ad-hoc JSON handling everywhere.
- Use atomic writes (temp file + rename) to prevent half-written files.
- Be tolerant: on read errors, log and return a default instead of crashing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def read_json_safely(
    path: Path,
    default: Optional[T] = None,
    *,
    log_missing: bool = False,
) -> Optional[T]:
    """
    Read JSON from a file and return the parsed object.

    Behaviour:
    - If the file does not exist:
        - returns `default`
        - optionally logs at INFO level when log_missing=True
    - If parsing fails:
        - logs at WARNING level
        - returns `default`

    This is meant for small config/state files, not large datasets.
    """
    if not path.is_file():
        if log_missing:
            logger.info("read_json_safely: file not found: %s", path)
        return default

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("read_json_safely: failed to read %s: %s", path, exc)
        return default

    try:
        return json.loads(text)  # type: ignore[return-value]
    except json.JSONDecodeError as exc:
        logger.warning("read_json_safely: invalid JSON in %s: %s", path, exc)
        return default


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON to disk in a safe, atomic-ish way:

    - ensures parent directory exists
    - writes to a temporary file next to the target
    - renames the temp file to the final path

    If anything fails, an exception is raised so the caller can decide
    how to respond (e.g. HTTP 500).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("write_json_atomic: failed to create dir %s: %s", path.parent, exc)
        raise

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
        tmp_path.write_text(json_text, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        logger.error("write_json_atomic: failed to write %s: %s", path, exc)
        raise


def read_text_safely(
    path: Path,
    default: Optional[str] = None,
    *,
    strip: bool = False,
) -> Optional[str]:
    """
    Read a UTF-8 text file and return its content.

    - On failure, logs and returns `default`.
    - If strip=True, leading/trailing whitespace is removed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("read_text_safely: failed to read %s: %s", path, exc)
        return default

    return text.strip() if strip else text


def write_text_atomic(path: Path, content: str) -> None:
    """
    Write a UTF-8 text file using the same atomic strategy as write_json_atomic.

    Intended for small config or log snapshot files, not large blobs.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("write_text_atomic: failed to create dir %s: %s", path.parent, exc)
        raise

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        logger.error("write_text_atomic: failed to write %s: %s", path, exc)
        raise
