# app/core/safety.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Safety helpers
--------------------------------------
Central place for:
- Cleaning / normalizing user text before intent + LLM.
- Clamping reply text length before sending to the Pi.

These functions are *pure* (no network, no I/O) so they are easy to test.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Hard cap for *user* text we accept into the pipeline.
# (Independently of model reply limit.)
MAX_USER_CHARS: int = 512


@dataclass
class SanitizedTextResult:
    """
    Result of sanitize_user_text().

    Attributes
    ----------
    original:
        Original raw text from the client (may be None -> "").
    sanitized:
        Cleaned version used for intent classification + LLM.
    truncated:
        True if we had to cut the text at MAX_USER_CHARS.
    too_short:
        True if sanitized text is empty (or effectively empty).
    """
    original: str
    sanitized: str
    truncated: bool
    too_short: bool


# ---------------------------------------------------------------------------
# User text sanitization
# ---------------------------------------------------------------------------

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_user_text(raw_text: Optional[str]) -> SanitizedTextResult:
    """
    Clean up user text before it touches intent.py or the LLM.

    Steps:
    - Convert None -> "" so callers never see None.
    - Remove control characters (non-printable).
    - Collapse whitespace (multiple spaces/newlines -> single space).
    - Trim leading/trailing whitespace.
    - Truncate to MAX_USER_CHARS.

    Returns a SanitizedTextResult with flags for truncation/emptiness.
    """
    original = raw_text if isinstance(raw_text, str) else ""

    # 1) Strip control characters
    cleaned = _CONTROL_CHARS_RE.sub("", original)

    # 2) Normalize whitespace: " hi \n\n  there " -> "hi there"
    cleaned = " ".join(cleaned.split())

    truncated = False
    if len(cleaned) > MAX_USER_CHARS:
        cleaned = cleaned[:MAX_USER_CHARS]
        truncated = True

    sanitized = cleaned.strip()
    too_short = len(sanitized) == 0

    if truncated:
        logger.debug(
            "sanitize_user_text: truncated user text from %d to %d chars",
            len(original),
            len(sanitized),
        )

    if too_short and original:
        logger.debug(
            "sanitize_user_text: sanitized text became empty; original=%r",
            original,
        )

    return SanitizedTextResult(
        original=original,
        sanitized=sanitized,
        truncated=truncated,
        too_short=too_short,
    )


# ---------------------------------------------------------------------------
# Reply text clamping
# ---------------------------------------------------------------------------

def clamp_reply_text(reply_text: str) -> str:
    """
    Ensure the final reply text is not too long for TTS / UI.

    Uses:
        settings.max_reply_chars

    Behavior:
    - If limit <= 0: returns an empty string.
    - If reply length <= limit: returns as-is.
    - If too long: cuts to (limit - 3) and appends "..." if possible.
    """
    text = reply_text if isinstance(reply_text, str) else str(reply_text or "")

    limit = getattr(settings, "max_reply_chars", 512)
    if limit <= 0:
        logger.warning("clamp_reply_text: max_reply_chars <= 0, returning empty string.")
        return ""

    if len(text) <= limit:
        return text

    # We want a small visual indicator that we truncated ("..."),
    # but we also avoid negative slicing if limit < 3.
    if limit > 3:
        clamped = text[: limit - 3].rstrip() + "..."
    else:
        clamped = text[:limit]

    logger.debug(
        "clamp_reply_text: truncated reply from %d to %d chars (limit=%d)",
        len(text),
        len(clamped),
        limit,
    )
    return clamped


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Robot Savo — safety.py self-test\n")

    samples = [
        "  hello   Robot   Savo  ",
        "\x00\x01weird\x02text\n\n line 2   ",
        "x" * 600,
        "",
        None,
    ]

    for s in samples:
        res = sanitize_user_text(s)  # type: ignore[arg-type]
        print(f"Original : {repr(res.original)}")
        print(f"Sanitized: {repr(res.sanitized)}")
        print(f"truncated={res.truncated} too_short={res.too_short}")
        print("-" * 60)

    long_reply = "This is a very long reply " * 40
    print("\nClamp test:")
    print("Before:", len(long_reply))
    clamped = clamp_reply_text(long_reply)
    print("After :", len(clamped), repr(clamped[:80] + ("..." if len(clamped) > 80 else "")))
