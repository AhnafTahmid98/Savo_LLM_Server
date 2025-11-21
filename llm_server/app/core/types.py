# app/core/types.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Shared type helpers
-------------------------------------------
Central place for small shared type definitions used across the core:

- IntentLabel   : string literal for STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT
- TierLabel     : which tier was used ("tier1"|"tier2"|"tier3")
- ModelCallResult : result of a single model call (before JSON parsing)
- ParsedJsonResult: result after parsing the final JSON block
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

# ---------------------------------------------------------------------------
# Intent / tier labels (string forms)
# ---------------------------------------------------------------------------

# These mirror:
# - app.core.intent.IntentType (Literal[...] in the classifier)
# - app.models.chat_response.IntentType (Enum used in responses)
IntentLabel = Literal["STOP", "FOLLOW", "NAVIGATE", "STATUS", "CHATBOT"]

# Which tier produced the text
TierLabel = Literal["tier1", "tier2", "tier3"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelCallResult:
    """
    Result of a single model call (Tier1 / Tier2 / Tier3),
    BEFORE we parse the final JSON block.

    Attributes
    ----------
    text:
        Full text returned by the model (spoken reply + JSON at the end).
    used_tier:
        "tier1" | "tier2" | "tier3"
    raw:
        Optional backend metadata (e.g. model name, URL, provider).
    """
    text: str
    used_tier: TierLabel
    raw: Dict[str, Any]


@dataclass
class ParsedJsonResult:
    """
    Result AFTER parsing the final JSON block from the model output.

    Attributes
    ----------
    reply_text:
        Final text to speak (already truncated to max_reply_chars).
    intent:
        Final intent string ("STOP"/"FOLLOW"/"NAVIGATE"/"STATUS"/"CHATBOT").
    nav_goal:
        Canonical navigation goal (or None if not applicable).
    used_tier:
        Which tier produced the text ("tier1"/"tier2"/"tier3").
    parse_error:
        Optional error code if JSON parsing failed or intent was invalid.
    """
    reply_text: str
    intent: IntentLabel
    nav_goal: Optional[str]
    used_tier: TierLabel
    parse_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.types
    """
    print("Robot Savo — core.types self-test\n")

    example = ModelCallResult(
        text="Okay, I will guide you to A201. {\"reply_text\": \"...\", \"intent\": \"NAVIGATE\", \"nav_goal\": \"A201\"}",
        used_tier="tier1",
        raw={"backend": "openrouter", "model": "x-ai/grok-4.1-fast:free"},
    )

    print("ModelCallResult example:")
    print(f"  text      : {example.text[:60]}...")
    print(f"  used_tier : {example.used_tier}")
    print(f"  raw       : {example.raw}")
    print("-" * 60)

    parsed = ParsedJsonResult(
        reply_text="Okay, I will guide you to A201. Please follow me.",
        intent="NAVIGATE",
        nav_goal="A201",
        used_tier="tier1",
        parse_error=None,
    )

    print("ParsedJsonResult example:")
    print(f"  reply_text : {parsed.reply_text}")
    print(f"  intent     : {parsed.intent}")
    print(f"  nav_goal   : {parsed.nav_goal}")
    print(f"  used_tier  : {parsed.used_tier}")
    print(f"  parse_error: {parsed.parse_error!r}")
    print("\nSelf-test completed OK.")
