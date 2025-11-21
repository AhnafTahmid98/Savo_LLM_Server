# app/core/pipeline.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Pipeline
--------------------------------
High-level orchestration for one chat turn.

Flow:
  ChatRequest ->
    1) classify_intent()        (STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT)
    2) extract_nav_goal()       (heuristic nav goal, e.g. "A201")
    3) generate_reply_text()    (Tier1/Tier2/Tier3 -> ModelCallResult)
    4) parse final JSON block   (-> ParsedJsonResult)
    5) build ChatResponse       (reply_text, intent, nav_goal)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Union

from app.core.config import settings
from app.core.intent import IntentType, classify_intent, extract_nav_goal
from app.core.generate import generate_reply_text
from app.core.types import ModelCallResult, ParsedJsonResult, TierLabel
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse

logger = logging.getLogger(__name__)

_INTENT_VALUES = {"STOP", "FOLLOW", "NAVIGATE", "STATUS", "CHATBOT"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text_and_tier(result: Union[ModelCallResult, Dict[str, Any]]) -> tuple[str, TierLabel]:
    """
    Small compatibility helper.

    - Preferred: result is ModelCallResult with .text and .used_tier.
    - Legacy/defensive: result may be a dict with 'text' and 'used_tier' keys.
    """
    if hasattr(result, "text") and hasattr(result, "used_tier"):
        return result.text, result.used_tier  # type: ignore[attr-defined]
    if isinstance(result, dict):
        text = str(result.get("text", ""))
        tier = result.get("used_tier", "tier3")
        return text, tier  # type: ignore[return-value]
    raise TypeError(f"Unsupported ModelCallResult type: {type(result)!r}")


def _fallback_reply_text(raw_text: str) -> str:
    """
    Last-resort spoken reply if JSON is missing or broken.
    """
    text = (raw_text or "").strip()
    if not text:
        return "Hello, I am Robot Savo. How can I help you?"
    max_chars = getattr(settings, "max_reply_chars", 512)
    return text[:max_chars]


def _parse_model_output(
    result: Union[ModelCallResult, Dict[str, Any]],
    default_intent: IntentType,
    default_nav_goal: Optional[str],
) -> ParsedJsonResult:
    """
    Parse the final JSON block from the model output.

    Expected format (last line / last block):
      {
        "reply_text": "...",
        "intent": "NAVIGATE",
        "nav_goal": "A201"
      }

    If anything fails, we fall back to:
      - reply_text: spoken part or generic fallback
      - intent    : default_intent
      - nav_goal  : default_nav_goal
    """
    raw_text, used_tier = _get_text_and_tier(result)

    idx = raw_text.rfind("{")
    if idx == -1:
        logger.warning("No JSON block found in model output (tier=%s).", used_tier)
        return ParsedJsonResult(
            reply_text=_fallback_reply_text(raw_text),
            intent=default_intent,
            nav_goal=default_nav_goal,
            used_tier=used_tier,
            parse_error="no_json_block",
        )

    json_part = raw_text[idx:]
    spoken_part = raw_text[:idx].strip()

    try:
        data = json.loads(json_part)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error from model output (tier=%s): %s", used_tier, exc)
        return ParsedJsonResult(
            reply_text=_fallback_reply_text(raw_text),
            intent=default_intent,
            nav_goal=default_nav_goal,
            used_tier=used_tier,
            parse_error=f"json_decode_error: {exc}",
        )

    reply_text = data.get("reply_text")
    intent_str = data.get("intent")
    nav_goal = data.get("nav_goal", default_nav_goal)

    if not isinstance(reply_text, str) or not reply_text.strip():
        reply_text = spoken_part or _fallback_reply_text(raw_text)

    if not isinstance(intent_str, str) or intent_str not in _INTENT_VALUES:
        logger.warning(
            "Invalid or missing intent in JSON (%r); falling back to %s",
            intent_str,
            default_intent,
        )
        intent_value: IntentType = default_intent
        parse_error = "invalid_or_missing_intent"
    else:
        intent_value = intent_str  # type: ignore[assignment]
        parse_error = None

    if nav_goal is not None and not isinstance(nav_goal, str):
        nav_goal = str(nav_goal)

    max_chars = getattr(settings, "max_reply_chars", 512)
    if len(reply_text) > max_chars:
        reply_text = reply_text[:max_chars].rstrip()

    return ParsedJsonResult(
        reply_text=reply_text,
        intent=intent_value,
        nav_goal=nav_goal,
        used_tier=used_tier,
        parse_error=parse_error,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    request: ChatRequest,
    context: Optional[Dict[str, Any]] = None,
) -> ChatResponse:
    """
    High-level pipeline for one user turn.
    """
    # 1) Deterministic intent classification
    intent: IntentType = classify_intent(request.user_text)

    # 2) Heuristic nav goal
    nav_goal_guess: Optional[str] = None
    if intent == "NAVIGATE":
        nav_goal_guess = extract_nav_goal(request.user_text)

    # 3) Call generation chain
    model_result = generate_reply_text(
        request=request,
        intent=intent,
        nav_goal_guess=nav_goal_guess,
    )

    # 4) Parse final JSON block
    parsed: ParsedJsonResult = _parse_model_output(
        result=model_result,
        default_intent=intent,
        default_nav_goal=nav_goal_guess,
    )

    # 5) Build ChatResponse
    response = ChatResponse(
        reply_text=parsed.reply_text,
        intent=parsed.intent,
        nav_goal=parsed.nav_goal,
    )

    logger.debug(
        "run_pipeline: tier=%s intent=%s nav_goal=%r parse_error=%r",
        parsed.used_tier,
        parsed.intent,
        parsed.nav_goal,
        parsed.parse_error,
    )

    return response


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.pipeline
    """
    from app.models.chat_request import InputSource

    print("Robot Savo — pipeline.py self-test\n")

    req = ChatRequest(
        user_text="Can you take me to the info desk please?",
        source=InputSource.KEYBOARD,
        language="en",
        meta={"session_id": "demo-pipeline-001"},
    )

    resp = run_pipeline(req, context={})
    print("ChatResponse:")
    print(f"  reply_text : {resp.reply_text!r}")
    print(f"  intent     : {resp.intent!r}")
    print(f"  nav_goal   : {resp.nav_goal!r}")
