# app/core/pipeline.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — High-level pipeline
-------------------------------------------
This module glues everything together for the /chat endpoint:

1) Takes a ChatRequest from FastAPI.
2) Classifies intent using our deterministic classifier.
3) (NEW) Optionally fetches live data (weather, time, crypto) via tools_web
   and injects it into request.meta["live_context"] so the LLM can use it.
4) Calls generate_reply_text() to run the Tier1/Tier2/Tier3 chain.
5) Parses the final JSON block from the model output.
6) Returns a ChatResponse model (reply_text, intent, nav_goal) for FastAPI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.intent import (
    classify_intent,
    extract_nav_goal,
    IntentType,  # Literal type alias ("STOP", "FOLLOW", ...)
)
from app.core.types import ModelCallResult, ParsedJsonResult
from app.core.generate import generate_reply_text
from app.core.tools_web import (
    get_weather_current,
    get_local_time,
    get_crypto_price,
    ToolsWebError,
)
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse, IntentType as ResponseIntentType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for parsing the model's JSON block
# ---------------------------------------------------------------------------

def _extract_json_suffix(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Try to find and parse a JSON object at the end of the model's reply.

    We search for the last '{' and attempt json.loads() from there.
    If parsing fails, return None.
    """
    if not raw_text:
        return None

    idx = raw_text.rfind("{")
    if idx == -1:
        return None

    candidate = raw_text[idx:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON suffix from model output.")
        return None


def _parse_model_output(
    result: ModelCallResult,
    intent_hint: IntentType,
    nav_goal_guess: Optional[str],
) -> ParsedJsonResult:
    """
    Take ModelCallResult (text + tier info) and extract:

    - reply_text : what TTS should speak
    - intent     : STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT (string)
    - nav_goal   : canonical goal or None
    - used_tier  : tier1 / tier2 / tier3
    - parse_error: None or error string
    """
    raw_text: str = result.text or ""
    used_tier: str = result.used_tier

    json_obj = _extract_json_suffix(raw_text)
    if not json_obj:
        # No valid JSON block → fall back to simple behavior:
        return ParsedJsonResult(
            reply_text=raw_text.strip()[: settings.max_reply_chars],
            intent=intent_hint,
            nav_goal=nav_goal_guess,
            used_tier=used_tier,
            parse_error="json_missing_or_invalid",
        )

    # Extract fields from JSON with safe defaults
    reply_text = str(json_obj.get("reply_text") or "").strip()
    intent = str(json_obj.get("intent") or intent_hint).strip().upper()
    nav_goal = json_obj.get("nav_goal", nav_goal_guess)

    if not reply_text:
        # If JSON didn't provide reply_text, fall back to full text
        reply_text = raw_text.strip()

    # Trim reply_text for safety
    reply_text = reply_text[: settings.max_reply_chars]

    # Normalize intent to one of our known labels; otherwise fall back.
    valid_intents = {"STOP", "FOLLOW", "NAVIGATE", "STATUS", "CHATBOT"}
    if intent not in valid_intents:
        intent = intent_hint

    # If model returned null/None for nav_goal, keep the original guess.
    if nav_goal is None:
        nav_goal = nav_goal_guess

    # If nav_goal is a string, normalize spacing a bit.
    if isinstance(nav_goal, str):
        nav_goal = nav_goal.strip()
        if not nav_goal:
            nav_goal = None

    return ParsedJsonResult(
        reply_text=reply_text,
        intent=intent,  # still string here; ChatResponse will coerce
        nav_goal=nav_goal,
        used_tier=used_tier,
        parse_error=None,
    )


# ---------------------------------------------------------------------------
# Live data integration (tools_web)
# ---------------------------------------------------------------------------

def _attach_live_context(request: ChatRequest) -> None:
    """
    Detect if the user is asking about live info (weather, time, crypto),
    call tools_web, and inject results into request.meta["live_context"].

    This function mutates `request` in-place.
    """
    text = (request.user_text or "").lower()
    live_context: Dict[str, Any] = {}

    # Example: simple keyword-based routing.
    # You can make this smarter later.
    try:
        if "weather" in text or "temperature" in text:
            # Kuopio approx; later you can take this from robot config or Pi.
            live_context["weather"] = get_weather_current(
                lat=62.89,
                lon=27.68,
            )
    except ToolsWebError as exc:
        logger.warning("Weather tool failed: %s", exc)

    try:
        if "time" in text and "battery" not in text:
            live_context["time"] = get_local_time()
    except ToolsWebError as exc:
        logger.warning("Time tool failed: %s", exc)

    try:
        if any(k in text for k in ("bitcoin", "btc", "crypto", "cryptocurrency")):
            price = get_crypto_price("bitcoin", "eur")
            if price is not None:
                live_context["crypto"] = {"btc_eur": price}
    except ToolsWebError as exc:
        logger.warning("Crypto tool failed: %s", exc)

    if not live_context:
        return

    meta = dict(request.meta or {})
    meta["live_context"] = live_context
    request.meta = meta


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    request: ChatRequest,
    context: Optional[Dict[str, Any]] = None,
) -> ChatResponse:
    """
    High-level entry point used by /chat.

    - Classifies intent.
    - Optionally attaches live data into request.meta["live_context"].
    - Runs generation chain (Tier1/Tier2/Tier3).
    - Parses JSON block from model output.
    - Returns ChatResponse for FastAPI.
    """
    context = context or {}

    user_text = request.user_text or ""
    intent: IntentType = classify_intent(user_text)

    # Simple nav goal guess from text (string like "a201" or "info desk").
    nav_goal_guess = extract_nav_goal(user_text) if intent in ("NAVIGATE", "FOLLOW") else None

    # 1) Attach live data (only for CHATBOT-style questions, typically)
    if intent == "CHATBOT":
        _attach_live_context(request)

    # 2) Run generation chain
    model_result: ModelCallResult = generate_reply_text(
        request=request,
        intent=intent,
        nav_goal_guess=nav_goal_guess,
    )

    # 3) Parse JSON block from model output
    parsed: ParsedJsonResult = _parse_model_output(
        result=model_result,
        intent_hint=intent,
        nav_goal_guess=nav_goal_guess,
    )

    # 4) Convert intent string → ChatResponse.IntentType enum
    try:
        response_intent = ResponseIntentType(parsed.intent)
    except ValueError:
        response_intent = ResponseIntentType.CHATBOT

    # 5) Build final ChatResponse
    response = ChatResponse(
        reply_text=parsed.reply_text,
        intent=response_intent,
        nav_goal=parsed.nav_goal,
    )

    logger.debug(
        "run_pipeline: intent=%s nav_goal=%r used_tier=%s live_keys=%s",
        parsed.intent,
        parsed.nav_goal,
        parsed.used_tier,
        list((request.meta or {}).get("live_context", {}).keys()),
    )
    return response


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    from app.models.chat_request import InputSource

    print("Robot Savo — pipeline.py self-test\n")

    # Fake navigation request
    req = ChatRequest(
        user_text="Can you take me to info deskS?",
        source=InputSource.KEYBOARD,
        language="en",
        meta={"session_id": "demo-002"},
    )

    resp = run_pipeline(req, context={})

    print("ChatResponse:")
    print(f"  reply_text : {resp.reply_text!r}")
    print(f"  intent     : {resp.intent!r}")
    print(f"  nav_goal   : {resp.nav_goal!r}")
