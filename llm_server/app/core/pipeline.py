# app/core/pipeline.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Main chat pipeline
------------------------------------------
This module glues everything together for a single chat turn:

1) Take a ChatRequest from the Pi (user text, source, meta...).
2) Classify the intent using our deterministic intent.py:
       STOP / FOLLOW / NAVIGATE / STATUS / CHATBOT
3) (Optional) Extract a raw navigation goal phrase (e.g. "a201",
   "info desk") from the text.
4) Load any live context (nav_state, robot_status, known_locations)
   from JSON files under app/map_data/.
5) Call the generation core (generate_reply_text), which will:
       - Build prompts based on intent + text
       - Run Tier1 → Tier2 → Tier3
       - Return the raw model output text + which tier was used
6) Try to parse the final JSON block from the model output
   (as required by our prompts: {"reply_text": "...", "intent": "...", "nav_goal": ...})
7) Build and return a ChatResponse Pydantic model, which is what
   FastAPI /chat will send back to the Pi.

IMPORTANT SAFETY DECISION:
- We treat the deterministic classifier (intent.py) as the source of truth
  for INTENT. The model's JSON "intent" field is *not* allowed to override it.
- For navigation goals we currently trust our simple extractor and any
  later map-lookup, not the model. Model nav_goal in JSON is ignored
  for now (we may use it for debugging in the future).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from app.core.intent import (
    IntentType,
    classify_intent,
    classify_intent_debug,
    extract_nav_goal,
)
from app.core.generate import generate_reply_text
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context loading helpers (nav_state, robot_status, known_locations)
# ---------------------------------------------------------------------------

def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file and return dict or None.

    - Logs a debug message if file is missing.
    - Logs a warning if JSON is invalid.
    """
    if not path.is_file():
        logger.debug("Context file not found (ok for now): %s", path)
        return None

    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return None


def load_context() -> Dict[str, Any]:
    """
    Load optional runtime context from app/map_data/.

    Files (all optional):
        nav_state.json
        robot_status.json
        known_locations.json

    Returns a dict:
        {
          "nav_state": {...} or None,
          "robot_status": {...} or None,
          "known_locations": {...} or None,
        }
    """
    base: Path = settings.map_data_dir

    nav_state = _load_json_file(base / "nav_state.json")
    robot_status = _load_json_file(base / "robot_status.json")
    known_locations = _load_json_file(base / "known_locations.json")

    return {
        "nav_state": nav_state,
        "robot_status": robot_status,
        "known_locations": known_locations,
    }


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------

def _extract_final_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract the final JSON object from the model's reply.

    Our prompts instruct the model to end with a JSON object like:

        {
          "reply_text": "...",
          "intent": "NAVIGATE",
          "nav_goal": "Info Desk"
        }

    But models are not perfect, so we:
    - Search from the last '{' to the last '}'.
    - Attempt json.loads(...) on that substring.
    - Return the decoded dict, or None if parsing fails.

    We DO NOT trust any earlier '{ ... }' in the text, only the last.
    """
    if not text or "{" not in text or "}" not in text:
        return None

    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]

    try:
        data = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse JSON block from model reply: %s", exc)
        return None

    if not isinstance(data, dict):
        return None

    return data


def _truncate_reply(text: str) -> str:
    """
    Enforce the max_reply_chars limit from settings.

    We don't want extremely long replies; this keeps TTS friendly.
    """
    max_len = getattr(settings, "max_reply_chars", 512)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(request: ChatRequest) -> ChatResponse:
    """
    Main public entry point for one chat turn.

    Steps:
        1) Classify intent from request.user_text (deterministic).
        2) Extract a raw navigation goal guess (if NAVIGATE).
        3) Load context JSON files (nav_state, robot_status, known_locations).
        4) Call generate_reply_text() (Tier1 / Tier2 / Tier3 chain).
        5) Try to parse the final JSON block from the model's reply.
        6) Construct ChatResponse with:
             - reply_text (from JSON or from raw text)
             - intent (ALWAYS from our classifier, for safety)
             - nav_goal (currently from our extractor, not model)
    """
    user_text = request.user_text

    # 1) Classify intent (deterministic, safety-first)
    intent: IntentType = classify_intent(user_text)
    debug_info = classify_intent_debug(user_text)
    logger.debug("Intent debug: %s", debug_info)

    # 2) Extract a rough nav goal phrase if needed
    nav_goal_guess: Optional[str] = None
    if intent == "NAVIGATE":
        nav_goal_guess = extract_nav_goal(user_text)

    # 3) Load live context (currently not deeply used; reserved for future use)
    context = load_context()

    # 4) Call generation core (multi-tier)
    raw_model_text, used_tier = generate_reply_text(
        request=request,
        intent=intent,
        nav_goal_guess=nav_goal_guess,
    )
    logger.info("Generation used tier: %s", used_tier)

    # 5) Try to parse JSON block at the end of the model text
    raw_model_text = raw_model_text.strip()
    parsed_json = _extract_final_json_block(raw_model_text)

    # Split "human-readable" part vs JSON block for debugging,
    # but for now we only need a clean reply_text.
    reply_text: str

    if parsed_json is not None:
        # Prefer reply_text from JSON if present, else fall back to the
        # full raw text (minus any trailing JSON). This allows the model
        # to slightly shorten / rephrase for speech.
        json_reply = parsed_json.get("reply_text")
        if isinstance(json_reply, str) and json_reply.strip():
            reply_text = json_reply.strip()
        else:
            # If no reply_text key, just use the full raw text
            reply_text = raw_model_text
    else:
        # No JSON found → just use raw text
        reply_text = raw_model_text

    reply_text = _truncate_reply(reply_text)

    # INTENT SAFETY:
    # We *always* trust our deterministic classifier for the final intent.
    final_intent_str: str = intent  # "STOP"/"FOLLOW"/...

    # NAV GOAL:
    # For now, we trust nav_goal_guess from extract_nav_goal. In the future,
    # we will validate it against known_locations.json. Model's nav_goal in
    # JSON is ignored for safety.
    final_nav_goal: Optional[str] = nav_goal_guess

    # Build the ChatResponse Pydantic model
    response = ChatResponse(
        reply_text=reply_text,
        intent=final_intent_str,
        nav_goal=final_nav_goal,
    )

    # Optionally log everything for debugging
    logger.debug(
        "Pipeline result: intent=%s nav_goal=%r used_tier=%s reply_text=%r",
        final_intent_str,
        final_nav_goal,
        used_tier,
        reply_text,
    )

    return response


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.pipeline
    """
    from app.models.chat_request import InputSource

    print("Robot Savo — pipeline.py self-test\n")

    # Example request that should trigger NAVIGATE
    req = ChatRequest(
        user_text="Can you take me to the info desk please?",
        source=InputSource.KEYBOARD,
        language="en",
        meta={"session_id": "demo-pipeline-001"},
    )

    resp = run_pipeline(req)

    print("ChatResponse:")
    print(f"  reply_text : {resp.reply_text!r}")
    print(f"  intent     : {resp.intent!r}")
    print(f"  nav_goal   : {resp.nav_goal!r}")
