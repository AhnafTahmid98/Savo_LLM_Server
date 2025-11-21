# app/providers/tier3_pi.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Tier3 Template / Pi-style Fallback
----------------------------------------------------------
This module provides a *fully offline*, deterministic fallback for
Robot Savo's replies.

Design goals:
- NEVER depends on network or heavy models.
- ALWAYS returns a safe, short B1 English sentence.
- Mirrors the kind of logic we will also run on the Pi when the LLM
  server is unreachable.
- Keeps all template logic in one place, so generate.py can stay clean.

Usage from generate.py (planned wiring):

    from app.providers.tier3_pi import call_tier3_fallback

    reply = call_tier3_fallback(request, intent, nav_goal_guess)

This is Tier3 in the 3-tier pipeline:

    Tier1: Online LLM (OpenRouter)
    Tier2: Local LLM (Ollama / llama-cpp)
    Tier3: Template fallback (this module)
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.intent import IntentType
from app.models.chat_request import ChatRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_tier3_fallback(
    request: ChatRequest,
    intent: IntentType,
    nav_goal_guess: Optional[str] = None,
) -> str:
    """
    Generate a simple, safe, offline reply based on intent and user_text.

    Parameters
    ----------
    request:
        Original ChatRequest (we mainly use user_text).
    intent:
        High-level intent decided by our classifier:
        "STOP" / "FOLLOW" / "NAVIGATE" / "STATUS" / "CHATBOT".
    nav_goal_guess:
        Optional raw goal name (e.g. "A201", "info desk"). This is not
        validated here; the caller is responsible for checking it
        against known_locations.json if needed.

    Returns
    -------
    reply_text: str
        A short, B1-level English sentence suitable for TTS.
    """
    text = (request.user_text or "").strip()
    intent = intent or "CHATBOT"  # defensive default

    # Normalize intent just in case
    intent_upper = intent.upper()

    # ------------------------------------------------------------------
    # STOP
    # ------------------------------------------------------------------
    if intent_upper == "STOP":
        return "Okay, I stop here and wait."

    # ------------------------------------------------------------------
    # FOLLOW
    # ------------------------------------------------------------------
    if intent_upper == "FOLLOW":
        return "Okay, I follow you. Please walk in front of me slowly."

    # ------------------------------------------------------------------
    # NAVIGATE
    # ------------------------------------------------------------------
    if intent_upper == "NAVIGATE":
        if nav_goal_guess:
            return f"Okay, I will guide you to {nav_goal_guess}. Please follow me."
        return "I can guide you in the building. Please tell me the room or place name."

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------
    if intent_upper == "STATUS":
        # Keep it simple; real status mode will later use nav_state.json
        # and robot_status.json inside the pipeline/LLM.
        return (
            "I am Robot Savo, a guide robot. Right now I am just waiting here and ready to help."
        )

    # ------------------------------------------------------------------
    # CHATBOT / fallback
    # ------------------------------------------------------------------
    if text:
        # Echo pattern is simple but still acceptable as Tier3 fallback.
        return f"You said: {text}. I am Robot Savo, how can I help you more?"
    else:
        return "Hello, I am Robot Savo. How can I help you?"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.providers.tier3_pi
    """
    from app.models.chat_request import InputSource

    print("Robot Savo — tier3_pi.py self-test\n")

    samples = [
        ("stop please", "STOP", None),
        ("can you follow me", "FOLLOW", None),
        ("can you take me to A201", "NAVIGATE", "A201"),
        ("take me somewhere", "NAVIGATE", None),
        ("why did you stop", "STATUS", None),
        ("hi robot savo", "CHATBOT", None),
        # Removed the empty-string case because ChatRequest.user_text
        # requires at least 1 character (min_length=1).
        # If we want to simulate "silence", we can use a placeholder:
        ("(no speech detected)", "CHATBOT", None),
    ]

    for text, intent, goal in samples:
        req = ChatRequest(
            user_text=text,
            source=InputSource.KEYBOARD,
            language="en",
            meta={"session_id": "demo-tier3"},
        )
        reply = call_tier3_fallback(req, intent, goal)
        print(f"User text : {text!r}")
        print(f"Intent    : {intent}")
        print(f"Nav goal  : {goal!r}")
        print(f"Reply     : {reply}")
        print("-" * 60)
