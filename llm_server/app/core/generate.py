# app/core/generate.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Generation core
---------------------------------------
This module handles the actual text generation workflow:

- Build prompts based on:
    - intent (STOP / FOLLOW / NAVIGATE / STATUS / CHATBOT)
    - user text (ChatRequest)
    - optional navigation goal guess

- Run the 3-tier chain:
    1) Tier1: Online LLM (OpenRouter, multi-model with priority list)
    2) Tier2: Local LLM (GGUF via llama-cpp or similar)
    3) Tier3: Template fallback (fully offline, deterministic)
    
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.intent import IntentType
from app.models.chat_request import ChatRequest
from app.providers.tier1_online import call_tier1_model, Tier1Error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loading helpers
# ---------------------------------------------------------------------------

_PROMPT_CACHE: Dict[str, str] = {}


def _read_prompt_file(filename: str) -> str:
    """
    Read a prompt file from settings.prompts_dir with simple caching.

    If the file does not exist, returns an empty string and logs a warning.
    """
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]

    path: Path = settings.prompts_dir / filename
    if not path.is_file():
        logger.warning("Prompt file not found: %s", path)
        _PROMPT_CACHE[filename] = ""
        return ""

    text = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[filename] = text
    return text


def _build_system_prompt(intent: IntentType) -> str:
    """
    Build the system prompt by combining:
    - system_prompt.txt
    - style_guidelines.txt
    - one of navigation/chatbot/status prompts depending on intent.
    """
    base = _read_prompt_file("system_prompt.txt")
    style = _read_prompt_file("style_guidelines.txt")

    if intent in ("NAVIGATE", "FOLLOW", "STOP"):
        mode = _read_prompt_file("navigation_prompt.txt")
    elif intent == "STATUS":
        mode = _read_prompt_file("status_prompt.txt")
    else:
        # CHATBOT (and any unknown → safe default)
        mode = _read_prompt_file("chatbot_prompt.txt")

    parts = [p for p in (base, style, mode) if p.strip()]
    if not parts:
        # Super safe fallback if prompt files are missing
        return (
            "You are Robot Savo, a polite indoor guide robot. "
            "Speak in short B1 English sentences and be safe."
        )

    return "\n\n".join(parts)


def _build_user_prompt(
    request: ChatRequest,
    intent: IntentType,
    nav_goal_guess: Optional[str] = None,
) -> str:
    """
    Build the 'user' content for the chat completion.

    We explicitly tell the model:
    - the raw user text
    - the pre-classified intent (from our deterministic classifier)
    - any guessed navigation goal
    - optional source / language / meta
    """
    lines: List[str] = [
        f"USER_TEXT: {request.user_text}",
        f"INTENT_HINT: {intent}",
    ]
    if nav_goal_guess:
        lines.append(f"NAV_GOAL_GUESS: {nav_goal_guess}")

    # Optional routing/source info
    if request.source:
        lines.append(f"SOURCE: {request.source}")
    if request.language:
        lines.append(f"LANGUAGE_HINT: {request.language}")

    if request.meta:
        try:
            meta_json = json.dumps(request.meta, ensure_ascii=False)
            lines.append(f"META: {meta_json}")
        except TypeError:
            # If meta is not JSON-serializable, ignore
            logger.warning("ChatRequest.meta is not JSON-serializable; ignoring.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier2: Local LLM (GGUF / llama-cpp)
# ---------------------------------------------------------------------------

class Tier2Error(Exception):
    """Raised when Tier2 (local) fails in a recoverable way."""


def _call_tier2_local(messages: List[Dict[str, str]]) -> str:
    """
    Call Tier2 local model.

    This is a placeholder implementation that you can wire to llama-cpp,
    Ollama, or any other local model backend.

    For now, it raises Tier2Error so the caller falls back to Tier3.
    """
    if not settings.tier2_enabled:
        raise Tier2Error("Tier2 is disabled in config.")

    # TODO: integrate with your chosen local backend:
    # - llama-cpp-python
    # - vLLM
    # - Ollama HTTP API
    #
    # For now, we raise an error so the caller falls back to Tier3.
    raise Tier2Error("Tier2 local model is not implemented yet.")


# ---------------------------------------------------------------------------
# Tier3: Template fallback
# ---------------------------------------------------------------------------

def _call_tier3_templates(
    request: ChatRequest,
    intent: IntentType,
    nav_goal_guess: Optional[str] = None,
) -> str:
    """
    Simple, fully offline fallback generator using templates.

    This is guaranteed to succeed and must be safe in all cases.
    Returns short B1 English sentences.
    """
    text = (request.user_text or "").strip()

    if intent == "STOP":
        return "Okay, I stop here and wait."

    if intent == "FOLLOW":
        return "Okay, I follow you. Please walk in front of me slowly."

    if intent == "NAVIGATE":
        if nav_goal_guess:
            return f"Okay, I will guide you to {nav_goal_guess}. Please follow me."
        else:
            return "I can guide you in the building. Please tell me the room or place name."

    if intent == "STATUS":
        return (
            "I am Robot Savo, a guide robot. Right now I am just waiting here and ready to help."
        )

    # CHATBOT or anything else
    if text:
        return f"You said: {text}. I am Robot Savo, how can I help you more?"
    else:
        return "Hello, I am Robot Savo. How can I help you?"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reply_text(
    request: ChatRequest,
    intent: IntentType,
    nav_goal_guess: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate a reply text for Robot Savo using the 3-tier chain.

    Parameters
    ----------
    request:
        ChatRequest from the /chat endpoint.
    intent:
        High-level intent decided by our deterministic classifier
        (STOP / FOLLOW / NAVIGATE / STATUS / CHATBOT).
    nav_goal_guess:
        Optional raw goal phrase extracted from the text (e.g. "a201", "info desk").
        This is not yet validated against known_locations.json.

    Returns
    -------
    (reply_text, used_tier)
        reply_text: final B1-English sentence for TTS.
        used_tier:  "tier1:<model>" | "tier2" | "tier3"
    """
    # 1) Build messages for LLM-style APIs
    system_prompt = _build_system_prompt(intent)
    user_prompt = _build_user_prompt(request, intent, nav_goal_guess)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 2) Try Tier1 (online) with model priority list
    if settings.tier1_enabled and settings.tier1_api_key:
        model_list = getattr(settings, "tier1_model_candidates", None) or []

        if not model_list:
            logger.warning(
                "Tier1 is enabled and API key is set, but no tier1_model_candidates configured; skipping Tier1."
            )
        else:
            for model_name in model_list:
                try:
                    reply = call_tier1_model(messages, model_name)
                    return reply, f"tier1:{model_name}"
                except Tier1Error as exc:
                    logger.warning("Tier1 model %s failed: %s", model_name, exc)

    # 3) Try Tier2 (local) if enabled
    if settings.tier2_enabled:
        try:
            reply = _call_tier2_local(messages)
            return reply, "tier2"
        except Tier2Error as exc:
            logger.warning("Tier2 failed: %s", exc)

    # 4) Fallback to Tier3 templates (must never fail)
    reply = _call_tier3_templates(request, intent, nav_goal_guess)
    return reply, "tier3"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.generate
    """
    from app.models.chat_request import InputSource

    print("Robot Savo — generate.py self-test\n")

    # Fake request
    req = ChatRequest(
        user_text="Can you take me to info deskS?",
        source=InputSource.KEYBOARD,
        language="en",
        meta={"session_id": "demo-001"},
    )

    reply_text, used_tier = generate_reply_text(
        req,
        intent="NAVIGATE",
        nav_goal_guess="info desk",
    )

    print(f"Used tier : {used_tier}")
    print(f"Reply text: {reply_text}")
