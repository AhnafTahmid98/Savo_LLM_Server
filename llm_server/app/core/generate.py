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
    2) Tier2: Local LLM (Ollama, via providers.tier2_local)
    3) Tier3: Template fallback (fully offline, via providers.tier3_pi)

It does NOT parse the JSON block from the model — that is handled by
app/core/pipeline.py.

Public API:
    generate_reply_text(request, intent, nav_goal_guess) -> ModelCallResult
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.intent import IntentType
from app.core.types import ModelCallResult, TierLabel
from app.models.chat_request import ChatRequest
from app.providers.tier1_online import call_tier1_model, Tier1Error
from app.providers.tier2_local import call_tier2_model, Tier2Error
from app.providers.tier3_pi import call_tier3_fallback

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
    """
    lines: List[str] = [
        f"USER_TEXT: {request.user_text}",
        f"INTENT_HINT: {intent}",
    ]
    if nav_goal_guess:
        lines.append(f"NAV_GOAL_GUESS: {nav_goal_guess}")

    if request.source:
        lines.append(f"SOURCE: {request.source}")
    if request.language:
        lines.append(f"LANGUAGE_HINT: {request.language}")

    if request.meta:
        try:
            meta_json = json.dumps(request.meta, ensure_ascii=False)
            lines.append(f"META: {meta_json}")
        except TypeError:
            logger.warning("ChatRequest.meta is not JSON-serializable; ignoring.")

    return "\n".join(lines)


def _make_result(
    text: str,
    tier: TierLabel,
    raw: Optional[Dict[str, Any]] = None,
) -> ModelCallResult:
    """
    Helper to construct ModelCallResult in one place.
    """
    return ModelCallResult(text=text, used_tier=tier, raw=raw or {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reply_text(
    request: ChatRequest,
    intent: IntentType,
    nav_goal_guess: Optional[str] = None,
) -> ModelCallResult:
    """
    Generate a reply text for Robot Savo using the 3-tier chain.

    Returns a ModelCallResult:
      - text      : full model output (spoken part + JSON block at the end)
      - used_tier : "tier1" | "tier2" | "tier3"
      - raw       : backend metadata (model name, URL, provider)
    """
    # 1) Build messages
    system_prompt = _build_system_prompt(intent)
    user_prompt = _build_user_prompt(request, intent, nav_goal_guess)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 2) Tier1 (OpenRouter) with priority model list
    if settings.tier1_enabled and settings.tier1_api_key:
        model_list = getattr(settings, "tier1_model_candidates", None) or []

        if not model_list:
            logger.warning(
                "Tier1 is enabled and API key is set, but no tier1_model_candidates configured; skipping Tier1."
            )
        else:
            for model_name in model_list:
                try:
                    reply_text = call_tier1_model(messages, model_name)
                    return _make_result(
                        text=reply_text,
                        tier="tier1",
                        raw={"backend": "openrouter", "model": model_name},
                    )
                except Tier1Error as exc:
                    logger.warning("Tier1 model %s failed: %s", model_name, exc)

    # 3) Tier2 (local Ollama)
    if settings.tier2_enabled:
        try:
            reply_text = call_tier2_model(messages)
            return _make_result(
                text=reply_text,
                tier="tier2",
                raw={
                    "backend": "ollama",
                    "url": settings.tier2_ollama_url,
                    "model": settings.tier2_ollama_model,
                },
            )
        except Tier2Error as exc:
            logger.warning("Tier2 failed: %s", exc)

    # 4) Tier3 (templates)
    reply_text = call_tier3_fallback(request, intent, nav_goal_guess)
    return _make_result(
        text=reply_text,
        tier="tier3",
        raw={"backend": "templates"},
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.generate
    """
    from app.models.chat_request import InputSource

    print("Robot Savo — generate.py self-test\n")

    req = ChatRequest(
        user_text="Can you take me to info deskS?",
        source=InputSource.KEYBOARD,
        language="en",
        meta={"session_id": "demo-001"},
    )

    # IntentType is a Literal alias, so we pass the string:
    result = generate_reply_text(
        req,
        intent="NAVIGATE",
        nav_goal_guess="Info Desk",
    )

    print(f"Used tier : {result.used_tier}")
    print(f"Backend   : {result.raw}")
    print(f"Text      : {result.text}")
