# app/providers/tier2_local.py
# -*- coding: utf-8 -*-
"""

Robot Savo LLM Server — Tier2 Local Provider (Ollama)

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class Tier2Error(Exception):
    """Raised when Tier2 (local) fails in a recoverable way."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_tier2_model(messages: List[Dict[str, str]]) -> str:
    """
    Entry point for Tier2 local models.

    Parameters
    ----------
    messages:
        List of {"role": "system"|"user"|"assistant", "content": "..."} dicts.

    Behavior
    --------
    - Checks if Tier2 is enabled in config.
    - Uses Ollama as the Tier2 backend (configured via Settings).
    - If Ollama is not configured or the HTTP/JSON fails, raises
      Tier2Error so that the caller (generate.py) can fall back
      to Tier3 templates.

    Returns
    -------
    content: str
        The assistant's reply (already stripped).

    Raises
    ------
    Tier2Error
        If Tier2 is disabled, misconfigured, or the HTTP/JSON fails.
    """
    if not settings.tier2_enabled:
        raise Tier2Error("Tier2 is disabled in config.")

    return _call_ollama(messages)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _call_ollama(messages: List[Dict[str, str]]) -> str:
    """
    Call a local Ollama model via HTTP.

    Expected config (from app/core/config.Settings):
        settings.tier2_ollama_url   e.g. "http://localhost:11434/api/chat"
        settings.tier2_ollama_model e.g. "llama3.2:latest"
    """
    base_url = settings.tier2_ollama_url
    model = settings.tier2_ollama_model

    if not base_url or not model:
        raise Tier2Error(
            "Tier2 (Ollama) is not configured. "
            "Set TIER2_OLLAMA_URL and TIER2_OLLAMA_MODEL in your .env "
            "or disable Tier2."
        )

    # Ollama /api/chat expects OpenAI-style messages and usually streams;
    # we force stream=false so we get a single JSON object.
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Optionally you could add temperature / num_predict etc. here,
        # mapped from settings.tier2_temperature / tier2_max_tokens.
    }

    try:
        resp = requests.post(base_url, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise Tier2Error(f"Ollama HTTP error: {exc}") from exc

    if resp.status_code != 200:
        text_preview = resp.text[:200].replace("\n", " ")
        raise Tier2Error(f"Ollama HTTP {resp.status_code}: {text_preview}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise Tier2Error("Ollama returned non-JSON response.") from exc

    # Typical /api/chat (stream=false) format:
    # {
    #   "model": "...",
    #   "created_at": "...",
    #   "message": {"role": "assistant", "content": "..."},
    #   "done": true,
    #   ...
    # }
    message = data.get("message") or {}
    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise Tier2Error("Ollama returned empty content.")

    return content.strip()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.providers.tier2_local

    This will:
    - Print whether Tier2 is enabled in config.
    - Show settings.tier2_ollama_url / settings.tier2_ollama_model.
    - (Optional) you can uncomment the live call once Ollama is running.
    """
    print("Robot Savo — tier2_local.py self-test\n")
    print(f"Tier2 enabled              : {settings.tier2_enabled}")
    print(f"settings.tier2_ollama_url  : {settings.tier2_ollama_url!r}")
    print(f"settings.tier2_ollama_model: {settings.tier2_ollama_model!r}")

    # Example messages for future live test
    example_messages = [
        {"role": "system", "content": "You are a polite helper."},
        {"role": "user", "content": "Hello from Robot Savo Tier2 test."},
    ]

    # CAUTION: uncomment only after Ollama is running AND config is set
    # from pprint import pprint
    # try:
    #     reply = call_tier2_model(example_messages)
    #     print("\nLive Tier2 reply:")
    #     pprint(reply)
    # except Tier2Error as exc:
    #     print("\nTier2Error:", exc)
