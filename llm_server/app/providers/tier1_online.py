# app/providers/tier1_online.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Tier1 Online Provider (OpenRouter)
----------------------------------------------------------
This module is the ONLY place that knows how to talk to OpenRouter.

Responsibilities:
- Build the HTTP request (URL, headers, JSON payload).
- Handle different models (Grok, LLaMA, DeepSeek, etc.).
- Enable Grok reasoning when appropriate.
- Parse the response and return assistant text.

It is used by app/core/generate.py, which:
- Chooses which model to call (from tier1_model_candidates in config).
- Falls back to Tier2 / Tier3 if this provider raises Tier1Error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class Tier1Error(Exception):
    """Raised when Tier1 (online) fails in a recoverable way."""


def _build_openrouter_payload(
    messages: List[Dict[str, str]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Build the JSON payload for OpenRouter.

    Parameters
    ----------
    messages:
        List of {"role": "system"|"user"|"assistant", "content": "..."} dicts.
    model_name:
        Any OpenRouter model ID, for example:
            - "x-ai/grok-4.1-fast:free"
            - "meta-llama/llama-3.3-70b-instruct:free"
            - "deepseek/deepseek-chat-v3-0324:free"

    Notes
    -----
    - For Grok models, we automatically enable reasoning via:
          extra_body.reasoning.enabled = True
    - Other models just receive the normal chat payload.
    """
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }

    # If we are using a Grok model, enable reasoning like in the OpenRouter docs
    if model_name.startswith("x-ai/grok"):
        payload["extra_body"] = {"reasoning": {"enabled": True}}

    return payload


def call_tier1_model(
    messages: List[Dict[str, str]],
    model_name: str,
) -> str:
    """
    Call a Tier1 online model via OpenRouter and return the assistant's text.

    Parameters
    ----------
    messages:
        List of {"role": "system"|"user"|"assistant", "content": "..."} dicts.
    model_name:
        Which OpenRouter model to use (string).

    Returns
    -------
    content: str
        The assistant's reply (already stripped).

    Raises
    ------
    Tier1Error
        If Tier1 is disabled, misconfigured, or the HTTP/JSON fails.
    """
    if not settings.tier1_enabled:
        raise Tier1Error("Tier1 is disabled in config.")

    api_key = settings.tier1_api_key
    if not api_key:
        raise Tier1Error("Tier1 API key is missing.")

    url = settings.tier1_base_url

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = _build_openrouter_payload(messages, model_name)

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.tier1_timeout_s,
        )
    except requests.RequestException as exc:
        raise Tier1Error(f"Tier1 HTTP error: {exc}") from exc

    if resp.status_code != 200:
        text_preview = resp.text[:200].replace("\n", " ")
        raise Tier1Error(f"Tier1 HTTP {resp.status_code}: {text_preview}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise Tier1Error("Tier1 returned non-JSON response.") from exc

    try:
        # OpenAI/OpenRouter-style: choices[0].message.content
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Tier1Error(
            "Tier1 response JSON missing choices[0].message.content"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise Tier1Error("Tier1 returned empty content.")

    return content.strip()


if __name__ == "__main__":
    """
    Minimal self-test for Tier1 provider.

    This only checks that:
    - config can be loaded
    - API key visibility
    - payload building for Grok

    It does NOT actually call OpenRouter by default to avoid burning tokens.
    Uncomment the live-call section if you want to test with real API.
    """
    print("Robot Savo — tier1_online.py self-test\n")
    print(f"Tier1 enabled: {settings.tier1_enabled}")
    print(f"API key set  : {bool(settings.tier1_api_key)}")
    print(f"Base URL     : {settings.tier1_base_url}")

    example_messages = [
        {"role": "user", "content": "Hello Robot Savo, how are you?"},
    ]
    payload = _build_openrouter_payload(
        example_messages,
        model_name="x-ai/grok-4.1-fast:free",
    )
    print("Example payload keys:", list(payload.keys()))
    print("extra_body for Grok :", payload.get("extra_body"))

    # Live test (CAUTION: uses real API key and tokens!)
    # from pprint import pprint
    # reply = call_tier1_model(example_messages, "x-ai/grok-4.1-fast:free")
    # print("\nLive reply:")
    # pprint(reply)
