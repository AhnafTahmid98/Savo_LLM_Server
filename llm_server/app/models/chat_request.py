# app/models/chat_request.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — ChatRequest model
-----------------------------------------
Canonical request payload for the /chat endpoint.

This model is shared by all tiers (online, local, tier3). It carries:
- the raw user text
- minimal context about where it came from
- an explicit session_id for conversation tracking
- optional metadata for logging / routing
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, constr


class InputSource(str, Enum):
    """Where the text originally came from."""

    MIC = "mic"            # Robot microphone → STT
    KEYBOARD = "keyboard"  # Manual developer / operator input
    SYSTEM = "system"      # Internal/system-generated messages
    TEST = "test"          # Automated tests and health checks


class ChatRequest(BaseModel):
    """
    Canonical request body for /chat.

    Fields
    ------
    user_text:
        User utterance in plain text. This is already decoded from audio
        if it came from microphone + speech-to-text.
    source:
        Origin of the text. Helps with logging and behavior tuning.
    language:
        BCP-47 language code for the text, e.g. "en", "fi".
        For Robot Savo we mainly use "en" now, but it is future-proof.
    session_id:
        Stable identifier for this conversation. The Pi should generate
        and reuse the same session_id while talking to the same human,
        and start a new session_id when a new user arrives.
        If omitted, the pipeline may fall back to a default strategy
        (e.g. per-connection or one-shot).
    meta:
        Free-form metadata bag (client name, debug flags, environment info).
        This can be used by any tier for logging or routing decisions.
    """

    user_text: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description="User utterance in plain text.",
        example="Hello, can you help me?",
    )
    source: InputSource = Field(
        default=InputSource.MIC,
        description="Origin of the text (mic, keyboard, system, test).",
        example="mic",
    )
    language: Optional[str] = Field(
        default="en",
        description="BCP-47 language code of user_text (e.g. 'en', 'fi').",
        example="en",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable identifier for this conversation/session. "
            "The Pi should generate and reuse this while talking "
            "to the same human, and start a new one for a new user."
        ),
        example="robot-savo-session-001",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (client_id, flags, environment info, etc.).",
        example={"client": "pi5", "build": "2025-11-23"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_text": "Hello Robot Savo, how are you?",
                    "source": "mic",
                    "language": "en",
                    "session_id": "robot-savo-session-001",
                    "meta": {"client": "pi5", "env": "lab"},
                },
                {
                    "user_text": "Can you guide me to A201?",
                    "source": "keyboard",
                    "language": "en",
                    "session_id": "dev-session-001",
                    "meta": {"client": "dev-shell"},
                },
            ]
        }
    }


if __name__ == "__main__":
    # Minimal self-test so you can quickly verify the model works.
    print("Robot Savo — ChatRequest self-test")

    sample = ChatRequest(
        user_text="Hello Robot Savo, how are you?",
        source=InputSource.TEST,
        language="en",
        session_id="demo-session-001",
        meta={"client": "dev-shell", "env": "local"},
    )

    print("\nPython representation:")
    print(sample)

    print("\nAs JSON:")
    # Support both Pydantic v2 and v1
    try:
        # Pydantic v2
        print(sample.model_dump_json(indent=2))
    except AttributeError:
        # Pydantic v1
        print(sample.json(indent=2))
