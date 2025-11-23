# app/routers/chat.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — /chat router
------------------------------------
This router exposes the main HTTP endpoint that Robot Savo (or any client)
will call.

Flow:
  HTTP POST /chat  (ChatRequest JSON from Pi)
    -> run_pipeline(ChatRequest)
       - resolves session_id and loads conversation history
       - sanitizes user text
       - classifies intent (STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT)
       - loads NavState, RobotStatus, KnownLocations from app/map_data
       - attaches live web context (weather / time / BTC) when needed
       - calls Tier1/Tier2/Tier3 via generate.py
       - parses the model JSON block
       - updates runtime session state
    -> returns ChatResponse JSON (reply_text, intent, nav_goal, session_id, tier_used)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.pipeline import run_pipeline
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse

# `tags` is just for docs (Swagger / ReDoc), makes it grouped nicely.
router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat/navigation endpoint for Robot Savo.

    - Input JSON is validated as ChatRequest by Pydantic.
    - We call run_pipeline(), which:
        * resolves session_id and loads conversation history
        * sanitizes user_text
        * classifies intent (STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT)
        * enriches request.meta with nav_state / robot_status / locations
        * attaches live web context (weather / time / BTC) when needed
        * runs Tier1/Tier2/Tier3 generate chain
        * parses the model JSON block
        * updates runtime session state
        * returns a ChatResponse model
    - FastAPI then serializes ChatResponse back to JSON for the client.
    """
    # Prefer explicit field; fall back to legacy meta["session_id"]; else None.
    session_id = request.session_id
    if session_id is None and isinstance(request.meta, dict):
        session_id = request.meta.get("session_id")

    logger.info(
        "[/chat] source=%s session_id=%s text=%r",
        request.source.value,
        session_id,
        request.user_text,
    )

    try:
        response = await run_pipeline(request)
        logger.info(
            "[/chat] intent=%s nav_goal=%r tier=%s session_id=%s",
            response.intent.value,
            response.nav_goal,
            response.tier_used,
            response.session_id,
        )
        return response
    except Exception as exc:  # pragma: no cover - defensive guard
        # In development, we want the full stack trace to see the bug.
        logger.exception("Unhandled exception in /chat endpoint")
        if settings.debug:
            raise

        # In production we hide internal details from the client.
        raise HTTPException(
            status_code=500,
            detail="Internal server error in /chat pipeline.",
        ) from exc
