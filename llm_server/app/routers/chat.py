# app/routers/chat.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — /chat router
------------------------------------
This router exposes the main HTTP endpoint that Robot Savo (or any client)
will call.

Flow:
  HTTP POST /chat  (ChatRequest JSON from Pi)
    -> run_pipeline(request, context)
    -> returns ChatResponse JSON (reply_text, intent, nav_goal)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.pipeline import run_pipeline
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse


# `tags` is just for docs (Swagger / ReDoc), makes it grouped nicely.
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat/navigation endpoint for Robot Savo.

    - Input JSON is validated as ChatRequest by Pydantic.
    - We call run_pipeline(), which:
        * classifies intent (STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT)
        * runs Tier1/Tier2/Tier3 generate chain
        * parses the model JSON block
        * returns a ChatResponse model
    - FastAPI then serializes ChatResponse back to JSON for the client.
    """
    # Later this `context` dict will carry:
    #   - nav_state (from nav_state.json)
    #   - robot_status (from robot_status.json)
    #   - known_locations (from known_locations.json)
    # For now it's empty but the function signature is future-proof.
    context = {}

    try:
        response = run_pipeline(request, context=context)
        return response
    except Exception as exc:  # pragma: no cover - defensive guard
        # In development, we want the full stack trace to see the bug.
        if settings.debug:
            raise

        # In production we hide internal details from the client.
        raise HTTPException(
            status_code=500,
            detail="Internal server error in /chat pipeline.",
        ) from exc
