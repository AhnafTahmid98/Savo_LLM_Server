# app/routers/ws.py
# -*- coding: utf-8 -*-
"""
Robot Savo — WebSocket router
-----------------------------
WebSocket endpoints for:

- /ws/chat
    Persistent LLM chat with Robot Savo using the existing pipeline
    (ChatRequest -> run_pipeline -> ChatResponse).

- /ws/telemetry
    High-rate, low-overhead telemetry channel for:
      - navigation state (NavState)
      - robot status / health (RobotStatus)

Design goals
------------
- Keep the protocol simple and JSON-based.
- Re-use existing Pydantic models + pipeline.
- Be robust against malformed messages (never crash the server on bad input).
- Make it easy to test with a simple Python client or tools like websocat.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.pipeline import run_pipeline
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse
from app.models.nav_state_model import NavState
from app.models.robot_status_model import RobotStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _send_error(
    websocket: WebSocket,
    code: str,
    message: str,
    details: Any | None = None,
) -> None:
    """Send a structured error frame to the client."""
    payload: Dict[str, Any] = {
        "type": "error",
        "code": code,
        "message": message,
    }
    if details is not None:
        payload["details"] = details
    try:
        await websocket.send_json(payload)
    except Exception:  # noqa: BLE001
        # If we can't even send the error, just ignore.
        logger.debug("Failed to send error frame over WebSocket", exc_info=True)


# ---------------------------------------------------------------------------
# /ws/chat
# ---------------------------------------------------------------------------


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for LLM chat.

    - Accepts ChatRequest-shaped JSON frames from the client.
    - Runs the standard pipeline (run_pipeline).
    - Sends back ChatResponse-shaped JSON frames.

    This version is non-streaming (one request -> one response frame).
    Later we can extend this to stream partial tokens if needed.

    Session behaviour:
    - The client (Pi) is responsible for sending a stable `session_id`
      in the ChatRequest (field or meta["session_id"]) while talking to
      the same human.
    - The pipeline then uses runtime_state.session_store to preserve
      conversation history between frames.
    """
    await websocket.accept()
    logger.info("WebSocket /ws/chat connected")

    try:
        while True:
            raw = await websocket.receive_json()
            logger.debug("WS /ws/chat received: %r", raw)

            try:
                chat_req = ChatRequest(**raw)
            except ValidationError as exc:
                logger.warning("Invalid ChatRequest over WS: %s", exc)
                await _send_error(
                    websocket,
                    code="invalid_chat_request",
                    message="Payload does not match ChatRequest schema.",
                    details=exc.errors(),
                )
                # Keep connection open; allow client to retry.
                continue

            # Resolve session_id for logging (pipeline also resolves internally).
            session_id = chat_req.session_id
            if session_id is None and isinstance(chat_req.meta, dict):
                session_id = chat_req.meta.get("session_id")

            logger.info(
                "WS /ws/chat frame: source=%s session_id=%s text=%r",
                chat_req.source.value,
                session_id,
                chat_req.user_text,
            )

            try:
                # run_pipeline is async and returns ChatResponse.
                chat_resp: ChatResponse = await run_pipeline(chat_req)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline failure in WS /ws/chat: %s", exc)
                await _send_error(
                    websocket,
                    code="pipeline_error",
                    message="Internal error while generating response.",
                )
                # Close with server error code; client can reconnect.
                await websocket.close(code=1011)
                return

            # Log the outcome for debugging and telemetry.
            try:
                logger.info(
                    "WS /ws/chat response: intent=%s nav_goal=%r tier=%s session_id=%s",
                    chat_resp.intent.value,
                    chat_resp.nav_goal,
                    chat_resp.tier_used,
                    chat_resp.session_id,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to log WS /ws/chat response details", exc_info=True)

            # Convert ChatResponse to plain dict, excluding None fields.
            try:
                payload = chat_resp.model_dump(mode="json", exclude_none=True)
            except AttributeError:
                # Pydantic v1 fallback
                payload = chat_resp.dict(exclude_none=True)

            await websocket.send_json(payload)

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/chat disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in WS /ws/chat: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# /ws/telemetry
# ---------------------------------------------------------------------------


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for Robot Savo telemetry.

    Expected message format (client -> server):
    -------------------------------------------
    {
      "type": "status" | "navstate" | "ping" | ...,
      "payload": { ... }   # depends on type
    }

    Known types:
      - "status"   -> RobotStatus (writes robot_status.json)
      - "navstate" -> NavState    (writes nav_state.json)
      - "ping"     -> simple ping/pong connectivity check
    """
    await websocket.accept()
    logger.info("WebSocket /ws/telemetry connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            payload = data.get("payload")

            if not msg_type:
                await _send_error(
                    websocket,
                    code="missing_type",
                    message='Telemetry frame must include a "type" field.',
                )
                continue

            msg_type_lower = str(msg_type).lower().strip()

            # PING ----------------------------------------------------------------
            if msg_type_lower == "ping":
                # Simple ping/pong: no validation, just echo.
                ack = {
                    "type": "telemetry_ack",
                    "kind": "ping",
                    "ok": True,
                }
                await websocket.send_json(ack)
                continue

            # For status/navstate frames, we require a payload
            if payload is None:
                await _send_error(
                    websocket,
                    code="missing_payload",
                    message='Telemetry frame must include a "payload" object.',
                )
                continue

            # STATUS --------------------------------------------------------------
            if msg_type_lower == "status":
                try:
                    status = RobotStatus(**payload)
                except ValidationError as exc:
                    logger.warning("Invalid RobotStatus payload over WS: %s", exc)
                    await _send_error(
                        websocket,
                        code="invalid_status_payload",
                        message="Payload does not match RobotStatus schema.",
                        details=exc.errors(),
                    )
                    continue

                try:
                    path = status.save()
                    logger.debug("Saved RobotStatus to %s", path)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to save RobotStatus: %s", exc)
                    await _send_error(
                        websocket,
                        code="status_save_error",
                        message="Failed to persist RobotStatus on server.",
                    )
                    continue

                ack = {
                    "type": "telemetry_ack",
                    "kind": "status",
                    "ok": True,
                }
                await websocket.send_json(ack)
                continue

            # NAVSTATE ------------------------------------------------------------
            if msg_type_lower == "navstate":
                try:
                    nav_state = NavState(**payload)
                except ValidationError as exc:
                    logger.warning("Invalid NavState payload over WS: %s", exc)
                    await _send_error(
                        websocket,
                        code="invalid_navstate_payload",
                        message="Payload does not match NavState schema.",
                        details=exc.errors(),
                    )
                    continue

                try:
                    path = nav_state.save()
                    logger.debug("Saved NavState to %s", path)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to save NavState: %s", exc)
                    await _send_error(
                        websocket,
                        code="navstate_save_error",
                        message="Failed to persist NavState on server.",
                    )
                    continue

                ack = {
                    "type": "telemetry_ack",
                    "kind": "navstate",
                    "ok": True,
                }
                await websocket.send_json(ack)
                continue

            # UNKNOWN TYPE -------------------------------------------------------
            logger.warning("Unknown telemetry type received: %r", msg_type)
            await _send_error(
                websocket,
                code="unknown_type",
                message=f'Unknown telemetry type: {msg_type!r}',
            )

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/telemetry disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in WS /ws/telemetry: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass
