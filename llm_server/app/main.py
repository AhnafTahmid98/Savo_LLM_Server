# app/main.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — FastAPI application entrypoint
------------------------------------------------------
This file wires everything together:

- Sets up central logging (so you see pipeline tier logs in the console).
- Creates the FastAPI app.
- Adds middleware (CORS for dev).
- Mounts routers:
    * /chat          (HTTP)     → main LLM endpoint for Robot Savo
    * /map/*         (HTTP)     → Pi pushes NavState / RobotStatus snapshots
    * /status/*      (HTTP)     → read-only server + robot status views
    * /ws/chat       (WebSocket)→ dev / console chat (same pipeline as /chat)
    * /ws/telemetry  (WebSocket)→ fast telemetry channel from Pi
- Exposes an ASGI `app` object for uvicorn.

Typical run command (dev):

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers.chat import router as chat_router
from app.routers.map import router as map_router
from app.routers.status import router as status_router
from app.routers.ws import router as ws_router
from app.utils import setup_logging, get_logger


# ---------------------------------------------------------------------------
# Global logging config
# ---------------------------------------------------------------------------
# We configure logging once here so that:
# - app.core.pipeline INFO logs (used_tier, backend) are visible.
# - tools_web, map_lookup, runtime_state, etc. can also log useful messages.
#
# setup_logging() honours settings.debug, so in development you get DEBUG,
# and in production you can keep it quieter.
# ---------------------------------------------------------------------------
setup_logging(debug=settings.debug)
logger = get_logger(__name__)
logger.info(
    "Robot Savo LLM server starting (env=%s, tier1_enabled=%s, tier2_enabled=%s, tier3_enabled=%s)",
    settings.environment,
    settings.tier1_enabled,
    settings.tier2_enabled,
    settings.tier3_enabled,
)


def create_app() -> FastAPI:
    """
    Application factory.

    Returns a configured FastAPI instance ready for uvicorn.
    """
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------
    # CORS (mainly useful if you ever call this API from a browser UI)
    #
    # For Robot Savo + local dev, allowing all origins is fine.
    # In production you can restrict this list to known frontends.
    # ------------------------------------------------------------------
    if settings.environment != "production":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Routers (HTTP + WebSocket)
    # ------------------------------------------------------------------
    # Main chat/navigation endpoint (Pi will call this via HTTP POST /chat)
    app.include_router(chat_router)

    # Telemetry snapshots via HTTP:
    #   POST /map/navstate        -> app/map_data/nav_state.json
    #   POST /map/status          -> app/map_data/robot_status.json
    #   GET  /map/known_locations -> known_locations table
    app.include_router(map_router)

    # Read-only status views:
    #   GET /status/nav
    #   GET /status/robot
    #   GET /status/all
    app.include_router(status_router)

    # WebSocket endpoints:
    #   /ws/chat       -> dev console / future streaming chat
    #   /ws/telemetry  -> fast NavState/RobotStatus from Pi
    app.include_router(ws_router)

    # ------------------------------------------------------------------
    # Meta / health endpoints
    # ------------------------------------------------------------------

    @app.get("/", tags=["meta"])
    async def root():
        """
        Simple root endpoint so you can quickly see the server is alive.
        """
        return {
            "name": settings.app_name,
            "environment": settings.environment,
            "message": "Robot Savo LLM server is running.",
        }

    @app.get("/health", tags=["meta"])
    async def health_check():
        """
        Lightweight health check for Pi / monitoring scripts.
        """
        return {
            "status": "ok",
            "environment": settings.environment,
            "debug": settings.debug,
            "tier1_enabled": settings.tier1_enabled,
            "tier2_enabled": settings.tier2_enabled,
            "tier3_enabled": settings.tier3_enabled,
        }

    logger.info("FastAPI app created (env=%s)", settings.environment)
    return app


# ASGI app for uvicorn / gunicorn
app = create_app()


if __name__ == "__main__":
    """
    Allow `python3 -m app.main` during development.

    In production you normally use:

        uvicorn app.main:app --host 0.0.0.0 --port 8000
    """
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment != "production"),  # auto-reload only in non-prod
    )
