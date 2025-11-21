# app/main.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — FastAPI application entrypoint
------------------------------------------------------
This file wires everything together:

- Creates the FastAPI app
- Adds middleware (CORS for dev)
- Mounts routers (currently: /chat)
- Exposes a simple /health endpoint for monitoring
- Provides an ASGI `app` object for uvicorn
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers.chat import router as chat_router


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
    # For Robot Savo + local dev, allowing all origins is fine.
    # In production you can restrict this list.
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
    # Routers
    # ------------------------------------------------------------------
    # Main chat/navigation endpoint
    app.include_router(chat_router)

    # TODO (later):
    # from app.routers.ws import router as ws_router
    # from app.routers.map import router as map_router
    # from app.routers.status import router as status_router
    # app.include_router(ws_router)
    # app.include_router(map_router)
    # app.include_router(status_router)

    # ------------------------------------------------------------------
    # Health check + root
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

    return app


# ASGI app for uvicorn / gunicorn
app = create_app()


if __name__ == "__main__":
    # Allow `python3 app/main.py` during development.
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,       # auto-reload on code changes (dev only)
    )
