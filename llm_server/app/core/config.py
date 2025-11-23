# app/core/config.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Configuration
-------------------------------------
Central configuration for the LLM server, including:

- app metadata
- API host/port
- filesystem paths (prompts, map_data, logs)
- Tier1 (online via OpenRouter),
- Tier2 (local via Ollama HTTP),
- Tier3 (template fallback),
- basic safety limits.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# This file is: llm_server/app/core/config.py
APP_DIR: Path = Path(__file__).resolve().parents[1]   # .../llm_server/app
ROOT_DIR: Path = APP_DIR.parent                       # .../llm_server

PROMPTS_DIR: Path = APP_DIR / "prompts"
MAP_DATA_DIR: Path = APP_DIR / "map_data"
LOGS_DIR: Path = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Global configuration for the LLM server.

    This class is instantiated once at import time as `settings`
    and used everywhere in the codebase.
    """

    # Tell pydantic-settings where to read .env, and how to behave with extras.
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App / server basics -----------------------------------------------
    app_name: str = "Robot Savo LLM Server"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Filesystem paths ---------------------------------------------------
    # Base directories
    prompts_dir: Path = PROMPTS_DIR
    map_data_dir: Path = MAP_DATA_DIR
    logs_dir: Path = LOGS_DIR

    # Specific map_data files used by prompts + telemetry
    # These are the canonical locations other modules should use.
    nav_state_path: Path = MAP_DATA_DIR / "nav_state.json"
    robot_status_path: Path = MAP_DATA_DIR / "robot_status.json"
    known_locations_path: Path = MAP_DATA_DIR / "known_locations.json"

    # --- Tier toggles -------------------------------------------------------
    tier1_enabled: bool = True   # Online LLM (OpenRouter)
    tier2_enabled: bool = True   # Local LLM (Ollama HTTP)
    tier3_enabled: bool = True   # Template fallback

    # --- Tier1: Online provider (OpenRouter) -------------------------------
    tier1_provider: Literal["openrouter"] = "openrouter"
    tier1_base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # ENV: TIER1_API_KEY=sk-or-v1-...
    tier1_api_key: str | None = Field(
        default=None,
        description="API key for Tier1 online provider (env: TIER1_API_KEY).",
    )

    # Priority-ordered model list for Tier1 (first → last).
    # This is what generate.py uses in the Tier1 loop.
    tier1_model_candidates: list[str] = [
        "x-ai/grok-4.1-fast:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-chat-v3-0324:free",
    ]

    # Timeout (seconds) for Tier1 HTTP calls
    tier1_timeout_s: float = 18.0

    # --- Tier2: Local backend (Ollama HTTP) --------------------------------
    #
    # For Robot Savo right now, Tier2 is Ollama running on the PC/Mac.
    # Configuration comes from:
    #   TIER2_OLLAMA_URL   (e.g. http://localhost:11434/api/chat)
    #   TIER2_OLLAMA_MODEL (e.g. llama3.2:latest)
    #
    tier2_ollama_url: str | None = Field(
        default=None,
        description=(
            "Ollama chat endpoint, e.g. http://localhost:11434/api/chat "
            "(env: TIER2_OLLAMA_URL)."
        ),
    )
    tier2_ollama_model: str | None = Field(
        default=None,
        description=(
            "Ollama model name (env: TIER2_OLLAMA_MODEL), "
            "e.g. llama3.2:latest."
        ),
    )

    # Optional generation parameters for Tier2 (used by providers.tier2_local)
    tier2_temperature: float = 0.7
    tier2_max_tokens: int = 512

    # --- Tier3: Template-based fallback ------------------------------------
    tier3_language: str = "en"
    tier3_enable_status_mode: bool = True

    # --- Safety / limits ----------------------------------------------------
    # These are global logical limits; they can be referenced by safety.py
    # and pipeline.py to clamp outputs and context size.
    max_reply_chars: int = 512       # Hard cap on reply length
    max_history_turns: int = 8       # How many past turns to keep in context


# Single global settings instance used by the rest of the app.
settings = Settings()


if __name__ == "__main__":
    # Minimal self-test so you can quickly verify config loading.
    print("Robot Savo — Settings self-test")
    print(f"ROOT_DIR        : {ROOT_DIR}")
    print(f"APP_DIR         : {APP_DIR}")
    print(f"PROMPTS_DIR     : {settings.prompts_dir}")
    print(f"MAP_DATA_DIR    : {settings.map_data_dir}")
    print(f"LOGS_DIR        : {settings.logs_dir}")
    print(f"NavState path   : {settings.nav_state_path}")
    print(f"RobotStatus path: {settings.robot_status_path}")
    print(f"Locations path  : {settings.known_locations_path}")
    print(f"Environment     : {settings.environment}")
    print(f"Tier1 enabled   : {settings.tier1_enabled}, API key set: {bool(settings.tier1_api_key)}")
    print(f"Tier1 models    : {getattr(settings, 'tier1_model_candidates', [])}")
    print(f"Tier2 enabled   : {settings.tier2_enabled}")
    print(f"Tier2 Ollama    : url={settings.tier2_ollama_url!r}, model={settings.tier2_ollama_model!r}")
    print(f"Tier3 enabled   : {settings.tier3_enabled}, language: {settings.tier3_language}")
