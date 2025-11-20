# app/core/config.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Configuration
-------------------------------------
Central configuration for the LLM server, including:

- app metadata
- API host/port
- filesystem paths (prompts, map_data, logs)
- Tier1 (online), Tier2 (local), Tier3 (templates) settings
- basic safety limits

Configuration source order (lowest → highest priority):
1) Defaults in this file
2) Values from .env (at repo root)
3) Environment variables (TIER1_API_KEY, etc.)

NOTE:
- Real secrets go to `.env` (NOT committed).
- `.env.example` is committed, showing which keys are needed.
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

    # --- App / server basics ------------------------------------------------
    app_name: str = "Robot Savo LLM Server"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Filesystem paths ---------------------------------------------------
    prompts_dir: Path = PROMPTS_DIR
    map_data_dir: Path = MAP_DATA_DIR
    logs_dir: Path = LOGS_DIR

    # --- Tier toggles -------------------------------------------------------
    tier1_enabled: bool = True   # Online LLM (OpenRouter, etc.)
    tier2_enabled: bool = True   # Local GGUF LLM
    tier3_enabled: bool = True   # Template fallback

    # --- Tier1: Online provider (e.g. OpenRouter) --------------------------
    tier1_provider: Literal["openrouter"] = "openrouter"
    tier1_base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # ENV: TIER1_API_KEY=sk-...
    tier1_api_key: str | None = Field(
        default=None,
        description="API key for Tier1 online provider (env: TIER1_API_KEY).",
    )

    # Reasoning model (first pass) and reply model (second pass)
    tier1_model_reasoning: str = "deepseek/deepseek-chat"
    tier1_model_reply: str = "meta-llama/llama-3.3-8b-instruct"

    tier1_timeout_s: float = 18.0

    # --- Tier2: Local GGUF LLM (llama-cpp, etc.) ---------------------------
    tier2_model_path: str = "./models/local/chat-model.gguf"
    tier2_n_ctx: int = 4096
    tier2_temperature: float = 0.7
    tier2_max_tokens: int = 512
    tier2_gpu_layers: int = 0  # 0 = CPU only, >0 = use GPU if available

    # --- Tier3: Template-based fallback ------------------------------------
    tier3_language: str = "en"
    tier3_enable_status_mode: bool = True

    # --- Safety / limits ----------------------------------------------------
    max_reply_chars: int = 512       # Hard cap on reply length
    max_history_turns: int = 8       # How many past turns to keep in context


# Single global settings instance used by the rest of the app.
settings = Settings()


if __name__ == "__main__":
    # Minimal self-test so you can quickly verify config loading.
    print("Robot Savo — Settings self-test")
    print(f"ROOT_DIR    : {ROOT_DIR}")
    print(f"PROMPTS_DIR : {settings.prompts_dir}")
    print(f"MAP_DATA_DIR: {settings.map_data_dir}")
    print(f"LOGS_DIR    : {settings.logs_dir}")
    print(f"Environment : {settings.environment}")
    print(f"Tier1 enabled: {settings.tier1_enabled}, API key set: {bool(settings.tier1_api_key)}")
    print(f"Tier2 enabled: {settings.tier2_enabled}, model: {settings.tier2_model_path}")
    print(f"Tier3 enabled: {settings.tier3_enabled}, language: {settings.tier3_language}")
