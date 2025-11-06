"""
config.py
---------
Central runtime configuration for the Robot Savo LLM server.

Why this file exists:
- We don't want secrets (API keys) or machine-specific paths hardcoded
  all over the code.
- We want the exact same codebase to run on:
    - Your dev PC (Ubuntu)
    - Your MacBook
    - The Raspberry Pi 5
  ...with only .env differences, not code differences.

How it works:
- We read environment variables (from the real environment or from a .env file).
- We fill in safe defaults if env vars are missing.
- We expose a single `settings` object that the rest of the app imports.

Example usage in other modules:
    from app.core.config import settings

    if settings.ALLOW_TIER1_ONLINE:
        call_openrouter_api(key=settings.OPENROUTER_API_KEY, ...)

Key responsibilities in this config:
1. Server identity / mode
2. Tier1 (online model) config
3. Tier2 (local GGUF model on PC) config
4. Tier3 (template fallback on Pi)
5. Paths to robot state JSON files
6. Safety limits
"""

from pydantic import BaseModel
from pathlib import Path
import os


class Settings(BaseModel):
    # -------------------------------------------------
    # 1. General server identity / mode
    # -------------------------------------------------
    # Human-readable name for logs / responses.
    ROBOT_NAME: str = os.getenv("ROBOT_NAME", "Robot Savo")

    # Language mode. Right now we are English-only.
    LANGUAGE: str = os.getenv("LANGUAGE", "en")

    # Environment tag. Can be "dev", "prod", "demo_offline", etc.
    # This is not security. It's for logging and behavior choices.
    ENV: str = os.getenv("SAVO_ENV", "dev")

    # -------------------------------------------------
    # 2. Tier1: Online reasoning (OpenRouter, internet)
    #
    # Tier1 does the "smart LLM" step:
    # - Deep reasoning model (ex: DeepSeek V3.1)
    # - Then a "final answer" model (ex: Llama 3.3 8B Instruct)
    #
    # Tier1 can also browse for live info if allowed.
    # If Tier1 fails (timeout, no internet, or disabled), we fall back to Tier2.
    # -------------------------------------------------

    # API key for OpenRouter. This SHOULD NOT go to GitHub.
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # Which models we ask in Tier1:
    # "reasoning" model = do thinking / tool-use / intent planning
    TIER1_REASONING_MODEL: str = os.getenv(
        "TIER1_REASONING_MODEL",
        "deepseek/deepseek-chat-v3.1"
    )

    # "reply" model = final surface-level answer voice style
    TIER1_REPLY_MODEL: str = os.getenv(
        "TIER1_REPLY_MODEL",
        "meta-llama/llama-3.3-8b-instruct"
    )

    # Timeout budget for Tier1 online call in seconds.
    # If we don't get answer fast enough, we move to Tier2 to stay responsive.
    TIER1_TIMEOUT_S: float = float(os.getenv("TIER1_TIMEOUT_S", "1.8"))

    # Master switch: allow or block Tier1 completely.
    # For example in a no-internet demo, set ALLOW_TIER1_ONLINE=false
    ALLOW_TIER1_ONLINE: bool = os.getenv(
        "ALLOW_TIER1_ONLINE", "true"
    ).lower() == "true"

    # Can Tier1 access web to answer "live" questions?
    # true  = robot can say current weather / news (from Tier1 with web)
    # false = robot will say "I cannot check live info now."
    WEB_ACCESS_ALLOWED: bool = os.getenv(
        "WEB_ACCESS_ALLOWED", "true"
    ).lower() == "true"

    # -------------------------------------------------
    # 3. Tier2: Local model (llama-cpp on your laptop/PC)
    #
    # Tier2 runs offline on your own machine (no internet needed).
    # This is the backup brain when Tier1 fails or is disabled.
    #
    # We'll load a GGUF model with llama-cpp-python.
    # Tier2 MUST still output valid structured JSON so the robot can act.
    # -------------------------------------------------

    # Path to your local GGUF model for llama-cpp.
    # Can be relative (./models/whatever.gguf) or absolute (/home/you/...).
    TIER2_MODEL_PATH: str = os.getenv(
        "TIER2_MODEL_PATH",
        "./models/local_model.gguf"
    )

    # How many CPU threads llama-cpp may use on your machine.
    # Tune this per device (desktop vs MacBook Air).
    TIER2_THREADS: int = int(os.getenv("TIER2_THREADS", "6"))

    # Master switch for Tier2. On Pi you could set this false if Pi is too weak
    # and you want to skip Tier2 and jump directly to Tier3 templates.
    ALLOW_TIER2_LOCAL: bool = os.getenv(
        "ALLOW_TIER2_LOCAL", "true"
    ).lower() == "true"

    # -------------------------------------------------
    # 4. Tier3: Template fallback (runs even on the Pi)
    #
    # Tier3 does not call any LLM.
    # It builds a safe reply from:
    #   - nav_state.json
    #   - robot_status.json
    #   - known_locations.json
    #
    # Tier3 always exists. It's our "last resort"
    # so the robot can *always* answer and guide.
    # -------------------------------------------------
    # No special env vars here yet. Tier3 uses the paths below.

    # -------------------------------------------------
    # 5. Robot state storage (JSON snapshots)
    #
    # The Pi will POST /map/status and /map/navstate.
    # We save that info into JSON files on disk so the LLM server
    # and Tier3 can read it.
    #
    # By default we store them in llm_server/app/map_data/.
    # On the real robot we could move these paths somewhere like /var/lib/robot_savo/
    # without touching code, just changing .env
    # -------------------------------------------------

    # Base directory for map/status data.
    MAP_DATA_DIR: Path = Path(
        os.getenv(
            "MAP_DATA_DIR",
            # default: llm_server/app/map_data
            str(Path(__file__).resolve().parents[1] / "map_data")
        )
    )

    NAV_STATE_PATH: Path = MAP_DATA_DIR / "nav_state.json"
    ROBOT_STATUS_PATH: Path = MAP_DATA_DIR / "robot_status.json"
    KNOWN_LOCATIONS_PATH: Path = MAP_DATA_DIR / "known_locations.json"

    # -------------------------------------------------
    # 6. Safety / parsing knobs
    #
    # These are not "security", but they protect against garbage input
    # and keep responses short and consistent for speech synthesis.
    # -------------------------------------------------

    # If STT returns a giant paragraph (maybe background noise),
    # we do not trust that as a motion command. We'll treat it like CHATBOT.
    MAX_USER_TEXT_LEN: int = int(os.getenv("MAX_USER_TEXT_LEN", "200"))

    # When we extract nav_goal ("take me to A201 please"),
    # we only keep at most this many words in the location.
    # Example: "the big classroom near stairs"
    # -> cut to first 4 words so we don't send essays as a goal.
    MAX_NAV_GOAL_WORDS: int = int(os.getenv("MAX_NAV_GOAL_WORDS", "4"))


# Create ONE shared settings object for the whole app.
# Everyone imports this instead of calling os.getenv() everywhere.
settings = Settings()
