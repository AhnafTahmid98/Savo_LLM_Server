# app/routers/status.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — /status router
--------------------------------------
Read-only status endpoints for:

- Current navigation state   (NavState snapshot)
- Current robot health       (RobotStatus snapshot)
- Combined server snapshot   (LLM tiers + known locations + telemetry)

Data sources:
    app/map_data/nav_state.json
    app/map_data/robot_status.json
    app/map_data/known_locations.json

These are the same JSON snapshots that the LLM pipeline uses
to answer STATUS / NAVIGATION questions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.map_lookup import get_known_locations
from app.models.nav_state_model import NavState
from app.models.robot_status_model import RobotStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["status"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_load_nav_state() -> Dict[str, Any]:
    """
    Load NavState snapshot from nav_state.json.

    If the file does not exist yet, return an IDLE default so that
    callers still get a valid JSON object.
    """
    try:
        nav_state = NavState.load()
    except FileNotFoundError:
        logger.info("NavState file not found; returning idle default.")
        nav_state = NavState.idle(note="No nav_state.json found yet.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load NavState: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to load NavState snapshot.",
        ) from exc

    return nav_state.to_json_dict()


def _safe_load_robot_status() -> Dict[str, Any]:
    """
    Load RobotStatus snapshot from robot_status.json.

    If the file does not exist yet, return a default RobotStatus so that
    callers always get a valid JSON object.
    """
    try:
        status = RobotStatus.load()
    except FileNotFoundError:
        logger.info("RobotStatus file not found; returning default status.")
        status = RobotStatus()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load RobotStatus: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to load RobotStatus snapshot.",
        ) from exc

    return status.to_json_dict()


def _safe_load_known_locations() -> List[Dict[str, Any]]:
    """
    Load known locations (canonical campus map) for debugging / UI.

    Returns a list of dicts:
        [
          {"name": "A201", "display_name": "Room A201 (Lab)", ...},
          ...
        ]
    """
    try:
        locations = get_known_locations()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load known locations: %s", exc)
        # For status, better to degrade gracefully instead of hard error.
        return []

    result: List[Dict[str, Any]] = []
    for name in locations.list_canonical_names():
        loc = locations.get(name)
        if not loc:
            continue

        result.append(
            {
                "name": name,
                "display_name": loc.display_name,
                "type": getattr(loc.type, "value", str(loc.type)),
                "building": loc.building,
                "floor": loc.floor,
                "tags": loc.tags,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/nav", summary="Current navigation state")
async def get_nav_status() -> Dict[str, Any]:
    """
    Return the current NavState snapshot as JSON.

    This is the same data the LLM pipeline uses to explain:
    - what the robot is doing,
    - how far it is from the goal,
    - whether a safety stop or estop is active.
    """
    return _safe_load_nav_state()


@router.get("/robot", summary="Current robot health/status")
async def get_robot_status() -> Dict[str, Any]:
    """
    Return the current RobotStatus snapshot as JSON.

    Typical fields include:
    - battery level / state
    - temperature
    - uptime
    - optional notes / error flags
    """
    return _safe_load_robot_status()


@router.get("/all", summary="Combined LLM + robot status snapshot")
async def get_full_status() -> Dict[str, Any]:
    """
    Return a combined snapshot of:

    - Server / LLM tier configuration
    - Current NavState
    - Current RobotStatus
    - Known locations (canonical campus map summary)

    Useful for a simple operator dashboard or rich health checks.
    """
    nav_state = _safe_load_nav_state()
    robot_status = _safe_load_robot_status()
    locations = _safe_load_known_locations()

    server_info: Dict[str, Any] = {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "tier1_enabled": settings.tier1_enabled,
        "tier2_enabled": settings.tier2_enabled,
        "tier3_enabled": settings.tier3_enabled,
        "tier1_models": getattr(settings, "tier1_model_candidates", []),
        "tier2_ollama_url": settings.tier2_ollama_url,
        "tier2_ollama_model": settings.tier2_ollama_model,
    }

    return {
        "server": server_info,
        "nav_state": nav_state,
        "robot_status": robot_status,
        "known_locations": locations,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test (no HTTP).

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.routers.status
    """
    print("Robot Savo — status.py (router) self-test\n")

    nav = _safe_load_nav_state()
    print("NavState snapshot:")
    print(nav)

    robot = _safe_load_robot_status()
    print("\nRobotStatus snapshot:")
    print(robot)

    locs = _safe_load_known_locations()
    print("\nKnown locations (summary):")
    print(locs if locs else "(none loaded)")
