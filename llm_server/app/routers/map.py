# app/routers/map.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — /map router
-----------------------------------
HTTP endpoints that the *robot (Pi)* and tools can call to:

- Push live navigation and status data into the LLM server.
- Query static map-related data such as known locations.

Endpoints
---------
POST /map/navstate
    - Body: NavState JSON (current pose, goal, state, etc.)
    - Effect: write app/map_data/nav_state.json

POST /map/status
    - Body: RobotStatus JSON (battery, temps, errors, etc.)
    - Effect: write app/map_data/robot_status.json

GET /map/known_locations
    - Response: JSON view of known_locations.json as used by the LLM server.
    - Used by the Pi or tools to see which canonical locations exist.

Later, app/core/map_lookup.py reads these JSON snapshots so that
pipeline.py and the LLM can answer STATUS and NAVIGATION questions
with real data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.map_lookup import get_known_locations
from app.models.nav_state_model import NavState
from app.models.robot_status_model import RobotStatus
from app.utils import write_json_atomic

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/map", tags=["map"])

# Paths must match map_lookup.py
MAP_DIR: Path = settings.map_data_dir
NAV_STATE_PATH: Path = MAP_DIR / "nav_state.json"
ROBOT_STATUS_PATH: Path = MAP_DIR / "robot_status.json"


# ---------------------------------------------------------------------------
# /map/navstate — Pi posts Nav2 state here
# ---------------------------------------------------------------------------


@router.post("/navstate")
async def update_nav_state(nav_state: NavState) -> Dict[str, Any]:
    """
    Update the live navigation state snapshot.

    Typical call from the Pi:
        POST /map/navstate
        {
          "state": "NAVIGATING",
          "frame_id": "map",
          "pose": {"x": 1.2, "y": -0.5, "yaw_deg": 90.0},
          "current_goal": "A201",
          "near_obstacle": false,
          "is_emergency_stop": false,
          "timestamp": "2025-11-22T00:12:34Z"
        }

    The payload is validated by NavState (Pydantic) before saving.
    """
    data = nav_state.model_dump()
    try:
        write_json_atomic(NAV_STATE_PATH, data)
    except OSError:
        # Let FastAPI turn this into a proper 500 for the client.
        raise HTTPException(
            status_code=500,
            detail="Failed to write nav_state.json",
        )

    logger.debug("map router: nav_state updated: %s", data)
    return {
        "status": "ok",
        "saved_to": str(NAV_STATE_PATH),
    }


# ---------------------------------------------------------------------------
# /map/status — Pi posts robot health here
# ---------------------------------------------------------------------------


@router.post("/status")
async def update_robot_status(robot_status: RobotStatus) -> Dict[str, Any]:
    """
    Update the live robot status snapshot.

    Typical call from the Pi:
        POST /map/status
        {
          "battery_percent": 72.5,
          "battery_state": "OK",
          "temp_c": 45.0,
          "wifi_rssi": -60,
          "uptime_s": 3600,
          "errors": [],
          "timestamp": "2025-11-22T00:12:34Z"
        }

    The payload is validated by RobotStatus (Pydantic) before saving.
    """
    data = robot_status.model_dump()
    try:
        write_json_atomic(ROBOT_STATUS_PATH, data)
    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Failed to write robot_status.json",
        )

    logger.debug("map router: robot_status updated: %s", data)
    return {
        "status": "ok",
        "saved_to": str(ROBOT_STATUS_PATH),
    }


# ---------------------------------------------------------------------------
# /map/known_locations — Pi (or tools) read location table from server
# ---------------------------------------------------------------------------


@router.get("/known_locations")
async def get_known_locations_endpoint() -> Dict[str, Any]:
    """
    Return the current known locations table used by the LLM server.

    This reads the same underlying data that app/core/map_lookup.py uses
    for resolving navigation goals (e.g. "info desk" -> "Info Desk").

    Typical response (shape):
        {
          "status": "ok",
          "count": 2,
          "locations": {
            "A201": {
              "display_name": "Room A201 (Lab)",
              "frame": "map",
              "x": 3.12,
              "y": -1.45,
              "theta": 1.57,
              ...
            },
            "Info Desk": {
              "display_name": "Information Desk",
              "frame": "map",
              "x": 0.0,
              "y": 0.0,
              "theta": 0.0,
              ...
            }
          }
        }

    Notes:
    - If known_locations.json is missing or empty, we return status=ok
      with count=0 and locations={} (no fake data).
    - If there is a parsing or serialization error, we respond with 500.
    """
    try:
        known = get_known_locations()
    except FileNotFoundError:
        # No file yet → treat as "no known locations", not an error.
        logger.info("map router: known_locations.json not found; returning empty set.")
        return {
            "status": "ok",
            "count": 0,
            "locations": {},
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("map router: failed to load known_locations: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to load known_locations on server.",
        ) from exc

    # Build a plain dict keyed by canonical name.
    locations: Dict[str, Any] = {}
    try:
        names = known.list_canonical_names()
        for key in names:
            loc = known.get(key)
            if not loc:
                continue

            # Try common patterns for Pydantic / custom models:
            if hasattr(loc, "to_json_dict"):
                # Preferred: our own helper
                locations[key] = loc.to_json_dict()  # type: ignore[call-arg]
            elif hasattr(loc, "model_dump"):
                # Pydantic v2
                try:
                    locations[key] = loc.model_dump(mode="json")  # type: ignore[call-arg]
                except TypeError:
                    locations[key] = loc.model_dump()  # type: ignore[call-arg]
            else:
                # Fallback: best-effort conversion
                try:
                    locations[key] = dict(loc)  # type: ignore[arg-type]
                except TypeError:
                    locations[key] = getattr(loc, "__dict__", str(loc))

    except Exception as exc:  # noqa: BLE001
        logger.exception("map router: failed to serialise known_locations: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to serialise known_locations data.",
        ) from exc

    return {
        "status": "ok",
        "count": len(locations),
        "locations": locations,
    }


# ---------------------------------------------------------------------------
# Self-test (no real file writes unless you call endpoints)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test for path wiring.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.routers.map
    """
    print("Robot Savo — map.py (router) self-test\n")
    print("MAP_DIR          :", MAP_DIR)
    print("NAV_STATE_PATH   :", NAV_STATE_PATH)
    print("ROBOT_STATUS_PATH:", ROBOT_STATUS_PATH)
    print("\nNo HTTP calls are made in this self-test.")
