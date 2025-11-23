# app/models/nav_state_model.py
# -*- coding: utf-8 -*-
"""
Robot Savo — Navigation State Model
-----------------------------------
Represents the *current* high-level navigation state of Robot Savo as seen by
the LLM server.

Usage
-----
- The Pi / ROS2 side sends navigation telemetry (goal, state, distances, etc.).
- The LLM server stores this into nav_state.json (settings.nav_state_path).
- Prompts (navigation + status) read this JSON to explain what the robot is doing.

IMPORTANT
---------
- This model itself does NOT create real nav_state.json on import.
- The self-test at the bottom writes ONLY to a demo path:
      app/map_data/nav_state_demo.json
  so it cannot overwrite the real runtime snapshot that comes from the Pi.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict

from app.core.config import settings


class NavStateEnum(str, Enum):
    """High-level navigation state of Robot Savo."""

    IDLE = "IDLE"                # Not moving, no active goal
    NAVIGATING = "NAVIGATING"    # Moving toward a goal
    FOLLOWING = "FOLLOWING"      # Following a person
    STOPPED = "STOPPED"          # Stopped but can continue
    BLOCKED = "BLOCKED"          # Cannot continue due to obstacle
    ERROR = "ERROR"              # Some navigation error
    MAPPING = "MAPPING"          # Mapping / exploration mode
    DOCKING = "DOCKING"          # Going to / at docking/charging station


class NavState(BaseModel):
    """
    Snapshot of Robot Savo's navigation state.

    This is what the LLM uses to answer questions like:
    - "What are you doing?"
    - "Where are we going?"
    - "Why did you stop?"

    Pi-side code can fill only the fields it knows; everything else is optional.
    """

    # Pydantic v2 config
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    # --- Core metadata ------------------------------------------------------
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of this nav state in UTC.",
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session or robot id, e.g. 'robot-savo-01'.",
    )

    # --- High-level navigation state ---------------------------------------
    state: NavStateEnum = Field(
        default=NavStateEnum.IDLE,
        description="High-level navigation state (IDLE, NAVIGATING, ...).",
    )

    nav_goal: Optional[str] = Field(
        default=None,
        description="Canonical destination name, e.g. 'A201' or 'Info Desk'.",
    )

    nav_goal_display: Optional[str] = Field(
        default=None,
        description="Optional display name, e.g. 'Room A201 (Lab)'.",
    )

    # --- Pose (if known) ----------------------------------------------------
    frame_id: str = Field(
        default="map",
        description="Coordinate frame of x/y/yaw (e.g. 'map').",
    )

    x: Optional[float] = Field(
        default=None,
        description="Current x position in meters (map frame).",
    )

    y: Optional[float] = Field(
        default=None,
        description="Current y position in meters (map frame).",
    )

    yaw: Optional[float] = Field(
        default=None,
        description="Heading in radians (map frame, yaw).",
    )

    # --- Distance + speed ---------------------------------------------------
    dist_to_goal_m: Optional[float] = Field(
        default=None,
        description="Approximate straight-line distance to goal in meters.",
    )

    linear_speed_mps: Optional[float] = Field(
        default=None,
        description="Current linear speed (m/s) from odometry.",
    )

    angular_speed_radps: Optional[float] = Field(
        default=None,
        description="Current angular speed (rad/s) from odometry.",
    )

    # --- Obstacle / safety info --------------------------------------------
    min_front_m: Optional[float] = Field(
        default=None,
        description="Closest obstacle in front sector (meters).",
    )

    min_back_m: Optional[float] = Field(
        default=None,
        description="Closest obstacle in rear sector (meters).",
    )

    min_left_m: Optional[float] = Field(
        default=None,
        description="Closest obstacle in left sector (meters).",
    )

    min_right_m: Optional[float] = Field(
        default=None,
        description="Closest obstacle in right sector (meters).",
    )

    is_safety_stop: bool = Field(
        default=False,
        description="True if a safety layer stopped the robot (<~0.28 m, etc.).",
    )

    safety_reason: Optional[str] = Field(
        default=None,
        description="Short explanation of the last safety stop, e.g. 'person in front'.",
    )

    is_estop: bool = Field(
        default=False,
        description="True if emergency stop is engaged.",
    )

    # --- Command + debug ----------------------------------------------------
    last_command: Optional[str] = Field(
        default=None,
        description="Last high-level nav command (e.g. 'NAVIGATE', 'STOP').",
    )

    note: Optional[str] = Field(
        default=None,
        description="Freeform human-readable note about current state.",
    )

    # ----------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------
    @classmethod
    def idle(cls, note: Optional[str] = None) -> "NavState":
        """
        Create an IDLE nav state with no active goal.

        Useful for:
        - Resetting state when robot is not guiding anyone.
        - Startup default before any goal is set.
        """
        return cls(
            state=NavStateEnum.IDLE,
            nav_goal=None,
            nav_goal_display=None,
            is_safety_stop=False,
            is_estop=False,
            note=note or "Robot is idle with no active goal.",
        )

    def to_json_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dict representation.

        - Datetimes become ISO8601 strings.
        - Enums become their string values.
        """
        return self.model_dump(mode="json")

    def has_active_goal(self) -> bool:
        """
        True if there is a non-empty navigation goal set.
        """
        return bool(self.nav_goal and str(self.nav_goal).strip())

    def is_moving(self) -> bool:
        """
        Rough check whether the robot is moving (NAVIGATING/FOLLOWING and
        linear speed above a tiny epsilon).
        """
        if self.state not in (NavStateEnum.NAVIGATING, NavStateEnum.FOLLOWING):
            return False
        try:
            return (self.linear_speed_mps or 0.0) > 0.01
        except TypeError:
            return False

    # ----------------------------------------------------------------------
    # Persistence helpers
    # ----------------------------------------------------------------------
    def save(self, path: Optional[Path | str] = None) -> Path:
        """
        Save this NavState to a JSON file.

        If `path` is None, uses `settings.nav_state_path`.

        This is what the LLM server should call after receiving telemetry
        from the robot (via HTTP or WebSocket).
        """
        target = Path(path) if path is not None else settings.nav_state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_dict()
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "NavState":
        """
        Load NavState from a JSON file.

        If `path` is None, uses `settings.nav_state_path`.

        This is what prompts / status endpoints will call to get the latest
        navigation snapshot when answering user questions.

        NOTE:
        - If the file does not exist, FileNotFoundError is raised.
          The pipeline catches this and uses NavState.idle(...) instead, so
          the robot will say things like:
              "I am not moving right now. I do not have an active goal."
          instead of pretending to go to A201.
        """
        target = Path(path) if path is not None else settings.nav_state_path
        if not target.exists():
            raise FileNotFoundError(f"NavState JSON not found at: {target}")
        raw = json.loads(target.read_text(encoding="utf-8"))
        return cls(**raw)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _self_test() -> int:
    """
    Simple self-test:

    - Create a demo NavState representing navigation toward A201.
    - Save it to a *demo* path:
         app/map_data/nav_state_demo.json
    - Load it back and print the JSON.

    This does not talk to the robot; it only verifies:
    - config paths are correct
    - model <-> JSON conversion is correct

    IMPORTANT:
    - It does NOT overwrite the real runtime nav_state file used by the Pi
      (settings.nav_state_path).
    """
    print("Robot Savo — NavState self-test")
    print("--------------------------------")

    demo_path = settings.map_data_dir / "nav_state_demo.json"
    print(f"Demo path (safe): {demo_path}")

    nav_state = NavState(
        state=NavStateEnum.NAVIGATING,
        nav_goal="A201",
        nav_goal_display="Room A201 (Lab)",
        x=1.23,
        y=4.56,
        yaw=0.78,
        dist_to_goal_m=12.34,
        linear_speed_mps=0.15,
        angular_speed_radps=0.05,
        min_front_m=0.42,
        min_back_m=1.20,
        min_left_m=0.80,
        min_right_m=0.95,
        is_safety_stop=False,
        is_estop=False,
        last_command="NAVIGATE",
        note="Demo nav state created by self-test.",
    )

    path = nav_state.save(demo_path)
    print(f"Saved DEMO NavState to: {path}")

    loaded = NavState.load(demo_path)
    print("Loaded DEMO NavState:")
    print(json.dumps(loaded.to_json_dict(), indent=2, sort_keys=True))

    print("\nSelf-test OK (demo file only, real nav_state untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
