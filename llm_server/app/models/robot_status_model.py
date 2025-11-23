# app/models/robot_status_model.py
# -*- coding: utf-8 -*-
"""
Robot Savo — Robot Status Model
-------------------------------
Represents the *current* health and system status of Robot Savo as seen by
the LLM server.

Usage
-----
- The Pi / ROS2 side sends periodic status telemetry:
    - battery levels (UPS + kit battery)
    - temperatures
    - CPU load / memory
    - Wi-Fi signal
    - power / shutdown flags
- The LLM server stores this into robot_status.json (settings.robot_status_path).
- Prompts (status + navigation) read this JSON to answer things like:
    - "How is your battery?"
    - "Are you overheating?"
    - "Can you still continue guiding me?"

IMPORTANT
---------
- This model itself does NOT create real status files on import.
- The self-test at the bottom writes ONLY to a demo path:
      app/map_data/robot_status_demo.json
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


class PowerState(str, Enum):
    """High-level power state of the robot."""

    NORMAL = "NORMAL"          # Everything OK
    LOW_BATTERY = "LOW_BATTERY"
    CRITICAL = "CRITICAL"      # Very low, must stop soon
    CHARGING = "CHARGING"
    ON_DOCK = "ON_DOCK"
    UNKNOWN = "UNKNOWN"


class RobotStatus(BaseModel):
    """
    Snapshot of Robot Savo's system status.

    This is what the LLM uses to answer questions like:
    - "How is your battery?"
    - "Are you overheating?"
    - "Can you still continue guiding me?"
    """

    # Pydantic v2 config
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    # --- Core metadata ------------------------------------------------------
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of this status snapshot in UTC.",
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session or robot id, e.g. 'robot-savo-01'.",
    )

    # --- Battery / power ----------------------------------------------------
    # UPS (Pi power) measurements
    ups_voltage_v: Optional[float] = Field(
        default=None,
        description="UPS HAT output voltage (V) powering the Pi.",
    )
    ups_soc_pct: Optional[float] = Field(
        default=None,
        description="UPS state-of-charge estimate (0–100 %).",
    )

    # Kit battery (drive motors) measurements
    kit_voltage_v: Optional[float] = Field(
        default=None,
        description="Kit / motor battery voltage (V).",
    )
    kit_soc_pct: Optional[float] = Field(
        default=None,
        description="Kit battery state-of-charge estimate (0–100 %).",
    )

    # Human-friendly battery note (words, no signs)
    # Aligns with the style we use in battery_health.py:
    #   'UPS low', 'Kit low', 'Both low', 'Good condition', 'Needs charging'
    battery_note: Optional[str] = Field(
        default=None,
        description=(
            "Short descriptive label for battery status, e.g. "
            "'UPS low', 'Kit low', 'Both low', 'Good condition', 'Needs charging'."
        ),
    )

    power_state: PowerState = Field(
        default=PowerState.NORMAL,
        description="High-level power state (NORMAL, LOW_BATTERY, CRITICAL, ...).",
    )

    shutdown_soon: bool = Field(
        default=False,
        description="True if the robot expects to shut down soon due to low power.",
    )

    # --- Temperatures -------------------------------------------------------
    temp_cpu_c: Optional[float] = Field(
        default=None,
        description="CPU temperature in Celsius.",
    )
    temp_board_c: Optional[float] = Field(
        default=None,
        description="Board / ambient internal temperature in Celsius.",
    )
    temp_motor_driver_c: Optional[float] = Field(
        default=None,
        description="Motor driver or HAT temperature in Celsius, if available.",
    )

    thermal_throttling: bool = Field(
        default=False,
        description="True if the system is currently thermally throttled.",
    )

    # --- System load / memory ----------------------------------------------
    cpu_load_pct: Optional[float] = Field(
        default=None,
        description="Approximate CPU load percentage (0–100 %).",
    )

    mem_used_pct: Optional[float] = Field(
        default=None,
        description="Approximate RAM usage percentage (0–100 %).",
    )

    disk_used_pct: Optional[float] = Field(
        default=None,
        description="Approximate disk usage percentage (0–100 %) on main volume.",
    )

    # --- Connectivity -------------------------------------------------------
    wifi_ssid: Optional[str] = Field(
        default=None,
        description="Current Wi-Fi SSID, if connected.",
    )

    wifi_rssi_dbm: Optional[float] = Field(
        default=None,
        description="Wi-Fi signal strength in dBm (e.g. -45 is strong, -80 is weak).",
    )

    wifi_link_quality_pct: Optional[float] = Field(
        default=None,
        description="Wi-Fi link quality as a percentage (0–100 %).",
    )

    network_ok: bool = Field(
        default=True,
        description="True if basic network connectivity looks OK.",
    )

    # --- Error / warnings ---------------------------------------------------
    has_errors: bool = Field(
        default=False,
        description="True if there are current error conditions.",
    )

    error_summary: Optional[str] = Field(
        default=None,
        description="Short summary of current errors (e.g. 'IMU offline').",
    )

    warning_summary: Optional[str] = Field(
        default=None,
        description="Short summary of current warnings (e.g. 'Wi-Fi weak').",
    )

    # --- Human-readable health note ----------------------------------------
    health_note: Optional[str] = Field(
        default=None,
        description=(
            "One or two short sentences describing overall health in simple words "
            "for the LLM to speak (e.g. 'My battery is fine and I can continue.'"
            " or 'My battery is getting low, but I can still guide you a bit.')."
        ),
    )

    # ----------------------------------------------------------------------
    # Convenience helpers (for prompts / pipeline)
    # ----------------------------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dict representation.

        - Datetimes become ISO8601 strings.
        - Enums become their string values.
        """
        return self.model_dump(mode="json")

    def has_real_battery_data(self) -> bool:
        """
        True if we have *any* real battery measurement, not just defaults.
        """
        return any(
            v is not None
            for v in (self.ups_voltage_v, self.ups_soc_pct, self.kit_voltage_v, self.kit_soc_pct)
        )

    def has_real_temperature_data(self) -> bool:
        """
        True if we have at least one temperature sensor reading.
        """
        return any(
            v is not None
            for v in (self.temp_cpu_c, self.temp_board_c, self.temp_motor_driver_c)
        )

    # ----------------------------------------------------------------------
    # Persistence helpers
    # ----------------------------------------------------------------------
    def save(self, path: Optional[Path | str] = None) -> Path:
        """
        Save this RobotStatus to a JSON file.

        If `path` is None, uses `settings.robot_status_path`.

        This is what the LLM server should call after receiving status telemetry
        from the robot (via HTTP or WebSocket).
        """
        target = Path(path) if path is not None else settings.robot_status_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_dict()
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "RobotStatus":
        """
        Load RobotStatus from a JSON file.

        If `path` is None, uses `settings.robot_status_path`.

        This is what prompts / status endpoints will call to get the latest
        health snapshot when answering user questions.

        NOTE:
        - If the file does not exist, FileNotFoundError is raised.
          The pipeline catches this and uses a safe "no data" default instead,
          so the robot says "I cannot read my battery right now" instead of
          lying.
        """
        target = Path(path) if path is not None else settings.robot_status_path
        if not target.exists():
            raise FileNotFoundError(f"RobotStatus JSON not found at: {target}")
        raw = json.loads(target.read_text(encoding="utf-8"))
        return cls(**raw)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _self_test() -> int:
    """
    Simple self-test:

    - Create a demo RobotStatus with reasonable fake values.
    - Save it to a *demo* path:
         app/map_data/robot_status_demo.json
    - Load it back and print the JSON.

    This does not talk to the robot; it only verifies:
    - config paths are correct
    - model <-> JSON conversion is correct

    IMPORTANT:
    - It does NOT overwrite the real runtime status file used by the Pi
      (settings.robot_status_path).
    """
    print("Robot Savo — RobotStatus self-test")
    print("-----------------------------------")

    demo_path = settings.map_data_dir / "robot_status_demo.json"
    print(f"Demo path (safe): {demo_path}")

    status = RobotStatus(
        session_id="robot-savo-demo",
        ups_voltage_v=3.85,
        ups_soc_pct=92.0,
        kit_voltage_v=8.05,
        kit_soc_pct=80.0,
        battery_note="Good condition",
        power_state=PowerState.NORMAL,
        shutdown_soon=False,
        temp_cpu_c=52.5,
        temp_board_c=40.0,
        temp_motor_driver_c=38.0,
        thermal_throttling=False,
        cpu_load_pct=23.0,
        mem_used_pct=48.0,
        disk_used_pct=35.0,
        wifi_ssid="Campus-WiFi",
        wifi_rssi_dbm=-55.0,
        wifi_link_quality_pct=85.0,
        network_ok=True,
        has_errors=False,
        error_summary=None,
        warning_summary="Wi-Fi is okay but not perfect.",
        health_note="My battery and temperature are fine, and I can continue guiding you.",
    )

    path = status.save(demo_path)
    print(f"Saved DEMO RobotStatus to: {path}")

    loaded = RobotStatus.load(demo_path)
    print("Loaded DEMO RobotStatus:")
    print(json.dumps(loaded.to_json_dict(), indent=2, sort_keys=True))

    print("\nSelf-test OK (demo file only, real status untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
