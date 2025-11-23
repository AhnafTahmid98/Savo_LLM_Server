# app/models/location_model.py
# -*- coding: utf-8 -*-
"""
Robot Savo — Location Models
----------------------------
Defines Pydantic models for known locations on campus that Robot Savo can
navigate to or talk about.

This is the data backing things like:
- "Take me to A201."
- "Where is the info desk?"
- "What places can you guide me to?"

Data is stored in JSON at:
    settings.known_locations_path  (app/map_data/known_locations.json)

File format (JSON)
------------------
We store a mapping from canonical name → location object, e.g.:

{
  "A201": {
    "canonical_name": "A201",
    "display_name": "Room A201 (Lab)",
    "type": "ROOM",
    "building": "A",
    "floor": 2,
    "description": "Electronics / IoT lab.",
    "tags": ["lab", "electronics", "iot"],
    "synonyms": ["room a201", "a 201 lab"],
    "x": 12.3,
    "y": 4.5
  },
  "Info Desk": {
    "canonical_name": "Info Desk",
    "display_name": "Information Desk",
    "type": "SERVICE",
    "building": "Lobby",
    "floor": 1,
    "description": "Main information desk at the lobby.",
    "tags": ["info", "help"],
    "synonyms": ["reception", "info", "information desk"],
    "x": 2.0,
    "y": 1.0
  }
}

Self-test
---------
Run:

    python -m app.models.location_model

Behavior:
- If known_locations.json exists and is non-empty:
    - Load and print all canonical names.
- Else:
    - Create a small demo dataset (A201, Info Desk),
      save it, and print the result.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from app.core.config import settings


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class LocationType(str, Enum):
    """High-level type/category of a location."""

    ROOM = "ROOM"
    SERVICE = "SERVICE"
    AREA = "AREA"
    BUILDING = "BUILDING"
    OTHER = "OTHER"


class Location(BaseModel):
    """
    A single known location that Robot Savo can talk about or navigate to.

    - canonical_name: stable key used in code and JSON (e.g. "A201").
    - display_name: human-friendly name for speech/GUI.
    - type: category (ROOM, SERVICE, AREA, BUILDING, OTHER).
    - building, floor: optional structural info.
    - description: short explanation for LLM to speak.
    - tags: keywords to help lookup / filtering.
    - synonyms: alternative names users may say ("a201", "room a201", "reception").
    - x, y: optional coordinates in the map frame (if known).
    """

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    canonical_name: str = Field(
        ...,
        description="Stable canonical name, e.g. 'A201' or 'Info Desk'.",
    )

    display_name: str = Field(
        ...,
        description="Human-friendly name, e.g. 'Room A201 (Lab)'.",
    )

    type: LocationType = Field(
        default=LocationType.OTHER,
        description="High-level type of the location.",
    )

    building: Optional[str] = Field(
        default=None,
        description="Building or area identifier, e.g. 'A', 'Lobby'.",
    )

    floor: Optional[int] = Field(
        default=None,
        description="Floor number (e.g. 1, 2).",
    )

    description: Optional[str] = Field(
        default=None,
        description="Short description for the LLM to speak.",
    )

    tags: List[str] = Field(
        default_factory=list,
        description="Keywords related to this location.",
    )

    synonyms: List[str] = Field(
        default_factory=list,
        description="Alternative names users might say.",
    )

    x: Optional[float] = Field(
        default=None,
        description="Optional x coordinate in map frame (meters).",
    )

    y: Optional[float] = Field(
        default=None,
        description="Optional y coordinate in map frame (meters).",
    )

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

    def all_names_lower(self) -> List[str]:
        """
        Return a list of all names (canonical + synonyms) lowercased.

        Useful for simple case-insensitive matching.
        """
        names = [self.canonical_name]
        names.extend(self.synonyms)
        return [n.lower().strip() for n in names if n]


class KnownLocations(BaseModel):
    """
    Container for multiple known locations.

    Internally stored as:
        { canonical_name: Location, ... }

    This model is what we read/write to known_locations.json.
    """

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    locations: Dict[str, Location] = Field(
        default_factory=dict,
        description="Mapping from canonical_name to Location.",
    )

    # ----------------------------------------------------------------------
    # I/O helpers
    # ----------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "KnownLocations":
        """
        Load KnownLocations from JSON.

        If path is None, uses settings.known_locations_path.
        """
        target = Path(path) if path is not None else settings.known_locations_path

        if not target.exists():
            # No file yet → empty container
            return cls(locations={})

        raw = json.loads(target.read_text(encoding="utf-8"))

        # Expect a mapping canonical_name -> location dict
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict in {target}, got {type(raw)!r}")

        locs: Dict[str, Location] = {}
        for key, value in raw.items():
            # Ensure the canonical_name is consistent
            if "canonical_name" not in value:
                value["canonical_name"] = key
            locs[key] = Location(**value)

        return cls(locations=locs)

    def save(self, path: Optional[Path | str] = None) -> Path:
        """
        Save KnownLocations to JSON.

        If path is None, uses settings.known_locations_path.
        """
        target = Path(path) if path is not None else settings.known_locations_path
        target.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            key: loc.to_json_dict() for key, loc in self.locations.items()
        }

        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    # ----------------------------------------------------------------------
    # Lookup helpers
    # ----------------------------------------------------------------------
    def get(self, canonical_name: str) -> Optional[Location]:
        """Get a location by its canonical name (case-sensitive key)."""
        return self.locations.get(canonical_name)

    def find_by_name(self, name: str) -> Optional[Location]:
        """
        Simple case-insensitive search by canonical_name or any synonym.

        This is a lightweight helper; more advanced logic can live in
        app.core.map_lookup if needed.
        """
        if not name:
            return None
        needle = name.lower().strip()
        for loc in self.locations.values():
            if needle in loc.all_names_lower():
                return loc
        return None

    def list_canonical_names(self) -> List[str]:
        """Return a sorted list of canonical location names."""
        return sorted(self.locations.keys())


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """
    Self-test behavior:

    - If known_locations.json exists and is non-empty:
        - Load it and print the canonical names.
    - Else:
        - Create a small demo dataset (A201, Info Desk),
          save it, and print what was written.

    This is safe to run on a real system; it won't overwrite a non-empty file.
    """
    print("Robot Savo — LocationModel self-test")
    print("------------------------------------")
    print(f"Using known_locations_path: {settings.known_locations_path}")

    target = settings.known_locations_path
    if target.exists():
        text = target.read_text(encoding="utf-8").strip()
        if text:
            print("Existing known_locations.json found. Loading...")
            locations = KnownLocations.load()
            names = locations.list_canonical_names()
            print(f"Loaded {len(names)} locations:")
            for name in names:
                print(f"  - {name}")
            print("\nSelf-test OK (existing data preserved).")
            return 0

    # No file or empty file → write a demo set
    print("No known_locations.json or file is empty. Creating demo dataset...")

    demo = KnownLocations(
        locations={
            "A201": Location(
                canonical_name="A201",
                display_name="Room A201 (Lab)",
                type=LocationType.ROOM,
                building="A",
                floor=2,
                description="Electronics / IoT lab.",
                tags=["lab", "electronics", "iot"],
                synonyms=["room a201", "a201 lab", "a 201"],
                x=12.3,
                y=4.5,
            ),
            "Info Desk": Location(
                canonical_name="Info Desk",
                display_name="Information Desk",
                type=LocationType.SERVICE,
                building="Lobby",
                floor=1,
                description="Main information desk at the lobby entrance.",
                tags=["info", "help"],
                synonyms=["reception", "info", "information desk"],
                x=2.0,
                y=1.0,
            ),
        }
    )

    path = demo.save()
    print(f"Saved demo locations to: {path}")
    print("Demo canonical names:")
    for name in demo.list_canonical_names():
        print(f"  - {name}")

    print("\nSelf-test OK (demo data created).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
