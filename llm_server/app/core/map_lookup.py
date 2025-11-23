# app/core/map_lookup.py
# -*- coding: utf-8 -*-
"""
Robot Savo — Location / Map Lookup
----------------------------------
Helpers for resolving user-specified destinations into canonical location
names and accessing the known locations database.

Responsibility
--------------
This module does ONE job:

    Text from user  --->  Canonical location + metadata

Examples:
    "take me to room a201"  ->  "A201"
    "where is reception"    ->  "Info Desk"

Data source
-----------
All locations are stored in JSON at:

    settings.known_locations_path  (app/map_data/known_locations.json)

The structure and models are defined in:

    app.models.location_model
        - Location
        - KnownLocations

Public API
----------
- get_known_locations(force_reload: bool = False) -> KnownLocations
- resolve_nav_goal(text: str) -> str | None
- get_location(canonical_name: str) -> Location | None
- list_locations() -> list[str]

Self-test
---------
You can run a simple self-test with:

    python -m app.core.map_lookup

This will:
- Load known_locations.json (creating a demo if needed).
- Print the canonical names.
- Demonstrate resolve_nav_goal() on a few example strings.
"""

from __future__ import annotations

import logging
from typing import Optional, List

from app.core.config import settings
from app.models.location_model import KnownLocations, Location

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_known_locations_cache: KnownLocations | None = None


def get_known_locations(force_reload: bool = False) -> KnownLocations:
    """
    Load and return the KnownLocations object.

    - Uses an in-memory cache by default for speed.
    - Set force_reload=True to re-read the JSON file.

    This is the main entry point that other modules (pipeline, intent helpers)
    should use to access location definitions.
    """
    global _known_locations_cache

    if _known_locations_cache is None or force_reload:
        logger.info(
            "Loading known locations from %s", settings.known_locations_path
        )
        _known_locations_cache = KnownLocations.load()

        count = len(_known_locations_cache.locations)
        logger.info("Loaded %d known locations", count)

    return _known_locations_cache


# ---------------------------------------------------------------------------
# Public lookup helpers
# ---------------------------------------------------------------------------


def resolve_nav_goal(text: str) -> Optional[str]:
    """
    Resolve a free-form user text into a canonical nav_goal string.

    Examples:
        resolve_nav_goal("A201")          -> "A201"
        resolve_nav_goal("room a201")     -> "A201"
        resolve_nav_goal("info desk")     -> "Info Desk"
        resolve_nav_goal("reception")     -> "Info Desk"
        resolve_nav_goal("random place")  -> None

    Behavior:
    ---------
    - Case-insensitive.
    - Uses Location.canonical_name and synonyms via KnownLocations.find_by_name.
    - Returns the canonical name if found, else None.
    """
    if not text:
        return None

    locations = get_known_locations()
    loc = locations.find_by_name(text)
    if loc is None:
        return None

    return loc.canonical_name


def get_location(canonical_name: str) -> Optional[Location]:
    """
    Return the Location object for a canonical name, or None if not found.

    Example:
        loc = get_location("A201")
        if loc:
            print(loc.display_name, loc.x, loc.y)
    """
    if not canonical_name:
        return None

    locations = get_known_locations()
    return locations.get(canonical_name)


def list_locations() -> List[str]:
    """
    Return a sorted list of all canonical location names.

    Useful for:
    - Debugging
    - 'What places can you guide me to?' features
    """
    locations = get_known_locations()
    return locations.list_canonical_names()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """
    Self-test for map_lookup.

    - Ensures known_locations.json is present (via KnownLocations.load()).
    - Prints all canonical names.
    - Demonstrates resolve_nav_goal() on a few example strings.
    """
    from pathlib import Path

    print("Robot Savo — map_lookup.py self-test")
    print("-------------------------------------")
    print(f"Known locations path: {settings.known_locations_path}")

    # Ensure the file exists and can be loaded.
    # If known_locations.json is empty or missing, KnownLocations.load()
    # will return an empty container (the demo is created by location_model
    # self-test, not here).
    locs = get_known_locations(force_reload=True)
    names = locs.list_canonical_names()

    print(f"Known locations keys ({len(names)}): {names if names else '[]'}")
    print("------------------------------------------------------------")

    examples = [
        "A201",
        "a201",
        "room a201",
        "info desk",
        "Info Desk",
        "cafeteria",
        "random place",
    ]

    for text in examples:
        result = resolve_nav_goal(text)
        print(f"resolve_nav_goal({text!r}) -> {result!r}")

    # Example of using get_location()
    print("\nLocation details (if present):")
    for key in names:
        loc = get_location(key)
        if loc is None:
            continue
        print(f"- {key}: display_name={loc.display_name!r}, type={loc.type}, "
              f"building={loc.building!r}, floor={loc.floor!r}")

    print("\nSelf-test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
