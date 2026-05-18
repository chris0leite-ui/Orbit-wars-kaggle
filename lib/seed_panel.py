"""Canonical 128-seed evaluation panel — stratified by geometry archetype.

The panel is built by ``scripts/build_seed_panel.py`` and persisted to
``data/seed_panel_128.json``. This module exposes it as importable Python
constants for tournament harnesses.

Three primary stratification axes (4 x 4 x 2 = 32 archetypes, 4 seeds each):
- ``total_production`` (game pace): low_prod / med_low_prod / med_high_prod / high_prod
- ``rotating_share``: mostly_static / mixed_static / mixed_rotating / mostly_rotating
- ``size_split`` (sign of radius_rotating_mean - radius_static_mean):
  big_static / big_rotating

Usage:
    from lib.seed_panel import SEED_PANEL_128, SEED_PANEL_BY_ARCHETYPE
    result = tournament.run_tournament(agents, SEED_PANEL_128)
"""

from __future__ import annotations

import json
from pathlib import Path

_PANEL_JSON = Path(__file__).resolve().parents[1] / "data" / "seed_panel_128.json"

_panel_data = json.loads(_PANEL_JSON.read_text())

SEED_PANEL_128: list[int] = [entry["seed"] for entry in _panel_data["panel"]]

SEED_PANEL_BY_ARCHETYPE: dict[str, list[int]] = {}
for entry in _panel_data["panel"]:
    SEED_PANEL_BY_ARCHETYPE.setdefault(entry["archetype"], []).append(entry["seed"])

ARCHETYPE_OF_SEED: dict[int, str] = {entry["seed"]: entry["archetype"] for entry in _panel_data["panel"]}

PANEL_BIN_EDGES: dict[str, list[float]] = _panel_data["bin_edges"]
ARCHETYPE_NAMES: dict[str, list[str]] = _panel_data["archetype_names"]


def features_for(seed: int) -> dict | None:
    """Return cached geometry features for ``seed`` if it's in the panel."""
    for entry in _panel_data["panel"]:
        if entry["seed"] == seed:
            return entry["features"]
    return None
