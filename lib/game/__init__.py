"""Pure-Python rebuild of the orbit_wars game engine.

Drop-in replacement for `kaggle_environments.envs.orbit_wars.orbit_wars`.
Byte-exact parity is enforced by `tests/test_game_parity.py`.
"""

from lib.game.interpreter import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    Fleet,
    Planet,
    distance,
    generate_comet_paths,
    generate_planets,
    interpreter,
    point_to_segment_distance,
    swept_pair_hit,
)

__all__ = [
    "BOARD_SIZE",
    "CENTER",
    "COMET_PRODUCTION",
    "COMET_RADIUS",
    "COMET_SPAWN_STEPS",
    "MAX_PLANET_GROUPS",
    "MIN_PLANET_GROUPS",
    "MIN_STATIC_GROUPS",
    "PLANET_CLEARANCE",
    "ROTATION_RADIUS_LIMIT",
    "SUN_RADIUS",
    "Fleet",
    "Planet",
    "distance",
    "generate_comet_paths",
    "generate_planets",
    "interpreter",
    "point_to_segment_distance",
    "swept_pair_hit",
]
