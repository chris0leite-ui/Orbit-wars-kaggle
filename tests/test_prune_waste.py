"""LR_PRUNE_WASTE (2026-06-18): drop launches whose path crosses the sun.

The shipped agent's own candidates are sun-filtered, but the producer-fallback
move it often plays is not -- so fleets get launched straight into the sun and
destroyed (measured: up to ~18% of launches on some boards). `_drop_sun_launches`
prunes exactly those (mapping each launch to its target by the agent's own
intercept aim) while keeping every clear launch, so the ships of a doomed fleet
stay home instead of dying.

This tests the pure filter directly: one launch aimed across the sun (dropped) and
one aimed along a clear path (kept).
"""
import importlib.util
import math
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("lr_main_prune_waste_test", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_drop_sun_launches_drops_only_sun_crossers():
    lr = _load_agent_module()
    # Sun is at (50, 50) r10. My planet below it; one target directly across the
    # sun (path crosses it) and one to the side (clear path). omega=0 so the emit
    # angle is a straight bearing the filter can match exactly.
    planets = [
        Planet(0, 0, 50.0, 15.0, 1.0, 100, 1),    # my source, below the sun
        Planet(1, -1, 50.0, 85.0, 1.0, 5, 5),     # across the sun -> doomed
        Planet(2, -1, 20.0, 15.0, 1.0, 5, 5),     # to the side -> clear
    ]
    by_id = {int(p.id): p for p in planets}
    ang_sun = math.atan2(85.0 - 15.0, 50.0 - 50.0)   # straight up, through the sun
    ang_clear = math.atan2(15.0 - 15.0, 20.0 - 50.0)  # left along y=15, clear
    move = [[0, ang_sun, 30], [0, ang_clear, 10]]

    kept = lr._drop_sun_launches(move, planets, by_id, frozenset(), {}, 0.0)

    assert len(kept) == 1, f"should drop exactly the sun-crossing launch, kept {kept}"
    assert abs(float(kept[0][1]) - ang_clear) < 1e-6, "the clear launch must be the one kept"


def test_drop_sun_launches_keeps_everything_when_all_clear():
    lr = _load_agent_module()
    planets = [
        Planet(0, 0, 20.0, 15.0, 1.0, 100, 1),
        Planet(1, -1, 80.0, 15.0, 1.0, 5, 5),     # along y=15, clear of the sun
        Planet(2, -1, 20.0, 85.0, 1.0, 5, 5),     # along x=20, clear of the sun
    ]
    by_id = {int(p.id): p for p in planets}
    a1 = math.atan2(15.0 - 15.0, 80.0 - 20.0)
    a2 = math.atan2(85.0 - 15.0, 20.0 - 20.0)
    move = [[0, a1, 20], [0, a2, 20]]

    kept = lr._drop_sun_launches(move, planets, by_id, frozenset(), {}, 0.0)
    assert len(kept) == 2, f"no launch crosses the sun; all should be kept, got {kept}"
