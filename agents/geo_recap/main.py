"""geo_recap — geo v3.1 + propose_recapture_missions in base pool.

Hypothesis: in 25 lost games we lose ~100 % of captured planets back to
the enemy (median turns held = 13). Wiring recapture targets this directly.
Defaults match the post-revert calibration knobs in lib/missions/recapture.py
(score-denom snipe-aligned, top-k=5 per turn).

Single-axis variant: NO other change vs agents/geo/main.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lib.intent import World
from lib.missions.opening import propose_opening_missions
from lib.missions.recapture import propose_recapture_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.world_model import WorldModel

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "geo_base_for_recap", _REPO / "agents" / "geo" / "main.py",
)
_geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo)


def _build_base_missions_with_recap(
    world: World, model: WorldModel,
) -> list:
    missions = (
        propose_opening_missions(world, model)
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
        + propose_recapture_missions(world, model)
    )
    return _geo._drop_comet_missions(missions, world)


_geo._build_base_missions = _build_base_missions_with_recap


def agent(obs, configuration=None):
    return _geo.agent(obs, configuration)
