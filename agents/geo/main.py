"""geo — geometric sense + posture arbiter + joint LP allocator.

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> sense_state(world, model)                       # lib/geo/sense.py
        -> decide_posture(world, sense, model)             # lib/geo/posture.py
        -> collect_posture_weighted_missions(...)          # local
        -> allocate(missions, world, sense, posture, model)# lib/geo/allocator.py
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

The orchestrator's only job is wiring + applying posture multipliers
to existing mission proposers' scores. The mission VALUE FORMULAS are
unchanged from v3.5.1 / v7_0 — only the SETTLEMENT layer (per-source
greedy → joint LP) and the POSTURE BIASES are new.

The v7_4_hungarian killed-path lesson informs the design: pure global
assignment without per-mission-class scoring fails. We keep the
per-class scoring and only replace the settlement.
"""

from __future__ import annotations

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.opening import propose_opening_missions
from lib.missions.recapture import propose_recapture_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.world_model import WorldModel

from lib.geo.allocator import allocate
from lib.geo.posture import Posture, decide_posture
from lib.geo.sense import sense_state


# ---------------------------------------------------------------------------
# Posture × mission-class score multipliers
# ---------------------------------------------------------------------------
#
# Starting values from the plan (calibrated against top-10 fingerprint and
# loss-mode audit). Tunable via local A/B at n=64.
#
# Empty entries (mission_class not in dict) mean "do not run this proposer
# in this posture."

POSTURE_WEIGHTS: dict[Posture, dict[str, float]] = {
    Posture.OPENING: {
        "snipe":     1.0,
        "opening":   3.0,
    },
    Posture.EXPAND: {
        "snipe":     1.0,
        "reinforce": 1.0,
        "recapture": 1.0,
    },
    Posture.DEFEND: {
        "snipe":     0.5,   # only enemy targets (see _enemy_only_filter)
        "reinforce": 3.0,
        "recapture": 1.5,
    },
    Posture.BREAK: {
        "snipe":     1.5,   # enemy-only
        "reinforce": 0.5,
        "recapture": 2.5,
    },
}


def _aggressive_for(posture: Posture) -> bool:
    return posture in (Posture.OPENING, Posture.BREAK)


def _enemy_only_filter(missions: list[Mission], world: World) -> list[Mission]:
    """Keep only missions whose target is owned by an enemy (not neutral)."""
    return [
        m for m in missions
        if (world.planets_by_id.get(m.target_id) is not None
            and world.planets_by_id[m.target_id].owner not in (world.my_id, -1))
    ]


def _scale_scores(missions: list[Mission], mult: float) -> list[Mission]:
    if mult == 1.0:
        return missions
    if mult == 0.0:
        return []
    return [
        Mission(
            mission_class=m.mission_class,
            src_id=m.src_id,
            target_id=m.target_id,
            ships=m.ships,
            score=m.score * mult,
            eta=m.eta,
            note=m.note,
        )
        for m in missions
    ]


def collect_posture_weighted_missions(
    world: World, model: WorldModel, posture: Posture,
) -> list[Mission]:
    """Run posture-relevant proposers and apply per-class score multipliers."""
    weights = POSTURE_WEIGHTS.get(posture, {})
    bag: list[Mission] = []

    if "snipe" in weights:
        snipe = propose_snipe_missions(
            world, model, aggressive=_aggressive_for(posture)
        )
        if posture in (Posture.DEFEND, Posture.BREAK):
            snipe = _enemy_only_filter(snipe, world)
        bag.extend(_scale_scores(snipe, weights["snipe"]))

    if "reinforce" in weights:
        rein = propose_reinforce_missions(world, model)
        bag.extend(_scale_scores(rein, weights["reinforce"]))

    if "recapture" in weights:
        recap = propose_recapture_missions(world, model)
        bag.extend(_scale_scores(recap, weights["recapture"]))

    if "opening" in weights:
        opens = propose_opening_missions(world, model)
        bag.extend(_scale_scores(opens, weights["opening"]))

    return bag


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    posture = decide_posture(world, sense, model)
    missions = collect_posture_weighted_missions(world, model, posture)
    intents = allocate(missions, world, sense, posture, model, method="lp")
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
