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

from lib.planner import settle_plan

from lib.geo.allocator import allocate, allocate_greedy_multi
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
    # EXPAND (~90% of turns) must match v3.5.1 verbatim: snipe + reinforce,
    # weights 1.0, NO recapture (recapture is added in v1.5 once we can
    # bound its candidate explosion). DEFEND/BREAK/OPENING are the
    # situational overrides — small, targeted bias, not domination.
    Posture.OPENING: {
        "snipe":     1.0,
        "reinforce": 1.0,
    },
    Posture.EXPAND: {
        "snipe":     1.0,
        "reinforce": 1.0,
    },
    Posture.DEFEND: {
        "snipe":     0.5,   # only enemy targets (see _enemy_only_filter)
        "reinforce": 2.0,   # boost defense
    },
    Posture.BREAK: {
        "snipe":     1.5,   # enemy-only
        "reinforce": 0.5,
    },
}


def _aggressive_for(posture: Posture) -> bool:
    # Aggressive snipe sizing on by default (matches v3.5.1's known-good
    # config and the top-10 source-emptying / 1.9x-launch-density signal).
    # Only DEFEND backs off — there we want to keep ships at home.
    return posture is not Posture.DEFEND


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
    # LAYER 2: posture-weighted missions, settled by settle_plan.
    # In EXPAND (the default), this is FUNCTIONALLY EQUIVALENT to v3.5.1
    # (multipliers 1.0, same proposers). DEFEND / BREAK adjust mid-game
    # priorities. The greedy-multi allocator is deferred to v1.5 (Layer 1
    # bisect: it regressed 46.9% -> 15.6% due to over-concentration of
    # launches at strong sources).
    sense = sense_state(world, model)
    posture = decide_posture(world, sense, model)
    missions = collect_posture_weighted_missions(world, model, posture)
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
