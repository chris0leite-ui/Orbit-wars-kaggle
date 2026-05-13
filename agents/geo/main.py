"""geo v1 — geometric sense + posture decision, settled by settle_plan.

Status: parity with v3.5.1 (n=64: 31/64 = 48.4%, Wilson [0.366, 0.604]).
Architecture is wired end-to-end and verified safe; no value-add yet.

Pipeline:
    obs -> World.from_obs -> WorldModel.from_world
        -> sense_state(world, model)              # lib/geo/sense.py
        -> decide_posture(world, sense, model)    # lib/geo/posture.py
        -> collect_posture_weighted_missions(...) # local
        -> settle_plan(missions, world, model)    # lib/planner.py (PROVEN)
        -> realize(intents, mechanisms=DEFAULT_MECHANISMS)

The substrate (sense_state, decide_posture) is computed every turn and
available to the missions/allocator stages. The current v1 doesn't
*use* the posture beyond reading it — POSTURE_WEIGHTS are all 1.0 and
_aggressive_for is constant True. This makes geo v1 functionally
equivalent to `propose_snipe(aggressive=True) + propose_reinforce +
settle_plan` (i.e., v3.5.1).

Why this minimalism. During v1 bisect (this branch's history):

| Iteration                                       | n=32 winrate | Δ      |
| ----------------------------------------------- | ------------ | ------ |
| (bisect-2) v3.5.1-exact source pipeline         | 46.9%        | base   |
| greedy-multi allocator                          | 15.6%        | -31pp  |
| posture mults (DEFEND ×2 reinforce, ×0.5 snipe) | 9.4%         | -37pp  |
| _aggressive_for(DEFEND) = False                 | 25.0%        | -22pp  |
| (current) aggressive=True, mults 1.0            | 48.4% (n=64) | +1.5pp |

Every value-add attempt regressed. The architecture is sound; the
heuristics that "obviously" should have helped (multi-launch, posture
multipliers, defensive sizing) all hurt. v1.5 work is to find geometric
filters/priors that respect the existing well-tuned scoring rather than
override it.

Deferred to v1.5 (the lib/geo/{sense,posture,allocator}.py code is
already in place to support these):
- **Voronoi-aware target prior**: in OPENING posture, prefer neutrals
  closer to our cluster than to any enemy cluster (sense.voronoi).
- **Front-pressure reinforce**: scale reinforce score by `1 + front_pressure(pid)`
  so reinforce of front planets > reinforce of interior.
- **Multi-launch allocator with per-source launch cap**: `allocate_greedy_multi`
  but bounded by `min(N, garrison // ship_per_launch)`. Doesn't over-concentrate.
- **Comet-claim filter**: drop snipe missions targeting comets where
  `sense.comet_claims[c] is None or != my_cluster_idx` (H15-style).

POSTURE_WEIGHTS / _enemy_only_filter scaffolding is kept here so v1.5
iterations have a clear seam to bias against. NOT used today.
"""

from __future__ import annotations

from lib.intent import World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.opening import propose_opening_missions  # noqa: F401  (v1.5)
from lib.missions.recapture import propose_recapture_missions  # noqa: F401  (v1.5)
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel

from lib.geo.allocator import allocate_greedy_multi  # noqa: F401  (v1.5)
from lib.geo.posture import Posture, decide_posture
from lib.geo.sense import sense_state


# ---------------------------------------------------------------------------
# Posture × mission-class score multipliers (v1: all 1.0; v1.5 lever)
# ---------------------------------------------------------------------------

POSTURE_WEIGHTS: dict[Posture, dict[str, float]] = {
    Posture.OPENING: {"snipe": 1.0, "reinforce": 1.0},
    Posture.EXPAND:  {"snipe": 1.0, "reinforce": 1.0},
    Posture.DEFEND:  {"snipe": 1.0, "reinforce": 1.0},
    Posture.BREAK:   {"snipe": 1.0, "reinforce": 1.0},
}


def _scale_scores(missions: list[Mission], mult: float) -> list[Mission]:
    if mult == 1.0:
        return missions
    if mult == 0.0:
        return []
    return [
        Mission(
            mission_class=m.mission_class,
            src_id=m.src_id, target_id=m.target_id,
            ships=m.ships, score=m.score * mult,
            eta=m.eta, note=m.note,
        )
        for m in missions
    ]


def collect_posture_weighted_missions(
    world: World, model: WorldModel, posture: Posture,
) -> list[Mission]:
    """Run posture-relevant proposers and apply per-class score multipliers.

    With all multipliers 1.0 (v1), this is effectively
    `propose_snipe(aggressive=True) + propose_reinforce`.
    """
    weights = POSTURE_WEIGHTS.get(posture, {})
    bag: list[Mission] = []
    if "snipe" in weights:
        snipe = propose_snipe_missions(world, model, aggressive=True)
        bag.extend(_scale_scores(snipe, weights["snipe"]))
    if "reinforce" in weights:
        rein = propose_reinforce_missions(world, model)
        bag.extend(_scale_scores(rein, weights["reinforce"]))
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
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)
