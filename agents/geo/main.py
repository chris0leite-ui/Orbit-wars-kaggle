"""geo v2 — geometric sense + K=10 lookahead selection.

Combines:
- v7's K=10 forward-sim lookahead (`lib/v7_search.py:score_candidate`)
- v7's drop-one candidate enumeration (the proven floor)
- Geometric extras: front-reinforce tilt + Voronoi-filter tilt
- "Avoid comets" — all comet targets filtered before settlement

The K=10 lookahead is the SAFETY NET: each geo-tilt candidate is scored
against the incumbent. If the tilt makes things worse, the incumbent
wins the argmax and the tilt is silently dropped. This sidesteps the
"posture multipliers regress" problem found in v1 (see
knowledge-base/thoughts/2026-05-13-geo-v1-bisect-lessons.md): the
lookahead validates each tilt rather than blindly applying it.

Pipeline per turn:
    obs -> World + WorldModel + sense_state
    -> base missions (snipe aggressive + reinforce + opening), no comets
    -> incumbent_action via settle_plan
    -> candidate set:
         [incumbent] + drop_one(incumbent)
         + [front-tilt action]      # if any front planets
         + [voronoi-filter action]  # if any in-Voronoi neutrals
    -> score each via fast_sim K=10 + Tier-1 opp mirror
    -> argmax -> action

Compute budget. v7_0 with drop-one+K=10 ran p95 ~750 ms; adding 2 geo
extras adds at most ~200 ms (K=10 score ~80-100 ms each). Hard wallclock
gate at 700 ms cuts the loop early if needed — incumbent is the floor.
"""

from __future__ import annotations

import time
from typing import Callable

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.mission import Mission
from lib.missions.opening import propose_opening_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.v7_search import _action_from_intents, score_candidate
from lib.world_model import WorldModel

from lib.geo.sense import SenseState, sense_state


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

K_LOOKAHEAD = 10
WALLCLOCK_MS = 700.0
TIE_TOLERANCE = 1e-6
FRONT_REINFORCE_TILT = 1.3   # gentle bias; >=1.5 regressed in v1 bisect


# ---------------------------------------------------------------------------
# Base missions — comets filtered (user directive: "avoid comets for now")
# ---------------------------------------------------------------------------


def _drop_comet_missions(missions: list[Mission], world: World) -> list[Mission]:
    """Hard-drop every mission whose target is currently a comet."""
    return [m for m in missions if m.target_id not in world.comet_ids]


def _build_base_missions(world: World, model: WorldModel) -> list[Mission]:
    """v7_1-style mission set (opening + aggressive snipe + reinforce),
    with all comet targets dropped."""
    missions = (
        propose_opening_missions(world, model)
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    return _drop_comet_missions(missions, world)


# ---------------------------------------------------------------------------
# Tilts — small multiplicative biases on Mission.score, then re-settle.
# Score deltas are <=1.5x so they don't crush the natural class hierarchy
# (cross-class >=2x regressed -37pp in v1 bisect).
# ---------------------------------------------------------------------------


def _front_reinforce_tilt(sense: SenseState) -> Callable[[Mission], Mission | None]:
    """Boost reinforce missions whose TARGET is on the front."""
    front = sense.front_pids
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "reinforce" or m.target_id not in front:
            return m
        return Mission(
            mission_class=m.mission_class,
            src_id=m.src_id, target_id=m.target_id,
            ships=m.ships, score=m.score * FRONT_REINFORCE_TILT,
            eta=m.eta, note=m.note,
        )
    return tilt


def _voronoi_filter_tilt(sense: SenseState, world: World
                         ) -> Callable[[Mission], Mission | None]:
    """Drop snipe missions targeting neutrals NOT in our Voronoi cell.

    Targets where an enemy reaches first are wasted; the source's
    runner-up wins settle_plan's per-source slot instead.
    """
    voronoi = sense.voronoi
    pbi = world.planets_by_id
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "snipe":
            return m
        t = pbi.get(m.target_id)
        if t is None or t.owner != -1:
            return m  # enemy targets unaffected
        # Neutral target — keep only if in OUR voronoi cell.
        owner_cluster = voronoi.get(m.target_id)
        if owner_cluster is None or owner_cluster < 0:
            return None  # not in any of our cells (contested or enemy-faster)
        return m
    return tilt


def _settle_with_tilt(
    base: list[Mission], world: World, model: WorldModel,
    tilt: Callable[[Mission], Mission | None],
) -> list[list]:
    """Apply tilt to each mission, drop Nones, settle, convert to action."""
    tilted_missions = []
    for m in base:
        out = tilt(m)
        if out is not None:
            tilted_missions.append(out)
    intents = settle_plan(tilted_missions, world, model)
    # _action_from_intents requires the original obs to build a World.
    # We have world already but need to pass `model` through `realize`.
    # Mirror what lib/v7_search.py:_action_from_intents does.
    return _action_from_intents(intents, world.obs_raw, model)


# ---------------------------------------------------------------------------
# Candidate enumerator
# ---------------------------------------------------------------------------


def _drop_one(action: list[list]) -> list[list[list]]:
    if not action:
        return [[]]
    out: list[list[list]] = [list(action)]
    for i in range(len(action)):
        out.append([m for j, m in enumerate(action) if j != i])
    return out


def _action_key(action: list[list]) -> tuple:
    return tuple(sorted((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent(obs, configuration=None):
    t_start = time.perf_counter()

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    sense = sense_state(world, model)
    my_id = world.my_id

    # Build incumbent (= v7_1 with no comets).
    base_missions = _build_base_missions(world, model)
    incumbent_intents = settle_plan(base_missions, world, model)
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # Candidate set: incumbent + drop-one variants + geo tilts.
    candidates: list[list[list]] = _drop_one(incumbent_action)
    seen = {_action_key(c) for c in candidates}

    # Front-reinforce tilt — only if there are front planets to bias toward.
    if sense.front_pids:
        try:
            front_action = _settle_with_tilt(
                base_missions, world, model, _front_reinforce_tilt(sense)
            )
            key = _action_key(front_action)
            if key not in seen:
                candidates.append(front_action)
                seen.add(key)
        except Exception:
            pass

    # Voronoi-filter tilt — only if we have any neutrals classified.
    if sense.voronoi:
        try:
            vor_action = _settle_with_tilt(
                base_missions, world, model, _voronoi_filter_tilt(sense, world)
            )
            key = _action_key(vor_action)
            if key not in seen:
                candidates.append(vor_action)
                seen.add(key)
        except Exception:
            pass

    # Score each candidate via fast_sim K=10 + Tier-1 opp mirror.
    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    best_action = incumbent_action
    best_score = float("-inf")
    scored_any = False
    for cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > WALLCLOCK_MS:
            break
        try:
            score = score_candidate(snap, cand, my_id=my_id, K=K_LOOKAHEAD, opp_tier=1)
        except Exception:
            continue
        if not scored_any:
            scored_any = True
            best_score = score
            best_action = list(cand)
            continue
        if score > best_score + TIE_TOLERANCE:
            best_score = score
            best_action = list(cand)
    return best_action
