"""geo v2.1 — geometric sense + K=10 lookahead, ladder-safe wallclock.

Combines:
- v7's K=10 forward-sim lookahead (`lib/v7_search.py:score_candidate`)
- v7's drop-one candidate enumeration (the proven floor; capped to top 3)
- Geometric extras: 4 tilts, scored in priority order, dropped under budget pressure
- "Avoid comets" — all comet targets filtered before settlement

The K=10 lookahead is the SAFETY NET: each tilt is scored against the
incumbent. If a tilt regresses, the incumbent wins argmax and the tilt
is silently discarded. This sidesteps the v1 lesson that "2x cross-class
multipliers regress -37pp": with the lookahead validating, we can push
tilts up to 2x because the floor is protected.

Pipeline per turn:
    obs -> World + WorldModel + sense_state
    -> base missions (snipe aggressive + reinforce + opening), no comets
    -> incumbent_action via settle_plan
    -> priority-ordered candidate set:
         0. incumbent              (always; floor)
         1. opening-boost tilt     (steps 0-15; 68% opening losses)
         2. enemy-focus tilt       (top-10 targets enemy 2.3x more)
         3. front-reinforce tilt   (geometric defense bias)
         4. voronoi-filter tilt    (geometric attack bias)
         5. drop-one variants      (top 3 by score-per-ship)
    -> score each via fast_sim K=10 + Tier-1 opp mirror
    -> HARD wallclock gate: skip remaining if elapsed > WALLCLOCK_MS BEFORE scoring
    -> argmax -> action

Wallclock. WALLCLOCK_MS=500 (down from 700) and the gate is BEFORE
score_candidate (not after), so a slow score can't push the next one
into the ladder timeout. Test bench (v3.5.1 eval pre-tightening):
p95=820ms / max=1919ms. Target after tightening: p95<700ms / max<950ms.
"""

from __future__ import annotations

import time
from typing import Callable

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import Intent, World
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
WALLCLOCK_MS = 500.0          # hard pre-candidate gate (was 700; ladder limit 1000)
TIE_TOLERANCE = 1e-6
MAX_DROP_ONE_VARIANTS = 3     # cap drop-one to top-3 by ships dropped (smallest fleets first)

# Tilt magnitudes — the K=10 lookahead validates each, so 1.5-2x is safe.
# Reference: v1 bisect found cross-class >=2x regressed -37pp WITHOUT lookahead.
TILT_OPENING_BOOST = 2.0      # opening missions, steps 0-15 (68% opening losses signal)
TILT_OPENING_STEP_LIMIT = 15
TILT_ENEMY_FOCUS = 1.5        # snipe targeting enemy-owned planets (2.3x top-10 signal)
TILT_FRONT_REINFORCE = 1.5    # reinforce missions targeting front planets
# voronoi-filter tilt: drops snipe missions to neutrals NOT in our cell (binary)


# ---------------------------------------------------------------------------
# Comet filter (user directive: "avoid comets for now")
# ---------------------------------------------------------------------------


def _drop_comet_missions(missions: list[Mission], world: World) -> list[Mission]:
    return [m for m in missions if m.target_id not in world.comet_ids]


def _build_base_missions(world: World, model: WorldModel) -> list[Mission]:
    missions = (
        propose_opening_missions(world, model)
        + propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    return _drop_comet_missions(missions, world)


# ---------------------------------------------------------------------------
# Tilts — each returns a function `mission -> mission | None` (None drops)
# ---------------------------------------------------------------------------


def _opening_boost_tilt(world: World) -> Callable[[Mission], Mission | None]:
    """Boost opening missions in early game; otherwise no-op."""
    if int(world.step) > TILT_OPENING_STEP_LIMIT:
        return lambda m: m
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "opening":
            return m
        return _scaled(m, TILT_OPENING_BOOST)
    return tilt


def _enemy_focus_tilt(world: World) -> Callable[[Mission], Mission | None]:
    """Boost snipe missions targeting ENEMY-OWNED (not neutral) planets."""
    pbi = world.planets_by_id
    my_id = world.my_id
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "snipe":
            return m
        t = pbi.get(m.target_id)
        if t is None or t.owner in (my_id, -1):
            return m  # neutral or our own — unchanged
        return _scaled(m, TILT_ENEMY_FOCUS)
    return tilt


def _front_reinforce_tilt(sense: SenseState) -> Callable[[Mission], Mission | None]:
    front = sense.front_pids
    if not front:
        return None  # signal: skip tilt entirely
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "reinforce" or m.target_id not in front:
            return m
        return _scaled(m, TILT_FRONT_REINFORCE)
    return tilt


def _voronoi_filter_tilt(sense: SenseState, world: World
                         ) -> Callable[[Mission], Mission | None]:
    """Drop snipe missions targeting neutrals NOT in our Voronoi cell."""
    voronoi = sense.voronoi
    if not voronoi:
        return None  # nothing classified -> no filter signal
    pbi = world.planets_by_id
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "snipe":
            return m
        t = pbi.get(m.target_id)
        if t is None or t.owner != -1:
            return m  # enemy targets unaffected
        owner_cluster = voronoi.get(m.target_id)
        if owner_cluster is None or owner_cluster < 0:
            return None  # neutral not in our cell -> drop
        return m
    return tilt


def _scaled(m: Mission, mult: float) -> Mission:
    return Mission(
        mission_class=m.mission_class,
        src_id=m.src_id, target_id=m.target_id,
        ships=m.ships, score=m.score * mult,
        eta=m.eta, note=m.note,
    )


def _settle_with_tilt(
    base: list[Mission], world: World, model: WorldModel,
    tilt: Callable[[Mission], Mission | None],
) -> list[list]:
    tilted = [out for out in (tilt(m) for m in base) if out is not None]
    intents = settle_plan(tilted, world, model)
    return _action_from_intents(intents, world.obs_raw, model)


# ---------------------------------------------------------------------------
# Drop-one capped to top-N (drops the smallest fleet first — least
# impactful drops considered first, so we keep the meaningful variants).
# ---------------------------------------------------------------------------


def _drop_one_capped(action: list[list], cap: int) -> list[list[list]]:
    if not action:
        return []
    # Order by ascending ship-count: drop the smallest first.
    indexed = sorted(range(len(action)), key=lambda i: int(action[i][2]))
    out: list[list[list]] = []
    for i in indexed[:cap]:
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

    base_missions = _build_base_missions(world, model)
    incumbent_intents = settle_plan(base_missions, world, model)
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)

    # Build candidate list in PRIORITY ORDER (highest expected EV first).
    seen = {_action_key(incumbent_action)}
    candidates: list[tuple[str, list[list]]] = [("incumbent", incumbent_action)]

    def _add_tilt(name: str, tilt: Callable[[Mission], Mission | None] | None):
        if tilt is None:
            return
        try:
            act = _settle_with_tilt(base_missions, world, model, tilt)
        except Exception:
            return
        key = _action_key(act)
        if key not in seen:
            candidates.append((name, act))
            seen.add(key)

    _add_tilt("opening_boost", _opening_boost_tilt(world))
    _add_tilt("enemy_focus",   _enemy_focus_tilt(world))
    _add_tilt("front_reinforce", _front_reinforce_tilt(sense))
    _add_tilt("voronoi_filter",  _voronoi_filter_tilt(sense, world))

    # Drop-one variants (capped) — proven v7_0 floor.
    for variant in _drop_one_capped(incumbent_action, MAX_DROP_ONE_VARIANTS):
        key = _action_key(variant)
        if key not in seen:
            candidates.append(("drop_one", variant))
            seen.add(key)

    # Score in order; HARD pre-candidate gate so a slow score can't push the
    # next one over the ladder timeout.
    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=2)
    best_action = incumbent_action
    best_score = float("-inf")
    scored_any = False
    for _name, cand in candidates:
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
