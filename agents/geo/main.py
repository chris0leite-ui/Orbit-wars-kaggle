"""geo v2.4 — geo v2.3 + ladder-safe wallclock via lite_greedy follow-up.

EVAL VERDICT for v2.3 (n=64 each):
  vs v3.5.1 (2P):       40/64 = 62.5%, Wlo=0.503  [+12pp lift]
  vs v7_0   (2P):       37/64 = 57.8%, Wlo=0.456  [+8pp over our live agent]
  vs 3x v7_0 (4P FFA):  32/64 = 50.0% first-place, Wlo=0.381 (vs 25% baseline)

But v2.3 had max=1915ms in 2P which would forfeit on the 1000ms ladder
limit. v2.4 fixes this with two changes:

  1. score_candidate(followup_policy=lite_greedy_policy)
     ~1ms/call vs top_tier_mirror's ~10ms. K=10 with 2 seats means
     K-1 = 9 follow-up steps × 2 seats × ~9ms saved = ~160ms saved per
     score_candidate. The first-step opp policy stays Tier-1 (mirror).
     Score validation: lite_greedy bit-different from mirror but maintains
     the same strategic shape (ROI-greedy launches, 0.7 ship fraction).

  2. WALLCLOCK_MS 500 -> 350: tighter pre-candidate gate so we never
     enter a 5th candidate when we're already over budget.

KEY EDGE OVER v7_0: v7_0 falls back to v3.5.1 incumbent in 4P games
(33% of live ladder per HANDOVER). This agent runs lookahead-validated
candidate selection in BOTH 2P and 4P via score_candidate_4p
(K=8, "us minus best-other-opp" scoring head, all-3-opps as Tier-1 mirror).


Combines:
- v7's K=10 forward-sim lookahead (`lib/v7_search.py:score_candidate`)
- v7's drop-one candidate enumeration (capped to top 2)
- Geometric extras: 4 sense-driven tilts
- TOP-10 ARCHETYPE BLEND: concentrated artillery + saturation pressure
- "Avoid comets" — all comet targets filtered before settlement

The K=10 lookahead is the SAFETY NET: each candidate is scored against
the incumbent. If a candidate regresses, the incumbent wins argmax and
the candidate is silently discarded.

Pipeline per turn (priority-ordered candidate set):
    obs -> World + WorldModel + sense_state
    -> base missions (snipe aggressive + reinforce + opening), no comets
    -> incumbent_action via settle_plan
    -> candidates in priority order:
         0. incumbent              (always; floor)
         1. opening-boost tilt     (steps 0-15; 68% opening losses)
         2. enemy-focus tilt       (top-10 targets enemy 2.3x more)
         3. concentrated archetype (snipe ships scaled to 0.9 of garrison)
         4. saturation archetype   (multi-launch per source via greedy-multi)
         5. front-reinforce tilt   (geometric defense bias)
         6. voronoi-filter tilt    (geometric attack bias)
         7. drop-one variants      (top 2)
    -> score each via fast_sim K=10 + Tier-1 opp mirror
    -> HARD wallclock gate BEFORE each score; skip if elapsed > WALLCLOCK_MS
    -> argmax -> action

Both top-10 archetypes (concentrated artillery, saturation pressure)
co-exist at the top per knowledge-base/concepts/top-performer-strategies.md.
This agent generates one candidate of each, lets the lookahead pick.
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
from lib.opp_model import lite_greedy_policy
from lib.planner import settle_plan
from lib.v7_search import (
    _action_from_intents, _infer_num_seats,
    score_candidate, score_candidate_4p,
)
from lib.world_model import WorldModel

from lib.geo.sense import SenseState, sense_state


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

K_LOOKAHEAD = 10        # 2P depth
K_LOOKAHEAD_4P = 8      # 4P shallower (3 opponents = more compute per step)
WALLCLOCK_MS = 350.0    # hard pre-candidate gate (was 500; ladder limit 1000)
TIE_TOLERANCE = 1e-6
MAX_DROP_ONE_VARIANTS = 2   # capped to 2; 2 archetype tilts replace the budget

# Tilt magnitudes — the K=10 lookahead validates each, so 1.5-2x is safe.
TILT_OPENING_BOOST = 2.0      # opening missions, steps 0-15 (68% opening losses signal)
TILT_OPENING_STEP_LIMIT = 15
TILT_ENEMY_FOCUS = 1.5        # snipe targeting enemy-owned planets (2.3x top-10 signal)
TILT_FRONT_REINFORCE = 1.5    # reinforce missions targeting front planets
CONCENTRATED_FRACTION = 0.9   # snipe ship-fraction (vs default aggressive 0.7)
# saturation archetype: multi-launch per source via lib/geo/allocator.py:allocate_greedy_multi
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


def _concentrated_archetype_tilt(world: World) -> Callable[[Mission], Mission | None]:
    """Top-10 'concentrated artillery' — same targets, BIGGER fleets.

    Rescales snipe missions' ship counts toward CONCENTRATED_FRACTION (0.9)
    of source garrison, vs the default aggressive 0.7. Same target, same
    score (settle_plan picks the same mission); the source just sends a
    larger fraction of its garrison. Top-10 #1 / #4 / #6 fingerprint.
    """
    pbi = world.planets_by_id
    def tilt(m: Mission) -> Mission | None:
        if m.mission_class != "snipe":
            return m
        src = pbi.get(m.src_id)
        if src is None:
            return m
        garrison = int(src.ships)
        if garrison <= m.ships:
            return m
        big = max(m.ships, int(garrison * CONCENTRATED_FRACTION))
        big = min(big, garrison - 1)  # leave at least 1 to keep the planet
        if big <= m.ships:
            return m
        return Mission(
            mission_class=m.mission_class,
            src_id=m.src_id, target_id=m.target_id,
            ships=big, score=m.score, eta=m.eta, note=m.note,
        )
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


def _saturation_archetype_action(
    base: list[Mission], world: World, model: WorldModel, sense: SenseState,
) -> list[list]:
    """Top-10 'saturation pressure' — multi-launch per source.

    Uses lib/geo/allocator.py:allocate_greedy_multi (global score-sort,
    multi-launch-per-source within source garrison). v1 bisect found this
    settlement loses -31pp standalone, but as ONE candidate validated by
    K=10 lookahead, the lookahead drops it on turns where it overcommits.
    On turns where multi-launch is genuinely better (large garrisons,
    multiple high-value targets), the lookahead picks it.
    """
    # Local import keeps this candidate optional: any import error or
    # allocator failure falls back to the incumbent floor via try/except.
    from lib.geo.allocator import allocate_greedy_multi
    from lib.geo.posture import Posture
    intents = allocate_greedy_multi(base, world, sense, Posture.EXPAND, model)
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
    _add_tilt("concentrated",  _concentrated_archetype_tilt(world))
    # Saturation uses a different SETTLEMENT (multi-launch), not a per-mission tilt.
    try:
        sat_action = _saturation_archetype_action(base_missions, world, model, sense)
        key = _action_key(sat_action)
        if key not in seen:
            candidates.append(("saturation", sat_action))
            seen.add(key)
    except Exception:
        pass
    _add_tilt("front_reinforce", _front_reinforce_tilt(sense))
    _add_tilt("voronoi_filter",  _voronoi_filter_tilt(sense, world))

    # Drop-one variants (capped) — proven v7_0 floor.
    for variant in _drop_one_capped(incumbent_action, MAX_DROP_ONE_VARIANTS):
        key = _action_key(variant)
        if key not in seen:
            candidates.append(("drop_one", variant))
            seen.add(key)

    # Score in order; HARD pre-candidate gate so a slow score can't push
    # the next one over the ladder timeout.
    # 2P / 4P dispatch: v7_0 falls back to v3.5.1 incumbent in 4P (per
    # lib/v7_search.py:choose line 1384). 33% of live games are 4P, so
    # this is a real edge: we get lookahead-validated candidates in both.
    num_seats = _infer_num_seats(world)
    if num_seats not in (2, 4):
        # 3P or other oddity — bail to incumbent.
        return incumbent_action
    snap = fs_from_obs(obs, configuration, episode_seed=0, num_seats=num_seats)
    best_action = incumbent_action
    best_score = float("-inf")
    scored_any = False
    for _name, cand in candidates:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > WALLCLOCK_MS:
            break
        try:
            if num_seats == 2:
                # lite_greedy follow-up: 2P opp's first turn still uses Tier-1
                # (opp_tier=1); the K-1 mirror-mirror follow-up steps use the
                # cheap (~1-2ms) lite_greedy_policy instead of top_tier_mirror
                # (~10ms). Saves ~160ms per score_candidate at K=10.
                score = score_candidate(
                    snap, cand, my_id=my_id, K=K_LOOKAHEAD, opp_tier=1,
                    followup_policy=lite_greedy_policy,
                )
            else:  # 4P
                # score_candidate_4p has no followup_policy parameter — all
                # seats use top_tier_mirror_policy. 4P p95=712ms in our eval
                # was already under ladder limit, so no fix needed there.
                score = score_candidate_4p(
                    snap, cand, my_id=my_id, K=K_LOOKAHEAD_4P,
                )
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
