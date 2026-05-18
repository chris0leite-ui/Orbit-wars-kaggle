"""bundle — trajectory-native chooser using BundleSearch + mirror-search opp model.

Wallclock discipline: each turn carves the env's 1000ms actTimeout into
a mirror-search budget (default 250ms) and an own-search budget (default
550ms). BundleSearch honors the deadlines and returns the best bundle
seen so far when budget runs out; the empty-bundle floor is preserved.
The remaining ~200ms is margin for World construction + emit + env
overhead. Override via BUNDLE_TOTAL_MS, BUNDLE_MIRROR_MS.

Pipeline (per turn):
  1. World.from_obs               build the trajectory layer snapshot.
  2. predict opp bundles via mirror search (depth=1)
                                  predict each opponent's best bundle
                                  by running BundleSearch from their seat.
  3. BundleSearch.search(seed=last_bundle.shift_forward(1),
                         opp_overlays=mirror_dict)
                                  beam search; score against realistic
                                  counterplay, not a passive world.
  4. emit specs_at_turn(0)        Future-turn launches stay in the
                                  carry-over bundle and persist via
                                  seed-and-extend next turn.

Knobs (env var overrides, all optional):
  BUNDLE_OWN_MAX_DEPTH         our search beam depth.             default 2
  BUNDLE_OWN_BEAM_WIDTH                                            default 3
  BUNDLE_OWN_CANDS_PER_SOURCE                                      default 2
  BUNDLE_OWN_LAUNCH_TURNS      comma-separated turn offsets.       default "0"
  BUNDLE_OPP_MAX_DEPTH         mirror search beam depth.           default 1
  BUNDLE_OPP_BEAM_WIDTH                                            default 2
  BUNDLE_OPP_CANDS_PER_SOURCE                                      default 2
  BUNDLE_MIRROR_DEPTH          recursion depth for the opp model.  default 1
  BUNDLE_OPP_MODE              "mirror" (Phase 8a recursive bundle
                               search) or "event_driven" (Phase 8b
                               per-arrival lite_greedy snapshots).  default mirror
  BUNDLE_HORIZON               evaluator horizon (turns).          default 30
  BUNDLE_TOTAL_MS              total per-turn budget (own_deadline). default 750
  BUNDLE_MIRROR_MS             mirror sub-budget within total.       default 250
  BUNDLE_PLANET_WEIGHT                                             default 5.0
  BUNDLE_PRODUCTION_WEIGHT     coefficient on path-integrated      default 1.0
                               production_delta (turns of ownership).
  BUNDLE_ELIMINATION_BONUS                                         default 200.0
"""

from __future__ import annotations

import os
import time
from typing import Optional

from lib.trajectory_layer import (
    Bundle,
    BundleEvaluator,
    BundleSearch,
    World,
    predict_opp_bundles_via_mirror_search,
    predict_opp_via_event_driven_lite_greedy,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_turns(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if v >= 0:
            out.append(v)
    return tuple(out) if out else default


def _build_searches() -> tuple[BundleSearch, BundleSearch, int]:
    ev = BundleEvaluator(
        horizon=_env_int("BUNDLE_HORIZON", 15),
        planet_weight=_env_float("BUNDLE_PLANET_WEIGHT", 5.0),
        production_weight=_env_float("BUNDLE_PRODUCTION_WEIGHT", 1.0),
        elimination_bonus=_env_float("BUNDLE_ELIMINATION_BONUS", 200.0),
        my_followup_mode=os.environ.get("BUNDLE_ME_FOLLOWUP", "off").lower(),
    )
    # Tuned 2026-05-18 with with_candidate cache-inheritance perf
    # fix in place. Full 363-turn game vs sloppy random at depth=2:
    # p50=49ms p95=131ms max=160ms, 0/363 over 1000ms cap. Massive
    # headroom vs the live-env actTimeout=1000ms. Depth=2 enables
    # gang-up bundles (the structurally-invisible-to-greedy pattern
    # BundleSearch was designed for). Bumping further
    # (launch_turns=(0,5,10) + depth=3) is feasible but pushes
    # per-turn cost up; tune per-A/B once the depth=2 ceiling is
    # measured.
    own = BundleSearch(
        evaluator=ev,
        max_depth=_env_int("BUNDLE_OWN_MAX_DEPTH", 2),
        beam_width=_env_int("BUNDLE_OWN_BEAM_WIDTH", 3),
        candidates_per_source=_env_int("BUNDLE_OWN_CANDS_PER_SOURCE", 2),
        launch_turns=_env_turns("BUNDLE_OWN_LAUNCH_TURNS", (0,)),
    )
    opp = BundleSearch(
        evaluator=ev,
        max_depth=_env_int("BUNDLE_OPP_MAX_DEPTH", 1),
        beam_width=_env_int("BUNDLE_OPP_BEAM_WIDTH", 2),
        candidates_per_source=_env_int("BUNDLE_OPP_CANDS_PER_SOURCE", 2),
        launch_turns=(0,),  # mirror keeps it cheap — no delayed-launch enum
    )
    mirror_depth = _env_int("BUNDLE_MIRROR_DEPTH", 1)
    return own, opp, mirror_depth


# Per-process state. Kaggle reuses the agent process across turns of a
# single game; resets are detected via (player_id, step==0) so a process
# that hosts multiple games behaves cleanly.
_LAST_BUNDLE: dict[int, Bundle] = {}


def _carry_over(my_id: int, step: int) -> Optional[Bundle]:
    """Pop the previous turn's bundle (if any) and shift it forward by
    one game step. Returns None if there's nothing to carry or it's
    turn 0 of a new game."""
    if step == 0:
        # New game — clear any leftover from a prior episode in the
        # same process.
        _LAST_BUNDLE.pop(my_id, None)
        return None
    prev = _LAST_BUNDLE.get(my_id)
    if prev is None or prev.is_empty:
        return None
    shifted = prev.shift_forward(1)
    if shifted.is_empty:
        return None
    return shifted


def _as_obs_dict(obs) -> dict:
    """Coerce obs to a plain dict so World.from_obs handles both
    Kaggle's SimpleNamespace-like obs and direct dicts (test path)."""
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "initial_planets": list(getattr(obs, "initial_planets", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def agent(obs, configuration=None):
    t_start = time.perf_counter()
    total_budget_ms = _env_float("BUNDLE_TOTAL_MS", 750.0)
    mirror_budget_ms = _env_float("BUNDLE_MIRROR_MS", 250.0)

    obs_d = _as_obs_dict(obs)
    me = int(obs_d.get("player", 0))
    step = int(obs_d.get("step", 0))
    raw_planets = obs_d.get("planets", []) or []
    if not raw_planets:
        return []

    # Need at least one of our planets to launch from.
    my_planets = [p for p in raw_planets if int(p[1]) == me]
    if not my_planets:
        # We're eliminated — no actions possible. Reset carry-over.
        _LAST_BUNDLE.pop(me, None)
        return []

    world = World.from_obs(obs_d, configuration=configuration)
    own_search, opp_search, mirror_depth = _build_searches()
    seed = _carry_over(me, step)

    mirror_deadline = t_start + mirror_budget_ms / 1000.0
    own_deadline = t_start + total_budget_ms / 1000.0

    # Opp model: "mirror" (Phase 8a: BundleSearch recursion, the
    # current default — assumes opp is bundle-search-like) or
    # "event_driven" (Phase 8b: walks the trajectory layer's arrival
    # event stream, calls lite_greedy_policy at each event,
    # accumulates a per-opp reactive Bundle — analogous to
    # agents/baseline's per-step opp model but built without
    # stepping a simulator).
    opp_mode = os.environ.get("BUNDLE_OPP_MODE", "mirror").lower()
    if opp_mode == "event_driven":
        mirror = predict_opp_via_event_driven_lite_greedy(
            world,
            my_id=me,
            horizon=own_search.evaluator.horizon,
        )
    else:
        mirror = _mirror_with_deadline(
            world, me, opp_search, mirror_depth, mirror_deadline,
        )

    chosen = own_search.search(
        world,
        my_id=me,
        seed_bundle=seed,
        opp_overlays=mirror,
        deadline=own_deadline,
    )

    _LAST_BUNDLE[me] = chosen

    # Emit only the launches scheduled for THIS turn; future-turn
    # commitments persist via the carry-over for next turn.
    actions = []
    for spec in chosen.specs_at_turn(0):
        actions.append([int(spec.src_id), float(spec.aim_angle), int(spec.ships)])
    return actions


def _mirror_with_deadline(world: World, me: int,
                          opp_search: BundleSearch,
                          depth: int, deadline: float,
                          ) -> dict[int, Bundle]:
    """Run mirror-search per opponent under a wallclock budget. Split
    the remaining time evenly across opps; bail early if budget runs
    out. Returns whatever partial mapping we built — BundleEvaluator
    silently drops opps that aren't in the dict (they're treated as
    passive)."""
    if depth <= 0:
        return {}
    opp_ids: list[int] = []
    seen: set[int] = set()
    for p in world.planets:
        if p.is_comet:
            continue
        if p.owner == -1 or p.owner == me:
            continue
        if p.owner in seen:
            continue
        seen.add(p.owner)
        opp_ids.append(p.owner)
    if not opp_ids:
        return {}

    out: dict[int, Bundle] = {}
    for i, opp_id in enumerate(sorted(opp_ids)):
        now = time.perf_counter()
        if now >= deadline:
            break
        remaining_ms = (deadline - now) * 1000.0
        opps_left = len(opp_ids) - i
        per_opp_ms = remaining_ms / opps_left
        opp_deadline = now + per_opp_ms / 1000.0
        bundle = opp_search.search(
            world,
            my_id=opp_id,
            opp_overlays={} if depth == 1 else None,
            deadline=opp_deadline,
        )
        if not bundle.is_empty:
            out[opp_id] = bundle
    return out
