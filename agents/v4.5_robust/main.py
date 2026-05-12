"""v4.5_robust — v4_planner with maximin-over-opp-models scoring.

The "best of both worlds" merge:

- **v4_planner's framework** — 5 mission portfolios (incumbent /
  conservative / per_source_swap / drop_weakest_source / noop), goal-
  shaped value head (`evaluate_value`: production-share + denial +
  ships + survivor), adaptive K, comet-boundary truncation, watchdog +
  per-step deadline, 4P fallback to v3.5.1.
- **v7_minimax's principle** — instead of scoring each portfolio's
  action under a single fixed opponent policy (v3.5.1), score it
  against TWO opponent models {O0 = v3.5.1, O1 = v3.5.1 with smallest
  launch dropped} and take the **min**. Argmax over portfolios of this
  worst-case score. This is maximin at the action level, using v4's
  value head throughout (not v7's ship-delta head).

Why this is additive over v4_planner: v4 scored expected value under a
fixed opp; v4.5 scores worst-case value over an opp-model class. If
v4's chosen portfolio is dominated against the more aggressive opp
model, v4.5 picks something more defensible.

Budget: doubling rollouts from 1 to 2 opp models means K must drop.
v4 ran K=6-10; v4.5 runs K=4-7 (adaptive on entropy, capped). If even
that projects over the 750ms hard deadline, we **fall back to v4's
single-opp scoring** for the remaining portfolios — never crash.

4P fallback unchanged: v3.5.1, because the value head is 2P-tuned.

Submission-bundle constraint: agents/ is not on sys.path; everything
must be inlined or imported from lib/. v7-specific helpers
(`_drop_smallest`, `_swap_obs_player`) are inlined here.
"""

from __future__ import annotations

import time

from lib.candidate_portfolios import generate_portfolios
from lib.intent import World, realize
from lib.lookahead import env_from_obs, score_action, score_joint_action_symmetric
from lib.lookahead_planner import adaptive_K, evaluate_value, truncate_K_to_comet_boundary
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


# Same hard deadline as v4. The maximin scoring is up to 2x v4's cost
# but K is reduced and the start-watchdog cuts later candidates first.
_HARD_DEADLINE_MS = 750.0
_START_WATCHDOG_MS = 550.0

# K caps: 2 opp models × seat-symmetric × K steps. K=7 keeps a single
# (portfolio, opp) cell at ~80ms; 2 opps × 2 seats × 5 portfolios = 20
# cells * 80ms = 1600ms worst-case but the watchdog kills it after 550ms.
# In practice the start-watchdog ends the loop around portfolio 3-4.
_K_MIN = 4
_K_MAX = 7

# Fall-back-to-v4 threshold: if maximin cost for one portfolio cell
# (single opp, symmetric) exceeds this, switch the loop to v4-style
# single-opp scoring for remaining portfolios. Calibrated to leave
# ~150ms of budget after a cell.
_PER_CELL_BUDGET_MS = 180.0


def _v351_action(obs):
    """v3.5.1 incumbent — fallback action + rollout policy.

    Inlined for the submission bundle (agents/ not on sys.path).
    """
    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    return realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)


def _drop_smallest(action: list) -> list:
    """Drop the smallest-ship launch from `action`. Empty if ≤1 launch.

    Used to build the "less aggressive opp" model O1: v3.5.1's predicted
    action minus its smallest launch. Ties broken by earliest-index for
    σ-determinism.
    """
    if not action:
        return []
    if len(action) == 1:
        return []
    min_idx = 0
    min_ships = int(action[0][2])
    for i, la in enumerate(action[1:], start=1):
        if int(la[2]) < min_ships:
            min_ships = int(la[2])
            min_idx = i
    return [la for i, la in enumerate(action) if i != min_idx]


def _swap_obs_player(obs, opp_id: int):
    """Shallow-copy obs with `player` set to opp_id (run v3.5.1 from opp POV)."""
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    keys = (
        "player", "planets", "fleets", "angular_velocity",
        "initial_planets", "comet_planet_ids", "comets",
        "step", "next_fleet_id", "remainingOverageTime",
    )
    obs2 = {}
    for k in keys:
        v = getattr(obs, k, None)
        if v is not None:
            obs2[k] = v
    obs2["player"] = opp_id
    return obs2


def _num_distinct_players(obs) -> int:
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    owners: set[int] = set()
    for p in planets:
        if int(p[1]) >= 0:
            owners.add(int(p[1]))
    for f in fleets:
        if int(f[1]) >= 0:
            owners.add(int(f[1]))
    return len(owners)


def _opp_models(obs, opp_id: int) -> list[list]:
    """Build the 2-element opp-model class: {v3.5.1, v3.5.1 drop-smallest}.

    Dedup if the smaller class collapses (e.g. opp has ≤1 launch → both
    models predict the same empty action).
    """
    swapped = _swap_obs_player(obs, opp_id)
    try:
        o0 = _v351_action(swapped)
    except Exception:
        return [[]]
    o1 = _drop_smallest(o0)
    if repr(o0) == repr(o1):
        return [o0]
    return [o0, o1]


def _robust_K(world) -> int:
    """Adaptive K, capped lower than v4 because maximin doubles cost."""
    K = adaptive_K(world)
    return max(_K_MIN, min(K, _K_MAX))


def agent(obs, configuration=None):
    t_start = time.perf_counter()

    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    if not raw_planets:
        return []

    # 4P fallback — value head + maximin assumptions are 2P-only.
    if _num_distinct_players(obs) > 2:
        return _v351_action(obs)

    world = World.from_obs(obs)
    if not world.planets_by_id:
        return []
    model = WorldModel.from_world(world)
    my_id = world.my_id
    opp_id = 1 - my_id

    # Always-safe incumbent action (returned on any timeout / exception).
    incumbent_missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    incumbent_intents = settle_plan(incumbent_missions, world, model)
    incumbent_action = realize(
        incumbent_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
    )

    portfolios = generate_portfolios(world, model, incumbent_missions)

    # Short-circuit: trivial portfolio set → skip the env reconstruction.
    if len(portfolios) <= 2 and all(
        p.label in ("incumbent", "noop") for p in portfolios
    ):
        return incumbent_action

    K = truncate_K_to_comet_boundary(_robust_K(world), world.step)

    try:
        sim_env = env_from_obs(obs, configuration)
    except Exception:
        return incumbent_action

    # Build the 2 opp models ONCE, reuse across portfolios.
    O = _opp_models(obs, opp_id)

    hard_deadline = t_start + _HARD_DEADLINE_MS / 1000.0
    best_action = incumbent_action
    best_v = float("-inf")
    fallback_to_single_opp = False

    for portfolio in portfolios:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > _START_WATCHDOG_MS:
            break

        # Realize this portfolio's action.
        try:
            if portfolio.label == "incumbent":
                action = incumbent_action
            else:
                intents = settle_plan(portfolio.missions, world, model)
                action = realize(
                    intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model
                )
        except Exception:
            continue

        # Score this portfolio.
        try:
            if fallback_to_single_opp:
                # Budget-aware path: revert to v4-style single-opp scoring
                # for the remaining portfolios.
                v = score_action(
                    sim_env, action, K=K, my_id=my_id,
                    policy=_v351_action, value_fn=evaluate_value,
                    deadline=hard_deadline,
                )
            else:
                # Robust path: maximin over the opp-model class, using
                # seat-symmetric scoring (cancels the env's P1 bias) with
                # v4's value head as the leaf function.
                cell_start = time.perf_counter()
                scores = []
                for opp_action in O:
                    s = score_joint_action_symmetric(
                        sim_env, action, opp_action, K=K,
                        policy=_v351_action, value_fn=evaluate_value,
                        deadline=hard_deadline,
                    )
                    scores.append(s)
                cell_elapsed_ms = (time.perf_counter() - cell_start) * 1000.0
                # If a single cell ran long, switch downstream portfolios
                # to single-opp scoring rather than risk timeout.
                if cell_elapsed_ms > _PER_CELL_BUDGET_MS:
                    fallback_to_single_opp = True
                v = min(scores) if scores else float("-inf")
        except Exception:
            continue

        if v > best_v:
            best_v = v
            best_action = action

    return best_action
