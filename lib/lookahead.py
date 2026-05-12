"""K-step forward-simulation scorer + candidate enumerator.

The Phase 2 probe (audit/2026-05-11-lookahead-phase2-forward-sim.md)
showed that `env.clone()` + `env.step()` repeated for K turns under a
fixed rollout policy is statistically indistinguishable from the
perfect oracle (AUC matches O50 to 0.000 at probe step 50). This
module wraps that into a production-grade scorer the agent can use.

Two pieces:

1. `env_from_obs(obs, configuration) -> Environment`
   Reconstructs a steppable env from agent-visible state. Used because
   `kaggle_environments` doesn't hand the live env to the agent — only
   obs + configuration are exposed inside `agent(obs, configuration)`.
   The seed is not in the agent's configuration on the live ladder, but
   the orbit_wars env's future steps are determined by current state +
   action sequence, not seed. The single fidelity gap is future comet
   spawns (steps 50/150/250/350/450) which use the env's RNG; for sims
   that don't cross a spawn boundary the rebuild is bit-exact.

2. `score_action(env, action, K, my_id, policy) -> float`
   Clones `env`, applies `action` as our P{my_id} launch this turn,
   then rolls forward K-1 more turns under `policy` as both players.
   Returns the (us - them) ship-total scalar at the rollout's final
   state — the same scoring head the Phase 2 probe validated.

Plus `enumerate_drop_one_candidates(action)` — the simplest non-trivial
candidate set: the incumbent action + each "drop one launch" variant.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Sequence

from kaggle_environments import make


def env_from_obs(obs, configuration: dict | None = None):
    """Build a fresh steppable env mirroring the current obs.

    Both player-states get the same public observation; only the
    `player` field is per-seat. status/reward are reset; action is None
    (will be filled in by step()).
    """
    cfg = dict(configuration or {})
    env = make("orbit_wars", configuration=cfg, debug=False)
    env.reset(num_agents=2)
    snapshot_keys = (
        "planets", "fleets", "comets", "comet_planet_ids",
        "initial_planets", "angular_velocity", "step", "next_fleet_id",
    )
    public = {k: copy.deepcopy(obs[k]) for k in snapshot_keys if k in obs}
    for i in range(2):
        env.state[i].observation.update(public)
        env.state[i].observation["player"] = i
        env.state[i].observation["remainingOverageTime"] = obs.get(
            "remainingOverageTime", 60.0
        )
        env.state[i].status = "ACTIVE"
        env.state[i].reward = 0
        env.state[i].action = None
    return env


def _ship_total_by_owner(observation) -> dict[int, float]:
    """Sum ships on owned planets + in fleets per owner."""
    totals: dict[int, float] = {}
    for p in observation.get("planets", []):
        owner = int(p[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(p[5])
    for f in observation.get("fleets", []):
        owner = int(f[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(f[6])
    return totals


def score_action(
    env,
    action: list,
    K: int,
    my_id: int,
    policy: Callable,
    value_fn: Callable | None = None,
    deadline: float | None = None,
) -> float:
    """Sim<K> score of taking `action` this turn, then K-1 turns of
    `policy` as both players.

    If `value_fn` is None (default — backward-compat for v3_lookahead),
    returns (our - opp) total ships. If `value_fn(observation, my_id)`
    is supplied, returns its scalar applied to the leaf observation —
    used by v4_planner with the production-share/denial head.

    `deadline` is an optional `time.perf_counter()` timestamp; the
    rollout loop checks before each remaining step and aborts early
    (returning the value at the partial-leaf state). This is the
    load-bearing robustness mechanism — without it a slow rollout
    can blow past the 1 s actTimeout regardless of the caller's
    pre-call budget estimate.

    Caller is responsible for passing `env` already-reconstructed; this
    function clones it so the caller can reuse `env` across candidates.
    """
    clone = env.clone()
    opp_id = 1 - my_id
    # First step: our forced action; opp plays policy on their obs.
    a_opp = policy(clone.state[opp_id].observation)
    actions = [None, None]
    actions[my_id] = action
    actions[opp_id] = a_opp
    if not clone.done:
        clone.step(actions)
    # Remaining K-1 steps: both players use policy.
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        if deadline is not None and time.perf_counter() > deadline:
            break
        a0 = policy(clone.state[0].observation)
        a1 = policy(clone.state[1].observation)
        clone.step([a0, a1])
    leaf_obs = clone.state[my_id].observation
    if value_fn is not None:
        return value_fn(leaf_obs, my_id)
    totals = _ship_total_by_owner(leaf_obs)
    return totals.get(my_id, 0.0) - totals.get(opp_id, 0.0)


def score_joint_action(
    env,
    our_action: list,
    opp_action: list,
    K: int,
    my_id: int,
    policy: Callable,
    value_fn: Callable | None = None,
    deadline: float | None = None,
) -> float:
    """Sim<K> score with BOTH first-turn actions injected.

    Unlike `score_action` (which lets `policy` choose opp's first move),
    `score_joint_action` forces both `our_action` and `opp_action` on
    turn 0, then rolls forward K-1 turns under `policy` as both players.

    If `value_fn` is None (default — backward-compat for v7_minimax),
    returns (our_ships - opp_ships) at the rollout's leaf. If
    `value_fn(observation, my_id)` is supplied, returns its scalar
    applied to the leaf observation — used by v4.5_robust with v4's
    production-share / denial head.

    `deadline` is an optional `time.perf_counter()` timestamp; the
    rollout aborts early if exceeded, returning the value at the
    partial-leaf state.
    """
    clone = env.clone()
    opp_id = 1 - my_id
    actions = [None, None]
    actions[my_id] = our_action
    actions[opp_id] = opp_action
    if not clone.done:
        clone.step(actions)
    for _ in range(max(0, K - 1)):
        if clone.done:
            break
        if deadline is not None and time.perf_counter() > deadline:
            break
        a0 = policy(clone.state[0].observation)
        a1 = policy(clone.state[1].observation)
        clone.step([a0, a1])
    leaf_obs = clone.state[my_id].observation
    if value_fn is not None:
        return value_fn(leaf_obs, my_id)
    totals = _ship_total_by_owner(leaf_obs)
    return totals.get(my_id, 0.0) - totals.get(opp_id, 0.0)


def score_joint_action_symmetric(
    env,
    our_action: list,
    opp_action: list,
    K: int,
    policy: Callable,
    value_fn: Callable | None = None,
    deadline: float | None = None,
) -> float:
    """Seat-symmetric variant — averages over both seat assignments.

    Cancels the documented P1-favoring seat bias in `kaggle_environments`
    so v7_minimax's payoff matrix is invariant under seat-flip; without
    this, P0 and P1's maximin picks diverge and σ-equiv self-play
    breaks. Cost is 2x score_joint_action; callers must budget K
    accordingly (v7 drops K from 5 → 3 → 2 under deadline pressure).

    Forwards `value_fn` + `deadline` to both inner calls so v4.5_robust
    can score a maximin-over-opp-models matrix with v4's goal-shaped
    value head.
    """
    a = score_joint_action(
        env, our_action, opp_action, K, my_id=0, policy=policy,
        value_fn=value_fn, deadline=deadline,
    )
    b = score_joint_action(
        env, our_action, opp_action, K, my_id=1, policy=policy,
        value_fn=value_fn, deadline=deadline,
    )
    return (a + b) / 2.0


def enumerate_drop_one_candidates(action: list) -> list[list]:
    """Generate the smallest non-trivial candidate set.

    Returns [action] + [action with launch i removed for each i].
    A pure subset of the incumbent — we never propose new launches the
    incumbent didn't already consider. With N launches we evaluate
    N + 1 candidates, which controls budget at ~(N+1) × K × 5.6 ms.

    For N=0 (incumbent does nothing) returns [[]] — only the empty
    action; no rollouts needed.
    """
    if not action:
        return [[]]
    cands: list[list] = [list(action)]
    for i in range(len(action)):
        cands.append([m for j, m in enumerate(action) if j != i])
    return cands
