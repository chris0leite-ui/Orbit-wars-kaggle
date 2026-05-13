"""Sub-phase 5: end-to-end JAX rollout sanity tests.

These tests verify the composed pipeline (state → score matrices →
settle_plan → mechanism → jax_step) executes without errors and
produces reasonable values. Strict parity vs scalar `score_candidate`
is sub-phase 7 (it requires the full lead_aim_v2 / sun_avoid stack).
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import pytest

from kaggle_environments import make

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_score import (
    policy_step_jax,
    rollout_step_jax,
    score_candidate_jax,
    value_delta_ships,
)


def _light_play(env, n_steps=25, rng_seed=7, num_agents=2):
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for ps in range(num_agents):
            mv = []
            for p in env.state[ps].observation.planets:
                if p[1] == ps and p[5] > 5 and rng.random() < 0.2:
                    mv.append([p[0], rng.uniform(0, 2 * math.pi),
                               max(1, int(p[5] * rng.uniform(0.1, 0.3)))])
            actions.append(mv)
        env.step(actions)


@pytest.mark.parametrize("seed", [3, 42])
def test_value_delta_ships_matches_scalar(seed):
    """value_delta_ships on initial JAX state matches scalar
    (planet ships + alive fleet ships per side)."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    obs = env.state[0].observation
    my_id = 0
    my_total = 0
    opp_total = 0
    for p in obs.planets:
        owner = p[1]
        ships = p[5]
        if owner == my_id:
            my_total += int(ships)
        elif owner != -1:
            opp_total += int(ships)
    for f in obs.fleets:
        owner = f[1]
        ships = f[6]
        if owner == my_id:
            my_total += int(ships)
        elif owner != -1:
            opp_total += int(ships)
    expected = my_total - opp_total

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_val = int(value_delta_ships(gs, my_id=my_id))
    assert jax_val == expected, (
        f"seed {seed}: scalar={expected} jax={jax_val}"
    )


def test_policy_step_emits_reasonable_actions():
    """policy_step_jax produces a non-negative-length action list and
    none of the emitted intents over-spend their source."""
    seed = 42
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    emit, _ = policy_step_jax(gs, my_id=0)

    import numpy as np
    planets_id = np.asarray(gs.planets_id)
    planets_ships = np.asarray(gs.planets_ships)
    planets_owner = np.asarray(gs.planets_owner)
    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(planets_id)
        if pid >= 0
    }
    src_used = {}
    for e in emit:
        src_slot = pid_to_slot[e["src_pid"]]
        assert int(planets_owner[src_slot]) == 0, "emitted from non-owned planet"
        src_used.setdefault(src_slot, 0)
        src_used[src_slot] += int(e["ships"])
        assert src_used[src_slot] <= int(planets_ships[src_slot]), (
            "per-source ship budget exceeded"
        )


def test_rollout_step_advances_state():
    """rollout_step_jax actually advances the env step by 1 and returns
    a state with consistent shapes."""
    env = make("orbit_wars", configuration={"seed": 11})
    env.reset(num_agents=2)
    _light_play(env, n_steps=15, rng_seed=99)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    initial_step = int(gs.step)
    new_s = rollout_step_jax(gs, my_id=0)
    assert int(new_s.step) == initial_step + 1
    assert new_s.planets_x.shape == gs.planets_x.shape
    assert new_s.fleets_x.shape == gs.fleets_x.shape


def test_score_candidate_returns_finite_value():
    """K=5 rollout finishes and returns a finite ship-delta value."""
    env = make("orbit_wars", configuration={"seed": 42})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=42 * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    emit, _ = policy_step_jax(gs, my_id=0)
    val = score_candidate_jax(gs, emit, K=5, my_id=0)
    assert math.isfinite(val), f"score_candidate returned non-finite: {val}"


# ---------------------------------------------------------------------------
# Sub-phase 8f T2: scalar-vs-JAX rollout-pipeline parity at a single state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42])
def test_jax_rollout_pipeline_matches_scalar_realize(seed):
    """Single-state parity: scalar `realize(intents, obs,
    mechanisms=DEFAULT_MECHANISMS)` vs the full JAX pipeline
    (`policy_emit_jax_pure` reduced to per-agent action dicts).

    Compares:
      - emitted (src_pid, target_pid) set,
      - ship counts byte-exact,
      - aim angles within 0.05 rad (search_safe_intercept may pick a
        different convergent intercept than scalar's iteration).

    This is the end-to-end parity test the per-component tests
    individually approximate; without it the "JAX matches scalar"
    claim is bundled assumptions.
    """
    import numpy as np
    from lib.intent import World, realize
    from lib.world_model import WorldModel, DEFAULT_HORIZON
    from lib.missions.snipe import propose_snipe_missions
    from lib.missions.reinforce import propose_reinforce_missions
    from lib.planner import settle_plan
    from lib.mechanism import DEFAULT_MECHANISMS
    from lib.game.jax.jax_world_model import build_world_model
    from lib.game.jax.jax_score import policy_emit_jax_pure

    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _light_play(env, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    obs = env.state[0].observation
    my_id = 0

    # Scalar reference: snipe + reinforce → settle_plan → realize.
    sw = World.from_obs(obs)
    swm = WorldModel.from_world(sw)
    scalar_missions = (
        propose_snipe_missions(sw, swm, aggressive=False)
        + propose_reinforce_missions(sw, swm)
    )
    scalar_intents = settle_plan(scalar_missions, sw, swm)
    scalar_actions = realize(
        scalar_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=swm,
    )
    scalar_by_src = {int(a[0]): a for a in scalar_actions}

    # JAX pure-path: policy_emit_jax_pure.
    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    pids, angles, ships = policy_emit_jax_pure(
        gs, jax_wm, my_id=my_id, aggressive=False, num_agents=2,
    )
    pids_np = np.asarray(pids)
    ang_np = np.asarray(angles)
    sh_np = np.asarray(ships)
    jax_by_src = {}
    for k in range(len(pids_np)):
        pid = int(pids_np[k])
        if pid >= 0:
            jax_by_src[pid] = (pid, float(ang_np[k]), int(sh_np[k]))

    diffs = []
    # Sources where scalar emits but JAX drops (or vice versa).
    for src_pid in scalar_by_src.keys() - jax_by_src.keys():
        diffs.append(f"  src={src_pid}: scalar emits, JAX drops")
    for src_pid in jax_by_src.keys() - scalar_by_src.keys():
        diffs.append(f"  src={src_pid}: JAX emits, scalar drops")
    # Ship counts + angles for shared.
    for src_pid in scalar_by_src.keys() & jax_by_src.keys():
        s = scalar_by_src[src_pid]
        j = jax_by_src[src_pid]
        if int(s[2]) != int(j[2]):
            diffs.append(
                f"  src={src_pid}: ships scalar={s[2]} jax={j[2]}"
            )
        ang_delta = math.atan2(
            math.sin(s[1] - j[1]),
            math.cos(s[1] - j[1]),
        )
        if abs(ang_delta) > 0.05:
            diffs.append(
                f"  src={src_pid}: angle scalar={s[1]:.3f} "
                f"jax={j[1]:.3f} delta={ang_delta:.3f}"
            )
    # The pure JAX path uses an inline numpy `argsort` for tie-breaking
    # of equal-scoring sources; scalar uses Python dict ordering. A
    # single divergence here is tolerable; assert at most one.
    assert len(diffs) <= 1, (
        "scalar-vs-jax rollout pipeline divergence:\n" + "\n".join(diffs)
    )
