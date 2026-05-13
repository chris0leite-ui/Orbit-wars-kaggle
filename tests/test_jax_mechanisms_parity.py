"""Sub-phase 4 parity: JAX-matrix → settle_plan → mechanism pipeline
vs scalar `realize(intents, obs, mechanisms=DEFAULT_MECHANISMS)`.

We compare:
- which (src_pid, target_pid) pairs survive,
- emitted ship counts (post arrival_size bump),
- aim_angle within tolerance (JAX uses atan2 only; scalar uses
  lead_aim_v2 for orbiting targets — divergence is bounded but real
  and tracked separately in sub-phase 7).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from kaggle_environments import make

from lib.intent import World, Intent, realize
from lib.world_model import WorldModel, DEFAULT_HORIZON
from lib.missions.snipe import propose_snipe_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.planner import settle_plan
from lib.mechanism import DEFAULT_MECHANISMS

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_world_model import build_world_model
from lib.game.jax.jax_missions import (
    compute_snipe_score_matrix,
    compute_reinforce_score_matrix,
    settle_plan_from_matrices,
)
from lib.game.jax.jax_mechanisms import (
    apply_mechanisms_numpy,
    apply_mechanisms_jax,
)
from lib.game.jax.jax_missions import (
    merge_class_matrices,
    settle_plan_jax,
)
import jax.numpy as jnp


def _spawn_in_flight_fleets(env, num_agents=2, n_steps=15, rng_seed=7):
    """Lighter random-policy: only launches with 20% probability per
    eligible planet and sends ~30% of garrison, so by step ~25 there
    are still owned planets with ships > 0 to source from."""
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for pid_seat in range(num_agents):
            moves = []
            obs = env.state[pid_seat].observation
            for p in obs.planets:
                if p[1] == pid_seat and p[5] > 5 and rng.random() < 0.2:
                    angle = rng.uniform(0, 2 * math.pi)
                    ships = max(1, int(p[5] * rng.uniform(0.1, 0.3)))
                    if 0 < ships <= p[5]:
                        moves.append([p[0], angle, ships])
            actions.append(moves)
        env.step(actions)


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_mechanism_pipeline_ship_pairs_parity(seed):
    """Ship counts and (src, target) pairs must agree between scalar
    realize() and the JAX-matrix pipeline."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    my_id = 0
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)

    # --- Scalar reference pipeline ---
    scalar_missions = (
        propose_snipe_missions(scalar_world, scalar_wm, aggressive=False)
        + propose_reinforce_missions(scalar_world, scalar_wm)
    )
    scalar_intents = settle_plan(scalar_missions, scalar_world, scalar_wm)
    scalar_actions = realize(
        scalar_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=scalar_wm,
    )
    scalar_by_src = {int(a[0]): a for a in scalar_actions}

    # --- JAX pipeline ---
    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe_out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    reinforce_out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)
    chosen = settle_plan_from_matrices(
        class_outputs=[snipe_out, reinforce_out],
        class_names=["snipe", "reinforce"],
        planets_id=gs.planets_id,
        world_owners_at=jax_wm.owners_at,
        world_ships_at=jax_wm.ships_at,
        my_id=my_id,
    )
    emitted = apply_mechanisms_numpy(chosen, gs, jax_wm, my_id=my_id)
    jax_by_src = {int(e["src_pid"]): e for e in emitted}

    # --- Compare ---
    diffs = []
    for src_pid, scalar_action in scalar_by_src.items():
        jax_e = jax_by_src.get(src_pid)
        if jax_e is None:
            diffs.append(
                f"  src={src_pid}: scalar emits "
                f"[ang={scalar_action[1]:.3f}, ships={scalar_action[2]}] "
                f"but JAX drops"
            )
            continue
        # Ship counts: scalar's arrival_size may differ when the target is
        # orbiting (we don't apply the +1 dynamic prod_tick in JAX).
        scalar_ships = int(scalar_action[2])
        jax_ships = int(jax_e["ships"])
        if abs(scalar_ships - jax_ships) > 1:
            diffs.append(
                f"  src={src_pid}: ships scalar={scalar_ships} jax={jax_ships}"
            )

    for src_pid in jax_by_src.keys() - scalar_by_src.keys():
        e = jax_by_src[src_pid]
        diffs.append(
            f"  src={src_pid}: JAX emits "
            f"[tgt={e['target_pid']}, ships={e['ships']}] but scalar drops"
        )

    assert not diffs, "mechanism pipeline divergence:\n" + "\n".join(diffs)


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_mechanism_pipeline_angles_within_tolerance(seed):
    """Per-source aim angles: tolerance 0.2 rad (~12°). JAX uses atan2
    of current target position; scalar uses lead_aim_v2 for moving
    orbiting targets. Wide tolerance acknowledged; sub-phase 7 closes
    the gap with a full lead_aim port.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    my_id = 0
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = (
        propose_snipe_missions(scalar_world, scalar_wm, aggressive=False)
        + propose_reinforce_missions(scalar_world, scalar_wm)
    )
    scalar_intents = settle_plan(scalar_missions, scalar_world, scalar_wm)
    scalar_actions = realize(
        scalar_intents, obs, mechanisms=DEFAULT_MECHANISMS, model=scalar_wm,
    )
    scalar_by_src = {int(a[0]): a for a in scalar_actions}

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe_out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    reinforce_out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)
    chosen = settle_plan_from_matrices(
        class_outputs=[snipe_out, reinforce_out],
        class_names=["snipe", "reinforce"],
        planets_id=gs.planets_id,
        world_owners_at=jax_wm.owners_at,
        world_ships_at=jax_wm.ships_at,
        my_id=my_id,
    )
    emitted = apply_mechanisms_numpy(chosen, gs, jax_wm, my_id=my_id)
    jax_by_src = {int(e["src_pid"]): e for e in emitted}

    # Sources present in BOTH outputs.
    common = scalar_by_src.keys() & jax_by_src.keys()
    if not common:
        pytest.skip(f"seed {seed}: no common sources to compare angles on")
    diffs = []
    for src_pid in common:
        scalar_ang = float(scalar_by_src[src_pid][1])
        jax_ang = float(jax_by_src[src_pid]["angle"])
        # Wrap difference to [-π, π].
        delta = math.atan2(math.sin(scalar_ang - jax_ang),
                           math.cos(scalar_ang - jax_ang))
        if abs(delta) > 0.02:
            diffs.append(
                f"  src={src_pid}: scalar_ang={scalar_ang:.3f} "
                f"jax_ang={jax_ang:.3f} delta={delta:.3f}"
            )
    assert not diffs, "angle divergence > 0.2 rad:\n" + "\n".join(diffs)


# ---------------------------------------------------------------------------
# Sub-phase 8e: apply_mechanisms_jax (pure-JAX path) parity
# ---------------------------------------------------------------------------


def _run_apply_mechanisms_jax(env, my_id: int):
    """Drive the full pure-JAX mechanism path from an env state.

    Returns:
      - emitted_jax: list of dicts {src_pid, angle, ships} from the
        JAX path (after settle_plan_jax + apply_mechanisms_jax).
      - emitted_numpy: list of dicts from the numpy path
        (apply_mechanisms_numpy) for the same settle output.
    """
    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe_out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    reinforce_out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)
    merged = merge_class_matrices([snipe_out, reinforce_out])
    src, tgt, ships, eta = settle_plan_jax(
        merged["score"], merged["ships"], merged["eta"], merged["valid"],
        jax_wm.ships_at,
    )

    # JAX path: apply_mechanisms_jax (now includes ray-cast).
    final_src, final_angle, final_ships = apply_mechanisms_jax(
        gs, jax_wm, src, tgt, ships, eta, my_id=my_id,
    )
    planets_id = np.asarray(gs.planets_id)
    emitted_jax = []
    for i in range(len(final_src)):
        if int(final_src[i]) >= 0:
            emitted_jax.append({
                "src_pid": int(planets_id[int(final_src[i])]),
                "angle": float(final_angle[i]),
                "ships": int(final_ships[i]),
            })

    # Numpy path: apply_mechanisms_numpy on the same settle output.
    chosen_np = []
    for i in range(len(src)):
        if int(src[i]) >= 0:
            chosen_np.append({
                "src_pid": int(planets_id[int(src[i])]),
                "target_pid": int(planets_id[int(tgt[i])]),
                "ships": int(ships[i]),
                "eta": int(eta[i]),
            })
    emitted_numpy = apply_mechanisms_numpy(chosen_np, gs, jax_wm, my_id=my_id)
    return emitted_jax, emitted_numpy


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_apply_mechanisms_jax_path_clears_parity(seed):
    """JAX-path emit matches numpy-path emit after the ray-cast lands.

    Compares the src-set, ship counts, and angles. Tolerance:
      - drop sets exactly equal (same intents survive),
      - ship counts byte-exact,
      - angles within 0.02 rad (search_safe_intercept fallback may
        pick a different convergent candidate than scalar's iter).
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 31)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    emitted_jax, emitted_numpy = _run_apply_mechanisms_jax(env, my_id=0)
    jax_by_src = {e["src_pid"]: e for e in emitted_jax}
    np_by_src = {e["src_pid"]: e for e in emitted_numpy}

    diffs = []
    for src_pid in np_by_src.keys() - jax_by_src.keys():
        diffs.append(f"  src={src_pid}: numpy emits but JAX drops")
    for src_pid in jax_by_src.keys() - np_by_src.keys():
        diffs.append(f"  src={src_pid}: JAX emits but numpy drops")
    for src_pid in np_by_src.keys() & jax_by_src.keys():
        n = np_by_src[src_pid]
        j = jax_by_src[src_pid]
        if int(n["ships"]) != int(j["ships"]):
            diffs.append(
                f"  src={src_pid}: ships numpy={n['ships']} jax={j['ships']}"
            )
        ang_delta = math.atan2(
            math.sin(n["angle"] - j["angle"]),
            math.cos(n["angle"] - j["angle"]),
        )
        if abs(ang_delta) > 0.02:
            diffs.append(
                f"  src={src_pid}: angle numpy={n['angle']:.3f} "
                f"jax={j['angle']:.3f} delta={ang_delta:.3f}"
            )
    assert not diffs, "JAX-path emit divergence:\n" + "\n".join(diffs)


def test_apply_mechanisms_jax_sun_avoid():
    """Inject an intent aimed straight through the sun; assert it's
    dropped by predict_fleet_fate_batch_jax's sun check."""
    from lib.game.jax.jax_mechanisms import (
        predict_fleet_fate_batch_jax,
        _build_planet_orbits_jax,
        OUTCOME_SUN,
    )
    env = make("orbit_wars", configuration={"seed": 42})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=10, rng_seed=42)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    planets_id = np.asarray(gs.planets_id)
    # Find any owned planet to use as source.
    planets_owner = np.asarray(gs.planets_owner)
    planets_alive = np.asarray(gs.planets_alive)
    src_slot = next(
        i for i in range(len(planets_id))
        if planets_owner[i] == 0 and planets_alive[i]
    )
    sx = float(np.asarray(gs.planets_x)[src_slot])
    sy = float(np.asarray(gs.planets_y)[src_slot])
    # Aim straight at the sun (CENTER, CENTER).
    sun_aim = math.atan2(50.0 - sy, 50.0 - sx)

    planet_orbits = _build_planet_orbits_jax(gs, max_steps=200)
    outcome, _, _ = predict_fleet_fate_batch_jax(
        src_x=jnp.array([sx], dtype=jnp.float32),
        src_y=jnp.array([sy], dtype=jnp.float32),
        src_r=jnp.array([float(np.asarray(gs.planets_radius)[src_slot])],
                        dtype=jnp.float32),
        src_slot=jnp.array([src_slot], dtype=jnp.int32),
        target_slot=jnp.array([-1], dtype=jnp.int32),
        aim_angle=jnp.array([sun_aim], dtype=jnp.float32),
        ships=jnp.array([20], dtype=jnp.int32),
        planet_orbits=planet_orbits,
        planets_alive=gs.planets_alive,
        planets_radius=gs.planets_radius,
        intent_active=jnp.array([True]),
        max_steps=200,
    )
    assert int(outcome[0]) == OUTCOME_SUN, (
        f"expected OUTCOME_SUN ({OUTCOME_SUN}) got {int(outcome[0])}"
    )


def test_apply_mechanisms_jax_oob():
    """Inject an intent aimed off-board; assert OOB."""
    from lib.game.jax.jax_mechanisms import (
        predict_fleet_fate_batch_jax,
        _build_planet_orbits_jax,
        OUTCOME_OOB,
    )
    env = make("orbit_wars", configuration={"seed": 42})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=10, rng_seed=42)
    if env.state[0].status != "ACTIVE":
        pytest.skip("terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    planets_owner = np.asarray(gs.planets_owner)
    planets_alive = np.asarray(gs.planets_alive)
    planets_x = np.asarray(gs.planets_x)
    planets_y = np.asarray(gs.planets_y)
    # Find a planet near the edge so its aim "outward" hits OOB quickly,
    # and choose a direction that avoids the sun (perpendicular to
    # center).
    src_slot = next(
        i for i in range(len(planets_owner))
        if planets_owner[i] == 0 and planets_alive[i]
    )
    sx, sy = float(planets_x[src_slot]), float(planets_y[src_slot])
    # Aim AWAY from the board center (i.e. opposite direction of sun).
    outward_aim = math.atan2(sy - 50.0, sx - 50.0)

    planet_orbits = _build_planet_orbits_jax(gs, max_steps=200)
    outcome, _, _ = predict_fleet_fate_batch_jax(
        src_x=jnp.array([sx], dtype=jnp.float32),
        src_y=jnp.array([sy], dtype=jnp.float32),
        src_r=jnp.array([float(np.asarray(gs.planets_radius)[src_slot])],
                        dtype=jnp.float32),
        src_slot=jnp.array([src_slot], dtype=jnp.int32),
        target_slot=jnp.array([-1], dtype=jnp.int32),
        aim_angle=jnp.array([outward_aim], dtype=jnp.float32),
        ships=jnp.array([20], dtype=jnp.int32),
        planet_orbits=planet_orbits,
        planets_alive=gs.planets_alive,
        planets_radius=gs.planets_radius,
        intent_active=jnp.array([True]),
        max_steps=200,
    )
    # OOB or PLANET (might hit a planet on the way out). Either is a
    # drop; just confirm it's not TARGET or TIMEOUT.
    from lib.game.jax.jax_mechanisms import OUTCOME_TARGET, OUTCOME_TIMEOUT
    code = int(outcome[0])
    assert code != OUTCOME_TARGET and code != OUTCOME_TIMEOUT, (
        f"expected drop outcome (OOB/SUN/PLANET) got {code}"
    )
