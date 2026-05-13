"""Sub-phase 3 parity: JAX missions vs scalar mission builders.

Currently tests:
- `compute_snipe_score_matrix` (base, non-aggressive) vs scalar
  `lib.missions.snipe.propose_snipe_missions(aggressive=False)`.

Future sub-phases add aggressive sizing (3b), leader-spoiler + comet
bonus (3c), reinforce + recapture (3d), and `settle_plan` (3e).
"""

from __future__ import annotations

import math
import random

import jax.numpy as jnp
import numpy as np
import pytest

from kaggle_environments import make

from lib.intent import World
from lib.world_model import WorldModel
from lib.missions.snipe import propose_snipe_missions
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.recapture import (
    propose_recapture_missions,
    _STATE as _RECAPTURE_STATE,
    _reset_state_for_tests as _recapture_reset,
)

from lib.game.jax import scalar_to_jax
from lib.game.jax.jax_world_model import build_world_model, DEFAULT_HORIZON
from lib.game.jax.jax_missions import (
    compute_snipe_score_matrix,
    compute_reinforce_score_matrix,
    compute_recapture_score_matrix,
    compute_opening_score_matrix,
    settle_plan_from_matrices,
    merge_class_matrices,
    settle_plan_jax,
)
from lib.missions.opening import propose_opening_missions
from lib.planner import settle_plan
import jax.numpy as jnp


def _spawn_in_flight_fleets(env, num_agents: int = 2, n_steps: int = 15, rng_seed: int = 7):
    """Run N steps with random-policy launches so the env has live fleets."""
    rng = random.Random(rng_seed)
    for _ in range(n_steps):
        if env.state[0].status != "ACTIVE":
            break
        actions = []
        for pid_seat in range(num_agents):
            moves = []
            obs = env.state[pid_seat].observation
            for p in obs.planets:
                if p[1] == pid_seat and p[5] > 0 and rng.random() < 0.5:
                    angle = rng.uniform(0, 2 * math.pi)
                    ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
                    if 0 < ships <= p[5]:
                        moves.append([p[0], angle, ships])
            actions.append(moves)
        env.step(actions)


def _check_snipe_parity(env, my_id: int, aggressive: bool, num_agents: int = 4):
    """Run scalar `propose_snipe_missions` and JAX `compute_snipe_score_matrix`
    on the same state, return (diffs, extra) lists.
    """
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = propose_snipe_missions(
        scalar_world, scalar_wm, aggressive=aggressive,
    )

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    out = compute_snipe_score_matrix(
        gs, jax_wm, my_id=my_id, aggressive=aggressive,
        num_agents=num_agents,
    )
    score = np.asarray(out["score"])
    ships = np.asarray(out["ships"])
    eta = np.asarray(out["eta"])
    valid = np.asarray(out["valid"])

    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }

    diffs = []
    matched_pairs = set()
    for m in scalar_missions:
        if m.src_id not in pid_to_slot or m.target_id not in pid_to_slot:
            continue
        s_slot = pid_to_slot[m.src_id]
        t_slot = pid_to_slot[m.target_id]
        matched_pairs.add((s_slot, t_slot))

        if not bool(valid[s_slot, t_slot]):
            # Scalar can produce a mission with time_to_hold=0 → score=0;
            # JAX validity is structural (src/tgt masks + redundancy filter
            # only). If scalar score is 0, mismatch is just because scalar
            # would launch a 0-value mission and JAX still marks it valid.
            # Otherwise this is a real diff.
            if m.score > 1e-6:
                diffs.append(
                    f"  pid={m.src_id}->{m.target_id}: scalar has mission "
                    f"(score={m.score:.4f}) but JAX valid=False"
                )
            continue
        if int(ships[s_slot, t_slot]) != int(m.ships):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: ships scalar={m.ships} "
                f"jax={int(ships[s_slot, t_slot])}"
            )
        if abs(int(eta[s_slot, t_slot]) - int(m.eta)) > 1:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: eta scalar={m.eta} "
                f"jax={int(eta[s_slot, t_slot])}"
            )
        jax_score = float(score[s_slot, t_slot])
        rel = abs(jax_score - m.score) / max(abs(m.score), 1e-6)
        if rel > 1e-3 and abs(jax_score - m.score) > 1e-3:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: score scalar={m.score:.6f} "
                f"jax={jax_score:.6f} rel={rel:.4e}"
            )
        if len(diffs) >= 6:
            break

    P = score.shape[0]
    extra = []
    for s in range(P):
        for t in range(P):
            if not bool(valid[s, t]):
                continue
            if (s, t) in matched_pairs:
                continue
            extra.append(
                f"  src_slot={s} (pid={int(gs.planets_id[s])}) "
                f"-> tgt_slot={t} (pid={int(gs.planets_id[t])}): "
                f"JAX valid, scalar absent. score={float(score[s, t]):.4f}"
            )
            if len(extra) >= 3:
                break
        if len(extra) >= 3:
            break

    return diffs, extra


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_compute_snipe_score_matrix_parity(seed):
    """Sub-phase 3a: JAX snipe score matrix matches scalar
    `propose_snipe_missions(aggressive=False)` per (src, target) pair.

    Scope: 2-player games (LEADER_MULTIPLIER never fires); base form.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 13 + 1)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated before mid-game")

    diffs, extra = _check_snipe_parity(env, my_id=0, aggressive=False)
    assert not diffs, "snipe score matrix divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX snipe matrix has extra valid pairs:\n" + "\n".join(extra)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_compute_snipe_score_matrix_aggressive_parity(seed):
    """Sub-phase 3b: aggressive=True (top-10 fraction sizing) matches scalar.

    Verifies src-conditioned base_ships:
      - garrison ≤ AGGRESSIVE_MIN_GARRISON → falls back to target_min
      - else → max(target_min, min(int(src.ships * 0.7), src.ships - 5))
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 17 + 5)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated before mid-game")

    diffs, extra = _check_snipe_parity(env, my_id=0, aggressive=True)
    assert not diffs, "aggressive snipe matrix divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX aggressive matrix has extra valid pairs:\n" + "\n".join(extra)


@pytest.mark.parametrize("seed", [42, 137])
def test_compute_snipe_score_matrix_with_comet_targets(seed):
    """Sub-phase 3c: comet-lifetime correction.

    Spawn comets by stepping past step=50 (first comet spawn), then
    verify JAX matches scalar on a state that has comet targets in
    `world.comet_ids`. Comet `time_to_hold = max(0, rem_lifetime - eta)`.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=55, rng_seed=seed)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated before comet spawn")

    # Confirm at least one comet exists at this state.
    scalar_world = World.from_obs(env.state[0].observation)
    if not scalar_world.comet_ids:
        pytest.skip(f"seed {seed}: no comets present after 55 steps")

    diffs, extra = _check_snipe_parity(env, my_id=0, aggressive=False)
    assert not diffs, "comet-aware snipe divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX has extra valid pairs:\n" + "\n".join(extra)


@pytest.mark.parametrize("seed", [0, 7, 42, 137])
def test_compute_opening_score_matrix_parity(seed):
    """H11 (2026-05-13): JAX opening proposer matches scalar
    `propose_opening_missions` per (src, target) pair.

    The opening proposer fires only at step <= 5; we run from a fresh
    reset (step 0) and check both step 0 and step 3 to exercise the
    in-window path. After step 5 the proposer must emit zero missions.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)

    # Two checks per seed: step 0 and step 5 (boundary).
    for step in (0, 5):
        # Advance the env without launches until target step.
        while env.state[0].observation.step < step:
            if env.state[0].status != "ACTIVE":
                pytest.skip(f"seed {seed} terminated before step {step}")
            env.step([[], []])
        if env.state[0].status != "ACTIVE":
            pytest.skip(f"seed {seed} terminated before step {step}")

        obs = env.state[0].observation
        sw = World.from_obs(obs)
        if not sw.planets_by_id:
            continue
        swm = WorldModel.from_world(sw)
        scalar_missions = propose_opening_missions(sw, swm)

        gs = scalar_to_jax(env.state, env.info["seed"])
        jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
        out = compute_opening_score_matrix(gs, jax_wm, my_id=0)
        score = np.asarray(out["score"])
        ships = np.asarray(out["ships"])
        valid = np.asarray(out["valid"])
        pid_to_slot = {
            int(pid): slot
            for slot, pid in enumerate(np.asarray(gs.planets_id))
            if pid >= 0
        }

        # Every scalar mission has a matching JAX valid cell.
        for m in scalar_missions:
            s_slot = pid_to_slot[m.src_id]
            t_slot = pid_to_slot[m.target_id]
            assert bool(valid[s_slot, t_slot]), (
                f"seed={seed} step={step} pid={m.src_id}->{m.target_id}: "
                f"scalar emits but JAX valid=False"
            )
            assert int(ships[s_slot, t_slot]) == int(m.ships), (
                f"seed={seed} step={step} pid={m.src_id}->{m.target_id}: "
                f"ships scalar={m.ships} jax={int(ships[s_slot, t_slot])}"
            )
            rel = abs(float(score[s_slot, t_slot]) - m.score) / max(abs(m.score), 1e-6)
            assert rel < 1e-3, (
                f"seed={seed} step={step} pid={m.src_id}->{m.target_id}: "
                f"score scalar={m.score:.4f} jax={float(score[s_slot, t_slot]):.4f}"
            )

        # No extra JAX valid cells.
        scalar_pairs = {
            (pid_to_slot[m.src_id], pid_to_slot[m.target_id])
            for m in scalar_missions
        }
        P = score.shape[0]
        for s in range(P):
            for t in range(P):
                if not bool(valid[s, t]):
                    continue
                assert (s, t) in scalar_pairs, (
                    f"seed={seed} step={step}: JAX valid at slot ({s},{t}) "
                    f"(pid {int(gs.planets_id[s])}->{int(gs.planets_id[t])}) "
                    f"but scalar didn't emit"
                )


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_compute_opening_score_matrix_inactive_after_window(seed):
    """Opening proposer must emit nothing at step > OPENING_WINDOW (5)."""
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    # Advance past the opening window.
    for _ in range(8):
        if env.state[0].status != "ACTIVE":
            pytest.skip(f"seed {seed} terminated early")
        env.step([[], []])
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    out = compute_opening_score_matrix(gs, jax_wm, my_id=0)
    valid = np.asarray(out["valid"])
    assert int(valid.sum()) == 0, (
        f"seed={seed} step={int(gs.step)}: opening must emit nothing "
        f"after window, got {int(valid.sum())} valid cells"
    )


def test_compute_snipe_score_matrix_4p_spoiler():
    """Sub-phase 3c: 4P LEADER_MULTIPLIER fires when our_rank >= 2.

    Run a 4-agent game; pick a seat that is unlikely to lead. Verify
    JAX score matches scalar (which applies ×1.5 to leader's planets).
    """
    seed = 11
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=4)
    _spawn_in_flight_fleets(env, num_agents=4, n_steps=30, rng_seed=seed)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    # Pick whichever seat has our_rank >= 2 (so spoiler fires). We
    # need to check rankings from the scalar side first.
    from lib.missions.snipe import _leader_pid
    chosen_seat = None
    for seat in range(4):
        obs = env.state[seat].observation
        world = World.from_obs(obs)
        leader_pid, rank = _leader_pid(world)
        if leader_pid is not None and rank is not None and rank >= 2:
            chosen_seat = seat
            break
    if chosen_seat is None:
        pytest.skip("no seat is rank>=2; spoiler condition not exercised")

    diffs, extra = _check_snipe_parity(
        env, my_id=chosen_seat, aggressive=False, num_agents=4,
    )
    assert not diffs, "spoiler snipe divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX has extra valid pairs:\n" + "\n".join(extra)


# ---------------------------------------------------------------------------
# Sub-phase 3d: reinforce parity
# ---------------------------------------------------------------------------


def _check_reinforce_parity(env, my_id: int):
    obs = env.state[my_id].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = propose_reinforce_missions(scalar_world, scalar_wm)

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)
    score = np.asarray(out["score"])
    ships = np.asarray(out["ships"])
    eta = np.asarray(out["eta"])
    valid = np.asarray(out["valid"])

    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }

    diffs = []
    matched = set()
    for m in scalar_missions:
        if m.src_id not in pid_to_slot or m.target_id not in pid_to_slot:
            continue
        s_slot = pid_to_slot[m.src_id]
        t_slot = pid_to_slot[m.target_id]
        matched.add((s_slot, t_slot))
        if not bool(valid[s_slot, t_slot]):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: scalar reinforce "
                f"(score={m.score:.4f}) but JAX valid=False"
            )
            continue
        if int(ships[s_slot, t_slot]) != int(m.ships):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: ships scalar={m.ships} "
                f"jax={int(ships[s_slot, t_slot])}"
            )
        if abs(int(eta[s_slot, t_slot]) - int(m.eta)) > 1:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: eta scalar={m.eta} "
                f"jax={int(eta[s_slot, t_slot])}"
            )
        jax_score = float(score[s_slot, t_slot])
        rel = abs(jax_score - m.score) / max(abs(m.score), 1e-6)
        if rel > 1e-3 and abs(jax_score - m.score) > 1e-3:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: score scalar={m.score:.6f} "
                f"jax={jax_score:.6f} rel={rel:.4e}"
            )
        if len(diffs) >= 6:
            break

    # Extras: JAX cells valid but no scalar mission.
    P = score.shape[0]
    extra = []
    for s in range(P):
        for t in range(P):
            if not bool(valid[s, t]):
                continue
            if (s, t) in matched:
                continue
            extra.append(
                f"  src_slot={s} (pid={int(gs.planets_id[s])}) "
                f"-> tgt_slot={t} (pid={int(gs.planets_id[t])}): "
                f"JAX valid, scalar absent. score={float(score[s, t]):.4f}"
            )
            if len(extra) >= 3:
                break
        if len(extra) >= 3:
            break
    return diffs, extra


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_compute_reinforce_score_matrix_parity(seed):
    """JAX reinforce matrix matches scalar propose_reinforce_missions.

    Needs both an under-threat planet (inbound enemy fleet that captures)
    AND another planet of ours that can arrive in time. The 25-step
    random-policy spawn usually produces these conditions.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 23)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    diffs, extra = _check_reinforce_parity(env, my_id=0)
    if not diffs and not extra:
        # Either side produced no candidates — fine, but verify symmetry.
        return
    assert not diffs, "reinforce divergence:\n" + "\n".join(diffs)
    assert not extra, "JAX reinforce extras:\n" + "\n".join(extra)


# ---------------------------------------------------------------------------
# Sub-phase 3d: recapture parity
# ---------------------------------------------------------------------------


def _build_lost_at_step_array(planets_id_arr, lost_at_dict: dict[int, int]):
    """Map scalar `lost_at` dict (planet_id -> step) to a JAX-shape int32
    array indexed by JAX planet slot. -1 elsewhere."""
    P = len(planets_id_arr)
    out = -np.ones(P, dtype=np.int32)
    for pid, step in lost_at_dict.items():
        for slot, slot_pid in enumerate(planets_id_arr):
            if int(slot_pid) == int(pid):
                out[slot] = int(step)
                break
    return jnp.asarray(out)


def test_compute_recapture_score_matrix_parity():
    """JAX recapture matches scalar after we've engineered a recapture
    scenario (we lose a planet, then attempt to recover it next turn).

    Steps:
    1. Reset module-level recapture state.
    2. Step the env a few turns to spawn fleets.
    3. Force-capture one of our planets by sending an attack ourselves
       — too convoluted. Easier: run random play long enough that an
       enemy fleet flips one of our planets, then run propose to
       populate `_STATE.lost_at`.
    """
    _recapture_reset()
    seed = 42
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)

    # Drive the env until we record at least one ownership loss. We have
    # to call propose_recapture_missions each turn to keep _STATE current.
    rng = random.Random(seed * 19 + 1)
    found_loss = False
    for step_count in range(250):
        if env.state[0].status != "ACTIVE":
            break
        obs = env.state[0].observation
        scalar_world = World.from_obs(obs)
        scalar_wm = WorldModel.from_world(scalar_world)
        _ = propose_recapture_missions(scalar_world, scalar_wm)
        if _RECAPTURE_STATE.lost_at:
            found_loss = True
            break
        actions = []
        for pid_seat in range(2):
            moves = []
            obs_seat = env.state[pid_seat].observation
            for p in obs_seat.planets:
                if p[1] == pid_seat and p[5] > 0 and rng.random() < 0.6:
                    angle = rng.uniform(0, 2 * math.pi)
                    ships = max(1, int(p[5] * rng.uniform(0.2, 0.9)))
                    if 0 < ships <= p[5]:
                        moves.append([p[0], angle, ships])
            actions.append(moves)
        env.step(actions)

    if not found_loss:
        pytest.skip("no ownership losses observed within 120 turns")

    # Now run scalar + JAX recapture on this state.
    obs = env.state[0].observation
    scalar_world = World.from_obs(obs)
    scalar_wm = WorldModel.from_world(scalar_world)
    scalar_missions = propose_recapture_missions(scalar_world, scalar_wm)

    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    lost_at = _build_lost_at_step_array(
        np.asarray(gs.planets_id), _RECAPTURE_STATE.lost_at
    )
    out = compute_recapture_score_matrix(gs, jax_wm, my_id=0, lost_at_step=lost_at)
    score = np.asarray(out["score"])
    ships = np.asarray(out["ships"])
    eta = np.asarray(out["eta"])
    valid = np.asarray(out["valid"])

    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }

    # Scalar caps top-K at 5 missions globally. JAX returns the full
    # matrix; verify the scalar-emitted pairs all match a JAX cell.
    diffs = []
    matched = set()
    for m in scalar_missions:
        if m.src_id not in pid_to_slot or m.target_id not in pid_to_slot:
            continue
        s_slot = pid_to_slot[m.src_id]
        t_slot = pid_to_slot[m.target_id]
        matched.add((s_slot, t_slot))
        if not bool(valid[s_slot, t_slot]):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: scalar emits "
                f"(score={m.score:.4f}) but JAX valid=False"
            )
            continue
        if int(ships[s_slot, t_slot]) != int(m.ships):
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: ships scalar={m.ships} "
                f"jax={int(ships[s_slot, t_slot])}"
            )
        if abs(int(eta[s_slot, t_slot]) - int(m.eta)) > 1:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: eta scalar={m.eta} "
                f"jax={int(eta[s_slot, t_slot])}"
            )
        jax_score = float(score[s_slot, t_slot])
        rel = abs(jax_score - m.score) / max(abs(m.score), 1e-6)
        if rel > 1e-3 and abs(jax_score - m.score) > 1e-3:
            diffs.append(
                f"  pid={m.src_id}->{m.target_id}: score scalar={m.score:.6f} "
                f"jax={jax_score:.6f} rel={rel:.4e}"
            )
        if len(diffs) >= 6:
            break

    assert not diffs, "recapture divergence:\n" + "\n".join(diffs)


# ---------------------------------------------------------------------------
# Sub-phase 3e: settle_plan parity (snipe + reinforce only;
# recapture stays in Python state for now)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_settle_plan_from_matrices_parity_snipe_reinforce(seed):
    """JAX-matrix-driven settle_plan picks the same (src, target) pairs
    as scalar `settle_plan` over the combined snipe+reinforce list.

    Scope: 2P games (no spoiler), non-aggressive sizing. Comet
    targets are filtered to keep parity with scalar; recapture is
    deferred to its own test since it depends on per-turn state.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 29)
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

    # Build (src_pid, target_pid) sets — order-independent comparison.
    scalar_pairs = {(int(i.src_id), int(i.target_id)) for i in scalar_intents}
    jax_pairs = {(c["src_pid"], c["target_pid"]) for c in chosen}

    only_scalar = scalar_pairs - jax_pairs
    only_jax = jax_pairs - scalar_pairs

    msg_lines = []
    for pair in only_scalar:
        msg_lines.append(f"  scalar picked {pair} but JAX did not")
    for pair in only_jax:
        msg_lines.append(f"  JAX picked {pair} but scalar did not")

    assert not msg_lines, (
        f"settle_plan divergence (seed {seed}):\n" + "\n".join(msg_lines)
    )


# ---------------------------------------------------------------------------
# Sub-phase 8b: JAX-scan settle_plan parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_settle_plan_jax_matches_numpy(seed):
    """JAX-scan settle_plan picks the same (src, target) set as the
    numpy reference (settle_plan_from_matrices).

    Scope: the JAX form takes a `merge_class_matrices` collapse of
    per-cell max-class score, NOT the full per-source candidate list
    walk that the numpy form does. The two are byte-equivalent ONLY
    when class target sets are disjoint — which they are for the
    current (snipe, reinforce) pair: snipe excludes our planets,
    reinforce only targets our planets. See the comment in
    `merge_class_matrices` for the full argument. (T3)
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=2)
    _spawn_in_flight_fleets(env, num_agents=2, n_steps=25, rng_seed=seed * 41)
    if env.state[0].status != "ACTIVE":
        pytest.skip(f"seed {seed} terminated early")

    my_id = 0
    gs = scalar_to_jax(env.state, env.info["seed"])
    jax_wm = build_world_model(gs, max_horizon=DEFAULT_HORIZON, num_agents=4)
    snipe_out = compute_snipe_score_matrix(gs, jax_wm, my_id=my_id)
    reinforce_out = compute_reinforce_score_matrix(gs, jax_wm, my_id=my_id)

    # Numpy reference (separate classes, internal merge).
    chosen_numpy = settle_plan_from_matrices(
        class_outputs=[snipe_out, reinforce_out],
        class_names=["snipe", "reinforce"],
        planets_id=gs.planets_id,
        world_owners_at=jax_wm.owners_at,
        world_ships_at=jax_wm.ships_at,
        my_id=my_id,
    )
    pid_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(np.asarray(gs.planets_id))
        if pid >= 0
    }
    numpy_pairs = {
        (pid_to_slot[c["src_pid"]], pid_to_slot[c["target_pid"]])
        for c in chosen_numpy
    }

    # JAX-scan version (merge → scan).
    merged = merge_class_matrices([snipe_out, reinforce_out])
    src_arr, tgt_arr, ships_arr, eta_arr = settle_plan_jax(
        merged["score"], merged["ships"], merged["eta"], merged["valid"],
        jax_wm.ships_at,
    )
    src_arr = np.asarray(src_arr)
    tgt_arr = np.asarray(tgt_arr)
    jax_pairs = {
        (int(s), int(t))
        for s, t in zip(src_arr, tgt_arr)
        if s >= 0 and t >= 0
    }

    only_numpy = numpy_pairs - jax_pairs
    only_jax = jax_pairs - numpy_pairs
    msg = []
    for pair in only_numpy:
        msg.append(f"  numpy picked {pair} but JAX did not")
    for pair in only_jax:
        msg.append(f"  JAX picked {pair} but numpy did not")
    assert not msg, "settle_plan_jax divergence:\n" + "\n".join(msg)
