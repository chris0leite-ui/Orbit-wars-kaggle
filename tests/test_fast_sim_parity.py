"""Bit-exact parity gate: fast_sim vs env.clone()+step().

Builds a Snapshot from a real env state, then steps both in lockstep
for K turns under random-policy actions. After each tick, asserts:
- env step count == snap.step_idx
- planets (id, owner, x, y, ships) match to ≤ 1e-6
- fleets (id, owner, x, y, angle, from_pid, ships) match to ≤ 1e-6
- comet path indices match
- ship_totals match

Covers:
- 2P self-play across the step-50 comet spawn boundary (5 seeds).
- 4P self-play (2 seeds).
- Clone independence: mutating the clone leaves the source untouched.

Marked SLOW because it spins up the env multiple times; ~30 s total.
"""

from __future__ import annotations

import random

import pytest

pytestmark = pytest.mark.slow

from kaggle_environments import make

from lib.fast_sim import Snapshot, clone, from_obs, ship_totals, step


def _make_actions(obs0, num_seats: int, rng: random.Random) -> list[list]:
    """Random-launch policy. Each owned planet with >5 ships fires with
    probability 0.3, at a uniform angle, sending half its garrison."""
    actions: list[list] = [[] for _ in range(num_seats)]
    for p in obs0["planets"]:
        owner = p[1]
        if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
            actions[owner].append([p[0], rng.uniform(0.0, 6.283), int(p[5] // 2)])
    return actions


def _planet_sig(obs):
    """(id, owner, x, y, ships) per planet — full state up to <1e-6 jitter."""
    return tuple(
        (int(p[0]), int(p[1]), round(p[2], 6), round(p[3], 6), int(p[5]))
        for p in obs["planets"]
    )


def _fleet_sig(obs):
    return tuple(
        (
            int(f[0]),
            int(f[1]),
            round(f[2], 6),
            round(f[3], 6),
            round(f[4], 6),
            int(f[5]),
            int(f[6]),
        )
        for f in obs["fleets"]
    )


def _comet_sig(obs):
    return tuple(
        (g["path_index"], tuple(g["planet_ids"])) for g in obs.get("comets", []) or []
    )


def _assert_parity(env, snap: Snapshot, *, label: str):
    """All four state slices must match bit-exactly."""
    env_obs = env.state[0].observation
    assert env_obs.step == snap.step_idx, f"{label}: step mismatch"
    assert _planet_sig(env_obs) == _planet_sig(snap.obs), f"{label}: planets mismatch"
    assert _fleet_sig(env_obs) == _fleet_sig(snap.obs), f"{label}: fleets mismatch"
    assert _comet_sig(env_obs) == _comet_sig(snap.obs), f"{label}: comets mismatch"
    e_tot = {}
    for p in env_obs["planets"]:
        if p[1] >= 0:
            e_tot[p[1]] = e_tot.get(p[1], 0.0) + p[5]
    for f in env_obs["fleets"]:
        if f[1] >= 0:
            e_tot[f[1]] = e_tot.get(f[1], 0.0) + f[6]
    s_tot = ship_totals(snap)
    assert e_tot == s_tot, f"{label}: ship_totals {e_tot} != {s_tot}"


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 42, 100, 314, 2026])
def test_2p_parity_100_steps(seed: int):
    """5 seeds × 100 lockstep ticks. Crosses step-50 comet spawn boundary."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    action_rng = random.Random(seed * 7 + 1)

    # Warm up the env for 20 turns so the snapshot starts from a non-trivial state.
    for _ in range(20):
        acts = _make_actions(env.state[0].observation, 2, action_rng)
        env.step(acts)

    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    _assert_parity(env, snap, label=f"seed={seed} init")

    for tick in range(100):
        acts = _make_actions(env.state[0].observation, 2, action_rng)
        env.step(acts)
        snap = step(snap, acts)
        _assert_parity(env, snap, label=f"seed={seed} tick={tick}")


@pytest.mark.parametrize("seed", [1, 11])
def test_4p_parity_60_steps(seed: int):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=4)
    action_rng = random.Random(seed * 13 + 5)

    for _ in range(10):
        acts = _make_actions(env.state[0].observation, 4, action_rng)
        env.step(acts)

    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=4,
    )
    _assert_parity(env, snap, label=f"4P seed={seed} init")

    for tick in range(60):
        acts = _make_actions(env.state[0].observation, 4, action_rng)
        env.step(acts)
        snap = step(snap, acts)
        _assert_parity(env, snap, label=f"4P seed={seed} tick={tick}")


def test_clone_is_independent():
    """Mutating the clone's planets / fleets / comets must NOT touch source."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(0)
    for _ in range(15):
        acts = _make_actions(env.state[0].observation, 2, rng)
        env.step(acts)

    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    snap_copy = clone(snap)
    snap_copy.obs.planets[0][5] = 99999
    snap_copy.obs.fleets.append([9999, 0, 50, 50, 0.0, 0, 1])
    assert snap.obs.planets[0][5] != 99999
    assert all(f[0] != 9999 for f in snap.obs.fleets)


def test_empty_actions_does_not_crash():
    """No-action turns should advance step and apply production / orbits."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    s0_before = snap.step_idx
    snap = step(snap, [[], []])
    assert snap.step_idx == s0_before + 1


def test_done_short_circuits():
    """step() on a done snapshot is a no-op (matches Environment.step's
    'cannot step a done env' guard without raising)."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    snap.fake_env.done = True
    s_before = snap.step_idx
    snap2 = step(snap, [[], []])
    assert snap2.step_idx == s_before
