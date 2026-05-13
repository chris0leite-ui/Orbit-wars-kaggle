"""Parity tests: lib.game.batch_interpreter.batch_step must produce
byte-exact same state as N sequential lib.fast_sim.step calls.

This is the parity contract for the batched interpreter. Any drift
between batched and scalar paths is a bug.
"""

from __future__ import annotations

import copy
import math
import random

import pytest

from kaggle_environments import make
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv, clone as fs_clone, from_obs, step as fs_step
from lib.game.batch_interpreter import batch_step


def _fresh_batch(N: int, base_seed: int, num_agents: int = 2):
    """Build N independent Snapshots from N distinct kaggle envs (so they
    each start with different planet layouts — the harshest stress test).
    """
    snaps = []
    for i in range(N):
        env = make(
            "orbit_wars", configuration={"seed": base_seed + i}
        )
        env.reset(num_agents=num_agents)
        snap = from_obs(
            env.state[0].observation,
            env.configuration,
            episode_seed=env.info["seed"],
            num_seats=num_agents,
        )
        snaps.append(snap)
    return snaps


def _fresh_batch_same_state(N: int, seed: int, num_agents: int = 2):
    """N clones of a SINGLE starting Snapshot. This is the agent-inference
    use case: lookahead across candidates from the same root.
    """
    env = make("orbit_wars", configuration={"seed": seed})
    env.reset(num_agents=num_agents)
    base = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=num_agents,
    )
    return [fs_clone(base) for _ in range(N)]


def _rand_actions_for_obs(obs, pid, rng, num_agents):
    moves = []
    for p in obs.planets:
        if p[1] == pid and p[5] > 0 and rng.random() < 0.4:
            angle = rng.uniform(0, 2 * math.pi)
            ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
            if 0 < ships <= p[5]:
                moves.append([p[0], angle, ships])
    return moves


def _rand_actions_per_lane(snaps, rng, num_agents):
    actions_per_lane = []
    for snap in snaps:
        lane_actions = []
        for pid in range(num_agents):
            lane_actions.append(
                _rand_actions_for_obs(
                    snap.state[pid].observation, pid, rng, num_agents
                )
            )
        actions_per_lane.append(lane_actions)
    return actions_per_lane


def _state_diff(snap_a, snap_b) -> str:
    a = snap_a.state[0].observation
    b = snap_b.state[0].observation
    if snap_a.fake_env.done != snap_b.fake_env.done:
        return f"done: {snap_a.fake_env.done} vs {snap_b.fake_env.done}"
    if snap_a.state[0].status != snap_b.state[0].status:
        return f"status: {snap_a.state[0].status} vs {snap_b.state[0].status}"
    if snap_a.state[0].reward != snap_b.state[0].reward:
        return f"reward: {snap_a.state[0].reward} vs {snap_b.state[0].reward}"
    if a.planets != b.planets:
        for j, (pa, pb) in enumerate(zip(a.planets, b.planets)):
            if pa != pb:
                return f"planets[{j}]: {pa} vs {pb}"
        return f"planets length: {len(a.planets)} vs {len(b.planets)}"
    if a.fleets != b.fleets:
        for j, (fa, fb) in enumerate(zip(a.fleets, b.fleets)):
            if fa != fb:
                return f"fleets[{j}]: {fa} vs {fb}"
        return f"fleets length: {len(a.fleets)} vs {len(b.fleets)}"
    if list(a.comet_planet_ids) != list(b.comet_planet_ids):
        return f"comet_planet_ids: {a.comet_planet_ids} vs {b.comet_planet_ids}"
    if len(a.comets) != len(b.comets):
        return f"comets length: {len(a.comets)} vs {len(b.comets)}"
    return ""


@pytest.mark.parametrize("N", [2, 4, 8, 16])
@pytest.mark.parametrize("num_agents", [2])
def test_batch_vs_scalar_short_60(N, num_agents):
    """N=2..16 lanes × 60 random-policy steps × 2 agents. Each lane starts
    from a DIFFERENT kaggle env (max diversity)."""
    snaps_batch = _fresh_batch(N, base_seed=100, num_agents=num_agents)
    snaps_scalar = [
        fs_clone(s) if False else _deep_clone_snapshot(s) for s in snaps_batch
    ]
    # Force complete independence between the two paths.
    snaps_scalar = _independent_copy(snaps_batch)

    rng = random.Random(1234 + N)
    for step_idx in range(60):
        actions_per_lane = _rand_actions_per_lane(snaps_batch, rng, num_agents)
        # Apply to both paths
        snaps_batch = batch_step(snaps_batch, actions_per_lane)
        snaps_scalar = [
            fs_step(s, actions_per_lane[i]) for i, s in enumerate(snaps_scalar)
        ]
        for lane in range(N):
            diff = _state_diff(snaps_batch[lane], snaps_scalar[lane])
            assert not diff, (
                f"batch vs scalar diverge: lane={lane} step={step_idx} "
                f"N={N} num_agents={num_agents}: {diff}"
            )
        if all(s.fake_env.done for s in snaps_batch):
            break


@pytest.mark.parametrize("N", [4, 16])
@pytest.mark.parametrize("seed", [0, 42])
def test_batch_vs_scalar_same_root_500(N, seed):
    """Agent-inference flavour: N lanes cloned from one root snapshot,
    each taking independent random actions. Full 500-step episode crosses
    all 5 comet spawn boundaries.
    """
    snaps_batch = _fresh_batch_same_state(N, seed=seed)
    snaps_scalar = _independent_copy(snaps_batch)

    rng = random.Random(seed * 31 + N)
    for step_idx in range(500):
        actions_per_lane = _rand_actions_per_lane(snaps_batch, rng, 2)
        snaps_batch = batch_step(snaps_batch, actions_per_lane)
        snaps_scalar = [
            fs_step(s, actions_per_lane[i]) for i, s in enumerate(snaps_scalar)
        ]
        for lane in range(N):
            diff = _state_diff(snaps_batch[lane], snaps_scalar[lane])
            assert not diff, (
                f"same-root divergence: lane={lane} step={step_idx} "
                f"N={N} seed={seed}: {diff}"
            )
        if all(s.fake_env.done for s in snaps_batch):
            break


def _deep_clone_snapshot(snap):
    """Deep-copy a snapshot for parity-test isolation."""
    return copy.deepcopy(snap)


def _independent_copy(snaps):
    """Build a fully-independent parallel copy of a list of Snapshots.
    Caches are kept distinct so cache writes in one path don't leak to the
    other.
    """
    out = []
    for s in snaps:
        new_fake_env = _FakeEnv(s.fake_env.configuration, s.episode_seed)
        new_fake_env.done = s.fake_env.done
        # Empty caches — each path manages its own.
        new_fake_env.comet_path_cache = {}
        new_fake_env.planet_position_cache = {}
        # Copy planet_position_cache contents (it's read-only after init).
        for k, v in s.fake_env.planet_position_cache.items():
            new_fake_env.planet_position_cache[k] = v
        for k, v in s.fake_env.comet_path_cache.items():
            new_fake_env.comet_path_cache[k] = v
        from lib.fast_sim import Snapshot
        new_state = []
        for seat in s.state:
            obs_dict = {}
            for k in (
                "planets", "fleets", "initial_planets", "comet_planet_ids",
                "comets", "angular_velocity", "step", "next_fleet_id", "player",
            ):
                if hasattr(seat.observation, k):
                    v = getattr(seat.observation, k)
                    if isinstance(v, list):
                        obs_dict[k] = [
                            (x[:] if isinstance(x, list) else
                             ({**x, "planet_ids": list(x["planet_ids"])} if isinstance(x, dict) else x))
                            for x in v
                        ]
                    else:
                        obs_dict[k] = v
            new_obs = Struct(**obs_dict)
            new_state.append(Struct(
                observation=new_obs,
                action=None,
                status=seat.status,
                reward=seat.reward,
                info={},
            ))
        out.append(Snapshot(state=new_state, fake_env=new_fake_env, episode_seed=s.episode_seed))
    return out
