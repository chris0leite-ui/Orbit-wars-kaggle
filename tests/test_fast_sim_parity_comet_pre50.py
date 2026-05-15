"""Step 3 — document the comet-RNG pre-step-50 boundary in fast_sim.

The original plan called for a "fix" to the comet-RNG pre-step-50
rebuild, but on closer inspection this is an *information-theoretic*
limit, not a fixable defect: the live ladder scrubs `episode_seed`
from `configuration`, so when an agent calls `fast_sim.from_obs()`
without the seed, future comet spawns at turns 50/150/250/350/450
will be drawn from `random.Random("orbit_wars-comet-0-50")` rather
than from the real game's seed. No code change can reverse that.

What we CAN do — and what this test file does — is make the boundary
explicit and tested:

1. `test_pre50_correct_seed_bit_exact` — when the seed IS supplied
   (offline self-play, parity tests), fast_sim stays bit-exact even
   across the step-50 comet spawn boundary. This documents the
   contract for callers who can supply a seed.

2. `test_pre50_unknown_seed_diverges_after_spawn` — when the seed
   is NOT supplied (live ladder case), fast_sim's comets at step 50
   DIFFER from the real game's. Confirms the limit is real, and
   prevents regressions from "fixes" that quietly mask the divergence
   (e.g., always returning empty comets).

3. `test_pre50_unknown_seed_within_horizon_is_bit_exact` — within a
   horizon that doesn't cross any spawn boundary, fast_sim is bit-
   exact even without the seed. This is the practical guarantee for
   live-ladder lookahead with horizon < 50.

Rule 38: tests (2) and (3) reproduce the failure / non-failure states
the docstring's caveat block describes. They're not "fix verifications"
in the patch sense; they're regression dragnets for the limit.
"""

from __future__ import annotations

import random

import pytest
from kaggle_environments import make

from lib.fast_sim import clone, from_obs, step


def _comet_signature(obs):
    """A hashable fingerprint of all comet ships+positions in the obs.

    Lets us compare snap.obs vs env.obs across the comet-related
    fields that the spawn-RNG influences (ships count + path geometry).
    """
    groups: list[tuple] = []
    for group in obs.get("comets", []) or []:
        groups.append((
            int(group.get("path_index", -1)),
            tuple(int(pid) for pid in group.get("planet_ids", [])),
            tuple(tuple(round(float(c), 4) for c in pt)
                  for path in group.get("paths", [])
                  for pt in path),
        ))
    groups.sort()

    # Comet planet rows (ships counts) — keep separate so sort keys
    # don't mix types.
    comet_pids = set(int(p) for p in obs.get("comet_planet_ids", []) or [])
    planet_ships: list[tuple[int, int]] = []
    for p in obs.get("planets", []) or []:
        if int(p[0]) in comet_pids:
            planet_ships.append((int(p[0]), int(p[5])))
    planet_ships.sort()

    return (tuple(groups), tuple(planet_ships))


def _step_env_to(env, target_step, seats=2, rng_seed=None):
    """Step the env forward to `target_step` with random actions on
    each turn (deterministic via `rng_seed`)."""
    action_rng = random.Random(rng_seed)
    while int(env.steps[-1][0].observation.get("step", 0)) < target_step:
        actions = [[] for _ in range(seats)]
        # Random no-ops for now (action discovery isn't the point).
        env.step(actions)


@pytest.mark.parametrize("seed", [42, 137])
def test_pre50_correct_seed_bit_exact(seed):
    """With the correct seed supplied, fast_sim's comet signature at
    step 51 matches the real env's. Confirms the contract for offline
    self-play / parity tests where the seed IS known."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    _step_env_to(env, target_step=25, seats=2, rng_seed=seed)

    state_at_25 = env.steps[-1]
    obs25 = state_at_25[0].observation
    real_episode_seed = env.info["seed"]

    snap = from_obs(
        obs25,
        configuration=env.configuration,
        episode_seed=real_episode_seed,
        num_seats=2,
    )

    # Step fast_sim forward 30 turns (crosses step 50 boundary).
    for _ in range(30):
        env.step([[], []])
        snap = step(snap, [[], []])

    env_obs_final = env.steps[-1][0].observation
    assert _comet_signature(env_obs_final) == _comet_signature(snap.obs), (
        "fast_sim with correct seed should be bit-exact on comet RNG"
    )


@pytest.mark.parametrize("seed", [42, 137])
def test_pre50_unknown_seed_diverges_after_spawn(seed):
    """When fast_sim is built without the seed (live-ladder case),
    its comet schedule diverges from the real game's after the next
    spawn boundary. Documents the information-theoretic limit;
    regressions on this test (i.e., the signatures DO match) would
    indicate someone has falsely masked the divergence."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    _step_env_to(env, target_step=25, seats=2, rng_seed=seed)

    obs25 = env.steps[-1][0].observation

    # Live-ladder behavior: agent doesn't have episode_seed, passes 0.
    snap = from_obs(
        obs25,
        configuration=env.configuration,
        episode_seed=0,
        num_seats=2,
    )

    for _ in range(30):
        env.step([[], []])
        snap = step(snap, [[], []])

    env_obs_final = env.steps[-1][0].observation
    env_sig = _comet_signature(env_obs_final)
    snap_sig = _comet_signature(snap.obs)

    # We expect divergence — unless by coincidence seed=0 produced the
    # same comets as seed=42/137 (vanishingly unlikely; assert it
    # didn't happen in our test seeds).
    assert env_sig != snap_sig, (
        f"Expected comet divergence past spawn boundary with seed=0 "
        f"override but signatures matched (real_seed={seed}). If this "
        f"test starts failing, either someone fixed the unfixable or "
        f"the test seeds collided with seed=0 by chance — pick different "
        f"seeds."
    )


@pytest.mark.parametrize("seed", [42, 137])
def test_pre50_unknown_seed_within_horizon_is_bit_exact(seed):
    """Live-ladder lookahead with horizon < distance-to-next-spawn-
    boundary is bit-exact even without the seed. This is the
    practical guarantee for shallow planning.

    Snapshot at step 25, walk 20 turns (to step 45) — never crosses
    step 50, so comet RNG never fires, so the missing seed doesn't
    matter."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    _step_env_to(env, target_step=25, seats=2, rng_seed=seed)

    obs25 = env.steps[-1][0].observation

    snap = from_obs(
        obs25,
        configuration=env.configuration,
        episode_seed=0,  # live-ladder case
        num_seats=2,
    )

    for _ in range(20):  # 25 → 45, never crosses 50
        env.step([[], []])
        snap = step(snap, [[], []])

    env_obs_final = env.steps[-1][0].observation
    assert _comet_signature(env_obs_final) == _comet_signature(snap.obs), (
        "Within-horizon lookahead (no spawn boundary crossed) must be "
        "bit-exact even with episode_seed=0"
    )


def test_clone_preserves_seed_assumption():
    """`clone(snap)` shares the comet path cache with the source, so
    branched rollouts inherit the same comet-RNG verdict. Just a sanity
    check that the design contract holds."""
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    snap = from_obs(
        env.steps[-1][0].observation,
        configuration=env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    clone1 = clone(snap)
    clone2 = clone(snap)
    # Cache is shared at the object level (not a copy).
    assert clone1.fake_env.comet_path_cache is snap.fake_env.comet_path_cache
    assert clone2.fake_env.comet_path_cache is snap.fake_env.comet_path_cache
