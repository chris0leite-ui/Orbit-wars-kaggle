"""Step 4 — property-based bit-exact parity harness vs the official env.

Random-search complement to the fixed-seed parity tests
(`test_fast_sim_parity.py`, `test_game_parity.py`). For each randomly
generated `(episode_seed, num_agents, T_turns, action_policy)` example:

1. Build a fresh `kaggle_environments.make("orbit_wars")` env.
2. Build a parallel `lib.fast_sim.Snapshot` from the same initial obs,
   threading the env's `episode_seed` so comet RNG is bit-exact.
3. For T_turns: generate the same random actions for both, step both,
   assert per-turn equality of every field
   (`_assert_parity` from `test_fast_sim_parity`).

Settings come from `tests/conftest_hypothesis.py`:
- `foundation_fuzz_ci`     — 20 examples (fast lane, this file)
- `foundation_fuzz_nightly` — 200 examples (nightly, opt-in via env)

Marked `@pytest.mark.slow` so the default fast lane skips. To run:
    `python -m pytest tests/test_foundation_parity_fuzz.py -q -m slow`

To switch to the nightly profile:
    `HYPOTHESIS_PROFILE=foundation_fuzz_nightly python -m pytest ... -m slow`

Why this exists: 698 tests in the suite, but every parity test uses
fixed seeds (7, 42, 137, ...). Hypothesis fills the gap by exploring
random states; on failure it shrinks to a minimal repro automatically.
This is the regression dragnet that Steps 5/6/7/8/9 land against.
"""

from __future__ import annotations

import os
import random

import pytest

# Load the Hypothesis profile from our shared conftest_hypothesis module.
from tests import conftest_hypothesis  # noqa: F401 — registers profiles

from hypothesis import given, settings
from hypothesis import strategies as st
from kaggle_environments import make

from lib.fast_sim import from_obs, step
from tests.test_fast_sim_parity import _assert_parity

# Allow override via env var; default to CI profile.
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "foundation_fuzz_ci")
settings.load_profile(_PROFILE)

pytestmark = pytest.mark.slow


def _make_actions(
    obs0,
    num_seats: int,
    rng: random.Random,
    launch_prob: float,
    min_ships_threshold: int,
    ship_fraction: float,
) -> list[list]:
    """Parameterised random-launch policy — same shape as the fixed
    policy in `test_fast_sim_parity._make_actions` but with the three
    decision parameters drawn from Hypothesis."""
    actions: list[list] = [[] for _ in range(num_seats)]
    for p in obs0["planets"]:
        owner = p[1]
        if 0 <= owner < num_seats and p[5] > min_ships_threshold and rng.random() < launch_prob:
            ships = max(1, int(p[5] * ship_fraction))
            actions[owner].append([p[0], rng.uniform(0.0, 6.283), ships])
    return actions


@given(
    episode_seed=st.integers(min_value=1, max_value=10_000),
    num_agents=st.sampled_from([2, 4]),
    T_turns=st.integers(min_value=5, max_value=25),
    action_rng_seed=st.integers(min_value=1, max_value=10_000),
    launch_prob=st.floats(min_value=0.1, max_value=0.7),
    min_ships_threshold=st.integers(min_value=3, max_value=20),
    ship_fraction=st.floats(min_value=0.2, max_value=0.8),
    warm_up_steps=st.integers(min_value=0, max_value=15),
)
def test_random_actions_random_seeds_bit_exact(
    episode_seed,
    num_agents,
    T_turns,
    action_rng_seed,
    launch_prob,
    min_ships_threshold,
    ship_fraction,
    warm_up_steps,
):
    """Bit-exact parity vs env over random actions × random seeds ×
    random policies.

    For each example: walk the env forward `warm_up_steps` so the
    snapshot starts from a non-trivial mid-game state, build a
    Snapshot from the env's obs, then run T_turns of lockstep with
    the same random actions. Every field of every per-turn obs must
    match exactly (the parity contract in `_assert_parity`).
    """
    env = make("orbit_wars", configuration={"seed": episode_seed}, debug=False)
    env.reset(num_agents=num_agents)
    action_rng = random.Random(action_rng_seed)

    # Warm-up: random actions, env-only.
    for _ in range(warm_up_steps):
        acts = _make_actions(
            env.state[0].observation, num_agents, action_rng,
            launch_prob, min_ships_threshold, ship_fraction,
        )
        env.step(acts)
        if all(s.status == "DONE" for s in env.state):
            return  # game ended during warm-up; nothing to test

    # Snapshot at the warm-up state.
    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=num_agents,
    )
    _assert_parity(env, snap, label=f"init seed={episode_seed}")

    # Lockstep.
    for tick in range(T_turns):
        acts = _make_actions(
            env.state[0].observation, num_agents, action_rng,
            launch_prob, min_ships_threshold, ship_fraction,
        )
        env.step(acts)
        snap = step(snap, acts)
        _assert_parity(
            env, snap,
            label=f"seed={episode_seed} agents={num_agents} tick={tick}",
        )
        if all(s.status == "DONE" for s in env.state):
            break


@given(
    episode_seed=st.integers(min_value=1, max_value=10_000),
    T_turns=st.integers(min_value=3, max_value=15),
)
def test_no_actions_random_seeds_bit_exact(episode_seed, T_turns):
    """Parity under the trivial all-no-op policy. Catches drift in
    production / planet rotation / comet path advancement decoupled
    from the action-handling code path."""
    env = make("orbit_wars", configuration={"seed": episode_seed}, debug=False)
    env.reset(num_agents=2)

    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    _assert_parity(env, snap, label=f"init seed={episode_seed} no-op")

    for tick in range(T_turns):
        env.step([[], []])
        snap = step(snap, [[], []])
        _assert_parity(env, snap, label=f"seed={episode_seed} no-op tick={tick}")
