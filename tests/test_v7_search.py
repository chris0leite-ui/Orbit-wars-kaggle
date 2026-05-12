"""v7 search sanity tests.

Covers the load-bearing invariants without running full forward sims
in the inner loop (those are exercised in `test_fast_sim_parity.py`).

Tests:
- Every enumerator mode places the incumbent first (parity-floor invariant).
- `choose(wallclock_ms=0)` returns the incumbent (watchdog floor).
- `enumerate_candidates(ship_sweep)` never produces ships > src.garrison.
- `enumerate_candidates(archetype)` returns exactly 1 + len(non-baseline
  archetypes) bundles (baseline placeholder + concentrated/saturation/defensive).
- `enumerate_candidates(combined)` deduplicates identical bundles.
- `score_candidate` is deterministic (same call twice yields same score).
"""

from __future__ import annotations

import random

import pytest

from kaggle_environments import make

from lib import fast_sim
from lib.intent import World
from lib.mission import Mission
from lib.v7_search import (
    ARCHETYPE_PRESETS,
    _action_key,
    _build_incumbent_intents,
    choose,
    enumerate_candidates,
    score_candidate,
)
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _warmup_env(seed: int = 42, warmup: int = 25):
    """Drive the env forward to a non-trivial state."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(seed)
    for _ in range(warmup):
        acts = []
        for p in range(2):
            launches = []
            for pl in env.state[0].observation["planets"]:
                if pl[1] == p and pl[5] > 6 and rng.random() < 0.3:
                    launches.append(
                        [pl[0], rng.uniform(0.0, 6.283), int(pl[5] // 3)]
                    )
            acts.append(launches)
        env.step(acts)
    return env


def _world_model_obs(env):
    obs = env.state[0].observation
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    incumbent_intents = _build_incumbent_intents(world, model)
    from lib.v7_search import _action_from_intents
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)
    return obs, world, model, incumbent_intents, incumbent_action


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["drop_one", "target_swap", "ship_sweep", "archetype", "hungarian", "combined"],
)
def test_enumerator_places_incumbent_first(mode: str):
    env = _warmup_env()
    obs, world, model, incumbent_intents, incumbent_action = _world_model_obs(env)
    cands = enumerate_candidates(
        world, model,
        enumerator_mode=mode,
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    assert len(cands) >= 1
    assert _action_key(cands[0]) == _action_key(incumbent_action), (
        f"mode={mode!r}: incumbent must be cands[0]"
    )


def test_choose_returns_incumbent_when_watchdog_zero():
    """wallclock_ms=0 → no candidate gets scored, so the parity floor
    (incumbent) is returned."""
    env = _warmup_env()
    obs = env.state[0].observation
    action = choose(obs, env.configuration, enumerator_mode="drop_one",
                    K=10, wallclock_ms=0.0)
    # Should equal what the v3.5.1 incumbent emits — i.e., non-empty if
    # there are owned planets with ships and a valid target.
    # The exact equality is checked in the next test against the incumbent.
    # Here we just confirm we returned SOMETHING and didn't crash.
    assert isinstance(action, list)


def test_choose_watchdog_zero_matches_incumbent():
    env = _warmup_env(seed=7)
    obs, world, model, incumbent_intents, incumbent_action = _world_model_obs(env)
    action = choose(obs, env.configuration, enumerator_mode="drop_one",
                    K=10, wallclock_ms=0.0)
    assert _action_key(action) == _action_key(incumbent_action)


def test_ship_sweep_respects_garrison():
    env = _warmup_env()
    obs, world, model, incumbent_intents, incumbent_action = _world_model_obs(env)
    cands = enumerate_candidates(
        world, model,
        enumerator_mode="ship_sweep",
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    # Every launch in every candidate must respect the source's garrison.
    for cand in cands:
        for launch in cand:
            src_id, _angle, ships = launch
            src = world.planets_by_id.get(int(src_id))
            assert src is not None
            assert int(ships) <= int(src.ships), (
                f"ship_sweep produced ships={ships} > src.ships={src.ships}"
            )


def test_archetype_returns_baseline_plus_non_baseline_count():
    env = _warmup_env()
    obs, world, model, incumbent_intents, incumbent_action = _world_model_obs(env)
    cands = enumerate_candidates(
        world, model,
        enumerator_mode="archetype",
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    non_baseline_count = sum(1 for n in ARCHETYPE_PRESETS if n != "baseline")
    # 1 incumbent + len(non-baseline presets) extras.
    # Some may be deduped against the incumbent if a preset yields the
    # exact same action; we accept ≤ expected.
    assert 1 <= len(cands) <= 1 + non_baseline_count


def test_combined_deduplicates():
    env = _warmup_env()
    obs, world, model, incumbent_intents, incumbent_action = _world_model_obs(env)
    cands = enumerate_candidates(
        world, model,
        enumerator_mode="combined",
        incumbent_intents=incumbent_intents,
        incumbent_action=incumbent_action,
        obs=obs,
    )
    keys = [_action_key(c) for c in cands]
    assert len(keys) == len(set(keys)), "combined produced duplicates"


def test_score_candidate_is_deterministic():
    env = _warmup_env()
    obs = env.state[0].observation
    snap = fast_sim.from_obs(obs, env.configuration,
                             episode_seed=env.info["seed"], num_seats=2)
    # Pick the incumbent itself as the action under test.
    _, _, _, _, incumbent_action = _world_model_obs(env)
    s1 = score_candidate(snap, incumbent_action, my_id=0, K=8, opp_tier=1)
    s2 = score_candidate(snap, incumbent_action, my_id=0, K=8, opp_tier=1)
    assert s1 == s2, f"score_candidate not deterministic: {s1} vs {s2}"
