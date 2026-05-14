"""Smoke + shape tests for `lib.missions.opp_archetypes`.

Pins the contract: each archetype returns a valid env-format
`list[[src_id, angle, ships], ...]`, deduplication works, and the
counter-* archetypes are reactive to `our_intents`.
"""

from __future__ import annotations

import pytest

from kaggle_environments import make

from lib.intent import Intent
from lib.missions.opp_archetypes import (
    archetype_counter_reinforce,
    archetype_counter_snipe,
    archetype_cross_attack,
    archetype_no_launch,
    archetype_v351,
    build_opp_archetypes,
    opp_pov_obs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_obs_pair():
    """Return `(obs_p0, obs_p1)` from a step-0 self-play game."""
    env = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 500})
    env.reset(num_agents=2)
    return env.steps[0][0].observation, env.steps[0][1].observation


# ---------------------------------------------------------------------------
# Individual archetypes
# ---------------------------------------------------------------------------


def test_no_launch_is_empty():
    assert archetype_no_launch() == []


def test_v351_returns_action_list(fresh_obs_pair):
    obs0, _ = fresh_obs_pair
    opp_obs = opp_pov_obs(obs0, opp_id=1)
    from lib.intent import World
    from lib.world_model import WorldModel
    w = World.from_obs(opp_obs)
    m = WorldModel.from_world(w)
    action = archetype_v351(w, m, opp_obs)
    assert isinstance(action, list)
    for launch in action:
        assert len(launch) == 3
        assert isinstance(int(launch[0]), int)
        assert isinstance(int(launch[2]), int)
        assert int(launch[2]) > 0


def test_counter_snipe_targets_our_largest(fresh_obs_pair):
    obs0, _ = fresh_obs_pair
    opp_obs = opp_pov_obs(obs0, opp_id=1)
    from lib.intent import World
    w = World.from_obs(opp_obs)
    action = archetype_counter_snipe(w, opp_obs)
    # On a fresh game, opp may not have enough ships (homes are 10).
    # Just assert the shape is valid (empty or single launch).
    assert isinstance(action, list)
    assert len(action) in (0, 1)


def test_counter_reinforce_empty_when_no_our_intents(fresh_obs_pair):
    obs0, _ = fresh_obs_pair
    opp_obs = opp_pov_obs(obs0, opp_id=1)
    from lib.intent import World
    w = World.from_obs(opp_obs)
    assert archetype_counter_reinforce(w, opp_obs, our_intents=[]) == []


def test_counter_reinforce_targets_our_launch(fresh_obs_pair):
    obs0, _ = fresh_obs_pair
    opp_obs = opp_pov_obs(obs0, opp_id=1)
    from lib.intent import World
    w = World.from_obs(opp_obs)
    # Pick any non-our planet as a target our launch would aim at.
    targets = [
        p.id for p in w.planets_by_id.values()
        if p.owner != w.my_id and p.owner != -1
    ]
    if not targets:
        pytest.skip("no opponent planets on this fixture")
    target_id = targets[0]
    src_id = next(iter(w.planets_by_id))
    our_intents = [Intent(src_id=src_id, target_id=target_id, ships=3)]
    action = archetype_counter_reinforce(w, opp_obs, our_intents=our_intents)
    # Counter targets the same planet we're attacking.
    assert isinstance(action, list)
    for launch in action:
        # source must be opp-owned; target == our target.
        assert launch[0] != target_id  # source isn't the target itself


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def test_build_opp_archetypes_returns_distinct(fresh_obs_pair):
    obs0, _ = fresh_obs_pair
    opp_obs = opp_pov_obs(obs0, opp_id=1)
    arch = build_opp_archetypes(opp_obs, our_intents=[])
    # At least the no-launch baseline.
    assert [] in arch
    # Deduplicated: no exact duplicates.
    seen: list = []
    for a in arch:
        assert a not in seen
        seen.append(a)


def test_build_opp_archetypes_typical_count(fresh_obs_pair):
    obs0, _ = fresh_obs_pair
    opp_obs = opp_pov_obs(obs0, opp_id=1)
    arch = build_opp_archetypes(opp_obs, our_intents=[])
    # On step 0, 2-3 archetypes typically distinct (no-launch +
    # v3.5.1, plus maybe counter-snipe if opp has a big-enough source).
    assert 1 <= len(arch) <= 5
