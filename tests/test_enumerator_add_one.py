"""Invariants for `_enumerate_add_one` and `_enumerate_drop_or_add_one`."""

from __future__ import annotations

import pytest

from kaggle_environments import make

from lib.intent import World
from lib.v7_search import (
    _action_key,
    _build_incumbent_intents,
    _enumerate_add_one,
    _enumerate_drop_one,
    _enumerate_drop_or_add_one,
    _action_from_intents,
)
from lib.world_model import WorldModel


@pytest.fixture(scope="module")
def fresh_obs_pair():
    """A mid-game obs where some sources are idle in the incumbent."""
    env = make("orbit_wars", configuration={"seed": 2, "episodeSteps": 500})
    env.reset(num_agents=2)
    from agents.v7_ablations.v7_0_drop_one.main import agent as v7_0
    state = env.steps[0]
    for _ in range(10):
        obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs1 = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation
        a0 = v7_0(obs0, env.configuration)
        a1 = v7_0(obs1, env.configuration)
        state = env.step([a0, a1])
    obs = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
    return obs


def _build(obs):
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    incumbent_intents = _build_incumbent_intents(world, model)
    incumbent_action = _action_from_intents(incumbent_intents, obs, model)
    return world, model, incumbent_intents, incumbent_action


def test_add_one_includes_incumbent_first(fresh_obs_pair):
    obs = fresh_obs_pair
    world, model, intents, action = _build(obs)
    cands = _enumerate_add_one(world, model, intents, action, obs)
    assert len(cands) >= 1
    assert cands[0] == action  # incumbent first


def test_add_one_extends_with_idle_sources(fresh_obs_pair):
    obs = fresh_obs_pair
    world, model, intents, action = _build(obs)
    cands = _enumerate_add_one(world, model, intents, action, obs)
    incumbent_srcs = {int(i.src_id) for i in intents}
    # Each non-incumbent candidate has STRICTLY MORE launches than incumbent
    # (we appended one to the source-set).
    for cand in cands[1:]:
        assert len(cand) == len(action) + 1, (
            f"add_one variant has wrong launch count: {len(cand)} vs {len(action)}+1"
        )
        # The extra launch's source isn't in the incumbent.
        cand_srcs = {int(m[0]) for m in cand}
        new_srcs = cand_srcs - incumbent_srcs
        assert len(new_srcs) == 1, (
            f"expected exactly one new source, got {new_srcs}"
        )


def test_drop_or_add_one_is_union(fresh_obs_pair):
    obs = fresh_obs_pair
    world, model, intents, action = _build(obs)
    drop = _enumerate_drop_one(action)
    add = _enumerate_add_one(world, model, intents, action, obs)
    union = _enumerate_drop_or_add_one(world, model, intents, action, obs)
    drop_keys = {_action_key(c) for c in drop}
    add_keys = {_action_key(c) for c in add}
    union_keys = {_action_key(c) for c in union}
    assert drop_keys.issubset(union_keys)
    assert add_keys.issubset(union_keys)


def test_drop_or_add_one_dedupes(fresh_obs_pair):
    obs = fresh_obs_pair
    world, model, intents, action = _build(obs)
    union = _enumerate_drop_or_add_one(world, model, intents, action, obs)
    keys = [_action_key(c) for c in union]
    assert len(keys) == len(set(keys)), "union has duplicate candidates"
