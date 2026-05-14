"""Invariants for `_enumerate_split_source` and `_enumerate_drop_or_split`."""

from __future__ import annotations

import pytest

from kaggle_environments import make

from lib.intent import World
from lib.v7_search import (
    _action_key,
    _action_from_intents,
    _build_incumbent_intents,
    _enumerate_drop_one,
    _enumerate_drop_or_split,
    _enumerate_split_source,
)
from lib.world_model import WorldModel


@pytest.fixture(scope="module")
def midgame_obs():
    """Self-play 10 turns to get an obs where sources have non-trivial garrisons."""
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
    return state[0]["observation"] if isinstance(state[0], dict) else state[0].observation


def _build(obs):
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    intents = _build_incumbent_intents(world, model)
    action = _action_from_intents(intents, obs, model)
    return world, model, intents, action


def test_split_includes_incumbent_first(midgame_obs):
    world, model, intents, action = _build(midgame_obs)
    cands = _enumerate_split_source(world, model, intents, action, midgame_obs)
    assert cands[0] == action


def test_split_extends_by_one_launch(midgame_obs):
    world, model, intents, action = _build(midgame_obs)
    cands = _enumerate_split_source(world, model, intents, action, midgame_obs)
    # Each non-incumbent candidate has exactly one more launch than the incumbent.
    for cand in cands[1:]:
        assert len(cand) == len(action) + 1


def test_split_reuses_an_existing_source(midgame_obs):
    """The extra launch's src_id must be one already in the incumbent
    (this is the distinguishing feature vs `_enumerate_add_one`)."""
    world, model, intents, action = _build(midgame_obs)
    cands = _enumerate_split_source(world, model, intents, action, midgame_obs)
    incumbent_srcs = {int(m[0]) for m in action}
    for cand in cands[1:]:
        cand_srcs = [int(m[0]) for m in cand]
        # All sources in the candidate must be in the incumbent set
        # (we don't introduce a new source — that's add-one's job).
        for src in cand_srcs:
            assert src in incumbent_srcs, (
                f"split-source variant introduced new source {src}"
            )
        # And at least one source appears twice (the split).
        from collections import Counter
        c = Counter(cand_srcs)
        assert max(c.values()) >= 2, "no source appears twice — not a split"


def test_drop_or_split_is_union(midgame_obs):
    world, model, intents, action = _build(midgame_obs)
    drop = _enumerate_drop_one(action)
    split = _enumerate_split_source(world, model, intents, action, midgame_obs)
    union = _enumerate_drop_or_split(world, model, intents, action, midgame_obs)
    drop_keys = {_action_key(c) for c in drop}
    split_keys = {_action_key(c) for c in split}
    union_keys = {_action_key(c) for c in union}
    assert drop_keys.issubset(union_keys)
    assert split_keys.issubset(union_keys)
    # Dedup.
    keys = [_action_key(c) for c in union]
    assert len(keys) == len(set(keys))
