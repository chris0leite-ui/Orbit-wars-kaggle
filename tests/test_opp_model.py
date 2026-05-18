"""Sanity tests for `lib/opp_model.py`.

Smoke-level coverage:
- Tier 0 produces the same action as v3_snipe's `agent(obs)` body.
- Tier 1 produces the same action as v3.5.1's `agent(obs)` body.
- The registry rejects unknown tier ids.
- `opponent_action_distribution` returns N copies of the deterministic
  action under tiers 0/1.
- Tier-1 policy survives the edge case where the opponent has 0 owned
  planets (empty action expected, no crash).
"""

from __future__ import annotations

import pytest

from lib import opp_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _self_play_obs(seed: int = 42, warmup_turns: int = 5, seat: int = 0):
    """Drive the env forward a few turns and return the obs for `seat`."""
    from kaggle_environments import make
    import random

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(seed)
    for _ in range(warmup_turns):
        acts = []
        for p in range(2):
            launches = []
            for pl in env.state[0].observation["planets"]:
                if pl[1] == p and pl[5] > 5 and rng.random() < 0.3:
                    launches.append(
                        [pl[0], rng.uniform(0.0, 6.283), int(pl[5] // 2)]
                    )
            acts.append(launches)
        env.step(acts)
    return env.state[seat].observation


def _normalize_action(action: list) -> list[tuple]:
    """Convert env action [[pid, angle, ships], ...] to a sorted tuple
    set for stable comparison (set ordering can drift but our pipelines
    are deterministic so the list should match exactly; we normalise
    defensively anyway)."""
    return sorted(
        (int(m[0]), round(float(m[1]), 6), int(m[2])) for m in action
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tier0_matches_v3_snipe():
    """Tier 0 = mirror_self_policy. Must produce v3_snipe's action."""
    from lib.intent import World, realize
    from lib.mechanism import DEFAULT_MECHANISMS
    from lib.missions.reinforce import propose_reinforce_missions
    from lib.missions.snipe import propose_snipe_missions
    from lib.planner import settle_plan
    from lib.world_model import WorldModel

    obs = _self_play_obs()

    # v3_snipe's body (`agents/v3_snipe/main.py`) inlined to dodge
    # any agent-module import side effects.
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=False)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    expected = realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)

    got = opp_model.mirror_self_policy(obs)
    assert _normalize_action(got) == _normalize_action(expected)


def test_tier1_matches_v3_5_1():
    """Tier 1 = top_tier_mirror_policy. Must produce v3.5.1's action."""
    from lib.intent import World, realize
    from lib.mechanism import DEFAULT_MECHANISMS
    from lib.missions.reinforce import propose_reinforce_missions
    from lib.missions.snipe import propose_snipe_missions
    from lib.planner import settle_plan
    from lib.world_model import WorldModel

    obs = _self_play_obs()

    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    missions = (
        propose_snipe_missions(world, model, aggressive=True)
        + propose_reinforce_missions(world, model)
    )
    intents = settle_plan(missions, world, model)
    expected = realize(intents, obs, mechanisms=DEFAULT_MECHANISMS, model=model)

    got = opp_model.top_tier_mirror_policy(obs)
    assert _normalize_action(got) == _normalize_action(expected)


def test_make_opp_policy_rejects_unknown_tier():
    with pytest.raises(ValueError):
        opp_model.make_opp_policy(tier=99)


def test_action_distribution_single_point_under_deterministic_tiers():
    obs = _self_play_obs()
    samples = opp_model.opponent_action_distribution(obs, tier=1, samples=3)
    assert len(samples) == 3
    canonical = _normalize_action(samples[0])
    for s in samples[1:]:
        assert _normalize_action(s) == canonical


def test_tier_returns_empty_on_no_owned_planets():
    """Synthesize an obs where seat-0 owns nothing — policy must return []."""
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": 5}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation

    # Flip every planet to non-self.
    for p in obs["planets"]:
        p[1] = 1
    obs["fleets"] = []
    result_tier0 = opp_model.mirror_self_policy(obs)
    result_tier1 = opp_model.top_tier_mirror_policy(obs)
    assert result_tier0 == [] and result_tier1 == []


def test_predict_opponent_action_default_tier_is_1():
    obs = _self_play_obs()
    default = opp_model.predict_opponent_action(obs)
    explicit = opp_model.predict_opponent_action(obs, tier=1)
    assert _normalize_action(default) == _normalize_action(explicit)


# ---------------------------------------------------------------------------
# lite_greedy vulnerability-term tests (2026-05-18 Tier 1)
# ---------------------------------------------------------------------------


def _planet(pid: int, owner: int, x: float, y: float,
            ships: int = 10, production: int = 1, radius: float = 1.0):
    """Planet tuple in obs format: (id, owner, x, y, radius, ships, production)."""
    return [pid, owner, float(x), float(y), float(radius),
            int(ships), int(production)]


def test_lite_greedy_prefers_vulnerable_target():
    """Two opp targets at same distance + same production, but one has
    5 garrison and the other has 50. Vulnerability term should make
    the lite_greedy opp prefer the weaker one."""
    import math as _m
    src = _planet(0, 0, 50.0, 50.0, ships=100, production=1)
    weak = _planet(1, 1, 20.0, 50.0, ships=5, production=1)
    strong = _planet(2, 1, 80.0, 50.0, ships=50, production=1)
    obs = {"player": 0, "planets": [src, weak, strong], "fleets": []}
    moves = opp_model.lite_greedy_policy(obs)
    assert len(moves) == 1, f"expected 1 launch, got {moves}"
    _src_id, angle, _ships = moves[0]
    # Weak is at x=20 < src.x=50 → cos(angle) < 0.
    # Strong is at x=80 > src.x=50 → cos(angle) > 0.
    assert _m.cos(angle) < 0, (
        f"expected launch toward weak (x=20 < src.x=50); "
        f"got angle={angle}, cos={_m.cos(angle)}"
    )


def test_lite_greedy_still_skips_unaffordable():
    """Existing affordability check preserved: source can't afford
    capture → no launch emitted."""
    src = _planet(0, 0, 50.0, 50.0, ships=10, production=1)
    too_hard = _planet(1, 1, 55.0, 50.0, ships=30, production=1)
    obs = {"player": 0, "planets": [src, too_hard], "fleets": []}
    moves = opp_model.lite_greedy_policy(obs)
    assert moves == []


def test_lite_greedy_neutral_no_production_accrual():
    """Neutral target's predicted defenders == current garrison.
    Regression test for env rule preserved through the rewrite."""
    src = _planet(0, 0, 50.0, 50.0, ships=15, production=1)
    neutral = _planet(1, -1, 80.0, 50.0, ships=9, production=1)
    obs = {"player": 0, "planets": [src, neutral], "fleets": []}
    moves = opp_model.lite_greedy_policy(obs)
    assert len(moves) == 1, (
        f"expected 1 launch (neutral is affordable: needs 10 ≤ 15 budget); "
        f"got {moves}. If empty, neutral was treated as accreting (bug)."
    )
