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
# Top-K launch-budget tests for `lite_greedy_policy` (Fix A, 2026-05-28 PM4).
# ---------------------------------------------------------------------------


def _viable_lite_greedy_candidates(obs):
    """Replica of `lite_greedy_policy`'s per-source affordability+ROI scan,
    returning (src_id, roi_score) for every source that would emit a
    launch at K=0. Used as the ROI ground-truth oracle in K=1/K=2 tests.
    """
    import math as _m
    from lib.fleet import speed as _fs

    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    planets = obs.get("planets") if isinstance(obs, dict) else getattr(obs, "planets", None)
    if not planets:
        return []
    targets = [p for p in planets if p[1] != player]
    out: list[tuple[int, float]] = []
    for src in planets:
        if src[1] != player or src[5] < 10:
            continue
        best = None
        best_score = -1.0
        sx = src[2]; sy = src[3]
        for t in targets:
            if t[0] == src[0]:
                continue
            dx = t[2] - sx; dy = t[3] - sy
            d = _m.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                continue
            score = float(t[6]) / (d + 1.0)
            if score > best_score:
                best_score = score
                best = t
        if best is None:
            continue
        budget = int(src[5])
        agg_ships = max(5, int(budget * 0.7))
        if agg_ships > budget:
            agg_ships = budget
        spd = _fs(agg_ships)
        if spd <= 0:
            continue
        dx = best[2] - sx; dy = best[3] - sy
        d = _m.sqrt(dx * dx + dy * dy)
        flight = max(0.0, d - float(src[4]) - float(best[4]) - 0.1)
        eta = max(1, int(_m.ceil(flight / spd)))
        if int(best[1]) == -1:
            defenders_at_eta = float(best[5])
        else:
            defenders_at_eta = float(best[5]) + float(best[6]) * eta
        needed = int(_m.ceil(defenders_at_eta)) + 1
        if needed > budget:
            continue
        ships = max(agg_ships, needed)
        if ships > budget:
            ships = budget
        if ships < 5:
            continue
        out.append((int(src[0]), float(best_score)))
    return out


def test_lite_greedy_default_unlimited_byte_parity(monkeypatch):
    """K=0 (default) must emit IDENTICAL list to current behavior — no
    sort, no slice, planet-walk order preserved."""
    monkeypatch.setattr(opp_model, "OPP_MAX_LAUNCHES", 0)
    obs = _self_play_obs(warmup_turns=10)
    got = opp_model.lite_greedy_policy(obs)
    # Recompute the K=0 output by re-running the policy under the same
    # patched constant. Idempotent oracle: two K=0 calls must agree.
    again = opp_model.lite_greedy_policy(obs)
    assert got == again
    # And the source-id order must match the per-source viable scan
    # (planet-walk order, NOT ROI-desc order).
    viable_src_ids = [sid for sid, _roi in _viable_lite_greedy_candidates(obs)]
    emitted_src_ids = [int(m[0]) for m in got]
    assert emitted_src_ids == viable_src_ids


def test_lite_greedy_max_launches_k1(monkeypatch):
    """K=1 emits <=1 launch and (when 1) it's the highest-ROI viable source."""
    monkeypatch.setattr(opp_model, "OPP_MAX_LAUNCHES", 1)
    obs = _self_play_obs(warmup_turns=10)
    got = opp_model.lite_greedy_policy(obs)
    assert len(got) <= 1
    viable = _viable_lite_greedy_candidates(obs)
    if not viable:
        assert got == []
        return
    assert len(got) == 1
    top_src_id = max(viable, key=lambda t: t[1])[0]
    assert int(got[0][0]) == top_src_id


def test_lite_greedy_max_launches_k2(monkeypatch):
    """K=2 emits <=2 launches; emitted set equals top-2 by ROI; order is
    ROI-descending."""
    monkeypatch.setattr(opp_model, "OPP_MAX_LAUNCHES", 2)
    obs = _self_play_obs(warmup_turns=10)
    got = opp_model.lite_greedy_policy(obs)
    assert len(got) <= 2
    viable = _viable_lite_greedy_candidates(obs)
    expected = sorted(viable, key=lambda t: -t[1])[:2]
    expected_ids = [sid for sid, _ in expected]
    emitted_ids = [int(m[0]) for m in got]
    assert emitted_ids == expected_ids


def test_lite_greedy_k_above_viable_count_is_clamped(monkeypatch):
    """K=10 with fewer viable sources emits exactly viable-count launches
    (set-equality with the K=0 oracle; ordering not required)."""
    obs = _self_play_obs(warmup_turns=10)
    monkeypatch.setattr(opp_model, "OPP_MAX_LAUNCHES", 0)
    k0 = opp_model.lite_greedy_policy(obs)
    k0_ids = sorted(int(m[0]) for m in k0)

    monkeypatch.setattr(opp_model, "OPP_MAX_LAUNCHES", 10)
    k10 = opp_model.lite_greedy_policy(obs)
    k10_ids = sorted(int(m[0]) for m in k10)
    assert len(k10) == len(k0)
    assert k10_ids == k0_ids


def test_lite_greedy_k1_respects_affordability(monkeypatch):
    """Hand-crafted obs: one viable opp planet (ships=20, neutral target
    with 5 defenders) and one unaffordable opp planet (ships=10, target
    needs 30 to capture). K=1 must NOT emit the unaffordable source."""
    monkeypatch.setattr(opp_model, "OPP_MAX_LAUNCHES", 1)
    # Planet tuple shape: (id, owner, x, y, radius, ships, production).
    # Owner 0 = opp seat we're simulating; owner -1 = neutral.
    obs = {
        "player": 0,
        "planets": [
            # Viable opp source: 20 ships, target needs ~5 to capture.
            [0, 0, 0.0, 0.0, 1.0, 20, 0],
            # Easy-capture neutral: 5 defenders, prod 1.
            [1, -1, 10.0, 0.0, 1.0, 5, 1],
            # Unaffordable opp source: 10 ships, needs >10 to capture
            # its nearest target (defenders=30 on the neutral below).
            [2, 0, 0.0, 50.0, 1.0, 10, 0],
            [3, -1, 10.0, 50.0, 1.0, 30, 1],
        ],
    }
    got = opp_model.lite_greedy_policy(obs)
    assert len(got) <= 1
    if got:
        assert int(got[0][0]) == 0  # the viable source, never source-id 2
