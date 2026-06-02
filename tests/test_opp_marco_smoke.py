"""Smoke tests for the marco-v3-3 Tier-3 opponent model port.

Verifies:
1. `predict_marco_plan` returns a non-empty list at step 0 in a 2P game.
2. The Tier-3 policy emits the first fire-now commit when called at
   step 0, and an identical (or empty) emit on a re-call at the same
   step (cache hit, commit consumed).
3. The Tier-3 policy falls through to Tier 1 when the EAM gate fails
   (e.g. opp owns > 6 planets at step 0 — synthetic obs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root on path when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Disable noisy bedrock/sagemaker pre-loads.
os.environ.setdefault("KAGGLE_ENV_QUIET", "1")


@pytest.fixture(scope="module")
def step0_obs():
    """Real step-0 observation from a 2P orbit_wars game (seed 0)."""
    from kaggle_environments import make
    env = make("orbit_wars",
               configuration={"episodeSteps": 60, "actTimeout": 1.0, "seed": 0},
               debug=False)
    env.reset()
    return env.state[0]["observation"]


def test_predict_marco_plan_returns_nonempty_at_opening(step0_obs):
    from lib.opp_marco import predict_marco_plan
    plan = predict_marco_plan(step0_obs, opp_seat=0, time_budget_ms=300.0)
    assert plan is not None
    assert len(plan) > 0
    # Sanity: each commit names valid src/tgt ids and positive fleet.
    planet_ids = {int(p[0]) for p in step0_obs.get("planets", [])}
    for c in plan:
        assert int(c.src_id) in planet_ids
        assert int(c.tgt_id) in planet_ids
        assert int(c.src_id) != int(c.tgt_id)
        assert int(c.fleet) > 0
        assert float(c.eta) >= 0.0
        assert int(c.t_launch) >= 0


def test_tier3_policy_emit_then_consume(step0_obs):
    """First call: emit fire-now commits and consume them. Second call
    at the same obs.step: emit nothing further (no double-fire)."""
    from lib.opp_model import make_opp_policy
    policy = make_opp_policy(tier=3, opp_seat=0, time_budget_ms=300.0)
    emits_1 = policy(step0_obs)
    emits_2 = policy(step0_obs)
    # First call should return a list of [src, angle, ships].
    assert isinstance(emits_1, list)
    for e in emits_1:
        assert len(e) == 3
        assert isinstance(e[0], int)
        assert isinstance(e[1], float)
        assert isinstance(e[2], int)
    # Second call at the same step: nothing new (we already consumed
    # whatever was fire-now).
    assert emits_2 == [] or emits_2 == emits_1[:0]


def test_tier3_falls_through_when_gate_skips(step0_obs):
    """Synthetically violate the EAM gate (opp owns too many planets)
    and verify the policy falls through to Tier 1 without crashing.
    """
    from copy import deepcopy
    from lib.opp_model import make_opp_policy

    # Fabricate an obs where every planet is owned by seat 0 — EAM gate
    # fails because opp now owns 20+ planets.
    obs = deepcopy(step0_obs) if isinstance(step0_obs, dict) else dict(step0_obs)
    new_planets = []
    for p in obs.get("planets", []):
        np_ = list(p)
        np_[1] = 0  # owner -> seat 0
        new_planets.append(np_)
    obs["planets"] = new_planets

    policy = make_opp_policy(tier=3, opp_seat=0, time_budget_ms=200.0)
    out = policy(obs)
    # Tier-1 fallback returns a list (possibly empty). What we're really
    # asserting: no exception, no AttributeError on a missing cache slot.
    assert isinstance(out, list)


def test_tier3_cache_isolation_across_episodes(step0_obs):
    """Two policy instances must NOT share cache — required for
    rollouts that build a fresh opp_policy per candidate."""
    from lib.opp_model import make_opp_policy
    p1 = make_opp_policy(tier=3, opp_seat=0, time_budget_ms=200.0)
    p2 = make_opp_policy(tier=3, opp_seat=0, time_budget_ms=200.0)
    e1 = p1(step0_obs)
    # p1's cache is now populated for this episode + seat.
    e2 = p2(step0_obs)
    # p2's first call must also produce the fire-now emit (cache miss).
    assert e2 == e1
