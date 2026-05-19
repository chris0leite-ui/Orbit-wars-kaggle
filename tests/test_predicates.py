"""Unit tests for agents/baseline/predicates — Layer-0 closed-form checks.

Plan reference: /root/.claude/plans/take-the-lens-of-magical-shore.md

Covers W2 (provably-held reinforce). W1/L1/L2 stubs return UNCERTAIN
in this slice; their tests land with their implementations.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from agents.baseline.predicates import (
    UNCERTAIN,
    Verdict,
    l1_provably_wasted_launch,
    l2_dominance_prune,
    w1_provably_winning_capture,
    w2_provably_held_reinforce,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _fleet(fid, owner, x, y, angle, ships, from_planet_id=0):
    # Fleet schema: (id, owner, x, y, angle, from_planet_id, ships).
    return Fleet(fid, owner, x, y, angle, from_planet_id, ships)


def _world(my_id, planets, *, fleets=None, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [
            (f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
            for f in (fleets or [])
        ],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# W2 — provably-held reinforce
# ---------------------------------------------------------------------------


def test_w2_inapplicable_not_a_reinforce():
    """W2 only fires for `tgt.owner == me`. Captures defer to W1/rollout."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    enemy_tgt = _planet(1, 1, 30.0, 50.0, ships=5, production=2)
    world = _world(0, [src, enemy_tgt])
    model = WorldModel.from_world(world)
    v = w2_provably_held_reinforce(
        src, enemy_tgt, ships=10, wait_N=0, eta=3, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain"


def test_w2_inapplicable_no_threat():
    """No inbound threat → W2 abstains (proposer wouldn't even emit this)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    mine_tgt = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, mine_tgt])
    model = WorldModel.from_world(world)
    v = w2_provably_held_reinforce(
        src, mine_tgt, ships=10, wait_N=0, eta=1, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain"


def test_w2_abstains_when_at_rest_opp_in_reach():
    """Conservative abstain: an at-rest enemy planet within counter-reach
    of our threatened planet is NOT covered by the in-flight ledger.
    W2 defers to the rollout rather than commit unsoundly.
    """
    # Layout: src (mine, far rear), tgt (mine, threatened by inbound),
    # opp_at_rest (enemy, 100 ships, parked 20 units from tgt).
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, 0, 50.0, 50.0, ships=4, production=2)
    opp_at_rest = _planet(2, 1, 70.0, 50.0, ships=100, production=2)
    # In-flight enemy fleet aimed at tgt — gives `time_to_enemy_threat`
    # a non-None answer.
    inbound = _fleet(0, 1, 65.0, 50.0, angle=3.141592, ships=20)
    world = _world(0, [src, tgt, opp_at_rest], fleets=[inbound])
    model = WorldModel.from_world(world)
    v = w2_provably_held_reinforce(
        src, tgt, ships=50, wait_N=0, eta=7, world=world, model=model, me=0,
    )
    # opp_at_rest has 100 ships and is within counter-reach; W2 must abstain.
    assert v.kind == "uncertain", (
        f"expected abstain due to at-rest opp; got {v}"
    )


def test_w2_commits_when_threat_fully_inflight_and_we_arrive_in_time():
    """Positive case: in-flight enemy fleet is the only threat, our
    reinforce arrives in time to defend. No other opp planet in reach.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, 0, 50.0, 50.0, ships=4, production=2)
    # Far-away weak opp — below MIN_COUNTER_SHIPS=20 → not a counter threat.
    opp_far = _planet(2, 1, 95.0, 50.0, ships=10, production=1)
    # In-flight enemy fleet heading at tgt with eta ≈ 5.
    inbound = _fleet(0, 1, 80.0, 50.0, angle=3.141592, ships=15)
    world = _world(0, [src, tgt, opp_far], fleets=[inbound])
    model = WorldModel.from_world(world)
    # Our reinforce arrives before threat and overwhelms it.
    v = w2_provably_held_reinforce(
        src, tgt, ships=50, wait_N=0, eta=3, world=world, model=model, me=0,
    )
    assert v.kind == "commit", f"expected commit; got {v}"
    assert v.reason == "W2"


def test_w2_abstains_when_we_arrive_too_late():
    """Our reinforce arrives after the planet has already flipped. W2
    abstains; the rollout will score this as a likely recapture rather
    than a hold.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, 0, 50.0, 50.0, ships=2, production=1)
    opp_far = _planet(2, 1, 95.0, 50.0, ships=10, production=1)
    # Big inbound enemy fleet — will overwhelm tgt at eta ≈ 2.
    inbound = _fleet(0, 1, 60.0, 50.0, angle=3.141592, ships=50)
    world = _world(0, [src, tgt, opp_far], fleets=[inbound])
    model = WorldModel.from_world(world)
    # Our reinforce arrives at eta=10 — far too late.
    v = w2_provably_held_reinforce(
        src, tgt, ships=20, wait_N=0, eta=10, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain", (
        f"expected abstain (too-late reinforce); got {v}"
    )


def test_w2_verdict_is_immutable():
    """`Verdict` is a frozen dataclass; cannot be mutated post-creation."""
    v = Verdict(kind="commit", lower_bound=1.5, reason="W2")
    import pytest
    with pytest.raises((AttributeError, Exception)):
        v.kind = "discard"  # type: ignore[misc]


def test_uncertain_singleton_has_zero_lower_bound():
    assert UNCERTAIN.kind == "uncertain"
    assert UNCERTAIN.lower_bound == 0.0
    assert UNCERTAIN.reason == ""


# ---------------------------------------------------------------------------
# W1 / L1 / L2 — stubs return UNCERTAIN / passthrough until implemented.
# These tests pin the stub behaviour so later wiring can rely on them.
# ---------------------------------------------------------------------------


def test_w1_stub_returns_uncertain():
    src = _planet(0, 0, 10.0, 50.0)
    tgt = _planet(1, 1, 30.0, 50.0)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=10, wait_N=0, eta=3, world=world, model=model, me=0,
    )
    assert v is UNCERTAIN or v.kind == "uncertain"


def test_l1_stub_returns_uncertain():
    src = _planet(0, 0, 10.0, 50.0)
    tgt = _planet(1, 1, 30.0, 50.0)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    v = l1_provably_wasted_launch(
        src, tgt, ships=10, wait_N=0, eta=3, world=world, model=model, me=0,
    )
    assert v is UNCERTAIN or v.kind == "uncertain"


def test_l2_stub_passes_candidates_through():
    fake = [("a",), ("b",)]
    out = l2_dominance_prune(fake)
    assert out == fake
