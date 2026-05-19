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
# W1 — provably-winning capture
# ---------------------------------------------------------------------------


def test_w1_inapplicable_reinforce():
    """W1 defers to W2 for reinforces (tgt.owner == me)."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    mine_tgt = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, mine_tgt])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, mine_tgt, ships=10, wait_N=0, eta=1, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain"


def test_w1_uncertain_on_bounce():
    """Under-sized capture → owner_at_arrival != me → uncertain (L1's job)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    # Neutral with 50 ships, we send only 5 → bounce.
    tgt = _planet(1, -1, 50.0, 50.0, ships=50, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=5, wait_N=0, eta=7, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain"


def test_w1_commits_on_clean_capture():
    """Strong capture against a far-away weak opp, source not exposed."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    # Neutral close to src; we send 80 ships, easily takes garrison=10.
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    # Far-away weak opp — below MIN_COUNTER_SHIPS=20, no counter threat.
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10, production=1)
    world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=80, wait_N=0, eta=4, world=world, model=model, me=0,
    )
    assert v.kind == "commit", f"expected commit; got {v}"
    assert v.reason == "W1"
    assert v.lower_bound > 0.0


def test_w1_uncertain_when_counter_recapture_feasible():
    """Strong nearby opp can recapture before our garrison defends."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=3)
    tgt = _planet(1, -1, 50.0, 50.0, ships=10, production=2)
    # Nearby strong opp — can counter and recapture.
    opp_close = _planet(2, 1, 55.0, 50.0, ships=200, production=4)
    world = _world(0, [src, tgt, opp_close])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=40, wait_N=0, eta=8, world=world, model=model, me=0,
    )
    # Counter recaptures → _target_holdable_after_capture returns False → uncertain.
    assert v.kind == "uncertain"


def test_w1_uncertain_when_source_exposed():
    """Source has its own inbound threat and post-launch can't defend."""
    src = _planet(0, 0, 10.0, 50.0, ships=20, production=1)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=1)
    opp_far = _planet(2, 1, 95.0, 50.0, ships=5)
    # Big inbound to src — drains the source if we launch.
    inbound_at_src = _fleet(0, 1, 15.0, 50.0, angle=3.141592, ships=50)
    world = _world(0, [src, tgt, opp_far], fleets=[inbound_at_src])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=15, wait_N=0, eta=3, world=world, model=model, me=0,
    )
    # _source_survives_launch returns False → uncertain.
    assert v.kind == "uncertain"


# ---------------------------------------------------------------------------
# L1 — provably-wasted launch
# ---------------------------------------------------------------------------


def test_l1_inapplicable_reinforce():
    """L1 defers to W2 for reinforces."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    mine_tgt = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, mine_tgt])
    model = WorldModel.from_world(world)
    v = l1_provably_wasted_launch(
        src, mine_tgt, ships=10, wait_N=0, eta=1, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain"


def test_l1_discards_bounce():
    """Under-sized launch → never own target in window → discard."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    # Neutral with garrison too big for our 5-ship launch.
    tgt = _planet(1, -1, 50.0, 50.0, ships=100, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    v = l1_provably_wasted_launch(
        src, tgt, ships=5, wait_N=0, eta=7, world=world, model=model, me=0,
    )
    assert v.kind == "discard", f"expected discard; got {v}"
    assert v.reason == "L1"


def test_l1_uncertain_when_capture_succeeds():
    """Successful capture (we own at arrival) → uncertain; rollout scores."""
    src = _planet(0, 0, 10.0, 50.0, ships=200)
    tgt = _planet(1, -1, 50.0, 50.0, ships=10, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    v = l1_provably_wasted_launch(
        src, tgt, ships=80, wait_N=0, eta=7, world=world, model=model, me=0,
    )
    assert v.kind == "uncertain"


# ---------------------------------------------------------------------------
# L2 — dominance prune
# ---------------------------------------------------------------------------


def _candidate(src, tgt, *, cheap_delta=1.0, ships=10, eta=5, wait_N=0):
    """Build a proposer-style prerank tuple."""
    return (float(cheap_delta), src, tgt, int(ships), 0.0, int(eta),
            int(eta + 2), int(wait_N))


def test_l2_empty_input():
    assert l2_dominance_prune([]) == []


def test_l2_drops_strictly_dominated_same_src_tgt():
    """Two candidates, same (src,tgt): the strictly worse one is dropped."""
    src = _planet(0, 0, 10.0, 50.0)
    tgt = _planet(1, 1, 30.0, 50.0)
    # B dominates A on all three dims.
    a = _candidate(src, tgt, cheap_delta=1.0, ships=20, eta=10)
    b = _candidate(src, tgt, cheap_delta=2.0, ships=10, eta=5)
    out = l2_dominance_prune([a, b])
    assert len(out) == 1
    assert out[0] is b


def test_l2_keeps_both_when_incomparable():
    """Tradeoff candidates (one higher cheap_delta, other lower ships) → both kept."""
    src = _planet(0, 0, 10.0, 50.0)
    tgt = _planet(1, 1, 30.0, 50.0)
    a = _candidate(src, tgt, cheap_delta=1.0, ships=10, eta=5)
    # B has higher cheap_delta but uses MORE ships → incomparable.
    b = _candidate(src, tgt, cheap_delta=2.0, ships=20, eta=5)
    out = l2_dominance_prune([a, b])
    assert len(out) == 2


def test_l2_keeps_different_targets():
    """Different (src,tgt) pairs never dominate each other in v1."""
    src = _planet(0, 0, 10.0, 50.0)
    tgt1 = _planet(1, 1, 30.0, 50.0)
    tgt2 = _planet(2, 1, 40.0, 50.0)
    a = _candidate(src, tgt1, cheap_delta=1.0, ships=10, eta=5)
    b = _candidate(src, tgt2, cheap_delta=2.0, ships=5, eta=3)  # better on all
    out = l2_dominance_prune([a, b])
    # Different targets → no dominance check applies.
    assert len(out) == 2


def test_l2_preserves_order():
    """Survivors retain input order."""
    src = _planet(0, 0, 10.0, 50.0)
    tgt1 = _planet(1, 1, 30.0, 50.0)
    tgt2 = _planet(2, 1, 40.0, 50.0)
    a = _candidate(src, tgt1, cheap_delta=1.0, ships=10, eta=5)
    b = _candidate(src, tgt2, cheap_delta=0.5, ships=10, eta=5)
    out = l2_dominance_prune([a, b])
    assert out[0] is a and out[1] is b
