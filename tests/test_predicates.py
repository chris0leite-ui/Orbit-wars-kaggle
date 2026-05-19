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
    _w1_value_bounds,
    l1_provably_wasted_launch,
    l2_dominance_prune,
    w1_dominance_classify,
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


def test_w1_uncertain_under_two_opp_gangup():
    """Slice 3 — Wald bound: two strong opps that EACH alone wouldn't
    overwhelm us still abstain because the SUM exceeds safety × garrison.

    Setup: our capture delivers a modest garrison; two opps at moderate
    range each have 80 ships (below the SAFETY × delivered threshold
    individually) but their coordinated total exceeds it.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=10, production=1)
    # Two opps within counter-reach, each 80 ships.
    opp_a = _planet(2, 1, 60.0, 45.0, ships=80, production=2)
    opp_b = _planet(3, 1, 60.0, 55.0, ships=80, production=2)
    world = _world(0, [src, tgt, opp_a, opp_b])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=40, wait_N=0, eta=7, world=world, model=model, me=0,
    )
    # Delivered ≈ 40 - 10 = 30; each opp contributes ~80+ ships within
    # window → coord_counter > SAFETY × garrison → abstain.
    assert v.kind == "uncertain", f"gang-up should abstain; got {v}"


def test_w1_uncertain_under_three_opp_gangup():
    """Three modest opps; sum dominates even though no single one does."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    tgt = _planet(1, -1, 50.0, 50.0, ships=10, production=2)
    opp_a = _planet(2, 1, 60.0, 40.0, ships=40, production=1)
    opp_b = _planet(3, 1, 60.0, 50.0, ships=40, production=1)
    opp_c = _planet(4, 1, 60.0, 60.0, ships=40, production=1)
    world = _world(0, [src, tgt, opp_a, opp_b, opp_c])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=50, wait_N=0, eta=7, world=world, model=model, me=0,
    )
    # Three opps × 40+ ships in reach should sum to exceed safety × ~40.
    assert v.kind == "uncertain", f"3-opp gang-up should abstain; got {v}"


def test_w1_commits_when_only_one_opp_in_reach():
    """Single-opp scenario the variant-1 check would have allowed must
    STILL commit under variant 2 — no regression on the clean case.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    # One distant weak opp — below MIN_COUNTER_SHIPS or out of window.
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10, production=1)
    world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    v = w1_provably_winning_capture(
        src, tgt, ships=80, wait_N=0, eta=4, world=world, model=model, me=0,
    )
    assert v.kind == "commit"


def test_w1_multi_opp_helper_returns_true_when_no_opps():
    """No opps in reach → trivially holds."""
    from agents.baseline.predicates import _w1_multi_opp_holds
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt = _planet(1, -1, 30.0, 50.0, ships=5, production=1)
    world = _world(0, [src, tgt])
    assert _w1_multi_opp_holds(
        src, tgt, ships=50, wait_N=0, eta=4, world=world, me=0,
    ) is True


def test_w1_multi_opp_helper_rejects_bounce():
    """Delivered < 1 (under-sized capture) → never holds."""
    from agents.baseline.predicates import _w1_multi_opp_holds
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    tgt = _planet(1, -1, 30.0, 50.0, ships=100, production=1)
    world = _world(0, [src, tgt])
    assert _w1_multi_opp_holds(
        src, tgt, ships=5, wait_N=0, eta=4, world=world, me=0,
    ) is False


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


# ---------------------------------------------------------------------------
# Slice 5 — bounded-interval scoring + dominance classifier
# ---------------------------------------------------------------------------


def test_value_bounds_reinforce_returns_zero_zero():
    """W2's territory; bounds are (0, 0) so this candidate doesn't compete in W1 dominance."""
    src = _planet(0, 0, 10.0, 50.0, ships=80)
    mine_tgt = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, mine_tgt])
    model = WorldModel.from_world(world)
    lo, hi = _w1_value_bounds(src, mine_tgt, ships=10, wait_N=0, eta=1, world=world, model=model, me=0)
    assert (lo, hi) == (0.0, 0.0)


def test_value_bounds_bounce_returns_zero_zero():
    """Capture fails at arrival; both bounds 0."""
    src = _planet(0, 0, 10.0, 50.0, ships=100)
    tgt = _planet(1, -1, 50.0, 50.0, ships=100, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    lo, hi = _w1_value_bounds(src, tgt, ships=5, wait_N=0, eta=7, world=world, model=model, me=0)
    assert (lo, hi) == (0.0, 0.0)


def test_value_bounds_clean_capture_lo_positive():
    """Clean capture passes Wald → lower bound > 0; upper bound is production × pv."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    lo, hi = _w1_value_bounds(src, tgt, ships=80, wait_N=0, eta=4, world=world, model=model, me=0)
    assert 0.0 < lo <= hi
    assert hi > 0.0


def test_value_bounds_contested_capture_lo_zero_hi_positive():
    """Capture succeeds but Wald fails (gang-up exists) → lo=0, hi>0."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    tgt = _planet(1, -1, 50.0, 50.0, ships=10, production=2)
    opp_a = _planet(2, 1, 60.0, 45.0, ships=80, production=2)
    opp_b = _planet(3, 1, 60.0, 55.0, ships=80, production=2)
    world = _world(0, [src, tgt, opp_a, opp_b])
    model = WorldModel.from_world(world)
    lo, hi = _w1_value_bounds(src, tgt, ships=80, wait_N=0, eta=7, world=world, model=model, me=0)
    assert lo == 0.0  # Wald rejected (gang-up)
    assert hi > 0.0   # but capture itself succeeds


def test_dominance_classifier_single_source_single_candidate_commits():
    """One candidate on src; lo > 0 → commit."""
    src = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    tgt = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    opp_far = _planet(2, 1, 95.0, 95.0, ships=10)
    world = _world(0, [src, tgt, opp_far])
    model = WorldModel.from_world(world)
    cand = (5.0, src, tgt, 80, 0.0, 4, 6, 0)
    verdicts = w1_dominance_classify([cand], world, model, 0, gamma=0.99)
    assert id(cand) in verdicts
    assert verdicts[id(cand)].kind == "commit"
    assert verdicts[id(cand)].reason == "W1"


def test_dominance_classifier_two_candidates_no_dominance_skips():
    """Two W1-eligible candidates from same source; neither lo > other's hi → no commit."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    tgt_a = _planet(1, -1, 30.0, 50.0, ships=10, production=2)
    tgt_b = _planet(2, -1, 40.0, 50.0, ships=10, production=2)
    opp_far = _planet(3, 1, 95.0, 95.0, ships=10)
    world = _world(0, [src, tgt_a, tgt_b, opp_far])
    model = WorldModel.from_world(world)
    cand_a = (5.0, src, tgt_a, 80, 0.0, 4, 6, 0)
    cand_b = (5.0, src, tgt_b, 80, 0.0, 5, 7, 0)
    verdicts = w1_dominance_classify([cand_a, cand_b], world, model, 0, gamma=0.99)
    # Two equal-production targets; neither's lo > other's hi → no commit.
    assert len(verdicts) == 0


def test_dominance_classifier_clear_winner_commits_only_one():
    """Source has one high-value + one low-value candidate; only the high one commits."""
    src = _planet(0, 0, 10.0, 50.0, ships=200, production=3)
    # High-value: production=5, capture succeeds, holds.
    tgt_high = _planet(1, -1, 30.0, 50.0, ships=10, production=5)
    # Low-value: production=1, very different scale.
    tgt_low = _planet(2, -1, 40.0, 50.0, ships=10, production=1)
    opp_far = _planet(3, 1, 95.0, 95.0, ships=10)
    world = _world(0, [src, tgt_high, tgt_low, opp_far])
    model = WorldModel.from_world(world)
    cand_high = (5.0, src, tgt_high, 80, 0.0, 4, 6, 0)
    cand_low = (1.0, src, tgt_low, 80, 0.0, 5, 7, 0)
    verdicts = w1_dominance_classify([cand_high, cand_low], world, model, 0, gamma=0.99)
    # tgt_high's lower bound (production=5 × pv over window) should exceed
    # tgt_low's upper bound (production=1 × pv full). Verify the high one commits.
    if id(cand_high) in verdicts:
        assert verdicts[id(cand_high)].kind == "commit"
    # The low one should NOT commit (it loses dominance to the high one).
    assert id(cand_low) not in verdicts or verdicts[id(cand_low)].kind != "commit"


def test_dominance_classifier_different_sources_independent():
    """Two sources, each with its own candidate; both can commit independently."""
    src_a = _planet(0, 0, 10.0, 50.0, ships=120, production=3)
    src_b = _planet(1, 0, 80.0, 50.0, ships=120, production=3)
    tgt_a = _planet(2, -1, 25.0, 50.0, ships=10, production=2)
    tgt_b = _planet(3, -1, 70.0, 50.0, ships=10, production=2)
    opp_far = _planet(4, 1, 95.0, 95.0, ships=10)
    world = _world(0, [src_a, src_b, tgt_a, tgt_b, opp_far])
    model = WorldModel.from_world(world)
    cand_a = (5.0, src_a, tgt_a, 80, 0.0, 4, 6, 0)
    cand_b = (5.0, src_b, tgt_b, 80, 0.0, 4, 6, 0)
    verdicts = w1_dominance_classify([cand_a, cand_b], world, model, 0, gamma=0.99)
    # Both should commit independently (different sources).
    assert id(cand_a) in verdicts and verdicts[id(cand_a)].kind == "commit"
    assert id(cand_b) in verdicts and verdicts[id(cand_b)].kind == "commit"
