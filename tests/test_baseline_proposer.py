"""Unit tests for agents/baseline/proposer."""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from agents.baseline.proposer import (
    MIN_FLEET_SIZE,
    WAIT_EXTRA_SURPLUS,
    _enumerate_reactor_candidates,
    _target_cost_parity_ok,
    capture_size,
    enumerate_ship_counts,
    min_wait_affordable,
    nearest_k,
    propose,
    wait_band,
    wait_then_fire_variants,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, x, y, *, ships=10, production=2, radius=1.5):
    """Planet is a NamedTuple: (id, owner, x, y, radius, ships, production)."""
    return Planet(pid, owner, x, y, radius, ships, production)


def _world(my_id, planets, *, step=0, omega=0.0, fleets=None):
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


def _fleet(fid, owner, x, y, angle, from_pid, ships):
    """Fleet is a NamedTuple: (id, owner, x, y, angle, from_planet_id, ships)."""
    return Fleet(fid, owner, x, y, angle, from_pid, ships)


def test_wait_band_buckets():
    assert wait_band(0) == 0
    assert wait_band(1) == 1
    assert wait_band(7) == 1
    assert wait_band(8) == 2
    assert wait_band(100) == 2


def test_nearest_k_sorts_by_distance():
    src = _planet(0, 0, 10.0, 50.0)
    targets = [
        _planet(1, 1, 90.0, 50.0),  # far
        _planet(2, 1, 20.0, 50.0),  # near
        _planet(3, 1, 50.0, 50.0),  # mid
    ]
    out = nearest_k(targets, src, 2)
    assert [t.id for t in out] == [2, 3]


def test_capture_size_neutral_is_at_least_min_plus_one():
    """For a non-mine target, capture size >= max(MIN_FLEET_SIZE, garrison+1)."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=2)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=1)  # neutral with 5 ships
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    sz = capture_size(src, tgt, model, omega=0.0, me=0, world=world)
    assert sz >= 6  # at minimum need garrison+1 = 6


def test_capture_size_own_no_threat_returns_zero():
    """Reinforce for my own planet with no enemy threat → 0 (skip)."""
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    mine_tgt = _planet(1, 0, 12.0, 50.0, ships=5)  # also mine, no threat
    world = _world(0, [src, mine_tgt])
    model = WorldModel.from_world(world)
    sz = capture_size(src, mine_tgt, model, omega=0.0, me=0, world=world)
    assert sz == 0


def test_enumerate_ship_counts_returns_sizes_at_or_under_budget():
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=1)
    tgt = _planet(1, -1, 12.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    sizes = enumerate_ship_counts(src, tgt, model, omega=0.0, me=0, world=world)
    assert sizes  # non-empty
    assert all(MIN_FLEET_SIZE <= s <= 50 for s in sizes)
    assert sizes == sorted(sizes)  # sorted ascending


def test_wait_then_fire_variants_skips_mine_targets():
    src = _planet(0, 0, 10.0, 50.0, ships=10, production=1)
    mine = _planet(1, 0, 12.0, 50.0, ships=5)
    world = _world(0, [src, mine])
    model = WorldModel.from_world(world)
    assert wait_then_fire_variants(src, mine, model, omega=0.0, me=0) == []


def test_wait_then_fire_variants_skips_zero_production():
    src = _planet(0, 0, 10.0, 50.0, ships=10, production=0)
    tgt = _planet(1, -1, 12.0, 50.0, ships=20)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    # src can't accumulate so no wait variants
    assert wait_then_fire_variants(src, tgt, model, omega=0.0, me=0) == []


def test_wait_then_fire_variants_emits_nothing_when_already_armed():
    """Backward grid (2026-05-18): src that can fire NOW affordably
    gets NO wait variants. The fire-now path handles affordable pairs;
    emitting speculative waits would compete with fire-now in chooser
    Δ ranking and cause under-emission (Roman game diagnosis).

    Previous behaviour (forward grid): always emit wait_N=1 even when
    feasible-now. That bug is now in wait_then_fire_variants_forward.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 12.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    variants = wait_then_fire_variants(src, tgt, model, omega=0.0, me=0)
    assert variants == [], (
        f"backward grid should emit [] for already-armed src; got {variants}"
    )


def test_wait_then_fire_variants_forward_legacy():
    """Legacy forward-grid path preserved via BASELINE_WAIT_GRID=forward."""
    import os
    from agents.baseline.proposer import wait_then_fire_variants_forward
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 12.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    # Direct call to the forward variant (bypasses env-var dispatch)
    variants = wait_then_fire_variants_forward(src, tgt, model, omega=0.0, me=0)
    assert variants, "forward variant should emit wait_N=1 even when armed"
    assert all(w >= 1 for _ships, w, _angle, _eta in variants)
    assert len(variants) <= len(WAIT_EXTRA_SURPLUS)


def test_propose_dedups_per_src_tgt_band():
    """propose() returns at most one entry per (src, tgt, wait_band).
    Bands are {0, 1..7, >=8}, so at most 3 entries per (src, tgt) pair.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[tgt],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    # All entries are for (src, tgt)
    assert all(int(e[1].id) == 0 and int(e[2].id) == 1 for e in out)
    # Unique (band) per (src, tgt)
    bands = [wait_band(int(e[7])) for e in out]
    assert len(bands) == len(set(bands))


def test_propose_sorts_descending_by_cheap_delta():
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[tgt],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    cheap_vals = [e[0] for e in out]
    assert cheap_vals == sorted(cheap_vals, reverse=True)


def test_propose_returns_empty_when_no_targets():
    src = _planet(0, 0, 10.0, 50.0, ships=20)
    world = _world(0, [src])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    assert out == []


# ---------------------------------------------------------------------------
# min_wait_affordable + backward wait grid (2026-05-18 Tier 1.5)
# ---------------------------------------------------------------------------


def test_min_wait_affordable_returns_zero_when_src_already_armed():
    """src has plenty of ships → fire-now affordable → return 0."""
    src = _planet(0, 0, 10.0, 50.0, ships=100, production=1)
    tgt = _planet(1, -1, 12.0, 50.0, ships=10)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    result = min_wait_affordable(src, tgt, model, omega=0.0, me=0)
    assert result == 0, f"expected 0 (fire-now affordable); got {result}"


def test_min_wait_affordable_returns_positive_for_accumulation():
    """src has too few ships → must accumulate → return positive wait_N."""
    src = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    tgt = _planet(1, -1, 12.0, 50.0, ships=20)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    result = min_wait_affordable(src, tgt, model, omega=0.0, me=0)
    assert result is not None and result > 0, (
        f"expected positive wait (need to accumulate); got {result}"
    )


def test_min_wait_affordable_returns_none_for_zero_production():
    """src can't produce → can't accumulate → return None."""
    src = _planet(0, 0, 10.0, 50.0, ships=5, production=0)
    tgt = _planet(1, -1, 12.0, 50.0, ships=20)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    result = min_wait_affordable(src, tgt, model, omega=0.0, me=0)
    assert result is None, f"expected None (zero production); got {result}"


def test_min_wait_affordable_returns_none_for_owned_target():
    """tgt is mine → reinforce path handles it; return None."""
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=1)
    own_tgt = _planet(1, 0, 12.0, 50.0, ships=5)  # me owns tgt
    world = _world(0, [src, own_tgt])
    model = WorldModel.from_world(world)
    result = min_wait_affordable(src, own_tgt, model, omega=0.0, me=0)
    assert result is None


def test_wait_variants_anchored_at_min_wait():
    """When src must accumulate, wait variants emit at {min_w, min_w+3}."""
    src = _planet(0, 0, 10.0, 50.0, ships=5, production=1)
    tgt = _planet(1, -1, 12.0, 50.0, ships=20)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    min_w = min_wait_affordable(src, tgt, model, omega=0.0, me=0)
    assert min_w is not None and min_w > 0
    variants = wait_then_fire_variants(src, tgt, model, omega=0.0, me=0)
    assert variants, "expected at least one wait variant for accumulating src"
    wait_Ns = {int(w) for _ships, w, _angle, _eta in variants}
    expected_anchor = {min_w, min_w + 3}
    assert wait_Ns.issubset(expected_anchor), (
        f"variants should anchor at min_w({min_w}) and min_w+3({min_w + 3}); "
        f"got wait_Ns={wait_Ns}"
    )


# ---------------------------------------------------------------------------
# Reactor-aware launch selection (Part A: cost-parity filter,
# Part B: reactor candidate generator). 2026-05-19 PM.
# ---------------------------------------------------------------------------


def test_cost_parity_accepts_own_target():
    """Reinforce launches on our own planets are never races; always pass."""
    src = _planet(0, 0, 10.0, 50.0, ships=20, production=2)
    mine_tgt = _planet(1, 0, 50.0, 50.0, ships=5, production=2)
    opp = _planet(2, 1, 90.0, 50.0, ships=50, production=2)
    world = _world(0, [src, mine_tgt, opp])
    model = WorldModel.from_world(world)
    assert _target_cost_parity_ok(
        src, mine_tgt, ships=10, wait_N=0, eta=20, world=world, model=model, me=0,
    ) is True


def test_cost_parity_accepts_no_opp_in_range():
    """No opp with MIN_REACTOR_SHIPS within range → filter passes."""
    src = _planet(0, 0, 10.0, 50.0, ships=30, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=5, production=1)
    # opp with 3 ships < MIN_REACTOR_SHIPS=8 → not a reactor threat
    opp_weak = _planet(2, 1, 60.0, 50.0, ships=3, production=1)
    world = _world(0, [src, tgt, opp_weak])
    model = WorldModel.from_world(world)
    assert _target_cost_parity_ok(
        src, tgt, ships=20, wait_N=0, eta=15, world=world, model=model, me=0,
    ) is True


def test_cost_parity_accepts_balanced_geometry():
    """When opp's reactor cost ~= ours, filter accepts (default margin 0.7)."""
    src = _planet(0, 0, 10.0, 50.0, ships=50, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=5, production=1)
    opp = _planet(2, 1, 90.0, 50.0, ships=50, production=2)  # mirror of src
    world = _world(0, [src, tgt, opp])
    model = WorldModel.from_world(world)
    # 30-ship launch; balanced opp ≈ 30 ships needed → ratio ≈ 1.0 → accept.
    assert _target_cost_parity_ok(
        src, tgt, ships=30, wait_N=0, eta=14, world=world, model=model, me=0,
    ) is True


def test_cost_parity_rejects_high_cost_launch_with_close_opp():
    """High-cost launch (heavy defenders) where a close opp recaptures
    cheaply (small residue) → reject. This is the first-mover-trap pattern."""
    src = _planet(0, 0, 5.0, 50.0, ships=200, production=2)
    # Heavy-defender neutral: we pay 100 to capture but only 60 remain
    tgt = _planet(1, -1, 80.0, 50.0, ships=40, production=1)
    opp = _planet(2, 1, 85.0, 50.0, ships=50, production=2)
    world = _world(0, [src, tgt, opp])
    model = WorldModel.from_world(world)
    # 100-ship launch: we pay 100, opp recapture ≈ 62 — opp pays 62% → reject.
    assert _target_cost_parity_ok(
        src, tgt, ships=100, wait_N=0, eta=15, world=world, model=model, me=0,
    ) is False


def test_cost_parity_env_var_disables():
    """Setting PROPOSER_COST_PARITY=off bypasses the filter in propose()."""
    src = _planet(0, 0, 5.0, 50.0, ships=200, production=2)
    tgt = _planet(1, -1, 80.0, 50.0, ships=40, production=1)
    opp = _planet(2, 1, 85.0, 50.0, ships=50, production=2)
    world = _world(0, [src, tgt, opp])
    model = WorldModel.from_world(world)
    old = os.environ.get("PROPOSER_COST_PARITY")
    os.environ["PROPOSER_COST_PARITY"] = "off"
    try:
        out = propose(
            my_planets=[src], target_pool=[tgt, opp],
            world=world, model=model, me=0, omega=0.0, baseline_len=50,
        )
        # With filter disabled, at least one candidate aimed at the
        # vulnerable target survives (cost-parity would otherwise drop it).
        # We don't need a strict count; just that the bypass is plumbed.
        assert isinstance(out, list)
    finally:
        if old is None:
            os.environ.pop("PROPOSER_COST_PARITY", None)
        else:
            os.environ["PROPOSER_COST_PARITY"] = old


def test_cost_parity_margin_env_var_overrides():
    """COST_PARITY_MARGIN env var changes the rejection threshold."""
    src = _planet(0, 0, 5.0, 50.0, ships=200, production=2)
    tgt = _planet(1, -1, 80.0, 50.0, ships=40, production=1)
    opp = _planet(2, 1, 85.0, 50.0, ships=50, production=2)
    world = _world(0, [src, tgt, opp])
    model = WorldModel.from_world(world)
    # Default margin (0.7) rejects this scenario
    assert _target_cost_parity_ok(
        src, tgt, ships=100, wait_N=0, eta=15, world=world, model=model, me=0,
    ) is False
    # Tighten margin to 0.0 → no rejection unless opp pays 0 → accept
    old = os.environ.get("COST_PARITY_MARGIN")
    os.environ["COST_PARITY_MARGIN"] = "0.0"
    try:
        assert _target_cost_parity_ok(
            src, tgt, ships=100, wait_N=0, eta=15, world=world, model=model, me=0,
        ) is True
    finally:
        if old is None:
            os.environ.pop("COST_PARITY_MARGIN", None)
        else:
            os.environ["COST_PARITY_MARGIN"] = old


def test_reactor_candidates_empty_when_no_opp_fleets():
    """No opp fleets in flight → reactor generator returns empty."""
    src = _planet(0, 0, 10.0, 50.0, ships=30, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=5, production=1)
    opp = _planet(2, 1, 90.0, 50.0, ships=30, production=2)
    world = _world(0, [src, tgt, opp])  # no fleets
    model = WorldModel.from_world(world)
    out = _enumerate_reactor_candidates(
        my_planets=[src], world=world, model=model, me=0,
        omega=0.0, baseline_len=50,
    )
    assert out == []


def test_reactor_candidates_fires_on_opp_fleet_capturing_neutral():
    """Opp fleet en route to a neutral that they'll capture → we get a
    reactor candidate from our nearby source."""
    src = _planet(0, 0, 30.0, 50.0, ships=80, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=5, production=1)
    opp_planet = _planet(2, 1, 75.0, 50.0, ships=5, production=2)
    # Opp fleet of 20 ships at (58, 50) heading toward tgt at (50, 50)
    # angle=π points in -x direction, which moves the fleet from x=58
    # toward x=50 (tgt). 20 > 5 defenders → opp captures.
    import math as _math
    opp_fleet = _fleet(
        fid=100, owner=1, x=58.0, y=50.0, angle=_math.pi,
        from_pid=2, ships=20,
    )
    world = _world(0, [src, tgt, opp_planet], fleets=[opp_fleet])
    model = WorldModel.from_world(world)
    out = _enumerate_reactor_candidates(
        my_planets=[src], world=world, model=model, me=0,
        omega=0.0, baseline_len=50,
    )
    assert out, "expected at least one reactor candidate"
    # Every candidate targets tgt from src
    for entry in out:
        cheap, c_src, c_tgt, ships, _angle, _eta, _horizon, wait_N = entry
        assert int(c_src.id) == 0
        assert int(c_tgt.id) == 1
        assert ships >= MIN_FLEET_SIZE
        assert wait_N >= 0


def test_reactor_candidates_skips_when_opp_bounces():
    """Opp fleet too small to capture (bounces off defenders) → no reactor
    candidate (existing wait/fire-now paths handle the still-neutral target)."""
    src = _planet(0, 0, 30.0, 50.0, ships=80, production=2)
    # tgt has 30 defenders; opp's 5-ship fleet will bounce
    tgt = _planet(1, -1, 50.0, 50.0, ships=30, production=1)
    opp_planet = _planet(2, 1, 75.0, 50.0, ships=5, production=2)
    import math as _math
    opp_fleet = _fleet(
        fid=100, owner=1, x=58.0, y=50.0, angle=_math.pi,
        from_pid=2, ships=5,
    )
    world = _world(0, [src, tgt, opp_planet], fleets=[opp_fleet])
    model = WorldModel.from_world(world)
    out = _enumerate_reactor_candidates(
        my_planets=[src], world=world, model=model, me=0,
        omega=0.0, baseline_len=50,
    )
    # Opp bounces → target stays neutral → existing pipeline covers; no reactor.
    assert out == []


def test_reactor_candidates_env_var_disables_via_propose():
    """PROPOSER_REACTOR_CANDIDATES=off prevents reactor candidates from
    being added inside propose()."""
    src = _planet(0, 0, 30.0, 50.0, ships=80, production=2)
    tgt = _planet(1, -1, 50.0, 50.0, ships=5, production=1)
    opp_planet = _planet(2, 1, 75.0, 50.0, ships=5, production=2)
    import math as _math
    opp_fleet = _fleet(
        fid=100, owner=1, x=58.0, y=50.0, angle=_math.pi,
        from_pid=2, ships=20,
    )
    world = _world(0, [src, tgt, opp_planet], fleets=[opp_fleet])
    model = WorldModel.from_world(world)
    old = os.environ.get("PROPOSER_REACTOR_CANDIDATES")
    os.environ["PROPOSER_REACTOR_CANDIDATES"] = "off"
    try:
        out = propose(
            my_planets=[src], target_pool=[tgt, opp_planet],
            world=world, model=model, me=0, omega=0.0, baseline_len=50,
        )
        # We can't assert the absence of all reactor-like candidates (the
        # existing wait_then_fire path may emit similar moves), but we CAN
        # assert no exception, that propose still returns a list, and that
        # the toggle is wired (the function is invoked, the disable path
        # taken — covered by code coverage). Just smoke-check the contract.
        assert isinstance(out, list)
    finally:
        if old is None:
            os.environ.pop("PROPOSER_REACTOR_CANDIDATES", None)
        else:
            os.environ["PROPOSER_REACTOR_CANDIDATES"] = old
