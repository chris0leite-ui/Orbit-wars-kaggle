"""Unit tests for agents/baseline/proposer."""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from agents.baseline.proposer import (
    MIN_FLEET_SIZE,
    WAIT_EXTRA_SURPLUS,
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


def _world(my_id, planets, *, step=0, omega=0.0):
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
    }
    return World.from_obs(obs)


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
    """propose() dedups per (src, tgt, wait_band, tier_band).

    Pre-2026-05-21: at most one entry per (src, tgt, wait_band).
    Post-tier-aware: at most one per (src, tgt, wait_band, tier_band)
    where tier_band ∈ {0: spec_min, 1: buffered, 2: other_overkill}.
    Each wait_band can thus host up to 3 entries — one per tier_band.
    The split is what lets the LP pick efficiency vs robustness.
    """
    from collections import Counter
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
    # At most 3 entries per wait_band (one per tier_band).
    bands = Counter(wait_band(int(e[7])) for e in out)
    assert all(c <= 3 for c in bands.values()), (
        f"wait_band counts exceed tier-band split limit (3): {bands}"
    )


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
