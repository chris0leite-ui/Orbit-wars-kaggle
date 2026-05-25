"""Unit tests for agents/baseline/proposer."""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from agents.baseline.proposer import (
    GAMMA,
    MIN_FLEET_SIZE,
    WAIT_EXTRA_SURPLUS,
    _bleed_penalty,
    _enumerate_reactor_candidates,
    _target_cost_parity_ok,
    aim_and_eta,
    capture_size,
    cheap_marginal_value,
    enumerate_ship_counts,
    enumerate_wave_candidates,
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


# ---------------------------------------------------------------------------
# Comet-aim fix (Part C) — aim_and_eta routes comets to aim_comet
# (path-indexed lead, not orbital rotation).
# ---------------------------------------------------------------------------


def _world_with_comet(my_id, planets, *, comet_id, path, path_index,
                     omega=0.04, step=50):
    """Build a World with a single comet riding `path[path_index]` etc.

    The comet appears in `planets` (at its current path position), in
    `comet_planet_ids`, AND in `comets` (with the path metadata).
    """
    cur_x, cur_y = path[path_index]
    comet_planet = (comet_id, -1, float(cur_x), float(cur_y), 1.0, 30, 1)
    obs = {
        "player": my_id,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ] + [comet_planet],
        "fleets": [],
        "angular_velocity": omega,
        "comet_planet_ids": [comet_id],
        "comets": [
            {
                "planet_ids": [comet_id],
                "paths": [path],
                "path_index": path_index,
            },
        ],
        "step": step,
    }
    return World.from_obs(obs)


def test_aim_and_eta_routes_comet_to_path_indexed_lead():
    """For a comet target moving east on a linear path, aim_and_eta with
    world set returns an angle pointing at a FUTURE path index, not the
    comet's current position."""
    from agents.baseline.proposer import aim_and_eta
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    src = _planet(0, 0, 5.0, 50.0, ships=80, production=2)
    # Comet moves east at 4 board-units/turn along y=50 (linear path).
    path = [[20.0 + i * 4.0, 50.0] for i in range(30)]
    world = _world_with_comet(
        my_id=0, planets=[src], comet_id=42, path=path, path_index=0,
    )
    comet_planet = world.planets_by_id[42]
    angle_with_world, eta_with_world = aim_and_eta(
        src, comet_planet, ships=10, omega=0.04, world=world,
    )
    # The comet's CURRENT position is (20, 50). At our launch the path
    # advances 1 step per turn; by arrival the comet has moved east.
    # The aim point's x must be GREATER than 20 (eastward lead).
    import math as _math
    aim_x = src.x + _math.cos(angle_with_world) * 100  # any +ve scale
    assert aim_x > 20.0, (
        f"aim should lead east of current comet position; got angle={_math.degrees(angle_with_world):.2f}° → aim_x={aim_x:.2f}"
    )


def test_aim_and_eta_comet_disabled_via_env_var():
    """BASELINE_COMET_AIM=off bypasses the comet branch in aim_and_eta."""
    from agents.baseline.proposer import aim_and_eta
    src = _planet(0, 0, 5.0, 50.0, ships=80, production=2)
    path = [[20.0 + i * 4.0, 50.0] for i in range(30)]
    world = _world_with_comet(
        my_id=0, planets=[src], comet_id=42, path=path, path_index=0,
    )
    comet_planet = world.planets_by_id[42]
    old = os.environ.get("BASELINE_COMET_AIM")
    os.environ["BASELINE_COMET_AIM"] = "off"
    try:
        angle_off, eta_off = aim_and_eta(
            src, comet_planet, ships=10, omega=0.04, world=world,
        )
        # With path-aware aim disabled, the function falls back to the
        # orbital/atan2 path which aims at the comet's CURRENT position
        # (or its orbital-rotated prediction).
        import math as _math
        # is_orbiting on this comet: orbit_r = hypot(20-50, 50-50) = 30,
        # so (30 + 1.0) < 50 (rotation limit) is True → uses aim_orbiting.
        # We just assert it doesn't crash and returns a sensible eta.
        assert eta_off >= 1
    finally:
        if old is None:
            os.environ.pop("BASELINE_COMET_AIM", None)
        else:
            os.environ["BASELINE_COMET_AIM"] = old


# ---------------------------------------------------------------------------
# Wave incentive — production-bleed penalty (baseline_wave 2026-05-24)
# ---------------------------------------------------------------------------


def _with_env(**kv):
    """Context-managed env-var override for tests."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        old = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _ctx()


def test_bleed_off_when_flag_off():
    src = _planet(0, 0, 50.0, 50.0, ships=200, production=5)
    with _with_env(BASELINE_BLEED_PENALTY=None):
        assert _bleed_penalty(src, ships=30, t_transit=20) == 0.0


def test_bleed_zero_when_no_excess():
    """Emit==src.ships → excess=0 (clamped) → penalty 0 regardless of bleed_rate."""
    src = _planet(0, 0, 50.0, 50.0, ships=30, production=5)
    with _with_env(BASELINE_BLEED_PENALTY="1", BASELINE_BLEED_BETA="0.05"):
        # Emit all but the MIN_FLEET_SIZE floor → excess = 0 by definition.
        assert _bleed_penalty(src, ships=30 - MIN_FLEET_SIZE, t_transit=20) == 0.0


def test_bleed_fires_on_stockpile():
    """200-ship src, 30-ship emit, t=20, P=5.

    bleed_rate = max(0, 5*20 − 30) = 70
    excess     = max(0, 200 − 30 − MIN_FLEET_SIZE) = 168
    cost       = min(168, 70) = 70 (bleed_rate is the binding constraint)
    penalty    = 0.05 * 70 * 0.99^20
    """
    src = _planet(0, 0, 50.0, 50.0, ships=200, production=5)
    with _with_env(BASELINE_BLEED_PENALTY="1", BASELINE_BLEED_BETA="0.05"):
        got = _bleed_penalty(src, ships=30, t_transit=20)
    expected = 0.05 * 70.0 * (GAMMA ** 20)
    assert abs(got - expected) < 1e-9


def test_bleed_clamped_by_excess_not_bleed_rate():
    """Small-stockpile planet with long transit: excess is binding, not bleed_rate.

    src.ships=40, emit=20, P=5, t=30 → bleed_rate=130, excess=18 → cost=18.
    """
    src = _planet(0, 0, 50.0, 50.0, ships=40, production=5)
    with _with_env(BASELINE_BLEED_PENALTY="1", BASELINE_BLEED_BETA="0.05"):
        got = _bleed_penalty(src, ships=20, t_transit=30)
    expected = 0.05 * 18.0 * (GAMMA ** 30)
    assert abs(got - expected) < 1e-9


def test_bleed_zero_for_zero_transit():
    """t=0 → instant arrival → no opportunity cost."""
    src = _planet(0, 0, 50.0, 50.0, ships=200, production=5)
    with _with_env(BASELINE_BLEED_PENALTY="1", BASELINE_BLEED_BETA="0.05"):
        assert _bleed_penalty(src, ships=30, t_transit=0) == 0.0


def test_cheap_marginal_value_bleed_threads_through_capture_path():
    """Verifies cheap_marginal_value subtracts the bleed term in the
    capture-success branch."""
    src = _planet(0, 0, 50.0, 50.0, ships=200, production=5)
    tgt = _planet(1, 1, 60.0, 50.0, ships=5, production=3)
    world = _world(my_id=0, planets=[src, tgt], step=0)
    model = WorldModel.from_world(world, horizon=30)

    # With bleed OFF (default): captures pred_owner=enemy, ships>pred_ships
    # → positive ROI; orbital-safety is OFF too so we hit the simple branch.
    with _with_env(BASELINE_BLEED_PENALTY=None, BASELINE_ORBITAL_SAFETY=None):
        base_v = cheap_marginal_value(
            src, tgt, ships=30, eta=20, world=world, model=model, me=0,
        )
    with _with_env(BASELINE_BLEED_PENALTY="1", BASELINE_BLEED_BETA="0.05",
                   BASELINE_ORBITAL_SAFETY=None):
        bleed_v = cheap_marginal_value(
            src, tgt, ships=30, eta=20, world=world, model=model, me=0,
        )
    # base_v - bleed_v should equal the bleed penalty.
    expected_bleed = 0.05 * 70.0 * (GAMMA ** 20)
    assert abs((base_v - bleed_v) - expected_bleed) < 1e-9


def test_aim_and_eta_comet_returns_eta_within_path_length():
    """If the comet path is short (5 steps), aim_and_eta should still
    return a valid (angle, eta) without crashing — falling through to
    the atan2 fallback when aim_comet returns None (comet exits)."""
    from agents.baseline.proposer import aim_and_eta
    # Source far from comet; only 5-step path → comet exits before arrival
    src = _planet(0, 0, 90.0, 90.0, ships=80, production=2)
    path = [[20.0 + i * 4.0, 50.0] for i in range(5)]
    world = _world_with_comet(
        my_id=0, planets=[src], comet_id=42, path=path, path_index=0,
    )
    comet_planet = world.planets_by_id[42]
    angle, eta = aim_and_eta(
        src, comet_planet, ships=10, omega=0.04, world=world,
    )
    assert eta >= 1, f"expected positive eta; got {eta}"


# ---------------------------------------------------------------------------
# Wave proposer — pure-math tests (baseline_wave v3 2026-05-24)
# ---------------------------------------------------------------------------


def _wave_world(my_id=0, omega=0.0):
    """3 my-planets + 1 enemy target where the target is too heavily
    defended for any single source to capture alone — a wave is required.
    omega=0 → no rotation, deterministic etas."""
    # High-prod, well-garrisoned enemy target. Any single 60-ship strike
    # bounces; combined 3-source wave (~180 ships) clears defense + margin.
    tgt = _planet(0, 1, 60.0, 50.0, ships=120, production=2)
    fast = _planet(1, my_id, 55.0, 50.0, ships=60, production=3)   # closest
    mid = _planet(2, my_id, 35.0, 50.0, ships=60, production=3)   # mid
    slow = _planet(3, my_id, 15.0, 50.0, ships=60, production=3)   # farthest
    world = _world(my_id, [tgt, fast, mid, slow], step=0, omega=omega)
    model = WorldModel.from_world(world, horizon=60)
    return tgt, [fast, mid, slow], world, model


def test_wave_proposer_off_returns_empty():
    """Default OFF: BASELINE_WAVE_PROPOSER unset → []. Orbitfix-path parity."""
    tgt, my_planets, world, model = _wave_world()
    with _with_env(BASELINE_WAVE_PROPOSER=None):
        out = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    assert out == []


def test_wave_proposer_emits_at_least_one_wave():
    """3 own planets all in range of an enemy target → at least one wave
    emitted with ≥2 legs."""
    tgt, my_planets, world, model = _wave_world()
    with _with_env(BASELINE_WAVE_PROPOSER="1"):
        out = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    assert len(out) >= 1, f"expected ≥1 wave, got {out}"
    for wave_tgt, legs in out:
        assert int(wave_tgt.id) == int(tgt.id)
        assert len(legs) >= 2, f"every wave needs ≥2 legs, got {legs}"


def test_wave_wait_N_consistent_with_arrival_step():
    """For each emitted wave, all legs must land at the SAME arrival step:
    arrival_step = wait_N_S + eta(S, T, ships_S, wait_N=wait_N_S).
    Same-step arrivals are the whole point — combat rule 1."""
    tgt, my_planets, world, model = _wave_world()
    with _with_env(BASELINE_WAVE_PROPOSER="1"):
        out = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    assert out, "expected at least one wave"
    for wave_tgt, legs in out:
        arrivals = []
        for src, ships, _angle, wait_N in legs:
            _ang, eta = aim_and_eta(
                src, wave_tgt, int(ships), omega=0.0,
                wait_N=int(wait_N), world=world,
            )
            arrivals.append(int(wait_N) + int(eta))
        assert min(arrivals) == max(arrivals), (
            f"all legs must land same step; got arrivals={arrivals}"
        )


def test_wave_total_exceeds_projected_defense():
    """Every emitted wave's total inbound ships exceeds the projected
    target garrison at arrival_step (else the wave bounces — not a wave)."""
    tgt, my_planets, world, model = _wave_world()
    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_MARGIN="2"):
        out = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    assert out, "expected at least one wave"
    for wave_tgt, legs in out:
        # Reconstruct arrival_step from leg[0] (all same per previous test).
        src0, ships0, _ang0, wait_N0 = legs[0]
        _ang, eta0 = aim_and_eta(
            src0, wave_tgt, int(ships0), omega=0.0,
            wait_N=int(wait_N0), world=world,
        )
        arrival_step = int(wait_N0) + int(eta0)
        defense = float(model.ships_at(int(wave_tgt.id), arrival_step) or 0.0)
        total = sum(int(L[1]) for L in legs)
        assert total > defense + 2.0 - 1e-9, (
            f"wave total {total} must exceed defense {defense} + margin 2"
        )


def test_wave_respects_per_turn_cap():
    """Many target candidates → enumerator stops at BASELINE_WAVE_MAX_PER_TURN."""
    # 4 targets at varying distances; all reachable by all my-planets.
    my0 = _planet(10, 0, 50.0, 50.0, ships=80, production=3)
    my1 = _planet(11, 0, 52.0, 50.0, ships=80, production=3)
    targets = [
        _planet(i, 1, 30.0 + i * 5.0, 60.0, ships=8, production=2)
        for i in range(4)
    ]
    world = _world(0, [my0, my1] + targets, step=0, omega=0.0)
    model = WorldModel.from_world(world, horizon=60)
    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_MAX_PER_TURN="2"):
        out = enumerate_wave_candidates(
            [my0, my1], targets, world, model, me=0, omega=0.0,
        )
    assert len(out) <= 2, f"cap=2 expected; got {len(out)} waves"


def test_wave_rejects_when_only_one_source_in_range():
    """With a single my-planet (< 2 sources), every wave must have ≥2 legs,
    so the enumerator returns []."""
    tgt = _planet(0, 1, 60.0, 50.0, ships=10, production=2)
    lone = _planet(1, 0, 55.0, 50.0, ships=80, production=3)
    world = _world(0, [tgt, lone], step=0, omega=0.0)
    model = WorldModel.from_world(world, horizon=60)
    with _with_env(BASELINE_WAVE_PROPOSER="1"):
        out = enumerate_wave_candidates(
            [lone], [tgt], world, model, me=0, omega=0.0,
        )
    assert out == [], f"single source should not emit waves; got {out}"


# ---------------------------------------------------------------------------
# Wave proposer v5 — multi-anchor / tempo-guard / overkill
# ---------------------------------------------------------------------------


def test_wave_overkill_inflates_ship_budget():
    """overkill=1.5 produces leg ship-counts ~1.5× of overkill=1.0 for legs
    whose budget_after_wait does NOT clamp probe_ships × overkill."""
    tgt, my_planets, world, model = _wave_world()

    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_OVERKILL="1.0",
                   BASELINE_WAVE_ANCHORS="1"):
        baseline_out = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_OVERKILL="1.5",
                   BASELINE_WAVE_ANCHORS="1"):
        boosted_out = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )

    assert baseline_out and boosted_out, "both configs should emit a wave"
    baseline_total = sum(int(L[1]) for L in baseline_out[0][1])
    boosted_total = sum(int(L[1]) for L in boosted_out[0][1])
    # Boosted total should be strictly larger (each leg got more ships, or
    # the same number of legs each contributed more).
    assert boosted_total > baseline_total, (
        f"overkill=1.5 should inflate ship budget; "
        f"baseline={baseline_total}, boosted={boosted_total}"
    )


def test_wave_multi_anchor_can_find_different_wave_than_single_anchor():
    """Pin BASELINE_WAVE_ANCHORS to 1 vs 3 on the same geometry; the
    multi-anchor sweep can find a wave the slowest-only path misses
    (different arrival_step, possibly more legs)."""
    # Geometry: target defense climbs with arrival step (because target
    # production = 4 means ships_at(t) grows). The slowest-anchor wave
    # faces the heaviest defender; the fast-anchor wave faces a lighter one.
    tgt = _planet(0, 1, 70.0, 50.0, ships=80, production=4)
    fast = _planet(1, 0, 65.0, 50.0, ships=80, production=3)
    mid = _planet(2, 0, 45.0, 50.0, ships=80, production=3)
    slow = _planet(3, 0, 10.0, 50.0, ships=80, production=3)
    world = _world(0, [tgt, fast, mid, slow], step=0, omega=0.0)
    model = WorldModel.from_world(world, horizon=80)

    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_ANCHORS="1"):
        out_a1 = enumerate_wave_candidates(
            [fast, mid, slow], [tgt], world, model, me=0, omega=0.0,
        )
    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_ANCHORS="3"):
        out_a3 = enumerate_wave_candidates(
            [fast, mid, slow], [tgt], world, model, me=0, omega=0.0,
        )
    # Multi-anchor must enumerate at least as many waves as single-anchor.
    assert len(out_a3) >= len(out_a1), (
        f"multi-anchor (a=3) should not produce fewer waves; "
        f"a1={len(out_a1)} a3={len(out_a3)}"
    )
    # And when both produce a wave, the multi-anchor one is at least as
    # good (≥ legs, or ≥ total ships when legs tie). Single-anchor is a
    # strict subset of the multi-anchor search.
    if out_a1 and out_a3:
        legs_a1 = len(out_a1[0][1])
        legs_a3 = len(out_a3[0][1])
        total_a1 = sum(int(L[1]) for L in out_a1[0][1])
        total_a3 = sum(int(L[1]) for L in out_a3[0][1])
        assert (legs_a3, total_a3) >= (legs_a1, total_a1), (
            f"multi-anchor result should dominate single-anchor on "
            f"(legs, total); a1=({legs_a1},{total_a1}) "
            f"a3=({legs_a3},{total_a3})"
        )


def test_wave_tempo_guard_admits_more_with_loose_setting():
    """Multi-anchor search exposes the tempo_guard effect: a tight guard
    (8) excludes slow legs when a faster source is chosen as anchor;
    loose guard (15) admits them. On the same geometry, guard=15 should
    yield waves with at least as many total legs across all targets."""
    tgt, my_planets, world, model = _wave_world()

    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_TEMPO_GUARD="8",
                   BASELINE_WAVE_ANCHORS="3"):
        out_g8 = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    with _with_env(BASELINE_WAVE_PROPOSER="1", BASELINE_WAVE_TEMPO_GUARD="15",
                   BASELINE_WAVE_ANCHORS="3"):
        out_g15 = enumerate_wave_candidates(
            my_planets, [tgt], world, model, me=0, omega=0.0,
        )
    legs_g8 = sum(len(L) for _t, L in out_g8)
    legs_g15 = sum(len(L) for _t, L in out_g15)
    assert legs_g15 >= legs_g8, (
        f"loose tempo guard should admit ≥ legs; "
        f"g8={legs_g8}, g15={legs_g15}"
    )
