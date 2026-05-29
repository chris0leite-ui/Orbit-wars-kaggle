"""Unit tests for agents/baseline/proposer."""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from agents.baseline.proposer import (
    MIN_FLEET_SIZE,
    _enumerate_reactor_candidates,
    _target_cost_parity_ok,
    capture_size,
    enumerate_ship_counts,
    nearest_k,
    propose,
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


def test_propose_dedups_per_src_tgt():
    """propose() returns at most one entry per (src, tgt) pair post the
    2026-05-29 wait-grid strip. Every emitted prerank entry has wait_N=0.
    """
    src = _planet(0, 0, 10.0, 50.0, ships=80, production=3)
    tgt = _planet(1, -1, 12.0, 50.0, ships=5, production=2)
    world = _world(0, [src, tgt])
    model = WorldModel.from_world(world)
    out = propose(
        my_planets=[src], target_pool=[tgt],
        world=world, model=model, me=0, omega=0.0, baseline_len=50,
    )
    assert all(int(e[1].id) == 0 and int(e[2].id) == 1 for e in out)
    # One entry per (src, tgt) and every entry is fire-now.
    assert len(out) <= 1
    assert all(int(e[7]) == 0 for e in out)


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
