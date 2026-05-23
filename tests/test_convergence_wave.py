"""Multi-source staggered-attrition convergence wave.

Fills the gap between drain_combat_stack (stacks onto already-inbound
attacks) and emit_sniper_strikes (bails when no single source can crack
a target). Bundles 2+ idle sources into a staggered-arrival attack on
the highest-ROI enemy planet that no friendly fleet is inbound to.

Origin: PI 2026-05-23 "streamline ships to the opponent really
aggressively … snowball through the fastest path." Plan:
/root/.claude/plans/do-5-enchanted-metcalfe.md.
"""

from __future__ import annotations

import math

import pytest

import agents.baseline.main as bm
from agents.baseline.main import (
    WAVE_MIN_PER_SOURCE_SHIPS,
    WAVE_MIN_TOTAL_SHIPS,
    _simulate_staggered_capture,
    emit_convergence_wave,
)
from lib.intent import World
from lib.world_model import WorldModel


def _planet(pid, owner, ships, prod=1, x=50.0, y=50.0, radius=1.0):
    return [pid, owner, x, y, radius, ships, prod]


def _fleet(fid, owner, x, y, angle, from_pid, ships):
    return [fid, owner, x, y, angle, from_pid, ships]


def _world(planets, *, my_id=0, step=20, fleets=(), omega=0.0):
    """Default omega=0 keeps geometry static so naked atan2 aims true.
    Both sniper and combat_stack use naked atan2 (no lead-aim) in
    production; omega=0 isolates the wave's selection logic from
    leading-aim dynamics."""
    obs = {
        "player": my_id,
        "planets": planets,
        "angular_velocity": omega,
        "comet_planet_ids": [],
        "step": step,
        "comets": [],
        "fleets": list(fleets),
    }
    return World.from_obs(obs)


def _model(world):
    return WorldModel.from_world(world)


def _enable_wave(monkeypatch):
    monkeypatch.setattr(bm, "CONVERGENCE_WAVE_ENABLED", True)


# ---------------------------------------------------------------------------
# Gate-off invariance
# ---------------------------------------------------------------------------


def test_gate_off_is_noop():
    """With BASELINE_CONVERGENCE_WAVE unset, emit_convergence_wave returns
    moves unchanged byte-for-byte. Module-level constant defaults False
    in this process (no env override)."""
    assert bm.CONVERGENCE_WAVE_ENABLED is False
    planets_raw = [
        _planet(0, owner=0, ships=200, prod=2, x=10.0, y=10.0),
        _planet(1, owner=0, ships=200, prod=2, x=12.0, y=15.0),
        _planet(2, owner=0, ships=200, prod=2, x=8.0, y=12.0),
        _planet(3, owner=1, ships=40, prod=5, x=80.0, y=80.0),
    ]
    w = _world(planets_raw)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    in_moves = [[0, 0.5, 30]]
    out = emit_convergence_wave(in_moves, planets, my_id=0, world=w, model=m)
    assert out == in_moves


# ---------------------------------------------------------------------------
# Staggered-capture simulator (pure)
# ---------------------------------------------------------------------------


class _FakeTgt:
    def __init__(self, ships, production, owner=1):
        self.ships = ships
        self.production = production
        self.owner = owner


def test_simulator_single_source_captures():
    tgt = _FakeTgt(ships=20, production=1)
    sources = [(5, "src0", 0.0, 50)]
    prefix_idx, total, ok = _simulate_staggered_capture(tgt, my_id=0, sources_eta=sources)
    assert ok is True
    assert prefix_idx == 1
    assert total == 50


def test_simulator_two_source_staggered_capture():
    """Garrison 80, prod 2. Source A (eta=5, 40 ships) — needs 80+5*2=90,
    not enough → garrison drops to 50. Source B (eta=10, 60 ships) —
    garrison regrows to 50+5*2=60, 60*1.05=63 ≥ 60 captures."""
    tgt = _FakeTgt(ships=80, production=2)
    sources = [(5, "A", 0.0, 40), (10, "B", 0.0, 60)]
    prefix_idx, total, ok = _simulate_staggered_capture(tgt, my_id=0, sources_eta=sources)
    assert ok is True
    assert prefix_idx == 2
    assert total == 100


def test_simulator_fails_when_garrison_too_big():
    tgt = _FakeTgt(ships=500, production=10)
    sources = [(5, "A", 0.0, 30), (10, "B", 0.0, 40)]
    _, _, ok = _simulate_staggered_capture(tgt, my_id=0, sources_eta=sources)
    assert ok is False


# ---------------------------------------------------------------------------
# emit_convergence_wave behavior
# ---------------------------------------------------------------------------


def test_fires_multi_source_at_high_value_target(monkeypatch):
    """Three idle sources with enough excess each + a high-prod weak
    enemy → wave emits 1-3 launches and captures."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=120, prod=1, x=10.0, y=80.0),
        _planet(1, owner=0, ships=120, prod=1, x=15.0, y=85.0),
        _planet(2, owner=0, ships=120, prod=1, x=12.0, y=75.0),
        _planet(3, owner=1, ships=30, prod=5, x=70.0, y=80.0),
    ]
    w = _world(planets_raw)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    out = emit_convergence_wave([], planets, my_id=0, world=w, model=m)
    assert len(out) >= 1
    # All launches must target the high-prod enemy planet via correct angle
    # (we don't get target id back, but angle should point at (70,70) from
    # the source). Just confirm the bundle's total ships ≥ WAVE_MIN_TOTAL.
    total = sum(int(m[2]) for m in out)
    assert total >= WAVE_MIN_TOTAL_SHIPS
    # Each launch passes the per-source minimum.
    for ent in out:
        assert int(ent[2]) >= WAVE_MIN_PER_SOURCE_SHIPS


def test_no_double_fire_on_used_sources(monkeypatch):
    """Source already in `moves` from the chooser must not appear in
    the wave's output."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=120, prod=1, x=10.0, y=80.0),
        _planet(1, owner=0, ships=120, prod=1, x=15.0, y=85.0),
        _planet(2, owner=0, ships=120, prod=1, x=12.0, y=75.0),
        _planet(3, owner=1, ships=30, prod=5, x=70.0, y=80.0),
    ]
    w = _world(planets_raw)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    chooser_moves = [[0, 0.5, 50]]  # source 0 already used
    out = emit_convergence_wave(chooser_moves, planets, my_id=0, world=w, model=m)
    # All NEW launches (i.e. out beyond the chooser_moves prefix) must
    # not use source 0.
    extras = out[len(chooser_moves):]
    for ent in extras:
        assert int(ent[0]) != 0


def test_skips_target_with_friendly_inbound(monkeypatch):
    """Combat-stack handles targets with friendly fleets already inbound.
    The wave must skip them — otherwise we double-bundle."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=120, prod=1, x=10.0, y=80.0),
        _planet(1, owner=0, ships=120, prod=1, x=15.0, y=85.0),
        _planet(2, owner=0, ships=120, prod=1, x=12.0, y=75.0),
        _planet(3, owner=1, ships=30, prod=5, x=70.0, y=80.0),
    ]
    # Friendly fleet near planet 3, aimed at it — registers as inbound
    # in the WorldModel ledger so the wave should skip target 3.
    fleets = [_fleet(0, owner=0, x=60.0, y=80.0, angle=0.0,
                     from_pid=0, ships=20)]
    w = _world(planets_raw, fleets=fleets)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    out = emit_convergence_wave([], planets, my_id=0, world=w, model=m)
    # No fresh-attack target available → wave declines to fire.
    assert out == []


def test_skips_tiny_target_below_min_prod(monkeypatch):
    """Production-1 enemy planet is below WAVE_MIN_TGT_PROD — no fire."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=120, prod=1, x=10.0, y=10.0),
        _planet(1, owner=0, ships=120, prod=1, x=15.0, y=12.0),
        _planet(2, owner=0, ships=120, prod=1, x=12.0, y=15.0),
        _planet(3, owner=1, ships=30, prod=1, x=70.0, y=70.0),  # prod=1 < min=2
    ]
    w = _world(planets_raw)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    out = emit_convergence_wave([], planets, my_id=0, world=w, model=m)
    assert out == []


def test_skips_when_no_idle_sources(monkeypatch):
    """All sources thin (below STAGNANT threshold) → no wave."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=5, prod=1, x=10.0, y=10.0),
        _planet(1, owner=0, ships=5, prod=1, x=15.0, y=12.0),
        _planet(2, owner=0, ships=5, prod=1, x=12.0, y=15.0),
        _planet(3, owner=1, ships=30, prod=5, x=70.0, y=70.0),
    ]
    w = _world(planets_raw)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    out = emit_convergence_wave([], planets, my_id=0, world=w, model=m)
    assert out == []


def test_skips_when_enemy_inbound_to_source(monkeypatch):
    """Source with an inbound enemy fleet is excluded (need defense).
    With only one safe-and-fat source, total may fall below bundle min
    → wave declines OR fires the single fat source as a 1-launch wave
    if it captures alone. Either way: source 0 (under attack) must NOT
    appear in the output."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=200, prod=1, x=10.0, y=80.0),
        _planet(1, owner=0, ships=200, prod=1, x=15.0, y=85.0),
        _planet(2, owner=0, ships=200, prod=1, x=12.0, y=75.0),
        _planet(3, owner=1, ships=30, prod=5, x=70.0, y=80.0),
    ]
    # Enemy fleet near source 0 aimed at it — registers as inbound.
    fleets = [_fleet(0, owner=1, x=12.0, y=80.0, angle=math.pi, from_pid=3, ships=30)]
    w = _world(planets_raw, fleets=fleets)
    m = _model(w)
    # Sanity: WorldModel agrees source 0 has inbound enemy.
    assert m.incoming_enemy_eta(0, my_id=0) is not None
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    out = emit_convergence_wave([], planets, my_id=0, world=w, model=m)
    for ent in out:
        assert int(ent[0]) != 0


def test_appends_to_existing_moves(monkeypatch):
    """Output preserves the input moves and only appends new launches."""
    _enable_wave(monkeypatch)
    planets_raw = [
        _planet(0, owner=0, ships=120, prod=1, x=10.0, y=80.0),
        _planet(1, owner=0, ships=120, prod=1, x=15.0, y=85.0),
        _planet(2, owner=0, ships=120, prod=1, x=12.0, y=75.0),
        _planet(3, owner=1, ships=30, prod=5, x=70.0, y=80.0),
    ]
    w = _world(planets_raw)
    m = _model(w)
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
    planets = [Planet(*p) for p in planets_raw]
    in_moves = [[9, 1.23, 7]]  # nonsense src=9 (doesn't matter for preservation)
    out = emit_convergence_wave(in_moves, planets, my_id=0, world=w, model=m)
    assert out[: len(in_moves)] == in_moves


# ---------------------------------------------------------------------------
# Smoke: gate-on agent runs a full game without crash
# ---------------------------------------------------------------------------


def test_agent_runs_full_game_with_wave_on(monkeypatch):
    """End-to-end: gate-on baseline runs a complete game vs random
    without crashing. Real proof the wave doesn't break the pipeline."""
    monkeypatch.setenv("BASELINE_CONVERGENCE_WAVE", "1")
    monkeypatch.setattr(bm, "CONVERGENCE_WAVE_ENABLED", True)
    from kaggle_environments import make
    from agents.baseline.main import agent
    env = make("orbit_wars", configuration={"seed": 13}, debug=False)
    env.run([agent, "random"])
    final = env.steps[-1]
    assert final[0].reward is not None
    assert final[1].reward is not None
