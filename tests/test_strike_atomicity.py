"""Atomicity tests for `agents.buildup_planner.strike.step`.

The strike phase mirrors the joint-candidate atomic-drop precedent in
`agents/baseline/chooser_trajectory.py:612-628`: ANY single shot's
failure drops the whole wave. Two failure modes:

  1. **Budget overflow** — per-source cumulative `ship_count` exceeds
     the source's garrison. (Step-2 predicate over-counts across
     multi-target plans; strike.py closes the gap.)
  2. **Physics fail** — `lib.trajectory.predict_fleet_fate` returns
     `outcome != "target"`.

All four cases exercise the real `World` + `predict_fleet_fate`; no
mocking — atomicity is a physics contract.
"""
from __future__ import annotations

import logging
import math

from lib.intent import World
from lib.trajectory import predict_fleet_fate

from agents.buildup_planner import strike
from agents.buildup_planner.predicates import StrikePlan
from agents.precision.intercept import Shot


# Planet tuple: (id, owner, x, y, radius, ships, production).
# Use static (non-orbiting) layouts — orbital drift is irrelevant for
# the atomicity contract; we just need predict_fleet_fate to return
# deterministic outcomes.
def _world(planets, fleets=None) -> World:
    obs = {
        "player": 0,
        "step": 0,
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,   # static (no orbit) — deterministic.
    }
    return World.from_obs(obs)


def _aim(src, tgt) -> float:
    """Direct ray from src centre to tgt centre. Sufficient for static
    planets when src.radius + tgt.radius < distance."""
    return math.atan2(tgt.y - src.y, tgt.x - src.x)


def _shot(src_id: int, tgt_id: int, angle: float, ships: int,
          eta: int = 5) -> Shot:
    return Shot(
        src_id=src_id, tgt_id=tgt_id, eta=eta, ship_count=ships,
        angle=angle, arrival_xy=(0.0, 0.0), arrival_ships=ships,
    )


def test_all_pass_emits_all_shots():
    """3 shots, all physics-valid, all within source budgets → 3 moves.

    Layout: 3 sources along y=15, 3 targets along y=85. Each pair's
    direct-aim closest-approach to sun (50, 50, radius 10) is ≥ 17.5.
    """
    planets = [
        (0, 0,  15.0, 15.0, 3.0, 100, 1),   # me, source 0 (bottom-left)
        (1, 0,  50.0, 15.0, 3.0, 100, 1),   # me, source 1 (bottom-mid)
        (2, 0,  85.0, 15.0, 3.0, 100, 1),   # me, source 2 (bottom-right)
        (3, 1,  15.0, 85.0, 3.0,  10, 1),   # opp target (top-left)
        (4, 1,  50.0, 85.0, 3.0,  10, 1),   # opp target (top-mid)
        (5, 1,  85.0, 85.0, 3.0,  10, 1),   # opp target (top-right)
    ]
    w = _world(planets)
    by_id = w.planets_by_id

    # Vertical pairs on the left/right edges (clear of sun), plus a
    # right-leaning diagonal (mid → top-right). Each path's closest
    # approach to (50, 50) is ≥ 17.5. Avoid the main 0→5 diagonal which
    # goes directly through the sun.
    pairs = ((0, 3), (1, 5), (2, 4))
    shots = []
    for sid, tid in pairs:
        src, tgt = by_id[sid], by_id[tid]
        angle = _aim(src, tgt)
        fate = predict_fleet_fate(src, tgt, angle, 20, w)
        assert fate.outcome == "target", \
            f"sanity fail src={sid} tgt={tid} outcome={fate.outcome}"
        shots.append(_shot(sid, tid, angle, 20))

    plan = StrikePlan(
        target_ids=frozenset({3, 4, 5}),
        arrival_step=10,
        shots=tuple(shots),
    )
    moves = strike.step(w, plan)
    assert len(moves) == 3
    for shot, move in zip(shots, moves):
        assert move[0] == shot.src_id
        assert move[1] == shot.angle
        assert move[2] == shot.ship_count


def test_one_physics_fail_drops_whole_wave(caplog):
    """3-shot plan; middle shot aimed straight through the sun → all dropped."""
    # Same layout as the all-pass test, but pair source 1 (50, 15) with
    # target 4 (50, 85) — a vertical line through the sun (50, 50).
    planets = [
        (0, 0,  15.0, 15.0, 3.0, 100, 1),
        (1, 0,  50.0, 15.0, 3.0, 100, 1),   # aims straight up through sun
        (2, 0,  85.0, 15.0, 3.0, 100, 1),
        (3, 1,  15.0, 85.0, 3.0,  10, 1),
        (4, 1,  50.0, 85.0, 3.0,  10, 1),
        (5, 1,  85.0, 85.0, 3.0,  10, 1),
    ]
    w = _world(planets)
    by_id = w.planets_by_id

    bad_angle = _aim(by_id[1], by_id[4])    # vertical, x=50
    bad_fate = predict_fleet_fate(by_id[1], by_id[4], bad_angle, 20, w)
    assert bad_fate.outcome != "target", \
        f"sanity: expected sun, got {bad_fate.outcome}"

    shots = (
        _shot(0, 3, _aim(by_id[0], by_id[3]), 20),
        _shot(1, 4, bad_angle, 20),    # physics-fail
        _shot(2, 5, _aim(by_id[2], by_id[5]), 20),
    )
    plan = StrikePlan(target_ids=frozenset({3, 4, 5}),
                      arrival_step=10, shots=shots)
    with caplog.at_level(logging.WARNING, logger="buildup_planner.strike"):
        moves = strike.step(w, plan)
    assert moves == []
    assert any("atomic-drop" in r.message and "physics_fail" in r.message
               for r in caplog.records)


def test_budget_overflow_drops_whole_wave(caplog):
    """Single shot exceeding source garrison → atomic drop."""
    planets = [
        (0, 0,  20.0, 50.0, 3.0,  5, 1),    # me — only 5 ships
        (1, 1,  80.0, 50.0, 3.0, 10, 1),    # opp target
    ]
    w = _world(planets)
    src, tgt = w.planets_by_id[0], w.planets_by_id[1]
    plan = StrikePlan(
        target_ids=frozenset({1}),
        arrival_step=10,
        shots=(_shot(0, 1, _aim(src, tgt), 6),),   # 6 ships > 5 garrison
    )
    with caplog.at_level(logging.WARNING, logger="buildup_planner.strike"):
        moves = strike.step(w, plan)
    assert moves == []
    assert any("budget_overflow" in r.message for r in caplog.records)


def test_duplicated_src_accumulates_budget(caplog):
    """Two shots from the same source whose SUM exceeds garrison → drop.

    Closes the Step-2 over-counting gap: the predicate may emit a plan
    where source S is counted toward both T1 and T2 in a |S|=2 wave;
    individually each shot is within budget, but together they exceed.
    """
    planets = [
        (0, 0,  20.0, 50.0, 3.0, 10, 1),    # me — 10 ships total
        (1, 1,  50.0, 80.0, 3.0,  5, 1),    # opp target T1
        (2, 1,  80.0, 50.0, 3.0,  5, 1),    # opp target T2
    ]
    w = _world(planets)
    src = w.planets_by_id[0]
    plan = StrikePlan(
        target_ids=frozenset({1, 2}),
        arrival_step=10,
        # 6 + 6 = 12 > 10 garrison; each individual shot is within budget.
        shots=(
            _shot(0, 1, _aim(src, w.planets_by_id[1]), 6),
            _shot(0, 2, _aim(src, w.planets_by_id[2]), 6),
        ),
    )
    with caplog.at_level(logging.WARNING, logger="buildup_planner.strike"):
        moves = strike.step(w, plan)
    assert moves == []
    # Must surface the duplicated-src case as budget_overflow on src 0.
    assert any("budget_overflow" in r.message and "src=0" in r.message
               for r in caplog.records)


def test_strike_log_emits_jsonl_line_per_call(tmp_path, monkeypatch):
    """Step 3b instrumentation: `BUILDUP_PLANNER_STRIKE_LOG=<path>` opt-in
    writes one JSONL line per `strike.step` call, with `outcome` matching
    the actual emit / atomic-drop reason. Unset env var = no log."""
    import json as _json

    log_path = tmp_path / "strike.jsonl"
    monkeypatch.setenv("BUILDUP_PLANNER_STRIKE_LOG", str(log_path))

    # (a) All-pass emit — should log outcome="emit".
    planets = [
        (0, 0,  15.0, 15.0, 3.0, 100, 1),
        (1, 1,  15.0, 85.0, 3.0,  10, 1),
    ]
    w = _world(planets)
    src, tgt = w.planets_by_id[0], w.planets_by_id[1]
    plan = StrikePlan(
        target_ids=frozenset({1}), arrival_step=10,
        shots=(_shot(0, 1, _aim(src, tgt), 20),),
    )
    moves = strike.step(w, plan, game_id="g1", step_now=42)
    assert len(moves) == 1

    # (b) Budget overflow — should log outcome="budget_overflow".
    plan_overflow = StrikePlan(
        target_ids=frozenset({1}), arrival_step=10,
        shots=(_shot(0, 1, _aim(src, tgt), 200),),  # 200 > 100 garrison
    )
    moves = strike.step(w, plan_overflow, game_id="g1", step_now=43)
    assert moves == []

    # (c) Empty plan — should log outcome="empty".
    plan_empty = StrikePlan(
        target_ids=frozenset(), arrival_step=10, shots=(),
    )
    moves = strike.step(w, plan_empty, game_id="g1", step_now=44)
    assert moves == []

    lines = log_path.read_text().splitlines()
    assert len(lines) == 3
    entries = [_json.loads(l) for l in lines]
    assert entries[0]["outcome"] == "emit"
    assert entries[0]["num_emitted"] == 1
    assert entries[0]["step"] == 42
    assert entries[0]["game_id"] == "g1"
    assert entries[1]["outcome"] == "budget_overflow"
    assert entries[1]["num_emitted"] == 0
    assert entries[2]["outcome"] == "empty"


def test_strike_log_disabled_when_env_unset(tmp_path, monkeypatch):
    """Default state (env var unset) writes nothing. Production behaviour
    must remain identical to pre-instrumentation."""
    monkeypatch.delenv("BUILDUP_PLANNER_STRIKE_LOG", raising=False)
    planets = [
        (0, 0,  15.0, 15.0, 3.0, 100, 1),
        (1, 1,  15.0, 85.0, 3.0,  10, 1),
    ]
    w = _world(planets)
    src, tgt = w.planets_by_id[0], w.planets_by_id[1]
    plan = StrikePlan(
        target_ids=frozenset({1}), arrival_step=10,
        shots=(_shot(0, 1, _aim(src, tgt), 20),),
    )
    moves = strike.step(w, plan)   # no extra args either — backward compat
    assert len(moves) == 1
    # No log file should have been created.
    assert not (tmp_path / "strike.jsonl").exists()
