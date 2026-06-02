"""Oracle tests for the sequential-greedy chooser (Rule 49).

Mechanism under test (agents/baseline/chooser_greedy.choose_greedy):
  - Rung 1: marginal gain is CONDITIONAL on the set already chosen, so a
    redundant launch (a second fleet at a target the first already takes)
    is declined even when the per-target lock is lifted.
  - Rung 3: coalition atoms (sync pairs) are pickable as single candidates,
    so a teamwork capture neither solo can make is assembled.
  - Determinism: the deterministic fast_sim oracle ⇒ identical output across
    repeated calls and across lazy/exact modes on non-superadditive boards.
  - Anytime: a tight deadline still returns a valid, non-crashing move set.

World construction mirrors tests/test_joint_sync.py.
"""

from __future__ import annotations

import agents.baseline.chooser_greedy as cg
from agents.baseline.chooser_greedy import choose_greedy
from agents.baseline.chooser_trajectory import score_candidate_v4
from agents.baseline.proposer import aim_and_eta
from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel

GAMMA = 0.99
HORIZON = 40


def _world(planets, *, num_seats=2):
    obs = {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    snap = fs_from_obs(obs, num_seats=num_seats)
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    by_id = {int(p.id): p for p in world.planets_by_id.values()}
    return snap, world, model, by_id


def _stack_scenario():
    """A+B each 25 ships; C is a 30-defender opponent planet neither can take
    alone but 50 stacked can. Mirrors test_joint_sync._scenario."""
    planets = [
        [0, 1, 85.0, 15.0, 1.5, 30, 1],   # C: target (opponent)
        [1, 0, 72.0, 25.0, 1.5, 25, 1],   # A: my near source
        [2, 0, 50.0, 28.0, 1.5, 25, 1],   # B: my far source
        [3, 1, 15.0, 80.0, 1.5, 30, 1],   # opp home
    ]
    snap, world, model, by_id = _world(planets)
    return snap, world, model, by_id[1], by_id[2], by_id[0]


def _prerank(world, *pairs):
    """Build solo prerank rows: each pair is (src, tgt, ships)."""
    rows = []
    for src, tgt, ships in pairs:
        angle, eta = aim_and_eta(src, tgt, int(ships), world.omega, world=world)
        rows.append((-1.0, src, tgt, int(ships), float(angle), int(eta),
                     HORIZON, 0))
    return rows


# ---------------------------------------------------------------------------
# (a) Coalition atom unlocks a teamwork capture; solo-only does not.
#     Scorer stubbed so ONLY a waiting-leg coalition scores positive — this
#     isolates the greedy's selection + coalition integration + emit.
# ---------------------------------------------------------------------------

def _stub_only_coalitions_score(monkeypatch):
    def _fake(snap_base, launches, *a, **k):
        if any(int(L[4]) > 0 for L in launches):   # has a waiting leg → coalition
            return (5.0, "scored")
        return (-1.0, "scored")
    monkeypatch.setattr(cg, "score_candidate_v4_joint", _fake)


def test_coalition_atom_makes_teamwork_capture(monkeypatch):
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    monkeypatch.setenv("BASELINE_GREEDY_COALITIONS", "1")
    monkeypatch.setenv("BASELINE_GREEDY_SHALLOW_H", "30")
    _stub_only_coalitions_score(monkeypatch)

    snap, world, model, A, B, C = _stack_scenario()
    prerank = _prerank(world, (A, C, 25), (B, C, 25))
    moves, commits = choose_greedy(
        snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model)

    sync = [c for c in commits if c.get("sync_joint")]
    assert len(sync) == 1, (moves, commits)
    assert sync[0]["tgt_id"] == 0          # target C
    assert sync[0]["src_id"] == 1          # near source (A) waits
    assert sync[0]["wait_remaining"] > 0
    assert any(int(m[0]) == 2 for m in moves)  # far source (B) fires now


def test_no_coalition_no_capture(monkeypatch):
    """Control: coalitions OFF ⇒ only solo singletons in the pool, all of
    which bounce (stub scores them −1) ⇒ nothing fired at C."""
    monkeypatch.setenv("BASELINE_GREEDY_COALITIONS", "0")
    monkeypatch.setenv("BASELINE_GREEDY_SHALLOW_H", "30")
    _stub_only_coalitions_score(monkeypatch)

    snap, world, model, A, B, C = _stack_scenario()
    prerank = _prerank(world, (A, C, 25), (B, C, 25))
    moves, commits = choose_greedy(
        snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model)

    assert not any(c.get("sync_joint") for c in commits)
    assert moves == [] and commits == []


# ---------------------------------------------------------------------------
# (b) Redundant double-spend declined by CONDITIONAL scoring, even with the
#     per-target lock lifted (BASELINE_JOINT_AGGR=1). Real favor rollout.
# ---------------------------------------------------------------------------

def _redundant_scenario():
    """Two of my sources (S1,S2) each able to solo-capture a weak neutral T;
    plus an opponent so favor has an opponent."""
    planets = [
        [0, -1, 85.0, 15.0, 1.5, 5, 1],   # T: weak neutral target
        [1, 0, 75.0, 15.0, 1.5, 20, 1],   # S1: close, captures T alone
        [2, 0, 78.0, 20.0, 1.5, 20, 1],   # S2: close, captures T alone
        [3, 1, 15.0, 80.0, 1.5, 20, 1],   # opp home
    ]
    snap, world, model, by_id = _world(planets)
    return snap, world, model, by_id[1], by_id[2], by_id[0]


def test_conditional_scoring_declines_redundant_launch(monkeypatch):
    monkeypatch.setenv("BASELINE_JOINT_AGGR", "1")   # LIFT the per-target lock
    monkeypatch.setenv("BASELINE_GREEDY_COALITIONS", "0")
    monkeypatch.setenv("BASELINE_GREEDY_SHALLOW_H", "20")
    monkeypatch.setenv("BASELINE_GREEDY_LAZY", "0")  # exact greedy for clarity

    snap, world, model, S1, S2, T = _redundant_scenario()
    prerank = _prerank(world, (S1, T, 20), (S2, T, 20))
    moves, _commits = choose_greedy(
        snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model)

    # Both solos score > 0 INDEPENDENTLY (the trajectory chooser would emit
    # both with the lock lifted) — but conditional greedy fires exactly one.
    favor_fn = cg.select_favor_fn()
    from agents.baseline.chooser_trajectory import build_trajectory_baseline
    favs = build_trajectory_baseline(snap, 0, 2, 20, favor_fn, GAMMA)
    a1, e1 = aim_and_eta(S1, T, 20, world.omega, world=world)
    d1, _st1, _ = score_candidate_v4(
        snap, S1, T, 20, float(a1), 0, 2, world, favs, favor_fn, GAMMA, 20)
    a2, e2 = aim_and_eta(S2, T, 20, world.omega, world=world)
    d2, _st2, _ = score_candidate_v4(
        snap, S2, T, 20, float(a2), 0, 2, world, favs, favor_fn, GAMMA, 20)
    assert d1 > 0 and d2 > 0, (d1, d2)  # independently both look good

    fired_at_T = [m for m in moves if _aim_targets(m, world, T)]
    assert len(fired_at_T) == 1, (moves, d1, d2)  # greedy declines the second


def _aim_targets(move, world, tgt):
    """True if `move` is a launch from a source whose straight-line aim is at
    `tgt` (both redundant sources aim at T, so source-id membership suffices)."""
    return int(move[0]) in (1, 2)


# ---------------------------------------------------------------------------
# (c) Determinism + lazy/exact equivalence on a non-superadditive board.
# ---------------------------------------------------------------------------

def test_determinism_repeated_calls(monkeypatch):
    monkeypatch.setenv("BASELINE_GREEDY_COALITIONS", "1")
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    monkeypatch.setenv("BASELINE_GREEDY_SHALLOW_H", "20")
    snap, world, model, A, B, C = _stack_scenario()
    prerank = _prerank(world, (A, C, 25), (B, C, 25))
    out1 = choose_greedy(snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA,
                         world, model)
    snap2, world2, model2, A2, B2, C2 = _stack_scenario()
    prerank2 = _prerank(world2, (A2, C2, 25), (B2, C2, 25))
    out2 = choose_greedy(snap2, prerank2, None, 0, 2, 2000.0, 5, HORIZON, GAMMA,
                         world2, model2)
    assert out1 == out2


def test_lazy_equals_exact_non_superadditive(monkeypatch):
    monkeypatch.setenv("BASELINE_GREEDY_COALITIONS", "0")
    monkeypatch.setenv("BASELINE_JOINT_AGGR", "1")
    monkeypatch.setenv("BASELINE_GREEDY_SHALLOW_H", "20")
    # Two independent capturable neutrals from two sources — purely additive.
    planets = [
        [0, -1, 85.0, 15.0, 1.5, 5, 1],
        [1, -1, 15.0, 85.0, 1.5, 5, 1],
        [2, 0, 75.0, 15.0, 1.5, 20, 1],
        [3, 0, 25.0, 85.0, 1.5, 20, 1],
        [4, 1, 50.0, 50.0 + 30, 1.5, 10, 1],
    ]
    snap, world, model, by_id = _world(planets)
    prerank = _prerank(world, (by_id[2], by_id[0], 20), (by_id[3], by_id[1], 20))

    monkeypatch.setenv("BASELINE_GREEDY_LAZY", "1")
    lazy = choose_greedy(snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA,
                         world, model)
    snap2, world2, model2, by_id2 = _world(planets)
    prerank2 = _prerank(world2, (by_id2[2], by_id2[0], 20),
                        (by_id2[3], by_id2[1], 20))
    monkeypatch.setenv("BASELINE_GREEDY_LAZY", "0")
    exact = choose_greedy(snap2, prerank2, None, 0, 2, 2000.0, 5, HORIZON,
                          GAMMA, world2, model2)
    assert sorted(lazy[0]) == sorted(exact[0])


# ---------------------------------------------------------------------------
# (d) Anytime: a tight deadline returns a valid, well-formed, non-crashing set.
# ---------------------------------------------------------------------------

def test_anytime_tight_deadline(monkeypatch):
    import time
    monkeypatch.setenv("BASELINE_GREEDY_COALITIONS", "1")
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    snap, world, model, A, B, C = _stack_scenario()
    prerank = _prerank(world, (A, C, 25), (B, C, 25))
    moves, commits = choose_greedy(
        snap, prerank, None, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model,
        agent_deadline=time.perf_counter() + 0.005)
    assert isinstance(moves, list) and isinstance(commits, list)
    for m in moves:
        assert len(m) == 3 and int(m[2]) >= 1   # [src_id, angle, ships]
