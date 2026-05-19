"""Phase A — Test 5: capture-math unit tests for trajectory_roi primitives.

Three constructed scenarios verify `_solve_single_source` + `_net_defenders`
+ `_aim_and_eta` against env-step ground truth:

- 5a (free capture): big source vs small neutral → solver returns Candidate,
  env-step flips ownership.
- 5b (bounce): small source vs big enemy → solver returns None (insufficient).
- 5c (wait-and-fire): small source can't capture now, but after enough
  production accrual the same solver returns a Candidate, and the env-step
  confirms the post-wait launch flips ownership.

Each scenario seeds a distant idle enemy planet so the env's terminal
predicate (`alive_players <= 1`) doesn't fire mid-test.

The verification step (env-step) is the load-bearing part: the solver
might say YES but the env disagrees, or vice versa. Closed-form claims
without env confirmation are exactly what Phase A was created to test.

Audit context: `audit/2026-05-19-analytics-verification.md`.
"""

from __future__ import annotations

import math

from lib import fast_sim
from lib.trajectory_layer import World
from tests.scenarios.base import _obs, _planet

from agents.trajectory_roi.main import (
    _solve_single_source,
    _build_centrality_cache,
)


# ---- shared fixture helpers ----------------------------------------------


def _world_and_planet(obs, planet_id):
    """Build World and pull a PlanetView by id."""
    world = World.from_obs(obs)
    p = next(p for p in world.planets if p.id == planet_id)
    return world, p


def _step_env(initial_obs, my_emits, opp_emits, n_turns):
    """Run `fast_sim` for n_turns with the given per-seat emit sequences.

    `my_emits` and `opp_emits` are lists of length `n_turns`, one emit per
    turn. Returns the final snapshot's seat-0 observation dict.
    """
    snap = fast_sim.from_obs(initial_obs, configuration=None)
    for t in range(n_turns):
        snap = fast_sim.step(snap, [my_emits[t], opp_emits[t]])
        if snap.fake_env.done:
            break
    seat0 = snap.state[0].observation
    return {
        "step": int(getattr(seat0, "step", 0)),
        "planets": [list(p) for p in seat0.planets],
        "fleets": [list(f) for f in seat0.fleets or []],
    }


# ---- Test 5a — free capture -----------------------------------------------


def test_5a_free_capture():
    """100 ships, prod 2 @ (50, 50) vs neutral 5-ship @ (70, 50). Solver
    must return a Candidate; env-step must flip ownership."""
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=100, production=2),
        _planet(1, owner=-1, x=30.0, y=50.0, ships=5, production=1),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=1),  # distant idle enemy
    ]
    obs = _obs(planets=planets, step=10, player=0)
    world, src = _world_and_planet(obs, 0)
    target = next(p for p in world.planets if p.id == 1)

    cache = _build_centrality_cache(world)
    cand = _solve_single_source(src, target, world, my_id=0,
                                centrality_cache=cache,
                                target_is_ours=False)

    assert cand is not None, "free capture must produce a Candidate"
    assert cand.flavor == "capture"
    assert cand.target_id == 1
    assert cand.total_ships >= 5, "must launch enough to overcome 5 defenders"
    assert cand.allocations[0].src_id == 0

    # Verify via env-step: launch on turn 0, run for arrival_turn+1 turns
    # with opp idle, target ownership must flip to my_id=0.
    my_emit_t0 = [[a.src_id, a.aim_angle, a.ships] for a in cand.allocations]
    eta = cand.arrival_turn
    my_emits = [my_emit_t0] + [[]] * (eta + 1)
    opp_emits = [[]] * (eta + 2)
    final = _step_env(obs, my_emits, opp_emits, eta + 2)
    target_after = next(p for p in final["planets"] if p[0] == 1)
    assert target_after[1] == 0, (
        f"target should be owned by player 0, got owner={target_after[1]}; "
        f"cand={cand}, target_after={target_after}"
    )


# ---- Test 5b — bounce ----------------------------------------------------


def test_5b_bounce():
    """10 ships @ (50, 50) vs enemy 100-ship @ (70, 50). Solver must
    return None (cannot afford the capture)."""
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=10, production=1),
        _planet(1, owner=1, x=30.0, y=50.0, ships=100, production=2),
    ]
    obs = _obs(planets=planets, step=10, player=0)
    world, src = _world_and_planet(obs, 0)
    target = next(p for p in world.planets if p.id == 1)

    cache = _build_centrality_cache(world)
    cand = _solve_single_source(src, target, world, my_id=0,
                                centrality_cache=cache,
                                target_is_ours=False)

    assert cand is None, (
        f"bounce scenario must return None, got Candidate={cand}"
    )

    # Sanity-check the env: launching everything we have should NOT flip
    # ownership (would-be bounce).
    src_xy = (50.0, 50.0)
    tgt_xy = (70.0, 50.0)
    ang = math.atan2(tgt_xy[1] - src_xy[1], tgt_xy[0] - src_xy[0])
    my_emit_t0 = [[0, ang, 10]]   # launch 10 — guaranteed bounce
    my_emits = [my_emit_t0] + [[]] * 10
    opp_emits = [[]] * 11
    final = _step_env(obs, my_emits, opp_emits, 11)
    target_after = next(p for p in final["planets"] if p[0] == 1)
    assert target_after[1] == 1, (
        f"target should remain owned by player 1, got owner={target_after[1]}"
    )


# ---- Test 5c — wait-and-fire ---------------------------------------------


def test_5c_wait_and_fire():
    """5 ships, prod=4 @ (50, 50) vs neutral 30-ship @ (70, 50).

    Immediate launch is infeasible (only 5 ships < 31 needed).
    After accruing 7 turns of production (5 + 7*4 = 33 ≥ 31), the same
    solver should return a Candidate.

    Verification: env-step 7 turns idle, then launch from the post-wait
    state, target ownership flips.
    """
    planets = [
        _planet(0, owner=0, x=10.0, y=50.0, ships=5, production=4),
        _planet(1, owner=-1, x=30.0, y=50.0, ships=30, production=1),
        _planet(2, owner=1, x=90.0, y=10.0, ships=10, production=1),  # distant idle enemy
    ]
    obs_now = _obs(planets=planets, step=10, player=0)
    world_now, src_now = _world_and_planet(obs_now, 0)
    target_now = next(p for p in world_now.planets if p.id == 1)

    cache_now = _build_centrality_cache(world_now)
    cand_now = _solve_single_source(src_now, target_now, world_now, my_id=0,
                                    centrality_cache=cache_now,
                                    target_is_ours=False)
    assert cand_now is None, (
        f"immediate launch must be infeasible, got Candidate={cand_now}"
    )

    # Wait 7 turns: both sides idle. Source accrues to 5 + 7*4 = 33 ships.
    my_emits_wait = [[]] * 7
    opp_emits_wait = [[]] * 7
    snap = fast_sim.from_obs(obs_now, configuration=None)
    for t in range(7):
        snap = fast_sim.step(snap, [my_emits_wait[t], opp_emits_wait[t]])
    seat0 = snap.state[0].observation
    obs_after_wait = {
        "player": 0,
        "step": int(getattr(seat0, "step", 7)),
        "planets": [list(p) for p in seat0.planets],
        "fleets": [list(f) for f in seat0.fleets or []],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": 0.0,
        "initial_planets": [list(p) for p in
                            getattr(seat0, "initial_planets", seat0.planets)],
    }
    src_ships_after = next(p for p in obs_after_wait["planets"] if p[0] == 0)[5]
    assert src_ships_after >= 31, (
        f"source should have accrued ≥31 ships, has {src_ships_after}"
    )

    # Now solver should produce a Candidate.
    world_after, src_after = _world_and_planet(obs_after_wait, 0)
    target_after = next(p for p in world_after.planets if p.id == 1)
    cache_after = _build_centrality_cache(world_after)
    cand_after = _solve_single_source(src_after, target_after, world_after,
                                      my_id=0, centrality_cache=cache_after,
                                      target_is_ours=False)
    assert cand_after is not None, (
        "after waiting 7 turns, solver must return a Candidate"
    )
    assert cand_after.flavor == "capture"
    assert cand_after.target_id == 1

    # Env-step verify: launch from post-wait state, run for arrival_turn+1
    # turns, target flips.
    my_emit_t0 = [[a.src_id, a.aim_angle, a.ships]
                  for a in cand_after.allocations]
    eta = cand_after.arrival_turn
    my_emits = [my_emit_t0] + [[]] * (eta + 1)
    opp_emits = [[]] * (eta + 2)
    final = _step_env(obs_after_wait, my_emits, opp_emits, eta + 2)
    target_final = next(p for p in final["planets"] if p[0] == 1)
    assert target_final[1] == 0, (
        f"after wait+launch, target should be owned by player 0, "
        f"got owner={target_final[1]}; cand={cand_after}, target={target_final}"
    )
