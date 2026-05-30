"""Phase-0 falsification + gate tests for synchronized-arrival JOINT
coalitions (BASELINE_JOINT_SYNC).

The mechanism: for a capture target NO single source can take alone, make
the closer source WAIT so all legs arrive the SAME tick and STACK their
ships (combat rule 1, lib/game/interpreter.py:809-812), capturing a planet
neither source could take solo.

Controlled scenario (omega=0 → synchronization is exact):
  - C: neutral planet at (90,50), 30 defenders, production 0 (static garrison).
  - A: my planet at (74,50), near C.    B: my planet at (44,50), far from C.
  - Each of A,B can field ~20 ships → neither's 20 beats C's 30 alone, but
    40 combined (arriving the same tick) captures with margin.
  - opp home at (10,10) so it's a valid 2P game and favor_fn has an opponent.
"""

from __future__ import annotations

import os

from agents.baseline.chooser_trajectory import (
    JOINT_SYNC_SETTLE,
    _solve_sync_wait,
    build_trajectory_baseline,
    choose_trajectory,
    score_candidate_v4,
    score_candidate_v4_joint,
    select_favor_fn,
)
from agents.baseline.proposer import aim_and_eta
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.intent import World
from lib.world_model import WorldModel

GAMMA = 0.99
HORIZON = 30


def _scenario(*, c_ships=30, a_ships=25, b_ships=25, c_owner=1, c_prod=1):
    """Build the controlled 2-source-stack world. Returns
    (obs, snap, world, model, A, B, C)."""
    # Cluster kept in the lower band (y<=28), well clear of the sun at
    # (50,50), and off each other's launch lines (no path_blocked / sun).
    # C is an opponent planet with production so capturing it is valuable
    # (gain + denial); neither A's nor B's 25 ships beats its arrival-time
    # garrison alone, but 50 stacked do.
    planets = [
        # id, owner,   x,   y,  radius, ships,   production
        [0, c_owner, 85.0, 15.0, 1.5, c_ships, c_prod],  # C: target
        [1, 0, 72.0, 25.0, 1.5, a_ships, 1],     # A: my near source
        [2, 0, 50.0, 28.0, 1.5, b_ships, 1],     # B: my far source
        [3, 1, 15.0, 80.0, 1.5, 30, 1],          # opp home
    ]
    obs = {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "step": 0,
    }
    snap = fs_from_obs(obs, num_seats=2)
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    by_id = {int(p.id): p for p in world.planets_by_id.values()}
    return obs, snap, world, model, by_id[1], by_id[2], by_id[0]


def _favors(snap):
    favor_fn = select_favor_fn()
    favs = build_trajectory_baseline(snap, 0, 2, HORIZON, favor_fn, GAMMA)
    return favor_fn, favs


# --------------------------------------------------------------------------
# Part B — the existence proof: the stack captures, neither solo does.
# --------------------------------------------------------------------------

def _owner_of(snap, pid):
    for p in snap.state[0].observation.planets:
        if int(p[0]) == pid:
            return int(p[1])
    return None


def _rollout_capture(snap, inject_by_tick, steps):
    """Roll the snapshot forward with my launches injected at the given
    ticks (opponent idle), return C's owner at the leaf. Opp-idle keeps
    this a clean mechanical proof of combat-rule-1 stacking."""
    s = fs_clone(snap)
    for t in range(steps):
        actions = [[], []]
        if t in inject_by_tick:
            actions[0] = inject_by_tick[t]
        s = fs_step(s, actions, in_place=True)
    return _owner_of(s, 0)


def test_neither_solo_captures_but_synchronized_stack_does():
    """Mechanical Phase-0 proof (favor-independent): two 25-ship launches
    that ARRIVE THE SAME TICK stack to 50 and capture C (garrison ~30+T),
    while either alone (25) bounces. Verified by C's owner at the leaf."""
    _obs, snap, world, model, A, B, C = _scenario()
    angle_a, eta_a = aim_and_eta(A, C, 25, world.omega, world=world)
    angle_b, eta_b = aim_and_eta(B, C, 25, world.omega, world=world)
    assert eta_a < eta_b, (eta_a, eta_b)  # A closer → B fires now, A waits
    T = int(eta_b)

    # Solo: neither single 25-ship launch captures C.
    assert _rollout_capture(snap, {0: [[1, float(angle_a), 25]]}, eta_a + 3) != 0
    assert _rollout_capture(snap, {0: [[2, float(angle_b), 25]]}, eta_b + 3) != 0

    # Synchronized stack: A waits so both land at T and combine to 50.
    solved = _solve_sync_wait(A, C, 25, world.omega, world, target_arrival=T)
    assert solved is not None, "static geometry must synchronize exactly"
    wait_a, angle_a_w = solved
    assert wait_a + int(aim_and_eta(A, C, 25, world.omega, wait_N=wait_a,
                                    world=world)[1]) == T
    captured_owner = _rollout_capture(
        snap,
        {0: [[2, float(angle_b), 25]], int(wait_a): [[1, float(angle_a_w), 25]]},
        T + 3,
    )
    assert captured_owner == 0, f"synchronized stack should capture C, got owner {captured_owner}"


# --------------------------------------------------------------------------
# _solve_sync_wait unit behavior
# --------------------------------------------------------------------------

def test_solve_sync_wait_hits_exact_tick_static():
    _obs, _snap, world, _model, A, _B, C = _scenario()
    _a, eta_a = aim_and_eta(A, C, 20, world.omega, world=world)
    target = int(eta_a) + 5
    solved = _solve_sync_wait(A, C, 20, world.omega, world, target_arrival=target)
    assert solved is not None
    wait, _angle = solved
    _a2, eta_w = aim_and_eta(A, C, 20, world.omega, wait_N=wait, world=world)
    assert wait + int(eta_w) == target


def test_solve_sync_wait_returns_none_when_target_before_arrival():
    _obs, _snap, world, _model, A, _B, C = _scenario()
    # A target arrival earlier than the soonest possible eta is impossible.
    assert _solve_sync_wait(A, C, 20, world.omega, world, target_arrival=0) is None


# --------------------------------------------------------------------------
# Gate discipline via choose_trajectory
# --------------------------------------------------------------------------

def _prerank_rows(world, model, A, B, C):
    """Two fire-now solo candidates (A→C, B→C) at each source's full 25
    ships — neither captures alone, but stacked (50) they beat C's
    arrival-garrison (~40)."""
    sa, sb = int(A.ships), int(B.ships)
    angle_a, eta_a = aim_and_eta(A, C, sa, world.omega, world=world)
    angle_b, eta_b = aim_and_eta(B, C, sb, world.omega, world=world)
    # tuple: (cheap, src, tgt, ships, angle, eta_hint, horizon, wait_N)
    return [
        (-1.0, A, C, sa, float(angle_a), int(eta_a), HORIZON, 0),
        (-1.0, B, C, sb, float(angle_b), int(eta_b), HORIZON, 0),
    ]


def test_off_by_default_emits_no_sync_commit(monkeypatch):
    monkeypatch.delenv("BASELINE_JOINT_SYNC", raising=False)
    monkeypatch.setenv("BASELINE_JOINT", "1")
    _obs, snap, world, model, A, B, C = _scenario()
    _favor_fn, favs = _favors(snap)
    prerank = _prerank_rows(world, model, A, B, C)
    _moves, commits = choose_trajectory(
        snap, prerank, favs, 0, 2, 600.0, 5, HORIZON, GAMMA, world, model,
    )
    assert not any(c.get("sync_joint") for c in commits)


def test_sync_on_emits_far_move_and_near_commit(monkeypatch):
    """Emit plumbing in isolation from favor scoring: stub the scorers so
    ONLY a synchronized coalition (a leg with wait_N>0) scores positive.
    The far leg must become a fire-now move; the near (waiting) leg a
    commit tagged sync_joint."""
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    monkeypatch.setenv("BASELINE_JOINT", "1")
    import agents.baseline.chooser_trajectory as ct
    # No solo scores positive (no solo_winners, no solo emits).
    monkeypatch.setattr(ct, "score_candidate_v4",
                        lambda *a, **k: (-1.0, "bounced", 1))
    # Only a coalition with a waiting leg (the sync coalition) scores > 0.
    def _fake_joint(snap_base, launches, *a, **k):
        if any(int(L[4]) > 0 for L in launches):
            return (5.0, "scored")
        return (-1.0, "scored")
    monkeypatch.setattr(ct, "score_candidate_v4_joint", _fake_joint)

    _obs, snap, world, model, A, B, C = _scenario()
    _favor_fn, favs = _favors(snap)
    prerank = _prerank_rows(world, model, A, B, C)
    moves, commits = ct.choose_trajectory(
        snap, prerank, favs, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model,
    )
    sync_commits = [c for c in commits if c.get("sync_joint")]
    # The near source (A, id=1) must be the waiting leg; far (B) fires now.
    assert len(sync_commits) == 1, (moves, commits)
    assert sync_commits[0]["src_id"] == 1
    assert sync_commits[0]["wait_remaining"] > 0
    assert any(int(m[0]) == 2 for m in moves)  # B (id=2) fires now


def test_sync_skipped_when_one_source_can_solo(monkeypatch):
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    monkeypatch.setenv("BASELINE_JOINT", "1")
    # B can solo-capture: give it 40 ships > C's 30 defenders.
    _obs, snap, world, model, A, B, C = _scenario(b_ships=40)
    _favor_fn, favs = _favors(snap)
    angle_a, eta_a = aim_and_eta(A, C, 20, world.omega, world=world)
    angle_b, eta_b = aim_and_eta(B, C, 40, world.omega, world=world)
    prerank = [
        (-1.0, A, C, 20, float(angle_a), int(eta_a), HORIZON, 0),
        (1.0, B, C, 40, float(angle_b), int(eta_b), HORIZON, 0),
    ]
    _moves, commits = choose_trajectory(
        snap, prerank, favs, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model,
    )
    # B alone captures → no synchronized coalition for this target.
    assert not any(c.get("sync_joint") for c in commits)
