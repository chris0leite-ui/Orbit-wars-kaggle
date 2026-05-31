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


def test_sync_generator_forms_coalition_without_prerank_entry(monkeypatch):
    """The fix (Phase 4): coalitions are built DIRECTLY from world geometry,
    NOT from `prerank`. The proposer cheap-rejects the bouncing single-source
    launches a coalition combines, so prerank often has NO row for the target.
    Even so, the generator must still assemble A+B→C from the map alone.

    Scorers are stubbed so only a waiting-leg coalition scores positive — this
    isolates the generator's enumeration (capture correctness is proven by
    test_neither_solo_captures_but_synchronized_stack_does)."""
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    monkeypatch.setenv("BASELINE_JOINT", "1")
    import agents.baseline.chooser_trajectory as ct
    from agents.baseline.proposer import MIN_FLEET_SIZE
    from lib.world_model import predict_garrison_at

    # No solo scores positive (no solo emits, no source reservation from solos).
    monkeypatch.setattr(ct, "score_candidate_v4",
                        lambda *a, **k: (-1.0, "bounced", 1))
    def _fake_joint(snap_base, launches, *a, **k):
        if any(int(L[4]) > 0 for L in launches):
            return (5.0, "scored")
        return (-1.0, "scored")
    monkeypatch.setattr(ct, "score_candidate_v4_joint", _fake_joint)

    _obs, snap, world, model, A, B, C = _scenario()
    _favor_fn, favs = _favors(snap)
    # prerank carries NO row for C (id 0) — only a single unrelated bounce
    # (B → opp home id 3), present solely so the `if not prerank` guard passes.
    opp_home = {int(p.id): p for p in world.planets_by_id.values()}[3]
    angle_d, eta_d = aim_and_eta(B, opp_home, int(B.ships), world.omega, world=world)
    prerank = [(-1.0, B, opp_home, int(B.ships), float(angle_d), int(eta_d), HORIZON, 0)]
    assert not any(int(r[2].id) == 0 for r in prerank)  # bypass premise: no C row

    moves, commits = ct.choose_trajectory(
        snap, prerank, favs, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model,
    )
    sync_commits = [c for c in commits if c.get("sync_joint")]
    # Generator built the coalition from world geometry despite the empty-of-C
    # prerank: A (id=1, nearer) waits; B (id=2) fires now; the target is C.
    assert len(sync_commits) == 1, (moves, commits)
    assert sync_commits[0]["src_id"] == 1
    assert sync_commits[0]["wait_remaining"] > 0
    assert sync_commits[0]["tgt_id"] == 0
    fire = [m for m in moves if int(m[0]) == 2]
    assert fire, (moves, commits)  # B fires now

    # Sizing: minimal but sufficient — both legs real, combined captures.
    near_ships = int(sync_commits[0]["ships_planned"])
    far_ships = int(fire[0][2])
    assert near_ships >= MIN_FLEET_SIZE and far_ships >= MIN_FLEET_SIZE
    wait_n = int(sync_commits[0]["wait_remaining"])
    _aw, eta_w = aim_and_eta(A, C, near_ships, world.omega, wait_N=wait_n, world=world)
    tarr = wait_n + int(eta_w)
    _o, garr_tarr = predict_garrison_at(C, tarr, model.ledger.get(0, []))
    assert near_ships + far_ships > garr_tarr  # stack beats arrival garrison


# --------------------------------------------------------------------------
# Lever 1 — size-to-hold (BASELINE_JOINT_SYNC_HOLD)
#
# hold_need inverts the _target_holdable_after_capture inequality to return the
# minimum total delivered ships that SURVIVES the predicted opp counter-attack
# (not just flips the planet). The core correctness pins below assert the
# self-consistency invariant: a hold_need-sized stack PASSES the very filter it
# is sized against, while a capture-only (garrison+1) stack FAILS it. The
# generator-side wiring (gate-skip under inflation) is exercised by the
# single-game trace; the OFF path is covered byte-identically by every test
# above (none set BASELINE_JOINT_SYNC_HOLD).
# --------------------------------------------------------------------------

from agents.baseline.proposer import (  # noqa: E402
    _target_holdable_after_capture,
    hold_need,
)


def _hold_world(*, c_ships=10, opp_ships=40, opp_prod=2, opp_near=True):
    """A 2P world where a small flip of neutral C is recapturable by a strong,
    nearby opp, and our only source is FAR (so the opp is the nearest planet to
    C → a counter is viable). With opp_near=False the strong opp is removed, so
    no viable counter exists. Returns (world, model, C)."""
    # C target (neutral) at (85,15); strong opp right next to it at (80,15);
    # our source far away at (20,15) — farther from C than the opp.
    planets = [
        [0, -1, 85.0, 15.0, 1.5, c_ships, 1],          # C: neutral target
        [1, 0, 20.0, 15.0, 1.5, 80, 1],                # our source (far)
    ]
    if opp_near:
        planets.append([2, 1, 80.0, 15.0, 1.5, opp_ships, opp_prod])  # strong opp, close
    else:
        # weak/no viable counter: a sub-threshold opp far away
        planets.append([2, 1, 15.0, 80.0, 1.5, 5, 1])
    obs = {
        "player": 0, "planets": planets, "fleets": [],
        "angular_velocity": 0.0, "comet_planet_ids": [], "step": 0,
    }
    world = World.from_obs(obs)
    model = WorldModel.from_world(world)
    C = {int(p.id): p for p in world.planets_by_id.values()}[0]
    src = {int(p.id): p for p in world.planets_by_id.values()}[1]
    return world, model, C, src


def test_hold_need_sizes_above_capture_to_survive_counter():
    """Core invariant: in a geometry where capture-only sizing gets recaptured,
    hold_need sizes UP, and a hold_need-sized stack is self-consistently
    HOLDABLE while the capture-only stack is NOT."""
    world, model, C, src = _hold_world()
    arrival_step = 10
    capture_need = int(C.ships) + 1  # neutral → garrison+1 flip threshold

    need = hold_need(C, arrival_step, world, 0, capture_need)
    assert need > capture_need, (need, capture_need)

    # A hold_need-sized stack survives the predicted counter (filter says True).
    assert _target_holdable_after_capture(
        src, C, need, 0, arrival_step, world, model, 0) is True
    # A capture-only stack does NOT (filter says False → it would be recaptured).
    assert _target_holdable_after_capture(
        src, C, capture_need, 0, arrival_step, world, model, 0) is False


def test_hold_need_returns_capture_when_no_counter():
    """With no viable counter in range, hold_need is a no-op = capture_need."""
    world, _model, C, _src = _hold_world(opp_near=False)
    capture_need = int(C.ships) + 1
    assert hold_need(C, 10, world, 0, capture_need) == capture_need


def test_sync_hold_on_does_not_break_emit(monkeypatch):
    """Wiring guard: with BASELINE_JOINT_SYNC_HOLD=1 in the standard scenario
    (where our sources are closer to C than the opp → no counter → no
    inflation), the coalition still emits exactly as with HOLD off. Proves the
    new import + env read + sizing branch + gate-skip don't crash or regress
    the ON path."""
    monkeypatch.setenv("BASELINE_JOINT_SYNC", "1")
    monkeypatch.setenv("BASELINE_JOINT_SYNC_HOLD", "1")
    monkeypatch.setenv("BASELINE_JOINT", "1")
    import agents.baseline.chooser_trajectory as ct
    monkeypatch.setattr(ct, "score_candidate_v4",
                        lambda *a, **k: (-1.0, "bounced", 1))

    def _fake_joint(snap_base, launches, *a, **k):
        if any(int(L[4]) > 0 for L in launches):
            return (5.0, "scored")
        return (-1.0, "scored")
    monkeypatch.setattr(ct, "score_candidate_v4_joint", _fake_joint)

    _obs, snap, world, model, A, B, C = _scenario()
    _favor_fn, favs = _favors(snap)
    angle_a, eta_a = aim_and_eta(A, C, 25, world.omega, world=world)
    angle_b, eta_b = aim_and_eta(B, C, 25, world.omega, world=world)
    prerank = [
        (-1.0, A, C, 25, float(angle_a), int(eta_a), HORIZON, 0),
        (-1.0, B, C, 25, float(angle_b), int(eta_b), HORIZON, 0),
    ]
    _moves, commits = ct.choose_trajectory(
        snap, prerank, favs, 0, 2, 2000.0, 5, HORIZON, GAMMA, world, model,
    )
    sync_commits = [c for c in commits if c.get("sync_joint")]
    assert len(sync_commits) == 1, (_moves, commits)
    assert sync_commits[0]["tgt_id"] == 0
