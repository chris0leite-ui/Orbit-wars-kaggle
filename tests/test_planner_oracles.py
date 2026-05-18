"""Synthetic planner oracle tests — coordination properties.

Each oracle is a synthetic obs (hand-built) for which the OBVIOUSLY
correct emit set is known. The planner is run and we assert the emit
satisfies the property.

These tests are EXPECTED TO FAIL currently. They define the target
behaviour for the next-architecture planner. As planner improves, more
oracles pass.

Reference: knowledge-base/concepts/coordination-oracle-testing.md

Run: pytest tests/test_planner_oracles.py -v --tb=short
Expected: most XFAIL today; the goal is to convert XFAIL → PASS
as the planner gains coordination ability.
"""

from __future__ import annotations

import math
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Synthetic obs builders
# ---------------------------------------------------------------------------


def _planet(pid: int, owner: int, x: float, y: float,
            ships: int = 10, production: int = 1,
            radius: float = 1.0) -> list:
    """Build a planet tuple in env obs format:
    [id, owner, x, y, radius, ships, production]."""
    return [int(pid), int(owner), float(x), float(y),
            float(radius), int(ships), int(production)]


def _obs(planets: list, fleets: list = None, step: int = 0,
         player: int = 0, angular_velocity: float = 0.0) -> dict:
    return {
        "player": player,
        "step": step,
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": angular_velocity,
        "initial_planets": [list(p) for p in planets],
        "next_fleet_id": len(fleets) if fleets else 0,
        "remainingOverageTime": 60.0,
    }


def _emit(obs):
    """Run our agent on obs and return its move list."""
    import os
    os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid")
    from agents.baseline.main import agent
    return agent(obs)


def _targets_of_emits(obs, moves) -> list[int]:
    """For each emit, return the FIRST planet on its straight-line path
    (best-effort target inference for static planets)."""
    if not moves:
        return []
    planets = obs["planets"]
    by_id = {p[0]: p for p in planets}
    targets = []
    for m in moves:
        src_id = int(m[0])
        angle = float(m[1])
        src = by_id.get(src_id)
        if src is None:
            continue
        sx, sy = src[2], src[3]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        best = None
        best_t = float("inf")
        for tgt in planets:
            if tgt[0] == src_id:
                continue
            dx, dy = tgt[2] - sx, tgt[3] - sy
            t = dx * cos_a + dy * sin_a
            if t <= 0:
                continue
            perp = abs(dx * (-sin_a) + dy * cos_a)
            if perp < tgt[4] + 1.0 and t < best_t:
                best_t = t
                best = tgt
        if best is not None:
            targets.append(int(best[0]))
    return targets


# ---------------------------------------------------------------------------
# Oracle 1 — Cleanup property (bug #13)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Bug #13: chooser stalls in dominant positions")
def test_oracle_cleanup_capture_last_opp_planet():
    """When we overwhelm opp 10×+, we must finish them off.

    Setup: 23 of our planets with 3000+ ships total, 1 opp planet
    with 100 ships. The PLANNER must emit at least one launch
    targeting the opp planet within 1-2 turns.

    This is the dekaineko scenario (step 150) abstracted.
    """
    # Build 23 our planets in a rough circle around the board
    our_planets = []
    for i in range(23):
        angle = 2 * math.pi * i / 23
        x = 50.0 + 35.0 * math.cos(angle)
        y = 50.0 + 35.0 * math.sin(angle)
        our_planets.append(_planet(i, 0, x, y, ships=150, production=2))
    # Opp's lone planet at a clear position
    opp_planet = _planet(23, 1, 85.0, 50.0, ships=100, production=2)
    obs = _obs(our_planets + [opp_planet], step=150)
    moves = _emit(obs)
    # At least one move toward opp's planet (pid=23)
    targets = _targets_of_emits(obs, moves)
    assert 23 in targets, (
        f"Cleanup oracle FAIL: with overwhelming force (23 planets, "
        f"~3450 ships) vs 1 opp planet (100 ships), planner emitted "
        f"{len(moves)} moves targeting {targets}. Expected at least "
        f"one move toward pid=23 (the lone opp planet)."
    )


# ---------------------------------------------------------------------------
# Oracle 2 — Coordinated capture (PI's two-150-vs-100 example)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Bug #14: planner can't model coordinated capture")
def test_oracle_coordinated_capture_two_sources():
    """PI's scenario (2026-05-18): two of our planets each with 150
    ships, opp has 1 planet with 100 ships. Distance is such that
    a single solo launch can win combat BUT loses the source to
    counter-attack. A coordinated joint or sequential plan wins.

    Properties checked (any one is acceptable):
      (a) two simultaneous launches at the opp planet (joint), OR
      (b) launch from one, with the other prepared to defend, OR
      (c) sequence: launch + reinforce-source on subsequent turn

    The chooser MUST emit at least one move toward the opp planet
    AND NOT leave either of our sources with zero ships post-launch.
    """
    # Two of our planets, far apart, each with 150 ships.
    # Opp at off-center position so neither A→opp nor B→opp
    # passes through the sun at (50,50).
    a = _planet(0, 0, 15.0, 15.0, ships=150, production=1)
    b = _planet(1, 0, 85.0, 15.0, ships=150, production=1)
    # Opp planet placed off-center to avoid sun-line from either source
    opp = _planet(2, 1, 50.0, 85.0, ships=100, production=1)
    obs = _obs([a, b, opp])
    moves = _emit(obs)
    targets = _targets_of_emits(obs, moves)
    # Must emit something targeting opp
    assert 2 in targets, (
        f"Coordinated-capture oracle FAIL: planner emitted {moves} → "
        f"targets {targets}. Expected at least one launch toward pid=2 "
        f"(opp planet). With 150+150=300 ships vs opp 100, this is "
        f"trivially winnable via coordination."
    )
    # No source should be drained to 0
    by_id = {p[0]: p for p in obs["planets"]}
    for m in moves:
        sid, _, ships = int(m[0]), float(m[1]), int(m[2])
        src = by_id.get(sid)
        if src is None:
            continue
        residue = src[5] - ships
        assert residue >= 5, (
            f"Drain-frontier oracle FAIL: launch from P{sid} of "
            f"{ships} ships leaves residue {residue} < 5. Source "
            f"would be vulnerable to counter-attack."
        )


# ---------------------------------------------------------------------------
# Oracle 3 — Solo CAN capture but loses source (forces coordination)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Bug #14: chooser is risk-averse about exposing sources")
def test_oracle_solo_capture_but_loses_source():
    """Single A → opp launch can win combat at opp (we have more
    ships) BUT leaves A with 0 → opp's launch from opp planet
    captures A before reinforcement.

    With ONE neighbor B that can reinforce A within the
    opp-counter-attack window, the coordinated plan wins.

    The planner should:
      (a) NOT do the bare solo from A (it loses), AND
      (b) emit a plan that captures opp without losing A
    """
    # Place off the y=50 axis to avoid sun-line.
    # A at (30, 25) with 110 ships
    a = _planet(0, 0, 30.0, 25.0, ships=110, production=1)
    # B (neighbor that can reinforce A) at (35, 25) with 80 ships
    b = _planet(1, 0, 35.0, 25.0, ships=80, production=1)
    # Opp at (70, 25) with 90 ships
    opp = _planet(2, 1, 70.0, 25.0, ships=90, production=1)
    obs = _obs([a, b, opp])
    moves = _emit(obs)
    # Must emit something toward opp
    targets = _targets_of_emits(obs, moves)
    assert 2 in targets, (
        f"Planner must emit toward opp (pid=2) but emitted {targets}"
    )
    # If only A launches solo (with > ~95 ships, leaving < 15), counter
    # would capture A. The planner must EITHER:
    # - launch from A AND B simultaneously toward opp, OR
    # - launch from A with smaller force (leaving 15+ residue), OR
    # - launch from B alone (80 ships ≥ 91 needed? actually 80 < 91 so
    #   B alone bounces — so B must contribute via coordination)
    by_id = {p[0]: p for p in obs["planets"]}
    # No emit from A with ships > 95 alone
    solo_a_emits = [m for m in moves if int(m[0]) == 0 and int(m[2]) > 95]
    if solo_a_emits and len(moves) == 1:
        # Only emit, and it's a draining solo from A → fail
        pytest.fail(
            f"Solo-drain FAIL: planner emitted only solo from A with "
            f"{solo_a_emits[0][2]} ships, leaving A vulnerable. Need "
            f"coordination with B (80 ships available)."
        )


# ---------------------------------------------------------------------------
# Oracle 4 — Defense-by-reinforcement: incoming threat that solo can't handle
# ---------------------------------------------------------------------------


def test_oracle_defense_against_incoming_multi_fleet():
    """Our planet P is under attack from TWO opp in-flight fleets.
    Combined enemy strength exceeds our garrison + production accrual.
    A reinforcement launch from a neighbor MUST be emitted.

    This is the asdf-game step 37 pattern abstracted. Used to be
    xfail (bugs #11 + #12 expected to suppress it). Bug #11 (orbital
    ray-cast) was fixed earlier and flipped this to xpass; the bug
    #12 fix (WAVE_LOOKAHEAD widening the inflight summation window)
    is the principled reason this should pass — both waves enter the
    enemy_inflight sum, shortfall is positive, reinforce candidate
    emitted. Unmarked 2026-05-18 PM after bug #12 fix.
    """
    # Place off y=50 axis to avoid sun. Our planet under threat:
    p_under = _planet(0, 0, 30.0, 25.0, ships=30, production=2)
    # Our neighbor (reinforcer) close enough to defend in time:
    p_help = _planet(1, 0, 25.0, 25.0, ships=200, production=2)
    # Opp planet (source of in-flight fleets), far from our planets:
    opp = _planet(2, 1, 75.0, 25.0, ships=10, production=1)
    # TWO in-flight opp fleets aimed at P0
    # Fleet 1: 5 units from P0, heading west, eta ~2 (close, ~40 ships)
    f1 = [10, 1, 35.0, 25.0, math.atan2(0, -5), 2, 40]
    # Fleet 2: 15 units away, heading west, eta ~7 (~60 ships)
    f2 = [11, 1, 45.0, 25.0, math.atan2(0, -15), 2, 60]
    obs = _obs([p_under, p_help, opp], fleets=[f1, f2])
    moves = _emit(obs)
    # Planner must emit a reinforcement from P1 to P0
    # Specifically: from src=1 (the helper) targeting pid=0 (under attack)
    targets = _targets_of_emits(obs, moves)
    assert 0 in targets, (
        f"Defense oracle FAIL: P0 is under attack from 100 ships "
        f"combined (40 at eta=5 + 60 at eta=10). P0 has 30 ships + "
        f"2*10=20 production = 50, vs 100. Needs reinforcement from "
        f"P1 (200 ships). Planner emitted {moves} → targets {targets}."
    )


# ---------------------------------------------------------------------------
# Oracle 5 — Bug #12 specific: WIDE-gap multi-wave attack
# ---------------------------------------------------------------------------


def test_oracle_defense_wide_gap_multi_wave():
    """Bug #12 verification: TWO opp fleets with a large eta gap (≥ 5
    ticks). Pre-fix the `enemy_eta + 1` window summed only the earliest
    wave; the later wave was silently excluded, shortfall went negative,
    no reinforce candidate emitted. Post-fix (`WAVE_LOOKAHEAD = 12`),
    both waves enter the sum, the shortfall is positive, and the
    proposer emits a reinforce candidate.

    Sized so the earliest wave ALONE cannot dominate our garrison
    (so the pre-fix code's shortfall is small/negative) but the
    combined wave clearly does. The reinforce neighbor has plenty of
    ships and is close enough to defend.
    """
    # Our planet under threat — 30 ships + 2/turn production
    p_under = _planet(0, 0, 30.0, 25.0, ships=30, production=2)
    # Reinforce neighbor — close, lots of ships
    p_help = _planet(1, 0, 25.0, 25.0, ships=200, production=2)
    # Opp planet (source) — sized so it can't single-launch
    opp = _planet(2, 1, 75.0, 25.0, ships=10, production=1)
    # First wave: 35 ships at eta ≈ 2 (close, would lose 1-vs-1 against
    # 30 + 2*2 = 34 garrison). Pre-fix shortfall ≈ 35 - 34 + 1 = 2 —
    # tiny, barely emits.
    f1 = [10, 1, 35.0, 25.0, math.atan2(0, -5), 2, 35]
    # Second wave: 50 ships at eta ≈ 8 (≥ 5 ticks later — outside the
    # pre-fix `enemy_eta + 1 = 3` window).
    f2 = [11, 1, 50.0, 25.0, math.atan2(0, -20), 2, 50]
    obs = _obs([p_under, p_help, opp], fleets=[f1, f2])
    moves = _emit(obs)
    targets = _targets_of_emits(obs, moves)
    assert 0 in targets, (
        f"Wide-gap multi-wave defense FAIL: combined opp force ≈ 85 vs "
        f"P0's 30 + accrual. Pre-fix the eta=8 wave was excluded from "
        f"the enemy_inflight sum (window was enemy_eta + 1 = 3). "
        f"Planner emitted {moves} → targets {targets}."
    )


# ---------------------------------------------------------------------------
# Sanity test — trivial 1-vs-1 (should always pass)
# ---------------------------------------------------------------------------


def test_oracle_sanity_trivial_capture():
    """Trivial: we have 100 ships, opp has 5 ships, easy capture.
    Planner MUST emit at least one move toward opp.

    Note: positions chosen to AVOID the sun at (50,50). The straight
    line from src to tgt must not pass within SUN_RADIUS (=10) of
    center — the proposer's trajectory filter would drop such
    candidates as 'sun-bound'.
    """
    # Place both planets off the central axis to avoid sun-line
    a = _planet(0, 0, 30.0, 30.0, ships=100, production=1)
    opp = _planet(1, 1, 70.0, 30.0, ships=5, production=1)
    obs = _obs([a, opp])
    moves = _emit(obs)
    targets = _targets_of_emits(obs, moves)
    assert 1 in targets, (
        f"Trivial-capture sanity FAIL: 100 ships vs 5 ships → "
        f"planner should emit. Got {moves} → targets {targets}."
    )
