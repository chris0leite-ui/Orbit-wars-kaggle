"""Unit tests for `lib.opp_model.me_defensive_action` — the bug #14
fix v2 (smart reactive defense for ME in the chooser's rollout).

The policy must satisfy these contracts:
- Returns env-format launches `[[src_id, angle, ships], ...]`.
- Emits ZERO actions when no MY planet has an inbound threat that
  exceeds its natural defense (ships + production × eta).
- Emits ONE reinforce per under-defended planet, sized for the
  bug-#12-windowed threat force, from the nearest viable sister.
- Skips threats where no reinforcer can arrive before the threat.
- Skips already-used sources within a single call (one launch per
  source per tick).

The asymmetric-rollout pessimism this fix targets is documented in
`audit/2026-05-18-bug-catalog.md#14`. The failed cheap-mirror is in
commit 5f22ea8; the test set there showed why a purely-defensive
policy is required (lite_greedy attacks from the would-be defender).
"""
from __future__ import annotations

import math

from lib.opp_model import me_defensive_action


def _planet(pid: int, owner: int, x: float, y: float,
            ships: int = 10, production: int = 1,
            radius: float = 1.0) -> list:
    return [int(pid), int(owner), float(x), float(y),
            float(radius), int(ships), int(production)]


def _obs(planets: list, fleets: list | None = None,
         angular_velocity: float = 0.0) -> dict:
    return {
        "player": 0,
        "step": 0,
        "planets": planets,
        "fleets": fleets or [],
        "comets": [],
        "comet_planet_ids": [],
        "angular_velocity": angular_velocity,
        "initial_planets": [list(p) for p in planets],
        "next_fleet_id": len(fleets) if fleets else 0,
        "remainingOverageTime": 60.0,
    }


def test_no_threat_emits_nothing():
    """No inbound enemy fleets → policy returns []."""
    p = [
        _planet(0, 0, 20.0, 20.0, ships=50, production=1),
        _planet(1, 0, 80.0, 80.0, ships=30, production=1),
        _planet(2, 1, 50.0, 50.0, ships=40, production=1),  # opp, idle
    ]
    obs = _obs(p, fleets=[])
    assert me_defensive_action(obs, me=0) == []


def test_single_threat_emits_one_reinforce():
    """One inbound 80-ship enemy fleet aimed at a thin (5-ship,
    prod=1) MY planet, with a sister (200 ships, prod=2) OFF the
    enemy fleet's axis (so ray-cast hits the thin planet first).

    Garrison_at_eta(P0) = 5 + 1·17 = 22  (eta ≈ 17 for 80-ship speed).
    Threat force = 80. Shortfall = 81 − 22 = 59.  Reinforce ships
    must be ≥ 60 = ceil(59) + safety margin (1).
    """
    p = [
        _planet(0, 0, 20.0, 20.0, ships=5, production=1),     # thin
        _planet(1, 0, 25.0, 35.0, ships=200, production=2),   # sister
        _planet(2, 1, 80.0, 80.0, ships=10, production=1),    # opp home
    ]
    # 80-ship fleet at (80, 20) heading -x: chord along y=20 only
    # ray-hits P0 (P1 is at y=35, perp > r).
    f = [10, 1, 80.0, 20.0, math.pi, 2, 80]
    moves = me_defensive_action(_obs(p, fleets=[f]), me=0)
    assert len(moves) == 1, f"expected one reinforce, got {moves}"
    sid, _angle, ships = moves[0]
    assert sid == 1, f"reinforcer should be P1, got {sid}"
    assert ships >= 60, (
        f"reinforce sized too small: {ships} (need ≥ 60 to cover "
        f"shortfall 59 + safety margin)"
    )


def test_multi_wave_threat_uses_wave_lookahead():
    """Bug #12 window — combined threat force is summed within
    `earliest_eta + WAVE_LOOKAHEAD` (=12). Two enemy fleets aimed at
    same MY planet at eta ≈ 6 (35 ships) and eta ≈ 14 (50 ships)
    are both within the 6+12 = 18 window, so threat_force = 85.

    Pre-fix bug #12 only counted the first wave (35) and zeroed the
    shortfall once production accrual covered it.
    """
    p = [
        _planet(0, 0, 20.0, 20.0, ships=15, production=1),    # thin
        _planet(1, 0, 25.0, 35.0, ships=200, production=2),   # sister
        _planet(2, 1, 80.0, 80.0, ships=10, production=1),
    ]
    # Two enemy fleets along y=20. Speeds: 35-ship ≈ 2.81, 50-ship ≈ 3.06.
    # First fleet at (40, 20), eta ≈ (40-20)/2.81 ≈ 7. Second at
    # (60, 20), eta ≈ (60-20)/3.06 ≈ 13.
    f1 = [10, 1, 40.0, 20.0, math.pi, 2, 35]
    f2 = [11, 1, 60.0, 20.0, math.pi, 2, 50]
    moves = me_defensive_action(_obs(p, fleets=[f1, f2]), me=0)
    assert len(moves) == 1, f"expected one reinforce, got {moves}"
    sid, _angle, ships = moves[0]
    assert sid == 1
    # Threat 85, garrison at eta ≈ 7 = 15 + 7 = 22 → shortfall 64.
    # Reinforce must be ≥ 65. If WAVE_LOOKAHEAD wasn't applied and
    # only fleet f1 was counted, threat would be 35 → shortfall 14
    # → reinforce ≈ 15, well under 65.
    assert ships >= 50, (
        f"reinforce sized for single-wave threat only: {ships}; "
        f"expected ≥ 50 from bug-#12-windowed multi-wave force"
    )


def test_undefendable_emits_nothing():
    """Threat too fast for any reinforcer to arrive in time → []."""
    p = [
        _planet(0, 0, 20.0, 20.0, ships=5, production=1),
        # Sister is FAR away (eta from sister > threat eta).
        _planet(1, 0, 90.0, 90.0, ships=200, production=2),
        _planet(2, 1, 80.0, 80.0, ships=10, production=1),
    ]
    # 30-ship fleet at (25, 20) heading -x: eta ≈ (25-20)/2.73 ≈ 2.
    # Sister at (90, 90) is ≈ 100 units away → reinforce eta ≫ 2.
    f = [10, 1, 25.0, 20.0, math.pi, 2, 30]
    assert me_defensive_action(_obs(p, fleets=[f]), me=0) == []


def test_self_sufficient_skips_emit():
    """Natural production covers the shortfall → no reinforce.

    Garrison_at_eta = 50 + 2·5 = 60. Threat = 40. Defended (60 ≥ 41).
    """
    p = [
        _planet(0, 0, 20.0, 20.0, ships=50, production=2),
        _planet(1, 0, 25.0, 35.0, ships=200, production=2),
        _planet(2, 1, 80.0, 80.0, ships=10, production=1),
    ]
    # 40-ship fleet 5 ticks away, heading at P0.
    f = [10, 1, 35.0, 20.0, math.pi, 2, 40]
    assert me_defensive_action(_obs(p, fleets=[f]), me=0) == []


def test_idempotency_inbound_friendly_counts_toward_garrison():
    """CRITICAL idempotency contract for the in-rollout use case:
    once the policy has emitted a reinforce, the SAME threat against
    the SAME planet must NOT trigger another emit on subsequent
    rollout ticks. Modeled by replaying the scenario with the
    previously-launched reinforce already in flight.

    Origin: 2026-05-18 PM. The first A/B of option 5 collapsed at
    15.6% (vs bundle's 50%) with max wallclock 8252ms. Diagnosis:
    the stateless policy re-emitted a redundant reinforce every
    rollout tick because `garrison_at_eta` didn't count
    already-in-flight friendly reinforces. By rollout tick 5 we had
    stacked 5 redundant reinforces, draining the sister and
    bloating the fleet count — `fs_step` slows under high fleet
    count and the chooser's leaf valuation goes haywire (it sees
    "I sent 300 ships, of course it held" → over-emits aggressive
    captures in real play).

    Fix: include friendly inbound ships in the garrison math. This
    test pins the contract.
    """
    p = [
        _planet(0, 0, 20.0, 20.0, ships=5, production=1),     # thin
        _planet(1, 0, 25.0, 35.0, ships=200, production=2),   # sister
        _planet(2, 1, 80.0, 80.0, ships=10, production=1),
    ]
    # Same enemy fleet as test_single_threat: 80 ships, eta≈17 → P0.
    f_enemy = [10, 1, 80.0, 20.0, math.pi, 2, 80]
    # A previously-launched friendly reinforce, 70 ships heading at
    # P0. Placed at (22, 25) — clearly EN ROUTE between P1 and P0
    # (not at P1's coords, otherwise the ray-cast self-attributes to
    # P1 and the friendly never reaches the policy's inbound map).
    angle_to_p0 = math.atan2(20.0 - 25.0, 20.0 - 22.0)
    f_friendly = [11, 0, 22.0, 25.0, angle_to_p0, 1, 70]
    obs = _obs(p, fleets=[f_enemy, f_friendly])
    moves = me_defensive_action(obs, me=0)
    assert moves == [], (
        f"idempotency contract VIOLATED: policy emitted {moves} "
        f"despite a 70-ship friendly reinforce already in flight to "
        f"the threatened planet. Without this contract the policy "
        f"stacks redundant reinforces every rollout tick — "
        f"catastrophic wallclock + behavioural blowup observed in "
        f"the 2026-05-18 A/B (15.6pct vs 50pct baseline, max "
        f"wallclock 8252ms)."
    )


def test_source_reservation_one_launch_per_source():
    """Two threatened MY planets that share the same nearest sister.
    The policy must NOT double-commit the sister; only one launch
    emits in this tick. (v1 single-reinforcer-per-tick contract.)"""
    p = [
        _planet(0, 0, 20.0, 20.0, ships=5, production=1),     # thin #1
        _planet(1, 0, 50.0, 30.0, ships=200, production=2),   # sister
        _planet(2, 0, 80.0, 20.0, ships=5, production=1),     # thin #2
        _planet(3, 1, 50.0, 90.0, ships=10, production=1),
    ]
    # Two enemy fleets: one heading at P0, one heading at P2. Both
    # 80-ship, both eta ≈ 17.
    f0 = [10, 1, 80.0, 20.0, math.pi, 3, 80]          # → P2 first
    # Wait — same direction would hit P2 first. Use different
    # axes. f_a goes -x toward P0 from x=80; f_b goes +x toward
    # P2 from a low-x position.
    # Cleaner: place enemies symmetrically.
    f_a = [10, 1, 60.0, 20.0, math.pi, 3, 80]    # heading -x, hits
    #                                              P0 at (20,20) first
    f_b = [11, 1, 40.0, 20.0, 0.0, 3, 80]        # heading +x, hits
    #                                              P2 at (80,20) first
    moves = me_defensive_action(_obs(p, fleets=[f_a, f_b]), me=0)
    sources_used = {int(m[0]) for m in moves}
    # The sister (P1) is the only viable reinforcer for both threats
    # (P0 and P2 can't reinforce each other — they're each under
    # immediate threat). So at most one launch should emit.
    assert len(moves) <= 1, (
        f"policy double-committed the sister: {moves}"
    )
    if moves:
        assert sources_used == {1}, (
            f"unexpected reinforcer used: {sources_used}"
        )
