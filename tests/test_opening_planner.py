"""Tests for the opening MILP planner.

Pin tests for the four opening fixes diagnosed against seed 384458460:

- Fix 1 (Bug A): cross-turn target dedup. The planner re-runs every turn
  and currently doesn't recognise that an in-flight friendly arrival
  already captures a target. Result: duplicate launches at the same
  target (seed 384458460 step 13: p16→p8 launched while p0→p8 is in
  flight, both land at p8).
- Fix 2 (Modeling gap C): per-(src, tgt) "diverse" cap kept top 3 by
  value, biasing toward early fire_steps that all conflict with the
  home corner-grab's ship budget. Post-fix: candidates span fire_steps
  with SPREAD_GAP=6 so the MILP sees a budget-feasible late fire.
- Fix 3 (Bug B): value model didn't discount flight time. Cross-board
  candidates (eta > 25 ticks) passed the ROI gate. Post-fix:
  value × OPENING_VALUE_GAMMA**(wait + eta).
- Fix 4 (Modeling gap D): `_expected_hold_duration` only consulted
  enemy threat ETA, not the predicted opp ship count at the contested
  target. Post-fix: if our residual < predicted opp force + margin,
  hold = 0.
"""

from __future__ import annotations

from lib.intent import World
from lib.joint_solver.opening_planner import _build_candidates
from lib.world_model import WorldModel


def _build_world_and_model(planets, *, step=0, omega=0.0, ledger=None):
    """Synthetic World + WorldModel from a list of [pid, owner, x, y, r,
    ships, prod] rows. `ledger` is `{pid: [(eta, owner, ships), ...]}`."""
    obs = {
        "player": 0,
        "planets": planets,
        "fleets": [],
        "angular_velocity": omega,
        "initial_planets": [],
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }
    world = World.from_obs(obs)
    base_ledger = {int(p[0]): [] for p in planets}
    if ledger:
        for pid, arrivals in ledger.items():
            base_ledger[int(pid)] = arrivals
    model = WorldModel(ledger=base_ledger, timelines={}, horizon=50)
    return world, model


# ---------------------------------------------------------------------------
# Fix 1 — cross-turn target dedup (Bug A)
# ---------------------------------------------------------------------------


def test_target_dedup_with_inflight_friendly_capture():
    """Pin: an in-flight friendly fleet captures p8 within ~7 ticks. The
    opening planner re-runs and proposes a SECOND launch at p8 from a
    different source. Pre-fix: candidates exist for p8. Post-fix: no
    p8 candidates (dedup recognises p8 will be ours via the in-flight
    arrival).

    Mirrors seed-384458460 step 13: at step 12 we fired p0→p8 (19
    ships, eta 8). At step 13's re-solve, the ledger contains an
    arrival at p8 (eta=7 from step-13 view). The planner shouldn't
    add a redundant p16→p8 launch.
    """
    planets = [
        [0, 0, 95.46, 89.06, 2.61, 30, 5],   # me: p0 (corner, captured)
        [8, -1, 72.65, 93.74, 2.39, 14, 4],  # neutral: p8 (already claimed)
        [16, 0, 83.41, 85.23, 1.69, 22, 2],  # me: p16 (home)
        [19, 1, 16.59, 14.77, 1.69, 10, 2],  # opp
    ]
    # In-flight friendly arrival at p8: eta=7 ticks, my fleet of 19 ships.
    ledger = {8: [(7, 0, 19)]}
    world, model = _build_world_and_model(planets, step=13, ledger=ledger)

    cands, _ = _build_candidates(world, model, my_id=0, num_seats=2)
    p8_cands = [c for c in cands if int(c.tgt_id) == 8]
    assert not p8_cands, (
        f"opening planner proposed {len(p8_cands)} candidate(s) targeting "
        f"p8 even though an in-flight friendly arrival captures p8 at "
        f"eta=7 (already in ledger). These create redundant attacks. "
        f"Got: {[(c.src_id, c.fire_step, c.ships) for c in p8_cands]}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — richer per-(src, tgt) candidate spread (Modeling gap C)
# ---------------------------------------------------------------------------


def test_candidate_spread_offers_budget_feasible_late_fires():
    """Pin: each (src, tgt) group should contain candidates SPREAD
    across fire_steps, not 3 consecutive ones. Pre-fix top-3-by-value
    keeps the earliest 3 (values monotonically decrease with fire_step).
    Post-fix: at least one (src, tgt) group has two kept candidates
    separated by ≥ SPREAD_GAP fire_steps so the MILP sees a
    budget-feasible LATE candidate distinct from the earliest one.

    Setup mirrors seed 384458460 layout from p16's perspective.
    """
    planets = [
        [16, 0, 83.41, 85.23, 1.69, 10, 2],  # me: home (cramped budget)
        [0, -1, 95.46, 89.06, 2.61, 14, 5],  # neutral: close corner
        [8, -1, 72.65, 93.74, 2.39, 18, 4],  # neutral: close ring
        [12, -1, 77.94, 74.02, 1.69, 22, 2], # neutral: close inner
        [19, 1, 16.59, 14.77, 1.69, 10, 2],  # opp
    ]
    world, model = _build_world_and_model(planets, step=0)

    cands, _ = _build_candidates(world, model, my_id=0, num_seats=2)
    # Group fire_steps per (src, tgt) pair.
    by_pair: dict[tuple, list[int]] = {}
    for c in cands:
        by_pair.setdefault((c.src_id, c.tgt_id), []).append(c.fire_step)
    # Post-fix: at least one pair should have two fire_steps separated
    # by ≥ 6 ticks (SPREAD_GAP). Pre-fix: each pair has 3 consecutive
    # fire_steps (max gap = 2).
    max_gap_seen = 0
    for pair, fires in by_pair.items():
        sf = sorted(fires)
        for i in range(len(sf) - 1):
            max_gap_seen = max(max_gap_seen, sf[i + 1] - sf[i])
    assert max_gap_seen >= 6, (
        f"no (src, tgt) pair had two candidates separated by ≥ 6 "
        f"fire_steps. The MILP only sees consecutive early candidates "
        f"that share the same ship-budget slot. Got fire_steps per "
        f"pair: {by_pair}; max gap: {max_gap_seen}"
    )


# ---------------------------------------------------------------------------
# Fix 3 — time-discount value model (Bug B)
# ---------------------------------------------------------------------------


def test_value_discounts_long_flights():
    """Pin: a close target and a far target with the same prod should
    differ by ≥ 3× in value post-fix. Pre-fix value model is `prod ×
    hold × bonus` (no eta dependence), so when neither target is
    opp-contested both score similarly. Post-fix: multiplied by
    γ^(wait+eta), the far target loses most of its value.

    Setup: NO opp planet at all — so `time_to_enemy_threat` returns
    None for every target, giving full hold credit. Two neutral
    targets with same prod=5: one close (dist ~13), one far (dist
    ~73). Pre-fix ratio should be ~ (T_END − close_arr) / (T_END −
    far_arr) ≈ 195/169 ≈ 1.15. Post-fix γ=0.95: ratio ≈ 4.4.
    """
    planets = [
        [16, 0, 83.41, 85.23, 1.69, 50, 2],  # me: home (lots of ships)
        [0, -1, 95.46, 89.06, 2.61, 14, 5],  # neutral: close (dist 12.6)
        [1, -1, 10.94, 95.46, 2.61, 14, 5],  # neutral: far (dist 73)
        # NO opp planet — full hold credit for both targets
    ]
    world, model = _build_world_and_model(planets, step=0)
    cands, _ = _build_candidates(world, model, my_id=0, num_seats=2)

    close_vals = [c.value for c in cands if int(c.tgt_id) == 0]
    far_vals = [c.value for c in cands if int(c.tgt_id) == 1]
    assert close_vals, (
        f"expected at least one close candidate (tgt=0). "
        f"Candidates seen: {[(c.tgt_id, c.fire_step) for c in cands]}"
    )
    assert far_vals, (
        f"expected at least one far candidate (tgt=1). "
        f"Candidates seen: {[(c.tgt_id, c.fire_step) for c in cands]}"
    )
    ratio = max(close_vals) / max(far_vals)
    assert ratio >= 3.0, (
        f"close target should outvalue far target by ≥ 3× post-fix "
        f"(time-discount). Got close max={max(close_vals):.1f}, "
        f"far max={max(far_vals):.1f}, ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Fix 4 — opp racing model (Modeling gap D)
# ---------------------------------------------------------------------------


def test_opp_race_rejects_overpowered_contested_target():
    """Pin: target reachable with positive eta-delta vs opp threat,
    but opp has OVERWHELMING ship count nearby. Pre-fix:
    `_expected_hold_duration` returns positive hold (delta-based);
    candidate accepted. Post-fix: predicted opp force at arrival
    exceeds our capture residual → hold = 0 → candidate dropped.

    Geometry: I'm slightly closer to target than opp, so opp_threat_eta
    > my arrival → pre-fix gives positive hold (small, but ROI-passing).
    But opp's source has 100 ships vs my 15-ship attack; my residual
    after capture is ~1, opp can easily recapture.
    """
    planets = [
        [16, 0, 50.0, 65.0, 1.69, 50, 2],   # me: 10u south of target
        [20, -1, 50.0, 75.0, 2.61, 14, 5],  # neutral target
        [25, 1, 50.0, 95.0, 2.39, 100, 4],  # opp: 20u north of target, 100 ships
    ]
    world, model = _build_world_and_model(planets, step=0)
    cands, _ = _build_candidates(world, model, my_id=0, num_seats=2)
    contested = [c for c in cands if int(c.tgt_id) == 20]
    assert not contested, (
        f"target p20 is reachable but heavily contested (opp source "
        f"20 units away with 100 ships; our residual ~1). Post-fix "
        f"the planner should reject this target. Got: "
        f"{[(c.src_id, c.fire_step, c.ships, c.value) for c in contested]}"
    )
