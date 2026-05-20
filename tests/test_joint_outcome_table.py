"""Unit tests for lib/joint_solver/outcome_table.

10 hand-crafted PB (per-planet battle) scenarios that exercise the
subset-enumeration plumbing against either:
  - closed-form simulate_planet_timeline (lib/world_model), which calls
    the same resolve_arrivals — proves enumeration produces the right
    arrival set; OR
  - end-to-end fast_sim stepping (lib/fast_sim), which exercises the
    env's interpreter — proves the per-tick semantics (production
    before combat) match the env exactly.

Scenarios:
   1. Solo capture of neutral (one candidate fires, planet flips).
   2. Solo capture of opp with garrison (capture sized to overrun).
   3. Solo under-capture (one candidate < garrison, no flip).
   4. Two-attacker tie at same eta (rule 4; planet keeps current owner).
   5. Reinforce + simultaneous opp attack.
   6. Multi-eta sequence (cap at t=3, reinforce at t=5).
   7. Empty subset baseline (no candidates fire; production only).
   8. Production stream attribution after mid-stream flip.
   9. Multi-flip owner sequence (production stream by owner).
  10. End-to-end fast_sim: hand-fire one fleet and verify planet state.

Plus 3 validation tests (column_id rules, MAX_ENUMERATION_BITS cap).
"""

from __future__ import annotations

import math

import pytest
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.fast_sim import from_obs, step as fs_step
from lib.joint_solver.outcome_table import (
    MAX_ENUMERATION_BITS,
    Arrival,
    empty_subset_outcome,
    enumerate_outcomes,
    winning_subsets,
)
from lib.world_model import simulate_planet_timeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _planet(pid, owner, *, ships=0, production=2, x=50.0, y=50.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _timeline_owner_ships(planet, arrivals_as_tuples, horizon):
    """Ground-truth via simulate_planet_timeline (lib/world_model).

    Both outcome_table and simulate_planet_timeline call resolve_arrivals;
    this comparison verifies the enumeration plumbing (correct subset →
    arrival list mapping, correct production-tick order)."""
    timeline = simulate_planet_timeline(planet, arrivals_as_tuples, horizon)
    return timeline["owner_at"][horizon], timeline["ships_at"][horizon]


# ---------------------------------------------------------------------------
# 1. Solo capture of neutral
# ---------------------------------------------------------------------------


def test_solo_capture_of_neutral():
    planet = _planet(0, owner=-1, ships=5, production=2)
    candidate = Arrival(eta=3, owner=0, ships=10, column_id=100)
    table = enumerate_outcomes(
        initial_owner=-1, initial_ships=5, production=2, horizon=10,
        fixed_arrivals=[], candidate_arrivals=[candidate],
    )
    # Empty subset: planet stays neutral (no production for neutrals).
    base = table[()]
    assert base.owner_T == -1
    assert base.ships_T == 5.0
    # Full subset: 10-ship attack beats garrison(5+0 prod, neutral doesn't produce)=5 → flip.
    fired = table[(100,)]
    assert fired.owner_T == 0
    # Once we own it (t=3), we get production for ticks 4..10 = 7*2 = 14.
    # Survivor 10−5=5 lands; from t=4 onwards +2 per tick.
    assert fired.ships_T == 5.0 + 7 * 2
    # Cross-check against simulate_planet_timeline
    o2, s2 = _timeline_owner_ships(planet, [(3, 0, 10)], 10)
    assert (fired.owner_T, fired.ships_T) == (o2, s2)


# ---------------------------------------------------------------------------
# 2. Solo capture of opp with garrison
# ---------------------------------------------------------------------------


def test_solo_capture_of_opp_with_garrison():
    planet = _planet(0, owner=1, ships=8, production=3)
    candidate = Arrival(eta=2, owner=0, ships=20, column_id=200)
    table = enumerate_outcomes(
        initial_owner=1, initial_ships=8, production=3, horizon=6,
        fixed_arrivals=[], candidate_arrivals=[candidate],
    )
    base = table[()]
    # Opp produces every tick: t=1..6 = 8 + 6*3 = 26.
    assert base.owner_T == 1
    assert base.ships_T == 8 + 6 * 3
    # Full subset: at t=2, opp has 8 + 2*3 = 14. 20 vs 14 → me wins, 6 survives.
    # Then ticks 3..6 = 4*3=12 production for me.
    fired = table[(200,)]
    assert fired.owner_T == 0
    assert fired.ships_T == 6 + 4 * 3


# ---------------------------------------------------------------------------
# 3. Solo under-capture (no flip)
# ---------------------------------------------------------------------------


def test_solo_under_capture_no_flip():
    planet = _planet(0, owner=1, ships=20, production=2)
    candidate = Arrival(eta=2, owner=0, ships=10, column_id=300)
    table = enumerate_outcomes(
        initial_owner=1, initial_ships=20, production=2, horizon=4,
        fixed_arrivals=[], candidate_arrivals=[candidate],
    )
    fired = table[(300,)]
    # At t=2 opp has 20+4=24; 10-ship attack → opp keeps with 24−10=14.
    # Then 2 more ticks production = 14 + 2*2 = 18.
    assert fired.owner_T == 1
    assert fired.ships_T == 14 + 2 * 2


# ---------------------------------------------------------------------------
# 4. Two-attacker tie at same eta (rule 4)
# ---------------------------------------------------------------------------


def test_two_attacker_tie_same_eta():
    planet = _planet(0, owner=-1, ships=0, production=0)
    me_attack = Arrival(eta=3, owner=0, ships=10, column_id=400)
    opp_attack = Arrival(eta=3, owner=1, ships=10, column_id=401)
    table = enumerate_outcomes(
        initial_owner=-1, initial_ships=0, production=0, horizon=5,
        fixed_arrivals=[], candidate_arrivals=[me_attack, opp_attack],
    )
    # Both fire → tie among attackers → no survivor; garrison untouched (still neutral 0).
    both = table[tuple(sorted([400, 401]))]
    assert both.owner_T == -1
    assert both.ships_T == 0.0
    # Only me fires → I capture (10 vs neutral 0).
    me_only = table[(400,)]
    assert me_only.owner_T == 0


# ---------------------------------------------------------------------------
# 5. Reinforce + simultaneous opp attack (mixed candidate + fixed)
# ---------------------------------------------------------------------------


def test_reinforce_plus_fixed_opp_attack():
    """My reinforce (candidate) + opp's already-in-flight attack (fixed)
    both arrive at t=4 at my planet."""
    planet = _planet(0, owner=0, ships=10, production=1)
    me_reinforce = Arrival(eta=4, owner=0, ships=8, column_id=500)
    opp_fixed = Arrival(eta=4, owner=1, ships=15, column_id=None)

    table = enumerate_outcomes(
        initial_owner=0, initial_ships=10, production=1, horizon=6,
        fixed_arrivals=[opp_fixed], candidate_arrivals=[me_reinforce],
    )
    # No reinforce: at t=4 garrison = 10 + 4 = 14, opp 15-ship attack lands.
    # Survivor = 15 (single attacker). 14 vs 15 → opp wins; remainder = 1.
    # Then ticks 5,6 production for opp = 1 + 2*1 = 3.
    base = table[()]
    assert base.owner_T == 1
    assert base.ships_T == 1 + 2
    # With reinforce: at t=4 garrison = 14, +8 reinforce (same owner → reinforce
    # rule 3a since survivor=me beats none). Actually resolve_arrivals groups
    # by owner: me=8, opp=15. Largest=opp 15, second=me 8 → opp survives with 7.
    # Then opp 7 vs garrison(me) 14 → me keeps with 14−7=7. +2 prod = 9.
    fired = table[(500,)]
    assert fired.owner_T == 0
    assert fired.ships_T == 7 + 2 * 1


# ---------------------------------------------------------------------------
# 6. Multi-eta sequence (capture at t=3, reinforce at t=5)
# ---------------------------------------------------------------------------


def test_multi_eta_capture_then_reinforce():
    planet = _planet(0, owner=-1, ships=2, production=2)
    cap = Arrival(eta=3, owner=0, ships=5, column_id=600)
    reinf = Arrival(eta=5, owner=0, ships=4, column_id=601)
    table = enumerate_outcomes(
        initial_owner=-1, initial_ships=2, production=2, horizon=7,
        fixed_arrivals=[], candidate_arrivals=[cap, reinf],
    )
    # Both fire: t=3 me captures (5 > 2 neutral); survivor 3. Ticks 4,5: +2,+2 → 7.
    # At t=5 me arrives with 4; resolve_arrivals → owner=me, garrison 7+4=11.
    # Ticks 6,7: +2,+2 → 15.
    both = table[tuple(sorted([600, 601]))]
    assert both.owner_T == 0
    assert both.ships_T == 15.0
    # Only cap: 3 + ticks 4..7 = 3 + 4*2 = 11.
    cap_only = table[(600,)]
    assert cap_only.owner_T == 0
    assert cap_only.ships_T == 11.0
    # Only reinforce (no capture first): at t=5, neutral 2-ship garrison
    # vs me 4 → me captures, survivor 2. Ticks 6,7: +2,+2 → 6.
    reinf_only = table[(601,)]
    assert reinf_only.owner_T == 0
    assert reinf_only.ships_T == 6.0


# ---------------------------------------------------------------------------
# 7. Empty subset baseline (no candidates fire)
# ---------------------------------------------------------------------------


def test_empty_subset_baseline_production_only():
    planet = _planet(0, owner=0, ships=5, production=3)
    table = enumerate_outcomes(
        initial_owner=0, initial_ships=5, production=3, horizon=10,
        fixed_arrivals=[], candidate_arrivals=[],
    )
    base = empty_subset_outcome(table)
    assert base.owner_T == 0
    assert base.ships_T == 5 + 10 * 3
    # All production credited to owner 0.
    assert base.prod_stream == {0: 30}


# ---------------------------------------------------------------------------
# 8. Production stream attribution after mid-stream flip
# ---------------------------------------------------------------------------


def test_production_stream_after_flip():
    planet = _planet(0, owner=-1, ships=0, production=2)
    cap = Arrival(eta=4, owner=0, ships=1, column_id=800)
    table = enumerate_outcomes(
        initial_owner=-1, initial_ships=0, production=2, horizon=10,
        fixed_arrivals=[], candidate_arrivals=[cap],
    )
    fired = table[(800,)]
    # Neutral t=1..3 (no production). t=4 capture (1 > 0). Then t=5..10 produce 2.
    # 6 ticks of production = 12 ships. Plus the surviving 1.
    assert fired.owner_T == 0
    assert fired.ships_T == 1 + 6 * 2
    # Prod stream: 6 ticks * 2 = 12 to owner 0; nothing to -1.
    assert fired.prod_stream.get(0, 0) == 12
    assert -1 not in fired.prod_stream


# ---------------------------------------------------------------------------
# 9. Multi-flip owner sequence: production credited per holder per interval
# ---------------------------------------------------------------------------


def test_multi_flip_production_stream():
    # Start owner=0 (me). Opp captures at t=3 (8 ships vs my 5+3*1=8 -- tie? Let's set
    # opp ships=9 to ensure flip). Then me re-captures at t=7.
    planet = _planet(0, owner=0, ships=5, production=1)
    opp_attack = Arrival(eta=3, owner=1, ships=9, column_id=900)
    me_retake = Arrival(eta=7, owner=0, ships=8, column_id=901)
    table = enumerate_outcomes(
        initial_owner=0, initial_ships=5, production=1, horizon=10,
        fixed_arrivals=[], candidate_arrivals=[opp_attack, me_retake],
    )
    both = table[tuple(sorted([900, 901]))]
    # t=1,2: me produces +1,+1 → garrison 7.
    # t=3: me produces +1 → 8. Opp 9 lands → survivor opp 1 → vs garrison(me)8
    # → me keeps 8-1=7. owner=0 still.
    # Hmm, opp_attack with ships=9 doesn't flip — me retains.
    # Adjust expectation: at t=3 garrison(me)=8, opp 9 attacks: opp wins with 9-8=1, owner=1.
    # Wait — resolve_arrivals(garrison_owner=0, garrison_ships=8, [(1,9)]):
    # ranked = [(1,9)]; top owner=1, ships=9. survivor=1 with 9 ships.
    # survivor owner (1) != garrison owner (0): garrison_ships=8-9=-1 → owner=1, ships=1.
    # So yes, flip at t=3. owner=1, ships=1.
    # t=4..6: opp produces +1,+1,+1 → 1+3=4.
    # t=7: opp produces +1 → 5. Me 8 attacks: survivor me 8. 5 vs 8 → me wins, 8-5=3, owner=0.
    # t=8..10: me produces +1,+1,+1 → 3+3=6.
    assert both.owner_T == 0
    assert both.ships_T == 6.0
    # Production stream: me holds t=1,2,3 (3 ticks before flip), then opp holds t=4,5,6,7 (4 ticks
    # before retake), then me holds t=8,9,10 (3 ticks). Total: me=6, opp=4.
    assert both.prod_stream.get(0, 0) == 6
    assert both.prod_stream.get(1, 0) == 4


# ---------------------------------------------------------------------------
# 10. End-to-end fast_sim: planet state matches outcome table after step.
# ---------------------------------------------------------------------------


def test_end_to_end_fast_sim_parity():
    """Fire a fleet via the env's action API, let the env compute the
    arrival tick from its own physics, observe it, and verify the outcome
    table predicts the same end-state when given that eta as a fixed arrival.

    This proves the outcome table is bit-exact against the env's per-tick
    semantics (production-then-combat order) — independent of any
    fleet-ETA prediction logic on our side."""
    fleet_ships = 20
    p0 = _planet(0, owner=0, ships=fleet_ships, production=0,
                 x=25.0, y=20.0, radius=1.5)
    p1 = _planet(1, owner=1, ships=8, production=2,
                 x=43.0, y=20.0, radius=1.5)
    obs = {
        "player": 0,
        "planets": [
            (p0.id, p0.owner, p0.x, p0.y, p0.radius, p0.ships, p0.production),
            (p1.id, p1.owner, p1.x, p1.y, p1.radius, p1.ships, p1.production),
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            [p0.id, p0.owner, p0.x, p0.y, p0.radius, p0.ships, p0.production],
            [p1.id, p1.owner, p1.x, p1.y, p1.radius, p1.ships, p1.production],
        ],
        "angular_velocity": 0.0,
        "step": 0,
        "next_fleet_id": 0,
    }
    snap = from_obs(obs, configuration=None, episode_seed=0, num_seats=2)

    # Fire toward p1 (east = angle 0). Env launches at step 0 → step 1.
    snap = fs_step(snap, [[[0, 0.0, fleet_ships]], []])

    # Observe arrival: the env keeps p1's ships changing only on fleet
    # arrival. Each step we record p1's (owner, ships); the moment owner
    # flips or ships drops vs the unperturbed trajectory, that's arrival.
    horizon_after = 4
    arrival_step = None
    max_steps = 30
    for _ in range(max_steps):
        snap = fs_step(snap, [[], []])
        # Arrival happens when the in-flight fleets list goes from 1 → 0.
        if not list(snap.obs.fleets):
            arrival_step = snap.step_idx  # step AFTER the arrival tick
            break
    assert arrival_step is not None, "fleet never arrived"
    # Run a few more ticks to exercise post-capture production.
    for _ in range(horizon_after):
        snap = fs_step(snap, [[], []])

    # Final p1 state from env.
    final_p1 = next(p for p in snap.obs.planets if int(p[0]) == 1)
    env_owner = int(final_p1[1])
    env_ships = int(final_p1[5])

    # The "fleet launched at step 0 → step 1" convention means arrival
    # at env step S corresponds to eta S in the outcome-table convention
    # (1-indexed ticks from the snapshot taken AFTER the launch step).
    # Horizon for the table = arrival_step + horizon_after.
    horizon = arrival_step + horizon_after
    table = enumerate_outcomes(
        initial_owner=1, initial_ships=8 + 1 * 2,  # +production for the launch tick
        production=2, horizon=horizon - 1,
        fixed_arrivals=[Arrival(eta=arrival_step - 1, owner=0, ships=fleet_ships,
                                column_id=None)],
        candidate_arrivals=[],
    )
    row = empty_subset_outcome(table)
    assert env_owner == row.owner_T, f"env={env_owner} vs table={row.owner_T}"
    assert env_ships == int(row.ships_T), \
        f"env={env_ships} vs table={row.ships_T}"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_rejects_overlarge_candidate_set():
    candidates = [Arrival(eta=2, owner=0, ships=5, column_id=i)
                  for i in range(MAX_ENUMERATION_BITS + 1)]
    with pytest.raises(ValueError, match="exceeds enumeration budget"):
        enumerate_outcomes(
            initial_owner=0, initial_ships=10, production=1, horizon=5,
            fixed_arrivals=[], candidate_arrivals=candidates,
        )


def test_rejects_fixed_arrival_with_column_id():
    with pytest.raises(ValueError, match="fixed_arrival has column_id"):
        enumerate_outcomes(
            initial_owner=0, initial_ships=10, production=1, horizon=5,
            fixed_arrivals=[Arrival(eta=2, owner=1, ships=3, column_id=1)],
            candidate_arrivals=[],
        )


def test_rejects_duplicate_candidate_column_ids():
    with pytest.raises(ValueError, match="duplicate column_ids"):
        enumerate_outcomes(
            initial_owner=0, initial_ships=10, production=1, horizon=5,
            fixed_arrivals=[],
            candidate_arrivals=[
                Arrival(eta=2, owner=0, ships=5, column_id=7),
                Arrival(eta=3, owner=0, ships=5, column_id=7),
            ],
        )


def test_winning_subsets_filter():
    candidate = Arrival(eta=2, owner=0, ships=10, column_id=42)
    table = enumerate_outcomes(
        initial_owner=-1, initial_ships=3, production=1, horizon=5,
        fixed_arrivals=[], candidate_arrivals=[candidate],
    )
    wins = winning_subsets(table, my_id=0)
    # Empty subset stays neutral; full subset captures.
    assert () not in wins
    assert (42,) in wins
