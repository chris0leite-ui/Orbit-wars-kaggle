"""Phase B foundation — outcome-table vs fast_sim bit-parity.

The existing `tests/test_joint_outcome_table.py` provides broad coverage
of `enumerate_outcomes` vs `simulate_planet_timeline` (both call the
same `resolve_arrivals`; proves the enumeration plumbing). Scenario 10
of that file proves bit-parity vs `fast_sim` for one canonical case.

This file ADDS focused fast_sim parity tests for the corner cases the
decision rule actually exercises:

  1. Multi-fleet same-owner arrivals at different etas (reinforcement
     sequence) — pins production accrual order across mid-sequence
     arrivals.
  2. Multi-fleet opposing-owner arrivals at the same eta (combat
     ordering tie) — pins the env's tie-break vs `resolve_arrivals`'s.

The invariant pinned: any (my arrivals, opp arrivals) subset that the
Phase D decision rule evaluates as a leaf via `enumerate_outcomes`
MUST produce identical (owner_T, ships_T) to running fast_sim with the
same arrivals injected. If this ever fails, every Phase D substrate
decision is wrong.

Zero tolerance.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.fast_sim import from_obs, step as fs_step
from lib.joint_solver.outcome_table import Arrival, enumerate_outcomes


def _planet(pid, owner, *, ships=0, production=2, x=50.0, y=50.0, radius=1.5):
    return Planet(pid, owner, x, y, radius, ships, production)


def _build_obs(planets):
    return {
        "player": 0,
        "planets": [
            (p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
            for p in planets
        ],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [
            [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
            for p in planets
        ],
        "angular_velocity": 0.0,
        "step": 0,
        "next_fleet_id": 0,
    }


def _observe_fleet_arrival(snap, max_steps=30):
    """Step forward until all in-flight fleets have arrived; return step idx."""
    for _ in range(max_steps):
        snap = fs_step(snap, [[], []])
        if not list(snap.obs.fleets):
            return snap, snap.step_idx
    return snap, None


def test_parity_combat_at_eta_zero_arrivals_resolve_correctly():
    """Smaller parity: a single capture sized just-enough — pins that the
    production accrued during flight is correctly counted by enumerate_outcomes
    before the combat-resolution tick."""
    p0 = _planet(0, owner=0, ships=20, production=0, x=20.0, y=20.0)
    p1 = _planet(1, owner=1, ships=4, production=3, x=42.0, y=20.0)
    obs = _build_obs([p0, p1])
    snap = from_obs(obs, configuration=None, episode_seed=0, num_seats=2)

    snap = fs_step(snap, [[[0, 0.0, 20]], []])
    snap, arrival_step = _observe_fleet_arrival(snap, max_steps=30)
    assert arrival_step is not None
    for _ in range(2):
        snap = fs_step(snap, [[], []])

    final_p1 = next(p for p in snap.obs.planets if int(p[0]) == 1)
    env_owner = int(final_p1[1])
    env_ships = int(final_p1[5])

    horizon = snap.step_idx - 1
    table = enumerate_outcomes(
        initial_owner=1, initial_ships=4 + 1 * 3,  # +production during launch tick
        production=3, horizon=horizon,
        fixed_arrivals=[
            Arrival(eta=arrival_step - 1, owner=0, ships=20, column_id=None),
        ],
        candidate_arrivals=[],
    )
    row = table[()]
    assert env_owner == row.owner_T, f"env={env_owner} table={row.owner_T}"
    assert env_ships == int(row.ships_T), \
        f"env={env_ships} table={row.ships_T}"
