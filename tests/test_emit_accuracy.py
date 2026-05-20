"""Emit-accuracy regression test.

The PI report on submission `52854094` (μ≈806): "ships do not hit
targets." This test makes that claim testable: drive a real
kaggle_environments game with the analytical agent at P0, and after
EVERY call to `solve_turn`, verify that each emitted move's intended
target — recovered via `MpcDiagnostics.emitted_targets` — would be
HIT according to `predict_fleet_fate(... wait_N=column.wait_N)`.

If any emitted move's predicted outcome ≠ "target", that's a "ship
that doesn't hit its target" — the exact failure mode the PI named.

Scope (kept small to fit pytest budgets):
  - 4 seeds: 0, 1, 7, 42.
  - 60 steps per seed (covers the opening + early mid-game).
  - vs trajectory baseline at P1.

If post-fix this test passes at ≥99% landing rate, we accept the
"ships hit targets" claim is closed; the remaining μ gap is in the
value function / opp model / strategy layers, not in the trajectory
plumbing.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
from kaggle_environments import make

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fast import _load_callable  # noqa: E402

from lib.intent import World  # noqa: E402
from lib.joint_solver import mpc  # noqa: E402
from lib.joint_solver.mpc import solve_turn  # noqa: E402
from lib.trajectory import predict_fleet_fate  # noqa: E402


def _trajectory_baseline():
    """Load the trajectory baseline agent as a callable."""
    return _load_callable(os.path.join(REPO, "agents", "baseline", "main.py"))


def _validate_emitted_move(world, et: dict) -> tuple[bool, str]:
    """Return (landed, outcome_str) for one emitted move dict from
    MpcDiagnostics.emitted_targets. Looks up src/tgt via world.planets_by_id.
    """
    src = world.planets_by_id.get(int(et["src_id"]))
    tgt = world.planets_by_id.get(int(et["tgt_id"]))
    if src is None or tgt is None:
        return False, f"src/tgt not in world (src_id={et['src_id']}, "\
                      f"tgt_id={et['tgt_id']})"
    fate = predict_fleet_fate(
        src, tgt, float(et["angle"]), int(et["ships"]), world,
        wait_N=int(et["wait_N"]),
    )
    return (fate.outcome == "target"), fate.outcome


def _run_seed(seed: int, max_steps: int = 60) -> dict:
    """Run one analytical-vs-baseline game for `max_steps` env steps,
    capturing per-turn emit-validity stats.

    Returns dict with: seed, n_emits, n_target, n_miss, miss_outcomes.
    """
    baseline_agent = _trajectory_baseline()

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    n_emits = 0
    n_target = 0
    miss_outcomes: list[str] = []
    miss_details: list[dict] = []

    # Step manually. Each iteration: read obs0, call solve_turn with
    # diagnostics, validate every emitted target, send the action pair
    # to env.step.
    for step_idx in range(max_steps):
        state = env.steps[-1]
        if state[0].status == "DONE" or state[1].status == "DONE":
            break

        obs0 = state[0].observation
        obs1 = state[1].observation

        # solve_turn with diagnostics. obs is a Struct; mpc handles it.
        moves_a, diag = solve_turn(obs0, return_diagnostics=True)

        # Re-build a World view for trajectory validation — independent
        # of the agent's internal world construction.
        world = World.from_obs(obs0)

        for et in diag.emitted_targets:
            n_emits += 1
            landed, outcome = _validate_emitted_move(world, et)
            if landed:
                n_target += 1
            else:
                miss_outcomes.append(outcome)
                miss_details.append({
                    "seed": seed, "step": step_idx,
                    "src_id": et["src_id"], "tgt_id": et["tgt_id"],
                    "outcome": outcome,
                })

        # Baseline takes its move at P1.
        action_b = baseline_agent(obs1, env.configuration)

        # env.step advances ONE turn given the pair of actions.
        env.step([moves_a, action_b])

    return {
        "seed": seed, "n_emits": n_emits, "n_target": n_target,
        "n_miss": len(miss_outcomes), "miss_outcomes": miss_outcomes,
        "miss_details": miss_details,
    }


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_emit_accuracy_per_seed(seed: int):
    """For each seed: at every step, every emitted move's predicted
    outcome MUST be 'target' (i.e., the fleet lands on its intended
    target). A non-target outcome (sun, oob, planet (wrong one), or
    timeout) is "a ship that does not hit its target."
    """
    result = _run_seed(seed, max_steps=60)
    n = result["n_emits"]
    hits = result["n_target"]
    misses = result["n_miss"]
    if n == 0:
        pytest.skip(f"seed={seed}: no moves emitted in 60 turns")

    landing_rate = hits / max(1, n)
    # Pre-fix: PI claim says ships don't hit; expect landing < 99%.
    # Post-fix: should be ≥99%.
    detail_summary = ", ".join(
        f"{o}:{result['miss_outcomes'].count(o)}"
        for o in sorted(set(result["miss_outcomes"]))
    )
    assert landing_rate >= 0.99, (
        f"seed={seed}: emit landing rate {landing_rate:.2%} "
        f"({hits}/{n} on target, {misses} missed). "
        f"Miss outcomes: {{{detail_summary}}}. "
        f"First 5 details: {result['miss_details'][:5]}")


def test_emit_accuracy_aggregate():
    """Aggregate gate across the 4 seeds. Even at small per-seed sample
    sizes, the cross-seed total must clear 99% landing.
    """
    seeds = [0, 1, 7, 42]
    total_n = 0
    total_hits = 0
    per_seed: list[tuple[int, int, int]] = []  # (seed, hits, total)
    for s in seeds:
        r = _run_seed(s, max_steps=60)
        total_n += r["n_emits"]
        total_hits += r["n_target"]
        per_seed.append((s, r["n_target"], r["n_emits"]))

    if total_n == 0:
        pytest.skip("no moves emitted across 4 seeds × 60 turns")

    rate = total_hits / max(1, total_n)
    detail = ", ".join(f"seed={s}:{h}/{n}" for s, h, n in per_seed)
    assert rate >= 0.99, (
        f"Aggregate landing rate {rate:.2%} ({total_hits}/{total_n}). "
        f"Per-seed: {detail}")
