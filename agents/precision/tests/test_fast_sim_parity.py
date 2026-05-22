"""Verify fast_sim.rollout produces same outputs as prediction.rollout.

Bit-parity on:
  - final_score_per_player
  - final_owner per planet
  - final_ships per planet

Plus benchmark: fast_sim should be measurably faster.
"""
from __future__ import annotations

import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from agents.precision import fast_sim, intercept, prediction


def _world(seed: int, advance_steps: int = 20):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": advance_steps + 50})
    env.reset(2)
    for _ in range(advance_steps):
        env.step([[], []])
    obs = env.steps[-1][0].observation
    return intercept.parse_world({
        "player": 0, "step": int(obs.step),
        "planets": list(obs.planets),
        "fleets": list(obs.fleets),
        "angular_velocity": obs.angular_velocity,
        "initial_planets": list(obs.initial_planets),
        "comets": list(obs.comets),
        "comet_planet_ids": list(obs.comet_planet_ids),
        "remainingOverageTime": 60.0,
    })


def test_fast_sim_parity_empty_plan():
    """No planned shots: pure production/projection rollout."""
    for seed in range(8):
        for adv in (5, 30, 60):
            w = _world(seed, advance_steps=adv)
            ref = prediction.rollout(w, [], horizon_steps=200)
            fast = fast_sim.rollout(w, [], horizon_steps=200)
            assert ref.final_score_per_player == fast.final_score_per_player, (
                f"seed={seed} adv={adv}: score mismatch ref={ref.final_score_per_player} "
                f"fast={fast.final_score_per_player}"
            )
            assert ref.final_owner == fast.final_owner, f"seed={seed} adv={adv}: owner mismatch"
            assert ref.final_ships == fast.final_ships, f"seed={seed} adv={adv}: ships mismatch"
    print("  empty-plan parity: 24/24 ok")


def test_fast_sim_parity_with_plan():
    """With our planned shots: arrivals + combat in the rollout."""
    for seed in range(6):
        w = _world(seed, advance_steps=15)
        my = [p for p in w["planets"] if p.owner == 0]
        if not my:
            continue
        src = my[0]
        targets = [p for p in w["planets"] if p.owner != 0][:3]
        plan = []
        for tgt in targets:
            shot = intercept.find_shot(src, tgt, max(1, min(src.ships // 3, tgt.ships + 5)), w)
            if shot is not None:
                plan.append(shot)
                break  # one shot per source, simple
        if not plan:
            continue
        ref = prediction.rollout(w, plan, horizon_steps=200)
        fast = fast_sim.rollout(w, plan, horizon_steps=200)
        assert ref.final_score_per_player == fast.final_score_per_player, (
            f"seed={seed}: score mismatch ref={ref.final_score_per_player} "
            f"fast={fast.final_score_per_player}"
        )
        assert ref.final_owner == fast.final_owner, f"seed={seed}: owner mismatch"
        assert ref.final_ships == fast.final_ships, f"seed={seed}: ships mismatch"
    print("  with-plan parity: ok")


def test_fast_sim_speed():
    """Fast sim should be measurably faster on mid-game boards."""
    w = _world(0, advance_steps=40)
    # Warm
    prediction.rollout(w, [], horizon_steps=200)
    fast_sim.rollout(w, [], horizon_steps=200)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        prediction.rollout(w, [], horizon_steps=200)
    ref_t = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        fast_sim.rollout(w, [], horizon_steps=200)
    fast_t = (time.perf_counter() - t0) / n
    speedup = ref_t / max(fast_t, 1e-9)
    print(f"  ref={ref_t*1000:.3f} ms/call, fast={fast_t*1000:.3f} ms/call, speedup={speedup:.1f}x")
    assert fast_t <= ref_t, "fast_sim must not be slower than prediction.rollout"


if __name__ == "__main__":
    test_fast_sim_parity_empty_plan()
    test_fast_sim_parity_with_plan()
    test_fast_sim_speed()
    print("\nAll fast_sim parity tests passed.")
