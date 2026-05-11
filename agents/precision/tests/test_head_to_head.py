"""Head-to-head benchmarks: precision agent vs baselines."""
from __future__ import annotations

import math
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from agents.precision.main import agent as precision_agent
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent


def run_match(a, b, seed: int, episode_steps: int = 500) -> int:
    """Return 1 if a wins, -1 if b wins, 0 draw."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": episode_steps})
    env.run([a, b])
    final = env.steps[-1]
    s_a = final[0].reward
    s_b = final[1].reward
    if s_a is None:
        # Need to compute from state
        obs = final[0].observation
        scores = [0, 0]
        for p in obs.planets:
            if p[1] in (0, 1):
                scores[p[1]] += p[5]
        for f in obs.fleets:
            scores[f[1]] += f[6]
        if scores[0] > scores[1]: return 1
        if scores[0] < scores[1]: return -1
        return 0
    if s_a is None or s_b is None: return 0
    if s_a > s_b: return 1
    if s_a < s_b: return -1
    return 0


def benchmark(name, opponent, n_seeds=20, episode_steps=200):
    wins = draws = losses = 0
    total_time = 0.0
    for seed in range(n_seeds):
        t0 = time.perf_counter()
        # Mirror match: also play the other side to remove position bias.
        r1 = run_match(precision_agent, opponent, seed, episode_steps=episode_steps)
        r2 = -run_match(opponent, precision_agent, seed, episode_steps=episode_steps)
        for r in (r1, r2):
            if r > 0: wins += 1
            elif r < 0: losses += 1
            else: draws += 1
        total_time += time.perf_counter() - t0
    total = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total else 0
    print(f"vs {name}: wins={wins} draws={draws} losses={losses} → win-rate={win_rate:.1%}  ({total_time:.1f}s for {total} games)")
    return win_rate


def test_vs_random():
    rate = benchmark("random", random_agent, n_seeds=10, episode_steps=200)
    assert rate >= 0.70, f"Win rate vs random only {rate:.1%}"


def test_vs_starter():
    rate = benchmark("starter", starter_agent, n_seeds=10, episode_steps=200)
    assert rate >= 0.55, f"Win rate vs starter only {rate:.1%}"


if __name__ == "__main__":
    print("Running head-to-head benchmarks (this will take a few minutes)...")
    test_vs_random()
    test_vs_starter()
    print("\nAll head-to-head tests passed.")
