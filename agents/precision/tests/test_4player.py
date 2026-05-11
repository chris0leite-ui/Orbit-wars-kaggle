"""4-player smoke test: confirm precision_v2 doesn't crash in a 4p match."""
from __future__ import annotations

import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
from agents.precision.main import agent as precision_v2


def test_4p_no_crash_precision_wins_or_survives():
    t0 = time.perf_counter()
    env = make("orbit_wars", configuration={"seed": 0, "episodeSteps": 80})
    env.run([precision_v2, random_agent, random_agent, random_agent])
    elapsed = time.perf_counter() - t0
    obs = env.steps[-1][0].observation
    scores = [0, 0, 0, 0]
    for p in obs.planets:
        if p[1] in (0, 1, 2, 3):
            scores[p[1]] += p[5]
    for f in obs.fleets:
        if f[1] in (0, 1, 2, 3):
            scores[f[1]] += f[6]
    print(f"4p game: {elapsed:.1f}s for {len(env.steps)} steps, scores={scores}")
    avg_ms = elapsed * 1000 / max(1, len(env.steps))
    print(f"  avg turn: {avg_ms:.0f}ms (4 agents called per step)")
    # Precision should at least not crash; ideally win, or be top-2.
    rank = sorted(range(4), key=lambda i: -scores[i])
    assert rank.index(0) <= 1, f"precision finished rank {rank.index(0)+1}/4 in 4p"


if __name__ == "__main__":
    test_4p_no_crash_precision_wins_or_survives()
    print("\n4-player smoke test passed.")
