"""Head-to-head: v2 planner vs v1 (frozen) baseline.

Gate: v2 wins ≥55% across mirror matches.
"""
from __future__ import annotations

import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.precision import intercept, planner
from kaggle_environments import make


def _make_agent(plan_fn):
    """Wrap a planner.plan_turn_* function as an `agent(obs)` callable."""
    def agent(obs):
        t0 = time.perf_counter()
        try:
            world = intercept.parse_world(obs)
        except Exception:
            return []
        if world["step"] == 0:
            return []
        deadline = t0 + 0.85
        try:
            plan = plan_fn(world, deadline=deadline)
        except Exception:
            plan = []
        return planner.emit_actions(plan)
    return agent


def _run(a, b, seed, eps=80):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": eps})
    env.run([a, b])
    obs = env.steps[-1][0].observation
    scores = [0, 0]
    for p in obs.planets:
        if p[1] in (0, 1):
            scores[p[1]] += p[5]
    for f in obs.fleets:
        scores[f[1]] += f[6]
    if scores[0] > scores[1]:
        return 1, scores
    if scores[0] < scores[1]:
        return -1, scores
    return 0, scores


def test_v2_beats_v1():
    v1 = _make_agent(planner.plan_turn_v1)
    v2 = _make_agent(planner.plan_turn)

    wins = draws = losses = 0
    t0 = time.perf_counter()
    for seed in range(3):
        for our_as_first in (True, False):
            if our_as_first:
                r, s = _run(v2, v1, seed)
            else:
                r, s = _run(v1, v2, seed)
                r = -r
                s = [s[1], s[0]]
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
            print(f"  seed={seed} v2_first={our_as_first}: "
                  f"{'WIN' if r>0 else 'LOSS' if r<0 else 'DRAW'} "
                  f"v2={s[0]} v1={s[1]}")

    total = wins + draws + losses
    rate = (wins + 0.5 * draws) / total
    print(f"\nv2 vs v1: {wins}W-{draws}D-{losses}L  win-rate={rate:.0%}  ({time.perf_counter()-t0:.0f}s)")
    assert rate >= 0.55, f"v2 win-rate {rate:.0%} below 55% gate"


if __name__ == "__main__":
    test_v2_beats_v1()
    print("\nv2-vs-v1 test passed.")
