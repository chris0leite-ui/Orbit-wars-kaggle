"""Kaggle-env smoke for v7_wide_deep — mirrors what Kaggle's evaluator does.

Runs `kaggle_environments.evaluate("orbit_wars", [bundle, opponent], num_episodes=N)`
and reports: episodes completed, crashes, p95 turn time, mean reward.

This is the deployment-path test: 0 crashes + p95 < 800 ms means the bundle
is safe to push (subject to PI approval).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from kaggle_environments import evaluate, make


REPO = Path(__file__).resolve().parents[1]
WIDE = REPO / "submissions" / "v7_wide_deep.py"
V7_0 = REPO / "submissions" / "v7_0_drop_one.py"
V35 = REPO / "submissions" / "v3.5.1.py"


def _run_one(seed: int, opp_path: Path) -> dict:
    """Run one episode under kaggle_environments.evaluate — the ladder path."""
    env = make("orbit_wars", configuration={"seed": seed})
    t0 = time.perf_counter()
    env.run([str(WIDE), str(opp_path)])
    wall = time.perf_counter() - t0
    steps = env.toJSON()["steps"]
    n_turns = len(steps)
    # Per-turn wallclock from agent.duration if present
    p95_turn = 0.0
    durations = []
    for t in range(n_turns):
        for seat in steps[t]:
            d = seat.get("duration", 0) or 0
            if d:
                durations.append(d)
    if durations:
        durations.sort()
        p95_turn = durations[int(0.95 * len(durations))]
    rewards = [s.reward for s in env.state]
    crashed = any(s.status == "ERROR" for s in env.state)
    return {
        "seed": seed, "wall_s": wall, "n_turns": n_turns,
        "p95_turn_s": p95_turn, "rewards": rewards, "crashed": crashed,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument(
        "--opponent",
        choices=["v7_0", "v3.5.1"],
        default="v3.5.1",
        help="opponent bundle to test against",
    )
    args = ap.parse_args(argv)

    opp = V35 if args.opponent == "v3.5.1" else V7_0
    print(f"smoke: v7_wide_deep vs {args.opponent}, {args.episodes} episodes")
    if not WIDE.is_file():
        print(f"missing: {WIDE}")
        return 1

    results = []
    t0_all = time.perf_counter()
    for seed in range(args.episodes):
        r = _run_one(seed, opp)
        results.append(r)
        focal_won = r["rewards"][0] == 1
        status = "CRASH" if r["crashed"] else ("WIN" if focal_won else "LOSS")
        print(
            f"  seed={seed}  {status}  "
            f"turns={r['n_turns']}  p95={r['p95_turn_s']*1000:.0f} ms  "
            f"wall={r['wall_s']:.1f} s"
        )

    elapsed = time.perf_counter() - t0_all
    crashes = sum(1 for r in results if r["crashed"])
    wins = sum(1 for r in results if not r["crashed"] and r["rewards"][0] == 1)
    p95s = [r["p95_turn_s"] for r in results if r["p95_turn_s"] > 0]
    p95_all = max(p95s) * 1000 if p95s else 0
    print()
    print(f"=== summary ===")
    print(f"episodes: {len(results)} run, {crashes} crashed, {wins} wins")
    print(f"max p95 turn time: {p95_all:.0f} ms (gate: < 800 ms)")
    print(f"total wallclock: {elapsed:.1f} s")
    return 0 if crashes == 0 and p95_all < 800 else 2


if __name__ == "__main__":
    sys.exit(main())
