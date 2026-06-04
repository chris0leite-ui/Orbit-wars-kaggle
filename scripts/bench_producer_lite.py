"""Speed gate for lib.producer_lite.producer_lite_policy.

Collects real in-game observations (by running a short game), then times the
policy on each board. Asserts mean < 3 ms/call (the rollout hot-path budget;
within ~1.5x Tier-0 lite_greedy). Also reports the p95 and max.

Usage:
    python scripts/bench_producer_lite.py [--seeds 4] [--budget-ms 3.0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.producer_lite import producer_lite_policy
from lib.opp_model import lite_greedy_policy


def _collect_boards(seeds):
    """Run cheap self-play games; return list of (obs-dict) snapshots.

    Board source is the lite_greedy wrapper agent (fast, ~1-2ms/turn) — it
    produces realistic mid/late-game positions (captured planets, in-flight
    fleets) without the heavy champion's ~500ms/turn lookahead cost, so the
    speed gate iterates in seconds, not minutes.
    """
    from kaggle_environments import make

    boards = []
    agent_path = "agents/lite_greedy/main.py"
    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
        env.run([agent_path, agent_path])
        for step in env.steps:
            for seat in (0, 1):
                obs = step[seat]["observation"]
                # Coerce to a plain dict with the player field set.
                d = dict(obs)
                d["player"] = seat
                if d.get("planets"):
                    boards.append(d)
    return boards


def _time_policy(policy, boards):
    times = []
    out_total = 0
    for d in boards:
        t0 = time.perf_counter()
        moves = policy(d)
        times.append((time.perf_counter() - t0) * 1000.0)
        out_total += len(moves)
    times.sort()
    n = len(times)
    mean = sum(times) / n if n else 0.0
    p95 = times[int(0.95 * n)] if n else 0.0
    return mean, p95, times[-1] if times else 0.0, out_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--budget-ms", type=float, default=3.0)
    args = ap.parse_args()

    boards = _collect_boards(range(args.seeds))
    print(f"collected {len(boards)} real boards from {args.seeds} games")

    pl_mean, pl_p95, pl_max, pl_moves = _time_policy(producer_lite_policy, boards)
    lg_mean, lg_p95, lg_max, lg_moves = _time_policy(lite_greedy_policy, boards)

    print(f"producer_lite: mean={pl_mean:.3f}ms p95={pl_p95:.3f}ms "
          f"max={pl_max:.3f}ms  ({pl_moves} total moves emitted)")
    print(f"lite_greedy:   mean={lg_mean:.3f}ms p95={lg_p95:.3f}ms "
          f"max={lg_max:.3f}ms  ({lg_moves} total moves emitted)")
    print(f"ratio producer_lite/lite_greedy mean = "
          f"{(pl_mean / lg_mean) if lg_mean else float('inf'):.2f}x")

    ok = pl_mean < args.budget_ms
    print(f"\nSPEED GATE: mean {pl_mean:.3f}ms {'<' if ok else '>='} "
          f"{args.budget_ms}ms  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
