"""4P A/B harness — focal bundle vs 3 background bundles, seat-balanced.

Closes the validation gap exposed by Phase 4: 8W/0L in 2P didn't translate to
the live ladder (4P games). For each seed, runs the focal in each of 4 seats
to remove seat bias.

Usage:
    python -m scripts.ab_4p_focal --focal submissions/_phase4_step1_FND.py \
        --bg submissions/baseline_joint_aggr_consolidated.py \
        --seeds 8 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt
from pathlib import Path

from kaggle_environments import make


def _wilson_lo(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denom


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _play_one(args: tuple) -> tuple[int, int, list[float], int]:
    """Run one 4P game with focal at seat `focal_seat`. Returns
    (seed, focal_seat, rewards, final_step)."""
    seed, focal_seat, focal_path, bg_path = args
    agents = [bg_path] * 4
    agents[focal_seat] = focal_path
    env = make("orbit_wars", configuration={"seed": seed})
    env.run(agents)
    rewards = [s.reward for s in env.state]
    final_step = env.state[0].observation.step
    return (seed, focal_seat, rewards, final_step)


def _focal_rank(rewards: list[float], focal_seat: int) -> int:
    """Rank of focal (1 = best). Higher reward = better.
    For Orbit Wars 4P, rewards are typically [1, -1, -1, -1] for winner
    or rank-based reward structure. We sort descending."""
    if any(r is None for r in rewards):
        return -1
    sorted_rs = sorted(rewards, reverse=True)
    return sorted_rs.index(rewards[focal_seat]) + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True, help="focal bundle path")
    ap.add_argument("--bg", required=True, help="background bundle path (x3)")
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of seeds (each played 4 times — one per seat)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="starting seed (default 0; use 100 for an unseen bag)")
    args = ap.parse_args(argv)

    focal = Path(args.focal).resolve()
    bg = Path(args.bg).resolve()
    if not focal.is_file() or not bg.is_file():
        print(f"missing bundle: focal={focal.is_file()} bg={bg.is_file()}",
              file=sys.stderr)
        return 1

    print(f"focal={focal.name} sha256={_file_hash(focal)}")
    print(f"bg   ={bg.name} sha256={_file_hash(bg)}")
    print(f"seeds={args.seeds} × 4 seats = {args.seeds * 4} games "
          f"(seed range {args.seed_offset}..{args.seed_offset + args.seeds - 1})")

    pairs: list[tuple] = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        for focal_seat in range(4):
            pairs.append((seed, focal_seat, str(focal), str(bg)))

    results: list[tuple] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_play_one, p) for p in pairs]
        for f in as_completed(futs):
            results.append(f.result())
    elapsed = time.perf_counter() - t0

    # Aggregate by rank.
    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, -1: 0}
    per_seat_wins = {0: 0, 1: 0, 2: 0, 3: 0}
    per_seat_total = {0: 0, 1: 0, 2: 0, 3: 0}
    errors = 0
    for seed, focal_seat, rewards, final_step in results:
        rank = _focal_rank(rewards, focal_seat)
        rank_counts[rank] += 1
        per_seat_total[focal_seat] += 1
        if rank == 1:
            per_seat_wins[focal_seat] += 1
        elif rank == -1:
            errors += 1

    total = len(results) - errors
    wins = rank_counts[1]
    wr = wins / total if total else 0.0
    wlo = _wilson_lo(wins, total)

    print()
    print(f"=== 4P FFA Results ({total} games, {errors} errors) ===")
    print(f"Focal win rate: {wins}/{total} = {100*wr:.1f}%  "
          f"(random baseline: 25.0%)")
    print(f"Wilson 95% lower: {wlo:.3f}  (gate: ≥ 0.300 → directional, "
          f"≥ 0.400 → strong)")
    print(f"Rank breakdown: rank1={rank_counts[1]} rank2={rank_counts[2]} "
          f"rank3={rank_counts[3]} rank4={rank_counts[4]}")
    print(f"Per-seat wins: " + " ".join(
        f"seat{s}={per_seat_wins[s]}/{per_seat_total[s]}" for s in range(4)))
    print(f"Wallclock: {elapsed:.1f}s ({elapsed/len(results):.2f}s/game)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
