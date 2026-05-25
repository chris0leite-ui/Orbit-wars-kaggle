"""Quick A/B harness: 5 games × 250 steps × no seat swap (2026-05-25 standard).

New A/B standard per PI: 5 games per opponent, capped at 250 environment
steps each, focal always P0 (no seat swap). Trades CRN coverage for ~25x
faster signal — typical full run 15 games × 50s ≈ 12 min vs 32-seed
balanced-pair eval at ~30 min.

Usage:
    python scripts/ab_quick.py FOCAL [--opps OPP1,OPP2,...] [--seeds 0,1,2,3,4]
                               [--max-steps 250] [--workers 1]

Defaults:
    --opps  = submissions/buildup_planner_phi1_only.py,
              submissions/baseline_joint_aggr_consolidated_orbitfix.py,
              v7_0
    --seeds = 0,1,2,3,4

Each opponent's row: wins/5 + Wilson 95% CI.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make
from fast import _load_callable, resolve_agent_spec, wilson_ci


def _play_one(args: tuple) -> tuple[int, str, str, float, int, int]:
    """Run one game. Returns (seed, opp_name, outcome, wallclock_s, n_steps,
    focal_max_turn_ms). focal always P0."""
    seed, focal_path, opp_path, opp_name, max_steps = args
    t0 = time.perf_counter()
    focal = _load_callable(focal_path)
    opp = _load_callable(opp_path)

    turn_ms: list[float] = []

    def focal_timed(obs, configuration=None):
        ts = time.perf_counter()
        try:
            return focal(obs, configuration)
        finally:
            turn_ms.append((time.perf_counter() - ts) * 1000.0)

    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": int(max_steps)},
               debug=False)
    try:
        env.run([focal_timed, opp])
    except Exception as exc:
        return (seed, opp_name, "error:" + type(exc).__name__,
                time.perf_counter() - t0, 0, 0)

    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    if r0 is None and r1 is None:
        outcome = "no_reward"
    elif r0 == r1:
        outcome = "draw"
    elif (r0 or 0) > (r1 or 0):
        outcome = "p0_win"
    else:
        outcome = "p1_win"

    elapsed = time.perf_counter() - t0
    max_ms = int(max(turn_ms)) if turn_ms else 0
    return (seed, opp_name, outcome, elapsed, len(env.steps), max_ms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="5-game / 250-step / no-swap A/B (PI standard 2026-05-25)."
    )
    parser.add_argument("focal", help="focal agent path or registry name")
    parser.add_argument(
        "--opps",
        default=(
            "submissions/buildup_planner_phi1_only.py,"
            "submissions/baseline_joint_aggr_consolidated_orbitfix.py,"
            "v7_0"
        ),
        help="comma-separated opp names/paths",
    )
    parser.add_argument(
        "--seeds", default="0,1,2,3,4",
        help="comma-separated seeds (default 5 seeds: 0,1,2,3,4)",
    )
    parser.add_argument("--max-steps", type=int, default=250,
                        help="episode step cap per game (default 250)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel workers (default 1 — keeps CPU contention low)")
    args = parser.parse_args(argv)

    _, focal_path = resolve_agent_spec(args.focal)
    opps: list[tuple[str, str]] = []
    for name in args.opps.split(","):
        name = name.strip()
        if not name:
            continue
        display, path = resolve_agent_spec(name)
        opps.append((display, path))
    seeds = [int(s) for s in args.seeds.split(",")]

    # Build the (seed × opp) job list.
    jobs = []
    for opp_name, opp_path in opps:
        for s in seeds:
            jobs.append((s, focal_path, opp_path, opp_name,
                         args.max_steps))

    print(f"== ab_quick  focal={Path(focal_path).name}  "
          f"opps={len(opps)}  seeds={len(seeds)}  cap={args.max_steps} ==")
    t_start = time.perf_counter()

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_play_one, jobs)
    else:
        results = [_play_one(j) for j in jobs]

    # Aggregate per opp.
    print()
    print(f"  {'opp':<55s}  {'wins':>5s}  {'wr':>6s}  "
          f"{'Wlo':>6s}  {'Whi':>6s}  {'avg_steps':>9s}  {'max_ms':>7s}")
    total_w = 0
    total_n = 0
    for opp_name, _opp_path in opps:
        rows = [r for r in results if r[1] == opp_name]
        wins = sum(1 for r in rows if r[2] == "p0_win")
        n = len(rows)
        wr = wins / n if n else 0.0
        lo, hi = wilson_ci(wins, n)
        avg_steps = sum(r[4] for r in rows) / n if n else 0
        max_ms = max(r[5] for r in rows) if rows else 0
        total_w += wins
        total_n += n
        print(f"  {opp_name:<55s}  {wins:>2d}/{n:<2d}  {100*wr:>5.1f}%  "
              f"{lo:>5.3f}  {hi:>5.3f}  {avg_steps:>9.1f}  {max_ms:>7d}")

    print()
    lo, hi = wilson_ci(total_w, total_n)
    print(f"  TOTAL: {total_w}/{total_n}  ({100*total_w/total_n:.1f}%)  "
          f"Wilson [{lo:.3f}, {hi:.3f}]  elapsed={time.perf_counter()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
