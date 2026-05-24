"""Small A/B driver for momentum_strike — PI-spec methodology.

≤ 8 games, NO seat swap (focal always P0), reports:
  - wins / losses / draws / errors
  - elimination-by-turn-250 count (focal won AND game ended <= 250)
  - avg turn of decisive game (rounded)
  - per-seed outcomes

Usage:
    python scripts/momentum_strike_ab.py [--vs <opponent>] [--seeds N] [--workers N]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse fast.py's play_one + agent-spec resolution.
from fast import play_one, resolve_agent_spec  # noqa: E402


def _run_one(args):
    seed, focal_path, opp_path = args
    result = play_one(seed, focal_path, opp_path, record_timing=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="momentum_strike 8-game A/B")
    parser.add_argument("--focal", default="agents/momentum_strike",
                        help="focal agent spec (default: agents/momentum_strike)")
    parser.add_argument("--vs", default="agents/simple/nearest.py",
                        help="opponent spec (default: nearest)")
    parser.add_argument("--seeds", type=int, default=8,
                        help="number of seeds (default: 8)")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first seed (default: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers (default: 4)")
    parser.add_argument("--max-turns", type=int, default=250,
                        help="cap for early-elimination tracking (default: 250)")
    args = parser.parse_args(argv)

    focal_name, focal_path = resolve_agent_spec(args.focal)
    opp_name, opp_path = resolve_agent_spec(args.vs)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"== momentum_strike A/B ==")
    print(f"  focal: {focal_name}  (always P0, no seat swap)")
    print(f"  opp:   {opp_name}")
    print(f"  seeds: {seeds}  workers: {args.workers}")
    print(f"  early-elim cap: turn {args.max_turns}")
    print()

    tasks = [(s, focal_path, opp_path) for s in seeds]
    t0 = time.perf_counter()
    results = []
    if args.workers <= 1:
        for t in tasks:
            results.append(_run_one(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one, t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())
    elapsed = time.perf_counter() - t0

    # Order results by seed for the table.
    results.sort(key=lambda r: r.seed)

    wins = losses = draws = errors = 0
    early_elims = 0
    win_turns = []
    print(f"{'seed':>5}  {'outcome':<10}  {'turns':>6}  {'early?':>7}")
    for r in results:
        focal_won = (r.outcome == "p0_win")
        focal_lost = (r.outcome == "p1_win")
        is_early = focal_won and r.n_steps <= args.max_turns
        if r.outcome == "error":
            errors += 1
            tag = "ERROR"
        elif focal_won:
            wins += 1
            win_turns.append(r.n_steps)
            tag = "WIN"
            if is_early:
                early_elims += 1
        elif focal_lost:
            losses += 1
            tag = "LOSS"
        else:
            draws += 1
            tag = "DRAW"
        flag = "YES" if is_early else ""
        print(f"{r.seed:>5}  {tag:<10}  {r.n_steps:>6}  {flag:>7}")

    n = len(results)
    print()
    print(f"  wins:   {wins}/{n}    losses: {losses}/{n}    "
          f"draws: {draws}/{n}    errors: {errors}/{n}")
    print(f"  win rate: {wins/n:.3f}  "
          f"early-elim (<= {args.max_turns}): {early_elims}/{n}")
    if win_turns:
        print(f"  avg turn-of-win: {mean(win_turns):.0f}  "
              f"(min={min(win_turns)}, max={max(win_turns)})")
    print(f"  elapsed: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
