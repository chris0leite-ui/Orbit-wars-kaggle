#!/usr/bin/env python
"""scripts/ab_quick.py — fast preview A/B with no seat swap, step-truncated.

Default protocol (set by user 2026-05-28):
  - 5 seeds, focal always at P0 (no seat swap)
  - episodeSteps = 250 (game truncated to half-length for speed)
  - parallel via ProcessPoolExecutor
  - works as single-opponent A/B (--vs path.py) or multi-opponent panel
    (--vs path1.py,path2.py,...)

Output: per-seed line, then per-opponent W-L-D + winrate.

Env vars (e.g. BASELINE_PV_ETA=1) are inherited by worker processes
through the spawning shell.

CAVEAT: Q6 (Rule 16) — step-250 truncation is NOT the live-ladder
metric (which is the full 500-step game). Use this as a fast preview;
confirm any lift candidate at the full 500-step length before submit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass
class Result:
    opp: str
    seed: int
    outcome: str          # "p0_win" | "p1_win" | "draw" | "error"
    rewards: tuple
    n_steps: int


def _play(args: tuple[str, str, int, int]) -> Result:
    """Worker: play one truncated game, focal at P0."""
    focal_path, opp_path, seed, episode_steps = args
    from kaggle_environments import make
    from fast import _load_callable, resolve_agent_spec

    _, focal_resolved = resolve_agent_spec(focal_path)
    _, opp_resolved = resolve_agent_spec(opp_path)
    p0 = _load_callable(focal_resolved)
    p1 = _load_callable(opp_resolved)
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": episode_steps},
        debug=False,
    )
    try:
        env.run([p0, p1])
    except Exception:
        return Result(
            opp=opp_path, seed=seed, outcome="error",
            rewards=(None, None),
            n_steps=len(env.steps) if hasattr(env, "steps") else 0,
        )
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    if r0 is None or r1 is None:
        outcome = "error"
    elif r0 > r1:
        outcome = "p0_win"
    elif r1 > r0:
        outcome = "p1_win"
    else:
        outcome = "draw"
    return Result(opp=opp_path, seed=seed, outcome=outcome,
                  rewards=(r0, r1), n_steps=len(env.steps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--focal", required=True, help="focal agent path")
    ap.add_argument("--vs", required=True,
                    help="opponent path(s), comma-separated for panel mode")
    ap.add_argument("--max-seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=250,
                    help="episodeSteps override (default 250)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="seeds = range(seed_base, seed_base+max_seeds)")
    args = ap.parse_args()

    opps = [o.strip() for o in args.vs.split(",") if o.strip()]
    seeds = list(range(args.seed_base, args.seed_base + args.max_seeds))
    jobs = [
        (args.focal, opp, seed, args.max_steps)
        for opp in opps for seed in seeds
    ]

    print(f"== ab_quick  focal={args.focal} ==")
    print(f"   opps={opps}")
    print(f"   seeds={seeds}  episodeSteps={args.max_steps}  "
          f"workers={args.workers}")
    if os.environ.get("BASELINE_PV_ETA"):
        print(f"   BASELINE_PV_ETA={os.environ['BASELINE_PV_ETA']}")
    print("")

    results: dict[str, list[Result]] = {opp: [] for opp in opps}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_play, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results[r.opp].append(r)
            tag = {"p0_win": "W", "p1_win": "L",
                   "draw": "D", "error": "E"}[r.outcome]
            short_opp = r.opp.split("/")[-1]
            print(f"   [{tag}] {short_opp:<50s} seed={r.seed:<3d} "
                  f"steps={r.n_steps:<4d} rewards={r.rewards}")
    elapsed = time.perf_counter() - t0

    print(f"\n== summary  ({elapsed:.1f}s wallclock) ==")
    for opp in opps:
        rs = results[opp]
        w = sum(1 for r in rs if r.outcome == "p0_win")
        l = sum(1 for r in rs if r.outcome == "p1_win")
        d = sum(1 for r in rs if r.outcome == "draw")
        e = sum(1 for r in rs if r.outcome == "error")
        n = w + l + d
        wr = (w / n * 100.0) if n > 0 else 0.0
        short = opp.split("/")[-1]
        print(f"   vs {short:<50s} W-L-D = {w}-{l}-{d}  "
              f"(err={e})  WR={wr:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
