"""Random-elimination gate: agent must beat `random` 100% via elimination
(i.e. opp ends with 0 planets), at n=16 over random seeds + random seat
assignment.

Per PI 2026-05-23: "first achieve 100% against nearest in random seats.
only accept wins by elimination."
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _play_one(args):
    """Play ONE game (clean subprocess) and return result dict."""
    seed, focal_path, focal_seat, opp = args
    p0_path, p1_path = (focal_path, opp) if focal_seat == 0 else (opp, focal_path)
    code = (
        "import json, sys, time;"
        "sys.path.insert(0, %r);"
        "from kaggle_environments import make;"
        "env = make('orbit_wars', configuration={'seed': %d}, debug=False);"
        "t0 = time.perf_counter();"
        "env.run([%r, %r]);"
        "wall = time.perf_counter() - t0;"
        "final = env.steps[-1];"
        "r0 = final[0]['reward']; r1 = final[1]['reward'];"
        "obs_final = final[0]['observation'];"
        # Count planets by owner at terminal step.
        "planets = obs_final.get('planets', []) if isinstance(obs_final, dict) else getattr(obs_final, 'planets', []);"
        "n_p0 = sum(1 for p in planets if int(p[1]) == 0);"
        "n_p1 = sum(1 for p in planets if int(p[1]) == 1);"
        "n_steps = len(env.steps);"
        "print(json.dumps({'r0': r0, 'r1': r1, 'n_p0': n_p0, 'n_p1': n_p1, 'n_steps': n_steps, 'wall': wall}))"
    ) % (str(REPO), int(seed), str(p0_path), str(p1_path))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as e:
        return {"seed": seed, "focal_seat": focal_seat, "outcome": "timeout",
                "stderr": f"timed out after {e.timeout}s"}
    out = (proc.stdout or "").strip().splitlines()
    line = next((l for l in reversed(out) if l.startswith("{")), "")
    if not line:
        return {"seed": seed, "focal_seat": focal_seat, "outcome": "error",
                "stderr": (proc.stderr or "")[:400]}
    d = json.loads(line)
    r0, r1 = d["r0"], d["r1"]
    focal_r = r0 if focal_seat == 0 else r1
    opp_r = r1 if focal_seat == 0 else r0
    focal_planets = d["n_p0"] if focal_seat == 0 else d["n_p1"]
    opp_planets = d["n_p1"] if focal_seat == 0 else d["n_p0"]
    if focal_r is None or opp_r is None:
        outcome = "error"
    elif focal_r > opp_r:
        outcome = "win"
    elif focal_r < opp_r:
        outcome = "loss"
    else:
        outcome = "draw"
    win_by_elim = (outcome == "win") and (opp_planets == 0)
    win_by_score = (outcome == "win") and (opp_planets > 0)
    return {
        "seed": int(seed),
        "focal_seat": int(focal_seat),
        "outcome": outcome,
        "focal_planets": int(focal_planets),
        "opp_planets": int(opp_planets),
        "win_by_elim": bool(win_by_elim),
        "win_by_score": bool(win_by_score),
        "n_steps": int(d["n_steps"]),
        "wall": float(d["wall"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("focal", help="Path to focal agent main.py")
    ap.add_argument("--n", type=int, default=16, help="Number of games (default 16)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rng-seed", type=int, default=2026, help="Meta-seed for game seeds + seat assignment")
    ap.add_argument("--opp", default="random",
                    help="Opponent: 'random' (default), 'starter', or path to agent main.py")
    args = ap.parse_args()

    focal = Path(args.focal).resolve()
    if not focal.exists():
        print(f"focal not found: {focal}", file=sys.stderr)
        return 2

    # Resolve opp: builtins stay as bare names; path strings resolved + validated.
    if args.opp in ("random", "starter"):
        opp = args.opp
    else:
        opp_path = Path(args.opp).resolve()
        if not opp_path.exists():
            print(f"opp not found: {opp_path}", file=sys.stderr)
            return 2
        opp = str(opp_path)

    rng = random.Random(args.rng_seed)
    tasks = []
    for _ in range(args.n):
        seed = rng.randint(0, 100000)
        focal_seat = rng.randint(0, 1)
        tasks.append((seed, str(focal), focal_seat, opp))

    print(f"== random-elim-gate focal={focal.name}  opp={opp}  n={args.n}  workers={args.workers} ==", flush=True)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_play_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "ELIM" if r.get("win_by_elim") else (
                "WIN(score)" if r.get("win_by_score") else r.get("outcome", "?").upper()
            )
            print(f"   seed={r.get('seed','?'):>6}  seat=P{r.get('focal_seat','?')}  "
                  f"{tag:<10}  steps={r.get('n_steps','?')}  "
                  f"my_planets={r.get('focal_planets','?')}  opp_planets={r.get('opp_planets','?')}  "
                  f"wall={r.get('wall', 0):.1f}s", flush=True)
    elapsed = time.perf_counter() - t0
    n = len(results)
    n_wins = sum(1 for r in results if r.get("outcome") == "win")
    n_elims = sum(1 for r in results if r.get("win_by_elim"))
    n_score_wins = sum(1 for r in results if r.get("win_by_score"))
    n_losses = sum(1 for r in results if r.get("outcome") == "loss")
    n_draws = sum(1 for r in results if r.get("outcome") == "draw")
    n_errs = n - n_wins - n_losses - n_draws

    print(f"\n   wins={n_wins}/{n}  (elim={n_elims}, by_score={n_score_wins})  "
          f"losses={n_losses}  draws={n_draws}  errs={n_errs}  elapsed={elapsed:.0f}s",
          flush=True)
    if n_elims == n:
        print(f"   ✅ GATE PASS — 100% win-by-elimination", flush=True)
        return 0
    else:
        print(f"   ❌ GATE FAIL — {n - n_elims} games not won by elimination", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
