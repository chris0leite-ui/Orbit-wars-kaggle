"""scripts/clean_ab.py — process-isolated A/B harness.

Why this exists: env-var pollution. Variant agent dirs use
`os.environ.setdefault(...)` at module-load time, then read env at
import. When two agents are loaded in the same process by fast.py /
quick_ab.py, the SECOND agent's `setdefault` is a no-op for already-set
keys → both agents end up running with the FIRST-loaded agent's env.
Module caching (`agents.baseline.chooser_trajectory` etc.) also locks
the FIRST-load constants for the lifetime of the worker.

This harness sidesteps both by running ONE game per Python subprocess.
Each subprocess starts with whatever env you pass (or none), loads the
agents fresh, and exits — no caching, no pollution.

Usage:
    python scripts/clean_ab.py focal.py opp.py --seeds 8 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _worker_play(args: tuple[int, str, str, bool]) -> dict:
    """Spawn a fresh subprocess that plays ONE game, returns its JSON result."""
    seed, focal_path, opp_path, focal_is_p0 = args
    p0_path, p1_path = (focal_path, opp_path) if focal_is_p0 else (opp_path, focal_path)
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
        "print(json.dumps({'r0': r0, 'r1': r1, 'n_steps': len(env.steps), 'wall': wall}))"
    ) % (str(REPO), int(seed), str(p0_path), str(p1_path))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ},  # clean shell env per game
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as e:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "timeout",
                "stderr": f"timed out after {e.timeout}s"}
    out = (proc.stdout or "").strip().splitlines()
    # The kaggle_environments registry prints chatter to stdout; the JSON is the LAST line.
    line = next((l for l in reversed(out) if l.startswith("{")), "")
    if not line:
        return {"seed": seed, "focal_is_p0": focal_is_p0, "outcome": "error",
                "stderr": (proc.stderr or "")[:400]}
    data = json.loads(line)
    r0, r1 = data["r0"], data["r1"]
    if r0 is None or r1 is None:
        outcome = "error"
    elif r0 > r1:
        outcome = "p0_win"
    elif r1 > r0:
        outcome = "p1_win"
    else:
        outcome = "draw"
    focal_won = (focal_is_p0 and outcome == "p0_win") or (not focal_is_p0 and outcome == "p1_win")
    return {
        "seed": seed,
        "focal_is_p0": focal_is_p0,
        "outcome": outcome,
        "focal_won": bool(focal_won),
        "n_steps": data["n_steps"],
        "wall": data["wall"],
    }


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    # Line-buffer stdout so prints flush per-game even under shell redirect.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("opp")
    ap.add_argument("--seeds", type=int, default=8, help="N seeds; each played twice (both seats)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    focal = str(Path(args.focal).resolve())
    opp = str(Path(args.opp).resolve())
    if not Path(focal).is_file():
        print(f"focal not found: {focal}", file=sys.stderr); return 2
    if not Path(opp).is_file():
        print(f"opp not found: {opp}", file=sys.stderr); return 2

    print(f"== clean_ab focal={Path(focal).name}  opp={Path(opp).name}  "
          f"seeds={args.seeds} (×2 seats = {2*args.seeds} games)  workers={args.workers} ==")

    tasks: list[tuple[int, str, str, bool]] = []
    for s in range(args.seeds):
        tasks.append((s, focal, opp, True))   # focal as P0
        tasks.append((s, focal, opp, False))  # focal as P1
    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker_play, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                # Don't let a single worker explosion kill the whole A/B.
                r = {"outcome": "error", "stderr": f"worker raised: {type(e).__name__}: {e}"[:400]}
            results.append(r)
            tag = "WIN" if r.get("focal_won") else ("LOSS" if r.get("outcome") in ("p0_win","p1_win") else r.get("outcome","?"))
            print(f"   seed={r.get('seed','?'):>4}  seat={'P0' if r.get('focal_is_p0') else 'P1' if r.get('focal_is_p0') is not None else '-'}  "
                  f"{tag:>7}  steps={r.get('n_steps','-')}  wall={r.get('wall',0):.1f}s")
    wins = sum(1 for r in results if r.get("focal_won"))
    errs = sum(1 for r in results if r.get("outcome") in ("error", "timeout"))
    n = len(results) - errs
    if n == 0:
        print("\n   ALL ERROR — no usable games")
        return 1
    lo, hi = wilson_ci(wins, n)
    elapsed = time.perf_counter() - t0
    print(f"\n   focal_wins={wins}/{n} ({100*wins/n:.1f}%)  errs/timeouts={errs}  "
          f"Wilson[{lo:.3f}, {hi:.3f}]  elapsed={elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
