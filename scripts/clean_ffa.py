"""clean_ffa.py — process-isolated 4-player FFA harness.

Same isolation rationale as clean_ab.py (one game per subprocess so env-var
gates can't leak between games), but for 4P: the focal agent plays one seat
against three copies of a background agent (default: vanilla producer, which
reads no PRODUCER_PLUS_* gates, so focal-bundle env baking is safe in the
shared process). The focal seat rotates across seeds so seat effects are
balanced. The metric is first-place rate (4P rewards are win-only: the three
non-winners all score -1).

Usage:
    python scripts/clean_ffa.py focal.py --seeds 8 --workers 2
    python scripts/clean_ffa.py focal.py --background path.py --seed-start 8
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
DEFAULT_BACKGROUND = str(REPO / "agents" / "producer" / "main.py")


def _worker_play(args: tuple[int, str, str, int]) -> dict:
    seed, focal_path, background_path, focal_seat = args
    paths = [background_path] * 4
    paths[focal_seat] = focal_path
    code = (
        "import json, sys, time;"
        "sys.path.insert(0, %r);"
        "from kaggle_environments import make;"
        "env = make('orbit_wars', configuration={'seed': %d}, debug=False);"
        "t0 = time.perf_counter();"
        "env.run(%r);"
        "wall = time.perf_counter() - t0;"
        "rw = [s['reward'] for s in env.state];"
        "print(json.dumps({'rewards': rw, 'n_steps': len(env.steps), 'wall': wall}))"
    ) % (str(REPO), int(seed), paths)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as e:
        return {"seed": seed, "focal_seat": focal_seat, "outcome": "timeout",
                "stderr": f"timed out after {e.timeout}s"}
    out = (proc.stdout or "").strip().splitlines()
    line = next((l for l in reversed(out) if l.startswith("{")), "")
    if not line:
        return {"seed": seed, "focal_seat": focal_seat, "outcome": "error",
                "stderr": (proc.stderr or "")[:400]}
    data = json.loads(line)
    rw = data["rewards"]
    if any(r is None for r in rw):
        return {"seed": seed, "focal_seat": focal_seat, "outcome": "error",
                "stderr": f"None reward: {rw}"}
    focal_won = rw[focal_seat] == max(rw) and rw.count(max(rw)) == 1
    return {"seed": seed, "focal_seat": focal_seat, "outcome": "done",
            "focal_won": bool(focal_won), "rewards": rw,
            "n_steps": data["n_steps"], "wall": data["wall"]}


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("--background", default=DEFAULT_BACKGROUND)
    ap.add_argument("--seeds", type=int, default=8,
                    help="N seeds; focal seat = seed %% 4 (rotates)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    focal = str(Path(args.focal).resolve())
    background = str(Path(args.background).resolve())
    if not Path(focal).is_file():
        print(f"focal not found: {focal}", file=sys.stderr); return 2
    if not Path(background).is_file():
        print(f"background not found: {background}", file=sys.stderr); return 2

    print(f"== clean_ffa focal={Path(focal).name}  background={Path(background).name}x3  "
          f"seeds={args.seeds}  workers={args.workers} ==")
    tasks = [
        (s, focal, background, s % 4)
        for s in range(args.seed_start, args.seed_start + args.seeds)
    ]
    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker_play, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"outcome": "error", "stderr": f"worker raised: {type(e).__name__}: {e}"[:400]}
            results.append(r)
            tag = ("WIN" if r.get("focal_won") else
                   "LOSS" if r.get("outcome") == "done" else r.get("outcome", "?"))
            print(f"   seed={r.get('seed','?'):>4}  seat={r.get('focal_seat','-')}  "
                  f"{tag:>7}  steps={r.get('n_steps','-')}  wall={r.get('wall',0):.0f}s")
    wins = sum(1 for r in results if r.get("focal_won"))
    errs = sum(1 for r in results if r.get("outcome") in ("error", "timeout"))
    n = len(results) - errs
    if n == 0:
        print("\n   ALL ERROR — no usable games")
        for r in results[:3]:
            print("   stderr:", r.get("stderr", "")[:200])
        return 1
    lo, hi = wilson_ci(wins, n)
    print(f"\n   focal_first_place={wins}/{n} ({100*wins/n:.1f}%)  errs/timeouts={errs}  "
          f"Wilson[{lo:.3f}, {hi:.3f}]  (4P random baseline = 25%)  "
          f"elapsed={time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
