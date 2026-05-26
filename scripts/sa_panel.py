"""Run the SA solver across the 32-archetype panel; report ceiling per-archetype.

For each archetype in data/seed_panel_128.json (one random seed per archetype,
controlled by --rng-seed), spawn `scripts/sa_solo_solver.py` as a subprocess
and collect its JSON result. Output: archetype × (ROI score, SA best score,
gap %) table sorted by archetype name, plus an overall summary.

This is the headline diagnostic: how much room is there above ROI per
archetype-class?
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO / "data" / "seed_panel_128.json"
SOLVER_PATH = REPO / "scripts" / "sa_solo_solver.py"


def _load_archetype_seeds(per_archetype: int, rng_seed: int):
    """Same selector as solo_bench.py — deterministic random per archetype."""
    with PANEL_PATH.open() as f:
        panel = json.load(f)
    by_arc = defaultdict(list)
    for entry in panel["panel"]:
        by_arc[entry["archetype"]].append(int(entry["seed"]))
    rng = random.Random(rng_seed)
    out = []
    for arc in sorted(by_arc.keys()):
        pool = list(by_arc[arc])
        rng.shuffle(pool)
        for s in pool[:per_archetype]:
            out.append((s, arc))
    return out


def _worker(args: tuple[int, str, int, int, float, float, int]) -> dict:
    seed, archetype, steps, iterations, t0, cooling, rng_seed = args
    cmd = [
        sys.executable, str(SOLVER_PATH),
        "--seed", str(seed),
        "--steps", str(steps),
        "--iterations", str(iterations),
        "--t0", str(t0),
        "--cooling", str(cooling),
        "--rng-seed", str(rng_seed),
        "--quiet",
    ]
    t0_wall = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return {"seed": seed, "archetype": archetype, "outcome": "timeout",
                "wall": time.perf_counter() - t0_wall}
    # JSON is the LAST stderr line starting with "{".
    err_lines = (proc.stderr or "").strip().splitlines()
    json_line = next((l for l in reversed(err_lines) if l.startswith("{")), "")
    if not json_line:
        return {"seed": seed, "archetype": archetype, "outcome": "error",
                "stderr": (proc.stderr or "")[:500],
                "wall": time.perf_counter() - t0_wall}
    data = json.loads(json_line)
    data["archetype"] = archetype
    data["outcome"] = "ok"
    data["wall"] = time.perf_counter() - t0_wall
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--per-archetype", type=int, default=1)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--t0", type=float, default=500.0)
    ap.add_argument("--cooling", type=float, default=0.99)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pairs = _load_archetype_seeds(args.per_archetype, args.rng_seed)
    print(f"[sa_panel] {len(pairs)} (seed × archetype) tasks, "
          f"iter={args.iterations}, steps={args.steps}, workers={args.workers}",
          file=sys.stderr)

    tasks = [(seed, arc, args.steps, args.iterations,
              args.t0, args.cooling, args.rng_seed)
             for seed, arc in pairs]

    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            elapsed = time.perf_counter() - t0
            print(f"[sa_panel] {done}/{len(tasks)}  arc={r.get('archetype'):<42s}  "
                  f"seed={r.get('seed')}  outcome={r.get('outcome')}  "
                  f"wall={r.get('wall', 0):.1f}s  elapsed={elapsed:.0f}s",
                  file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f"[sa_panel] DONE in {elapsed:.0f}s\n", file=sys.stderr)

    # ---- Tables ----
    results.sort(key=lambda r: r.get("archetype", ""))
    print(f"{'archetype':<42s} {'seed':>5s} {'ROI':>8s} {'SA':>8s} "
          f"{'gap':>+8s} {'gap%':>7s} {'init→best emissions':>22s} {'wall_s':>8s}")
    print("-" * 110)
    for r in results:
        if r.get("outcome") != "ok":
            print(f"{r.get('archetype'):<42s} {r.get('seed'):>5d} "
                  f"  {r.get('outcome'):>30s}")
            continue
        print(f"{r['archetype']:<42s} {r['seed']:>5d} "
              f"{r['initial_score']:>8.0f} {r['best_score']:>8.0f} "
              f"{r['gap_abs']:>+8.0f} {r['gap_pct']:>+6.1f}% "
              f"{r['initial_n_emissions']:>10d}→{r['best_n_emissions']:<10d}  "
              f"{r.get('wall', 0):>8.1f}")

    ok = [r for r in results if r.get("outcome") == "ok"]
    if ok:
        roi_total  = sum(r["initial_score"] for r in ok)
        sa_total   = sum(r["best_score"] for r in ok)
        gaps_pct   = [r["gap_pct"] for r in ok]
        nonzero_lift = [g for g in gaps_pct if g > 1.0]
        print("\n=== summary ===")
        print(f"  n archetypes:        {len(ok)}/{len(pairs)}")
        print(f"  ROI mean ships:      {roi_total / len(ok):>8.0f}")
        print(f"  SA  mean ships:      {sa_total  / len(ok):>8.0f}")
        print(f"  mean gap (%):        {statistics.mean(gaps_pct):>+7.1f}%")
        print(f"  median gap (%):      {statistics.median(gaps_pct):>+7.1f}%")
        print(f"  archetypes w/ lift:  {len(nonzero_lift)}/{len(ok)}")
        if nonzero_lift:
            print(f"  mean lift (% where lifted): {statistics.mean(nonzero_lift):>+6.1f}%")
        print(f"  max gap:             {max(gaps_pct):>+7.1f}%")

    # JSON dump to stderr for later analysis.
    print("\n[sa_panel] raw JSON:", file=sys.stderr)
    print(json.dumps(results), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
