"""Phase 0.5 — physics sanity probe.

Plays the topology-on bundle vs the no-topology baseline on 20 seeds,
using `check_fleet_outcomes` to inventory per-emit fleet fate via the
EXACT `predict_fleet_fate` primitive (parity-tested against env).
Subprocess-per-seed isolates the env-var-leak class.

Pass criterion (per /root/.claude/plans/composed-noodling-riddle.md
Phase 0.5): 0 sun + 0 oob across all 20 seeds.

Output: per-seed counts + aggregate summary. Logs any seed with
sun_hits > 0 OR oob_hits > 0.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [
    42, 1, 7, 13, 31, 100, 17, 23,           # canonical A/B harness seeds
    2, 5, 8, 11, 19, 29, 37, 53,             # primes for diversity
    101, 137, 211, 257,                       # additional primes
]
FOCAL = REPO / "submissions" / "analytical_phase_c.py"
BASELINE = REPO / "submissions" / "_phase4_step1_FND.py"


# Parse `check_fleet_outcomes` formatted output: lines like
#   "                  target : 102  (100.0%)"
# We capture each row's name + count.
COUNT_ROW_RE = re.compile(r"^\s+(target|planet|sun|oob|timeout|no_target_resolved)\s*:\s*(\d+)")


def _run_one(seed: int, timeout_s: int = 180) -> dict:
    """Run check_fleet_outcomes for one seed in a fresh subprocess."""
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "scripts.check_fleet_outcomes",
        "--seed", str(seed),
        "--focal", str(FOCAL),
        "--baseline", str(BASELINE),
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        return {"seed": seed, "error": "timeout", "wall_s": time.time() - t0}

    out = r.stdout + "\n" + r.stderr
    counts: dict | None = None
    for line in out.splitlines():
        m = COUNT_ROW_RE.match(line)
        if m:
            if counts is None:
                counts = {}
            counts[m.group(1)] = int(m.group(2))
    return {
        "seed": seed,
        "rc": r.returncode,
        "counts": counts,
        "wall_s": round(time.time() - t0, 1),
        "stderr_tail": r.stderr[-200:] if r.returncode != 0 else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20,
                    help="Number of seeds to probe; uses prefix of DEFAULT_SEEDS")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args(argv)

    seeds = DEFAULT_SEEDS[:args.seeds]
    assert FOCAL.exists(), f"focal bundle missing: {FOCAL}"
    assert BASELINE.exists(), f"baseline bundle missing: {BASELINE}"

    print(f"phase0.5 physics probe: {len(seeds)} seeds, workers={args.workers}")
    print(f"focal:    {FOCAL.relative_to(REPO)}")
    print(f"baseline: {BASELINE.relative_to(REPO)}")
    t_start = time.time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, s, args.timeout): s for s in seeds}
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            print(f"  seed={res['seed']:5d}  rc={res.get('rc')}  "
                  f"counts={res.get('counts')}  wall={res.get('wall_s')}s")

    # Aggregate.
    agg = {"target": 0, "planet": 0, "sun": 0, "oob": 0,
           "timeout": 0, "no_target_resolved": 0}
    errors = []
    bad_seeds = []
    for res in results:
        c = res.get("counts")
        if c is None:
            errors.append(res)
            continue
        for k in agg:
            agg[k] += int(c.get(k, 0))
        if c.get("sun", 0) > 0 or c.get("oob", 0) > 0:
            bad_seeds.append((res["seed"], c.get("sun", 0), c.get("oob", 0)))

    total_emits = sum(agg.values())
    target_rate = (agg["target"] / total_emits * 100) if total_emits else 0.0
    print()
    print(f"=== AGGREGATE ({len(results) - len(errors)}/{len(seeds)} seeds OK, "
          f"wallclock {time.time() - t_start:.0f}s) ===")
    print(f"  total emits:  {total_emits}")
    print(f"  target:       {agg['target']:5d}  ({target_rate:.1f}%)")
    print(f"  planet:       {agg['planet']:5d}  (path-blocked by other planet)")
    print(f"  sun:          {agg['sun']:5d}  <-- GATE: must be 0")
    print(f"  oob:          {agg['oob']:5d}  <-- GATE: must be 0")
    print(f"  timeout:      {agg['timeout']:5d}")
    print(f"  no_tgt:       {agg['no_target_resolved']:5d}")

    if errors:
        print(f"\n!! {len(errors)} seeds errored:")
        for e in errors:
            print(f"   seed={e['seed']}  rc={e.get('rc')}  err={e.get('stderr_tail')}")

    if bad_seeds:
        print(f"\n!! {len(bad_seeds)} seeds with sun/oob > 0:")
        for seed, sun, oob in bad_seeds:
            print(f"   seed={seed}  sun={sun}  oob={oob}")
        return 1

    print("\n>> Phase 0.5 GATE: 0 sun + 0 oob across all seeds — GREEN <<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
