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

**STANDARD PROCEDURE (2026-05-25, PI-ratified):**
- 5 games against EACH opponent in a panel
- Episode capped at 250 steps (game truncation; default-cap=500 too slow)
- NO seat switching (focal always P0). Seat asymmetry from FP rounding
  is a tiny signal vs between-strategy gaps; rotating doubles compute
  for negligible measurement gain (PI 2026-05-25). Per Rule 43, panel
  diversity is the real signal — not seat balancing.

Usage:
    # Standard panel A/B (5 games × N opponents = 5N games):
    python scripts/clean_ab.py focal.py opp1.py opp2.py opp3.py opp4.py

    # Single opponent (legacy):
    python scripts/clean_ab.py focal.py opp.py

    # Override defaults if needed:
    python scripts/clean_ab.py focal.py opp.py --seeds 8 --episode-steps 500 --swap-seats
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


def _worker_play(args: tuple[int, str, str, bool, int]) -> dict:
    """Spawn a fresh subprocess that plays ONE game, returns its JSON result."""
    seed, focal_path, opp_path, focal_is_p0, episode_steps = args
    p0_path, p1_path = (focal_path, opp_path) if focal_is_p0 else (opp_path, focal_path)
    code = (
        "import json, sys, time;"
        "sys.path.insert(0, %r);"
        "from kaggle_environments import make;"
        "env = make('orbit_wars', configuration={'seed': %d, 'episodeSteps': %d}, debug=False);"
        "t0 = time.perf_counter();"
        "env.run([%r, %r]);"
        "wall = time.perf_counter() - t0;"
        "final = env.steps[-1];"
        "r0 = final[0]['reward']; r1 = final[1]['reward'];"
        "print(json.dumps({'r0': r0, 'r1': r1, 'n_steps': len(env.steps), 'wall': wall}))"
    ) % (str(REPO), int(seed), int(episode_steps), str(p0_path), str(p1_path))
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


def _run_one_panel_entry(focal: str, opp: str, seeds: int, swap_seats: bool,
                          episode_steps: int, workers: int) -> dict:
    """Run focal vs one opp; return {wins, n, lo, hi, elapsed, per_game}."""
    print(f"== clean_ab focal={Path(focal).name}  opp={Path(opp).name}  "
          f"seeds={seeds}  swap_seats={swap_seats}  episode_steps={episode_steps}  workers={workers} ==")
    tasks: list[tuple[int, str, str, bool, int]] = []
    for s in range(seeds):
        tasks.append((s, focal, opp, True, episode_steps))   # focal as P0
        if swap_seats:
            tasks.append((s, focal, opp, False, episode_steps))  # focal as P1
    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker_play, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"outcome": "error", "stderr": f"worker raised: {type(e).__name__}: {e}"[:400]}
            results.append(r)
            tag = "WIN" if r.get("focal_won") else ("LOSS" if r.get("outcome") in ("p0_win","p1_win") else r.get("outcome","?"))
            print(f"   seed={r.get('seed','?'):>4}  seat={'P0' if r.get('focal_is_p0') else 'P1' if r.get('focal_is_p0') is not None else '-'}  "
                  f"{tag:>7}  steps={r.get('n_steps','-')}  wall={r.get('wall',0):.1f}s")
    wins = sum(1 for r in results if r.get("focal_won"))
    errs = sum(1 for r in results if r.get("outcome") in ("error", "timeout"))
    n = len(results) - errs
    elapsed = time.perf_counter() - t0
    if n == 0:
        print("   ALL ERROR — no usable games\n")
        return {"opp": Path(opp).name, "wins": 0, "n": 0, "lo": 0.0, "hi": 1.0,
                "elapsed": elapsed, "results": results, "errs": errs}
    lo, hi = wilson_ci(wins, n)
    print(f"   focal_wins={wins}/{n} ({100*wins/n:.1f}%)  errs/timeouts={errs}  "
          f"Wilson[{lo:.3f}, {hi:.3f}]  elapsed={elapsed:.0f}s\n")
    return {"opp": Path(opp).name, "wins": wins, "n": n, "lo": lo, "hi": hi,
            "elapsed": elapsed, "results": results, "errs": errs}


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("focal")
    ap.add_argument("opps", nargs="+", help="One or more opponent paths. Default: 5 games against each.")
    ap.add_argument("--seeds", type=int, default=5,
                    help="N seeds per opponent (default 5 per PI standard procedure 2026-05-25)")
    ap.add_argument("--episode-steps", type=int, default=250,
                    help="Episode truncation (default 250 per PI standard procedure 2026-05-25)")
    ap.add_argument("--swap-seats", action="store_true",
                    help="Play each seed twice (both seats). Default OFF per PI standard procedure 2026-05-25.")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    focal = str(Path(args.focal).resolve())
    if not Path(focal).is_file():
        print(f"focal not found: {focal}", file=sys.stderr); return 2

    opps = []
    for o in args.opps:
        p = Path(o).resolve()
        if not p.is_file():
            print(f"opp not found: {p}", file=sys.stderr); return 2
        opps.append(str(p))

    print(f"=== PANEL A/B  focal={Path(focal).name}  n_opp={len(opps)}  "
          f"seeds={args.seeds}  episode_steps={args.episode_steps}  "
          f"swap_seats={args.swap_seats}  ===")
    print()

    t_start = time.perf_counter()
    panel = []
    for opp in opps:
        entry = _run_one_panel_entry(
            focal, opp, args.seeds, args.swap_seats, args.episode_steps, args.workers,
        )
        panel.append(entry)

    total = time.perf_counter() - t_start
    print("=== PANEL SUMMARY ===")
    print(f"{'opponent':<55s}  {'wins':>5s}  {'win%':>5s}  {'Wilson_lo':>9s}  {'Wilson_hi':>9s}")
    pass_count = 0
    for e in panel:
        pct = 100 * e["wins"] / max(1, e["n"])
        ok = "PASS" if e["lo"] >= 0.50 else "----"
        if e["lo"] >= 0.50:
            pass_count += 1
        print(f"{e['opp']:<55s}  {e['wins']:>2d}/{e['n']:<2d}  {pct:>5.1f}  "
              f"{e['lo']:>9.3f}  {e['hi']:>9.3f}  {ok}")
    print(f"\n  passed (Wilson-lo >= 0.50): {pass_count}/{len(panel)}  "
          f"total elapsed={total:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
