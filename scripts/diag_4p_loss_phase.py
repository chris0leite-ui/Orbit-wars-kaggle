"""4P loss-phase diagnostic on the un-blinded panel.

Runs a focal agent vs a heterogeneous 4P background over seeds × seat
rotations, tracks the focal's planet-share and material-share (planet
ships + in-flight ships, focal / all-players) at checkpoint steps, and
splits the curves by WIN vs LOSS. The step where the win/loss curves
diverge is the phase we lose.

Serial (one game at a time) to avoid the torch-contention trap that
corrupts multi-worker eval (see audit/2026-06-13-referee-blindness-…).

Usage:
    python scripts/diag_4p_loss_phase.py --focal <path> \
        --bg producer,v7_0,nearest --seeds 7,11,13,21,42,101,202,303
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.play4p import _resolve  # noqa: E402

CHECKPOINTS = [25, 50, 100, 150, 250, 400]


def _share_at(step_obs, focal_pid: int):
    """planet-share, material-share for focal_pid from one seat's obs dict."""
    planets = step_obs.get("planets", []) or []
    fleets = step_obs.get("fleets", []) or []
    pl_mine = sum(1 for p in planets if int(p[1]) == focal_pid and int(p[0]) >= 0)
    pl_tot = sum(1 for p in planets if int(p[1]) >= 0 and int(p[0]) >= 0)
    mat_mine = sum(p[5] for p in planets if int(p[1]) == focal_pid)
    mat_mine += sum(f[6] for f in fleets if int(f[1]) == focal_pid)
    mat_tot = sum(p[5] for p in planets if int(p[1]) >= 0)
    mat_tot += sum(f[6] for f in fleets if int(f[1]) >= 0)
    return (pl_mine / pl_tot if pl_tot else 0.0,
            mat_mine / mat_tot if mat_tot else 0.0)


def run_game(focal_path, bg_paths, seed, focal_seat):
    from kaggle_environments import make
    agents = list(bg_paths)
    agents.insert(focal_seat, focal_path)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    steps = env.steps
    rewards = [s.reward for s in env.state]
    fr = rewards[focal_seat]
    won = fr is not None and fr == max(r for r in rewards if r is not None)
    curve = {}
    for cp in CHECKPOINTS:
        if cp >= len(steps):
            continue
        # any seat's observation carries the global board
        obs = steps[cp][0].get("observation", {}) or {}
        curve[cp] = _share_at(obs, focal_seat)
    return won, curve, len(steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True)
    ap.add_argument("--bg", required=True, help="comma-separated 3 bg agents")
    ap.add_argument("--seeds", required=True)
    args = ap.parse_args()

    focal = _resolve(args.focal) if "/" not in args.focal else args.focal
    bg = [_resolve(x) for x in args.bg.split(",")]
    assert len(bg) == 3, "4P needs exactly 3 background agents"
    seeds = [int(s) for s in args.seeds.split(",")]

    wins, losses = [], []
    n_first = 0
    for sd in seeds:
        for seat in range(4):
            won, curve, nsteps = run_game(focal, bg, sd, seat)
            (wins if won else losses).append(curve)
            n_first += int(won)
            print(f"  seed={sd} seat={seat}  {'WIN ' if won else 'loss'}  "
                  f"steps={nsteps}", file=sys.stderr)

    n = len(wins) + len(losses)
    print(f"\nfocal={Path(focal).name}  bg=[{args.bg}]")
    print(f"first-place: {n_first}/{n} ({100*n_first/n:.1f}%)\n")
    print(f"{'step':>5} | {'WIN planet/mat share':>22} | {'LOSS planet/mat share':>22} | n W/L")
    for cp in CHECKPOINTS:
        w = [c[cp] for c in wins if cp in c]
        l = [c[cp] for c in losses if cp in c]
        def avg(xs, i): return sum(x[i] for x in xs) / len(xs) if xs else float("nan")
        print(f"{cp:>5} | {avg(w,0):>9.3f} / {avg(w,1):>9.3f} | "
              f"{avg(l,0):>9.3f} / {avg(l,1):>9.3f} | {len(w)}/{len(l)}")
    print("\nRead: the step where LOSS material-share falls clearly below "
          "0.25 (even quarter) and below the WIN curve is the phase we lose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
