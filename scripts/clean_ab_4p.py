"""scripts/clean_ab_4p.py — process-isolated 4P A/B harness.

Companion to `scripts/clean_ab.py` (2P version). Solves the same
env-var / sys.modules contamination class, extended to 4P.

The artifact this fixes: `scripts/ab_4p_focal.py` runs `env.run(agents)`
with 4 agent paths in the SAME Python process. `kaggle_environments`
dedups loaded modules by path, so when 3 of 4 slots share the same
`bg_path`, they share module state (`_PS_DEFAULT`, `_TM_DEFAULT`,
`_KT_TABLE`) — the 3 bg agents "coordinate" via shared singletons,
while focal (different path) gets isolated state. Self-play probe
with identical files gave focal 5/8 = 62.5% vs the expected 25%
baseline.

Fix: one subprocess per game. Each subprocess starts with a fresh
`sys.modules`, so all 4 agents load as distinct module instances.
Even when focal_path == bg_path (true self-play), the 4 instances
have independent state.

Usage:
    python scripts/clean_ab_4p.py --focal F.py --bg B.py --seeds 8 --workers 4

The harness runs `seeds × 4` games (each seed played with focal in
each of the 4 seats) and reports the same metrics as `ab_4p_focal.py`
(win rate, rank histogram, per-seat wins, Wilson-lo).

Built-in sanity: if `--focal == --bg`, the harness logs an explicit
warning that the result is a self-play baseline and the win rate
should be ~25%; a tight Wilson CI excluding 0.25 indicates a residual
artifact even with subprocess isolation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _worker_play(args: tuple[int, int, str, str]) -> dict:
    """Spawn a fresh subprocess that plays ONE 4P game.

    `agents = [bg] * 4; agents[focal_seat] = focal` — exactly the
    layout from `ab_4p_focal.py:44-45`. Each subprocess re-loads all
    4 agents from disk into a clean `sys.modules`.
    """
    seed, focal_seat, focal_path, bg_path = args
    # Build the per-seat path tuple. Pass as 4 separate strings to
    # env.run so kaggle_environments treats them independently.
    paths = [bg_path] * 4
    paths[focal_seat] = focal_path
    code = (
        "import json, sys, time;"
        "sys.path.insert(0, %r);"
        "from kaggle_environments import make;"
        "env = make('orbit_wars', configuration={'seed': %d}, debug=False);"
        "t0 = time.perf_counter();"
        "env.run([%r, %r, %r, %r]);"
        "wall = time.perf_counter() - t0;"
        "final = env.steps[-1];"
        "rewards = [s['reward'] for s in final];"
        "n_steps = len(env.steps);"
        "print(json.dumps({"
        "    'rewards': rewards, 'n_steps': n_steps, 'wall': wall"
        "}))"
    ) % (str(REPO), int(seed), *paths)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=1200,
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
    rewards = data["rewards"]
    if any(r is None for r in rewards):
        return {"seed": seed, "focal_seat": focal_seat, "outcome": "error",
                "rewards": rewards}
    # Rank handling — counts STRICT inequality. If focal ties with
    # other players at the top, focal_won = False (we want unique
    # rank-1). The "tied at rank 1" case (no unique winner) is
    # recorded as outcome='tie' so it doesn't count as a win.
    max_reward = max(rewards)
    focal_reward = rewards[focal_seat]
    n_at_max = sum(1 for r in rewards if r >= max_reward)
    if focal_reward < max_reward:
        focal_rank = sum(1 for r in rewards if r > focal_reward) + 1
        focal_won = False
        tied = False
    elif n_at_max == 1:
        focal_rank = 1
        focal_won = True
        tied = False
    else:
        # focal is tied for the top — not a unique win.
        focal_rank = 1
        focal_won = False
        tied = True
    return {
        "seed": seed,
        "focal_seat": focal_seat,
        "rewards": rewards,
        "focal_rank": focal_rank,
        "focal_won": bool(focal_won),
        "tied_at_top": bool(tied),
        "n_steps": data["n_steps"],
        "wall": data["wall"],
    }


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", required=True, help="focal bundle path")
    ap.add_argument("--bg", required=True, help="background bundle path (x3)")
    ap.add_argument("--seeds", type=int, default=4,
                    help="number of seeds (each played in 4 seats)")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    focal = Path(args.focal).resolve()
    bg = Path(args.bg).resolve()
    if not focal.is_file() or not bg.is_file():
        print(f"missing bundle: focal={focal.is_file()} bg={bg.is_file()}",
              file=sys.stderr)
        return 2

    is_self_play = (str(focal) == str(bg))
    print(f"== clean_ab_4p focal={focal.name} bg={bg.name} "
          f"seeds={args.seeds} (×4 seats = {args.seeds * 4} games) "
          f"workers={args.workers} ==")
    if is_self_play:
        print("   [self-play sanity mode — expected win rate ~25%]")

    tasks: list[tuple[int, int, str, str]] = []
    for s in range(args.seed_offset, args.seed_offset + args.seeds):
        for seat in range(4):
            tasks.append((s, seat, str(focal), str(bg)))

    t0 = time.perf_counter()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker_play, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"outcome": "error",
                     "stderr": f"worker raised: {type(e).__name__}: {e}"[:400]}
            results.append(r)
            tag = ("WIN" if r.get("focal_won")
                   else "TIE" if r.get("tied_at_top")
                   else ("RANK%d" % r["focal_rank"]) if "focal_rank" in r
                   else r.get("outcome", "?"))
            print(f"   seed={r.get('seed','?'):>4}  seat={r.get('focal_seat','-')}  "
                  f"{tag:>6}  steps={r.get('n_steps','-')}  "
                  f"wall={r.get('wall',0):.1f}s")

    errs = sum(1 for r in results if r.get("outcome") in ("error", "timeout"))
    wins = sum(1 for r in results if r.get("focal_won"))
    ties = sum(1 for r in results if r.get("tied_at_top"))
    n = len(results) - errs
    if n == 0:
        print("\n   ALL ERROR — no usable games")
        return 1
    lo, hi = _wilson_ci(wins, n)
    rank_hist = {k: sum(1 for r in results if r.get("focal_rank") == k)
                 for k in (1, 2, 3, 4)}
    per_seat = {k: (
        sum(1 for r in results if r.get("focal_seat") == k and r.get("focal_won")),
        sum(1 for r in results if r.get("focal_seat") == k and "focal_rank" in r),
    ) for k in (0, 1, 2, 3)}
    elapsed = time.perf_counter() - t0

    print(f"\n   focal_wins={wins}/{n} ({100*wins/n:.1f}%)  ties_at_top={ties}  "
          f"errs/timeouts={errs}  "
          f"Wilson[{lo:.3f}, {hi:.3f}]  elapsed={elapsed:.0f}s")
    print(f"   rank breakdown: rank1={rank_hist[1]} rank2={rank_hist[2]} "
          f"rank3={rank_hist[3]} rank4={rank_hist[4]}")
    seat_str = " ".join(f"seat{k}={w}/{t}" for k, (w, t) in per_seat.items())
    print(f"   per-seat wins: {seat_str}")

    if is_self_play:
        # Sanity check — Wilson CI must include 0.25.
        if lo <= 0.25 <= hi:
            print(f"   [self-play] Wilson CI [{lo:.3f}, {hi:.3f}] includes "
                  f"0.25 → harness PARITY OK")
        else:
            print(f"   [self-play] Wilson CI [{lo:.3f}, {hi:.3f}] EXCLUDES "
                  f"0.25 → ARTIFACT STILL PRESENT")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
