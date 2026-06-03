"""Reframe B.1 self-play runner — pv_eta vs pv_eta with accepted-trace.

For each game in parallel:
  1. Set per-worker env vars (BASELINE_PV_ETA=1, BASELINE_WALLCLOCK_MS,
     BASELINE_ACCEPTED_TRACE pointing at this game's accepted.jsonl).
  2. Load `agents/baseline_pv_eta_probe/main.py`.
  3. Run env.run([agent, agent]) at the given seed.
  4. Persist replay.jsonl: per-tick {step, focal0_ships_total,
     focal1_ships_total, planets, fleets}.

The chooser's trace_accepted hook writes accepted.jsonl directly from
inside the worker process; the runner only handles replay.jsonl.

CLI:
    python scripts/probe_pveta_selfplay.py \
        --games 25 --seed 100 \
        --out-dir audit/2026-05-29-pveta-probe-data \
        --wallclock-ms 100 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_agent_callable(agent_path: Path):
    """Load the agent's `agent` symbol from a main.py path. Mirrors the
    convention in scripts/generate_selfplay_replays.py."""
    spec = importlib.util.spec_from_file_location(
        "_probe_agent_module", str(agent_path),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe_agent_module"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _focal_ships_total(obs: dict, seat: int) -> int:
    """Sum ships owned by `seat` across planets + in-flight fleets at
    this observation snapshot. Planet schema:
    (id, owner, x, y, radius, ships, production); Fleet schema:
    (id, owner, x, y, angle, from_planet_id, ships)."""
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    total = 0
    for p in planets:
        if int(p[1]) == seat:
            total += int(p[5])
    for f in fleets:
        if int(f[1]) == seat:
            total += int(f[6])
    return total


def _run_one_game(task: tuple) -> dict:
    """Worker: one self-play game; returns a small summary dict for
    parent-process logging. The heavy artifacts (replay.jsonl,
    accepted.jsonl) are written to disk from this worker."""
    seed, out_dir_str, wallclock_ms, agent_path_str = task
    out_dir = Path(out_dir_str)
    game_dir = out_dir / f"game-{seed:08d}"
    game_dir.mkdir(parents=True, exist_ok=True)

    # Env vars must be set BEFORE importing the agent so chooser
    # module-load gates pick them up.
    os.environ["BASELINE_WALLCLOCK_MS"] = str(int(wallclock_ms))
    os.environ["BASELINE_ACCEPTED_TRACE"] = str(game_dir / "accepted.jsonl")
    # The probe wrapper sets BASELINE_PV_ETA / BASELINE_ML_LAMBDA /
    # peak orbitfix preamble via setdefault — already correct, no
    # override needed here.

    from kaggle_environments import make  # noqa: E402

    agent_fn = _load_agent_callable(Path(agent_path_str))

    env = make(
        "orbit_wars", configuration={"seed": seed}, debug=False,
    )
    t0 = time.perf_counter()
    env.run([agent_fn, agent_fn])
    elapsed_s = time.perf_counter() - t0

    # Persist replay.jsonl: per-tick global state from seat-0's
    # observation (orbit_wars carries global state in each seat's POV).
    n_steps_written = 0
    replay_path = game_dir / "replay.jsonl"
    with replay_path.open("w") as fh:
        for t, step_seats in enumerate(env.steps):
            seat0 = step_seats[0] if step_seats else {}
            obs = seat0.get("observation") or {}
            planets = obs.get("planets") or []
            fleets = obs.get("fleets") or []
            rec = {
                "step": int(obs.get("step", t)),
                "focal0_ships_total": _focal_ships_total(obs, 0),
                "focal1_ships_total": _focal_ships_total(obs, 1),
                # Snapshots: keep planets as raw tuples (small; 12-40
                # entries) so the probe can resolve owner-at-launch.
                "planets": [list(p) for p in planets],
                "fleets": [list(f) for f in fleets],
            }
            fh.write(json.dumps(rec) + "\n")
            n_steps_written += 1

    accepted_path = game_dir / "accepted.jsonl"
    n_accepted = 0
    if accepted_path.exists():
        with accepted_path.open() as fh:
            for _ in fh:
                n_accepted += 1

    return {
        "seed": int(seed),
        "n_steps": n_steps_written,
        "n_accepted": n_accepted,
        "elapsed_s": elapsed_s,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, required=True,
                   help="Number of self-play games to run")
    p.add_argument("--seed", type=int, default=100,
                   help="Starting seed (incremented per game)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory root (per-game subdirs created)")
    p.add_argument("--wallclock-ms", type=int, default=100,
                   help="BASELINE_WALLCLOCK_MS per worker (default 100)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--agent",
        default=str(REPO / "agents" / "baseline_pv_eta_probe" / "main.py"),
        help="Agent path (default: baseline_pv_eta_probe wrapper)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (args.seed + i, str(out_dir), args.wallclock_ms, args.agent)
        for i in range(args.games)
    ]

    t0 = time.perf_counter()
    if args.workers <= 1:
        results_iter = (_run_one_game(t) for t in tasks)
    else:
        ctx = get_context("spawn")
        # maxtasksperchild=1 — trace_hook module-level state (env var
        # cached at module load, file handle cached on first write)
        # would otherwise leak the first game's accepted-trace path
        # across all subsequent games handled by the same worker.
        pool = ctx.Pool(processes=args.workers, maxtasksperchild=1)
        results_iter = pool.imap_unordered(_run_one_game, tasks, chunksize=1)

    total_accepted = 0
    total_steps = 0
    for k, r in enumerate(results_iter):
        total_accepted += r["n_accepted"]
        total_steps += r["n_steps"]
        print(
            f"  [{k+1}/{args.games}] seed={r['seed']:08d} "
            f"steps={r['n_steps']} accepted={r['n_accepted']} "
            f"game={r['elapsed_s']:.1f}s "
            f"elapsed={time.perf_counter()-t0:.0f}s",
            flush=True,
        )

    if args.workers > 1:
        pool.close()
        pool.join()

    print()
    print(
        f"=== summary === games={args.games} total_steps={total_steps} "
        f"total_accepted={total_accepted} "
        f"wall={time.perf_counter()-t0:.0f}s out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
