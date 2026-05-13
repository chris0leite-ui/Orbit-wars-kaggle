"""Orbit Wars A/B Kernel — sub-phase 8 deploy harness.

Runs an A/B between two Kaggle Simulations bundles using a multiprocessing
pool on the Kaggle Kernel's CPUs. Reports per-seed outcomes + Wilson 95%
lower bound, writes `/kaggle/working/result.json`.

This is the CPU baseline. Sub-phase 8b layers the JAX-GPU vmap path on
top once `score_candidate_jax` is vectorisable across N games.

Bundles must be uploaded as kernel inputs at `/kaggle/input/<dataset>/`.
The kernel reads them via the `BUNDLE_A_PATH` / `BUNDLE_B_PATH` env
variables (or the defaults below). For local testing, point the paths
at `submissions/` in the repo.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


# Bundle locations. On Kaggle Kernels, datasets land under /kaggle/input/.
# Defaults assume a single dataset named "orbit-wars-bundles" containing
# both bundle .py files.
BUNDLE_A_PATH = os.environ.get(
    "BUNDLE_A_PATH",
    "/kaggle/input/orbit-wars-bundles/v7_0_drop_one.py",
)
BUNDLE_B_PATH = os.environ.get(
    "BUNDLE_B_PATH",
    "/kaggle/input/orbit-wars-bundles/v3.5.1.py",
)

NUM_SEEDS = int(os.environ.get("NUM_SEEDS", "32"))
EPISODE_STEPS = int(os.environ.get("EPISODE_STEPS", "500"))


def _ensure_orbit_wars():
    """Kaggle Kernels ship kaggle_environments==1.27.3 which lacks
    orbit_wars. Pip-install the version our local code targets."""
    try:
        from kaggle_environments import make
        make("orbit_wars", configuration={"seed": 0})
        return None
    except Exception:
        pass
    print("Installing kaggle_environments==1.29.1 ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "kaggle_environments==1.29.1"],
    )
    import importlib, kaggle_environments
    importlib.reload(kaggle_environments)
    return "installed 1.29.1"


def _wilson_lo(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% lower bound on a binomial proportion."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (center - spread) / denom)


def _run_one_game(args):
    """Worker: one (seed, seat) pair. Returns
    (seed, seat, a_reward, b_reward, status_ok, wallclock_s)."""
    seed, a_seat, bundle_a, bundle_b, episode_steps = args
    from kaggle_environments import make
    # Place bundle_a at a_seat (0 or 1); bundle_b at 1 - a_seat.
    if a_seat == 0:
        agents = [bundle_a, bundle_b]
    else:
        agents = [bundle_b, bundle_a]
    t0 = time.perf_counter()
    try:
        env = make("orbit_wars", configuration={
            "seed": seed, "episodeSteps": episode_steps,
        })
        env.run(agents)
        wall = time.perf_counter() - t0
        a_idx = a_seat
        b_idx = 1 - a_seat
        return (
            seed, a_seat,
            int(env.state[a_idx].reward or 0),
            int(env.state[b_idx].reward or 0),
            True,
            wall,
        )
    except Exception as e:
        return (seed, a_seat, 0, 0, False, time.perf_counter() - t0)


def main():
    install_msg = _ensure_orbit_wars()
    info = {
        "install_msg": install_msg,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_mp": mp.cpu_count(),
        "bundle_a": BUNDLE_A_PATH,
        "bundle_b": BUNDLE_B_PATH,
        "num_seeds": NUM_SEEDS,
        "episode_steps": EPISODE_STEPS,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps({"setup": info}, indent=2))

    if not Path(BUNDLE_A_PATH).exists() or not Path(BUNDLE_B_PATH).exists():
        info["error"] = f"missing bundles: A={Path(BUNDLE_A_PATH).exists()} B={Path(BUNDLE_B_PATH).exists()}"
        info["files_in_input"] = sorted(os.listdir("/kaggle/input")) if Path("/kaggle/input").exists() else []
        _write(info)
        return

    # Build the task list: each seed runs at BOTH seats (mirror match).
    tasks = []
    for seed in range(NUM_SEEDS):
        for a_seat in (0, 1):
            tasks.append((seed, a_seat, BUNDLE_A_PATH, BUNDLE_B_PATH, EPISODE_STEPS))

    workers = min(os.cpu_count() or 4, 4)
    print(f"Running {len(tasks)} games on {workers} workers ...")
    t_pool_start = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(_run_one_game, tasks)
    pool_wall = time.perf_counter() - t_pool_start
    info["pool_wallclock_s"] = pool_wall

    # Tally.
    a_wins = sum(1 for r in results if r[2] > r[3])
    b_wins = sum(1 for r in results if r[3] > r[2])
    draws = len(results) - a_wins - b_wins
    failures = sum(1 for r in results if not r[4])
    avg_game_s = sum(r[5] for r in results) / max(1, len(results))

    info["a_wins"] = a_wins
    info["b_wins"] = b_wins
    info["draws"] = draws
    info["failures"] = failures
    info["n"] = len(results)
    info["a_winrate"] = a_wins / max(1, len(results))
    info["a_wilson_lo"] = _wilson_lo(a_wins, len(results))
    info["avg_game_s"] = avg_game_s
    info["pool_speedup"] = (sum(r[5] for r in results) / max(1e-9, pool_wall))
    print(json.dumps({"results": info}, indent=2))

    _write(info)


def _write(info):
    out_path = "/kaggle/working/result.json"
    if not Path("/kaggle/working").exists():
        out_path = "/tmp/result.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
