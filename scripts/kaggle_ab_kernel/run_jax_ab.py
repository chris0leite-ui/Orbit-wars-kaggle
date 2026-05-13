"""Orbit Wars JAX A/B Kernel — sub-phase 8c.

Vmap'd 64-game A/B between two JAX policies, designed for a Kaggle
Kernel with GPU enabled. Companion to `run_ab.py` (CPU baseline that
runs the scalar bundles through `kaggle_environments`).

Pipeline:
  1. Build N=64 initial `GameState`s by running `env.reset()` for 64
     different seeds and converting via `scalar_to_jax` (one-time
     setup, CPU/numpy).
  2. Stack the 64 states into a batched Pytree.
  3. `jax.vmap`'d 500-step rollout via `score_candidate_jax_pure`'s
     `lax.scan`. Half the games have A at seat 0, B at seat 1; the
     other half is mirrored.
  4. Tally winner per game; report A win rate + Wilson 95% lo.
  5. Write `/kaggle/working/result.json`.

The A/B configurable knobs:
  - A_AGGRESSIVE / B_AGGRESSIVE: snipe sizing flag per agent.

Future drop-one chooser A/B (next sub-phase) wraps this with a
candidate-enumeration loop, but the per-turn rollout machinery is
already in place.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


NUM_SEEDS = int(os.environ.get("NUM_SEEDS", "32"))      # × 2 mirror seats = 64 games
EPISODE_STEPS = int(os.environ.get("EPISODE_STEPS", "500"))
A_AGGRESSIVE = bool(int(os.environ.get("A_AGGRESSIVE", "0")))   # 0 = v7_0 style
B_AGGRESSIVE = bool(int(os.environ.get("B_AGGRESSIVE", "1")))   # 1 = v3.5.1 style
# H11 (2026-05-13): opening-landgrab proposer toggle per seat. v7_1 sets
# A_USE_OPENING=1 to test H11 + H15 (the comet reject is always-on inside
# `compute_snipe_score_matrix`). B_USE_OPENING=0 reproduces v7_0 baseline.
A_USE_OPENING = bool(int(os.environ.get("A_USE_OPENING", "1")))
B_USE_OPENING = bool(int(os.environ.get("B_USE_OPENING", "0")))


def _ensure_kaggle_environments():
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
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (center - spread) / denom)


def main():
    install_msg = _ensure_kaggle_environments()
    import jax
    import jax.numpy as jnp
    from kaggle_environments import make

    # Lazy-import our JAX code only after kaggle_environments is ready.
    # On Kaggle the repo lives under /kaggle/input; locally the path
    # resolves from this script's location.
    # Locate the dataset root: look for any directory under
    # /kaggle/input/** that contains lib/game/jax/.
    repo_root = None
    for candidate in Path("/kaggle/input").rglob("lib/game/jax") if Path("/kaggle/input").exists() else []:
        repo_root = str(candidate.parent.parent.parent)  # up past lib/game
        break
    if repo_root is None:
        # Local fallback: walk up from this script.
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
    sys.path.insert(0, repo_root)
    print(f"Inserted {repo_root} into sys.path")
    from lib.game.jax import scalar_to_jax
    from lib.game.jax.jax_score import rollout_step_jax_pure, value_delta_ships

    info = {
        "install_msg": install_msg,
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "jax_devices": [str(d) for d in jax.devices()],
        "num_seeds": NUM_SEEDS,
        "episode_steps": EPISODE_STEPS,
        "a_aggressive": A_AGGRESSIVE,
        "b_aggressive": B_AGGRESSIVE,
        "a_use_opening": A_USE_OPENING,
        "b_use_opening": B_USE_OPENING,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps({"setup": info}, indent=2))

    # ---- Build N initial states (CPU/scalar; one-time setup) ----
    t_init = time.perf_counter()
    states = []
    # A at seat 0 for the first half of seeds, A at seat 1 (mirror)
    # for the second half. Track per-game "a_seat" so we can score.
    a_seats = []
    for i in range(NUM_SEEDS):
        env = make("orbit_wars", configuration={
            "seed": i, "episodeSteps": EPISODE_STEPS,
        })
        env.reset(num_agents=2)
        # Thread cometSpeed from configuration (bug J).
        env_comet_speed = float(env.configuration.get("cometSpeed", 4.0))
        gs = scalar_to_jax(env.state, env.info["seed"], comet_speed=env_comet_speed)
        states.append(gs)
        a_seats.append(0)
    # Mirror half.
    for i in range(NUM_SEEDS):
        env = make("orbit_wars", configuration={
            "seed": i, "episodeSteps": EPISODE_STEPS,
        })
        env.reset(num_agents=2)
        env_comet_speed = float(env.configuration.get("cometSpeed", 4.0))
        gs = scalar_to_jax(env.state, env.info["seed"], comet_speed=env_comet_speed)
        states.append(gs)
        a_seats.append(1)
    N = len(states)
    info["N_games"] = N
    info["init_setup_s"] = time.perf_counter() - t_init
    print(f"Built {N} initial states in {info['init_setup_s']:.1f} s")

    # Stack into batched Pytree.
    batched_state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    a_seats_arr = jnp.asarray(a_seats, dtype=jnp.int32)

    # ---- Build vmap'd rollout ----
    # A is always the "my" side (at seat my_id ∈ {0,1} per the mirror
    # split). Plumb A_AGGRESSIVE → my_aggressive and B_AGGRESSIVE →
    # opp_aggressive so both knobs are real (bug G fix).
    def run_episode(state, my_id):
        """Single-game scan over EPISODE_STEPS turns."""
        def step(s, _):
            new_s = rollout_step_jax_pure(
                s, my_id=my_id, num_agents=2,
                opp_aggressive=B_AGGRESSIVE,
                my_aggressive=A_AGGRESSIVE,
                my_use_opening=A_USE_OPENING,
                opp_use_opening=B_USE_OPENING,
            )
            return new_s, None
        final, _ = jax.lax.scan(step, state, None, length=EPISODE_STEPS)
        return final

    run_batch_jit = jax.jit(jax.vmap(run_episode))

    print()
    print("Compiling vmap'd rollout ...")
    t_compile = time.perf_counter()
    final_batch = run_batch_jit(batched_state, a_seats_arr)
    # Force materialisation to flush the compile.
    final_batch.step.block_until_ready()
    info["compile_s"] = time.perf_counter() - t_compile
    print(f"  done in {info['compile_s']:.1f} s")

    # Already have the result from the compile call. Time a hot run
    # to record steady-state perf.
    print("Hot run ...")
    t_hot = time.perf_counter()
    final_batch = run_batch_jit(batched_state, a_seats_arr)
    final_batch.step.block_until_ready()
    info["hot_run_s"] = time.perf_counter() - t_hot
    print(f"  done in {info['hot_run_s']:.1f} s ({info['hot_run_s']*1000/N:.1f} ms/game amortized)")

    # ---- Score outcomes ----
    # Use the engine's terminate() reward directly (bug H). Scalar
    # `interpreter` and JAX `terminate` set rewards[i] = +1 for the
    # max-score seat (when max > 0), -1 otherwise; ties resolve to
    # BOTH sides losing. That matches kaggle_environments' winrate
    # accounting and is materially different from a ship-delta proxy
    # in exactly-zero edge cases.
    rewards_np = np.asarray(final_batch.rewards)             # (N, MAX_AGENTS)
    a_seats_np = np.asarray(a_seats_arr)
    a_rewards = rewards_np[np.arange(N), a_seats_np]         # (N,) ∈ {-1,0,+1}
    b_rewards = rewards_np[np.arange(N), 1 - a_seats_np]     # opp seat
    a_wins = int(np.sum(a_rewards > b_rewards))
    b_wins = int(np.sum(b_rewards > a_rewards))
    draws = N - a_wins - b_wins

    info["a_wins"] = a_wins
    info["b_wins"] = b_wins
    info["draws"] = draws
    info["a_winrate"] = a_wins / N if N else 0.0
    info["a_wilson_lo"] = _wilson_lo(a_wins, N)
    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print()
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
