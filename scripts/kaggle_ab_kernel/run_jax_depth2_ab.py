"""Orbit Wars JAX A/B Kernel — depth-2 maximin (v7_2) vs single-ply (v7_1).

Companion to `run_jax_ab.py`. The only difference is the per-step
rollout: this kernel uses `lib.game.jax.jax_depth2.rollout_step_depth2_jax_pure`
for seat A and the standard single-ply mirror for seat B.

That means A picks each turn's action via maximin over (our drop-one ×
opp drop-one) with a K_tail mirror tail; B picks via the single-ply
mission-stack pipeline. Both seats use H11 (`use_opening=True`) and
aggressive snipe sizing by default — the ONLY behavioural difference
is the depth-2 chooser.

Env vars:
  - NUM_SEEDS      — number of game seeds (× 2 mirror = N games). Default 32.
  - EPISODE_STEPS  — env step budget. Default 500.
  - K_TAIL         — mirror-tail depth for depth-2 cells. Default 4.
                     Total depth-2 horizon is 2 (forced turns) + K_TAIL.
  - A_AGGRESSIVE / B_AGGRESSIVE — snipe sizing per seat. Defaults 1 / 1
                     (both v7_0 baseline; depth-2 is the only delta).
  - A_USE_OPENING / B_USE_OPENING — H11 per seat. Defaults 1 / 1.

The dataset `chrisleitescha/orbit-wars-jax-repo` must include the
jax_depth2 module. Push only after updating the dataset.
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
K_TAIL = int(os.environ.get("K_TAIL", "4"))
A_AGGRESSIVE = bool(int(os.environ.get("A_AGGRESSIVE", "1")))   # 1 = v7_0 baseline
B_AGGRESSIVE = bool(int(os.environ.get("B_AGGRESSIVE", "1")))   # 1 = v7_0 baseline
A_USE_OPENING = bool(int(os.environ.get("A_USE_OPENING", "1")))
B_USE_OPENING = bool(int(os.environ.get("B_USE_OPENING", "1")))


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

    # Locate the dataset root: rglob for lib/game/jax under /kaggle/input.
    repo_root = None
    for candidate in (
        Path("/kaggle/input").rglob("lib/game/jax")
        if Path("/kaggle/input").exists() else []
    ):
        repo_root = str(candidate.parent.parent.parent)
        break
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
    sys.path.insert(0, repo_root)

    from lib.game.jax.conversions import scalar_to_jax
    from lib.game.jax.jax_score import rollout_step_jax_pure
    from lib.game.jax.jax_depth2 import rollout_step_depth2_jax_pure

    info = {
        "install_msg": install_msg,
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "jax_devices": [str(d) for d in jax.devices()],
        "num_seeds": NUM_SEEDS,
        "episode_steps": EPISODE_STEPS,
        "k_tail": K_TAIL,
        "a_aggressive": A_AGGRESSIVE,
        "b_aggressive": B_AGGRESSIVE,
        "a_use_opening": A_USE_OPENING,
        "b_use_opening": B_USE_OPENING,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(info, indent=2))

    # ---- Build initial states for N seeds × 2 mirror seats ----
    t_init = time.perf_counter()
    states: list = []
    a_seats: list = []
    for i in range(NUM_SEEDS):
        env = make("orbit_wars", configuration={
            "seed": i, "episodeSteps": EPISODE_STEPS,
        })
        env.reset(num_agents=2)
        env_comet_speed = float(env.configuration.get("cometSpeed", 4.0))
        gs = scalar_to_jax(env.state, env.info["seed"], comet_speed=env_comet_speed)
        states.append(gs)
        a_seats.append(0)
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

    batched_state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    a_seats_arr = jnp.asarray(a_seats, dtype=jnp.int32)

    # ---- Build vmap'd rollout: A=depth-2, B=single-ply ----
    # A is at seat `my_id`; B is at seat `1 - my_id`. Both seats step
    # together each turn, but A's action comes from the depth-2 chooser
    # (which already plays the *forced action* on its seat inside the
    # rollout helper) and B's action comes from the single-ply mirror.
    def run_episode(state, my_id):
        def step(s, _):
            new_s = rollout_step_depth2_jax_pure(
                s, my_id=my_id, K_tail=K_TAIL, num_agents=2,
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
    print("Compiling vmap'd depth-2 rollout ...")
    t_compile = time.perf_counter()
    final_batch = run_batch_jit(batched_state, a_seats_arr)
    final_batch.step.block_until_ready()
    info["compile_s"] = time.perf_counter() - t_compile
    print(f"  done in {info['compile_s']:.1f} s")

    print("Hot run ...")
    t_hot = time.perf_counter()
    final_batch = run_batch_jit(batched_state, a_seats_arr)
    final_batch.step.block_until_ready()
    info["hot_run_s"] = time.perf_counter() - t_hot
    print(
        f"  done in {info['hot_run_s']:.1f} s "
        f"({info['hot_run_s']*1000/N:.1f} ms/game amortized)"
    )

    # ---- Score outcomes ----
    rewards_np = np.asarray(final_batch.rewards)
    a_seats_np = np.asarray(a_seats_arr)
    a_rewards = rewards_np[np.arange(N), a_seats_np]
    b_rewards = rewards_np[np.arange(N), 1 - a_seats_np]
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
    with open(out_path, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"\nresult.json -> {out_path}")


if __name__ == "__main__":
    main()
