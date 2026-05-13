"""Drive Kaggle's orbit_wars interpreter and `lib.game.interpreter` through
complete 500-step random-policy episodes on N seeds in lockstep, asserting
byte-exact parity after every step. Reports the first divergence (if any)
with seed, step, and a diff line.

Usage:
    python -m scripts.full_episode_parity_sweep --seeds-2p 100 --seeds-4p 50

Exit 0 on full parity; non-zero on any divergence.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    interpreter as kaggle_interpreter,
)
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv
from lib.game.interpreter import interpreter as ours_interpreter


def _fresh(num_agents: int, seed: int):
    state = [
        Struct(observation=Struct(), action=[], status="ACTIVE", reward=0, info={})
        for _ in range(num_agents)
    ]
    conf = Struct(
        episodeSteps=500,
        shipSpeed=6.0,
        cometSpeed=4.0,
        actTimeout=1.0,
        agentTimeout=60.0,
        runTimeout=1200.0,
        seed=seed,
    )
    fe = _FakeEnv(configuration=conf, episode_seed=seed)
    fe.info = {}
    fe.done = False
    return state, fe


def _bookkeep(state, fe):
    obs0 = state[0].observation
    obs0.step = int(obs0.get("step", 0)) + 1
    for i in range(1, len(state)):
        state[i].observation.step = obs0.step
    if any(s.status == "DONE" for s in state):
        fe.done = True


def _rand_actions(state, rng: random.Random, n: int):
    out = []
    for i in range(n):
        obs = state[i].observation
        moves = []
        for p in obs.planets:
            if p[1] == i and p[5] > 0 and rng.random() < 0.4:
                angle = rng.uniform(0, 2 * math.pi)
                ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
                if 0 < ships <= p[5]:
                    moves.append([p[0], angle, ships])
        out.append(moves)
    return out


def _diff(state_a, state_b) -> str:
    for i, (sa, sb) in enumerate(zip(state_a, state_b)):
        if sa.status != sb.status:
            return f"seat[{i}].status: {sa.status!r} vs {sb.status!r}"
        if sa.reward != sb.reward:
            return f"seat[{i}].reward: {sa.reward} vs {sb.reward}"
    a, b = state_a[0].observation, state_b[0].observation
    if a.planets != b.planets:
        for j, (pa, pb) in enumerate(zip(a.planets, b.planets)):
            if pa != pb:
                return f"planets[{j}]: {pa} vs {pb}"
        return f"planets length: {len(a.planets)} vs {len(b.planets)}"
    if a.fleets != b.fleets:
        for j, (fa, fb) in enumerate(zip(a.fleets, b.fleets)):
            if fa != fb:
                return f"fleets[{j}]: {fa} vs {fb}"
        return f"fleets length: {len(a.fleets)} vs {len(b.fleets)}"
    if list(a.comet_planet_ids) != list(b.comet_planet_ids):
        return f"comet_planet_ids: {list(a.comet_planet_ids)} vs {list(b.comet_planet_ids)}"
    if len(a.comets) != len(b.comets):
        return f"comets length: {len(a.comets)} vs {len(b.comets)}"
    for j, (ga, gb) in enumerate(zip(a.comets, b.comets)):
        if ga["path_index"] != gb["path_index"]:
            return f"comets[{j}].path_index: {ga['path_index']} vs {gb['path_index']}"
        if ga["planet_ids"] != gb["planet_ids"]:
            return f"comets[{j}].planet_ids"
        if ga["paths"] != gb["paths"]:
            return f"comets[{j}].paths"
    return ""


def run_episode(seed: int, num_agents: int) -> tuple[bool, str, int, int]:
    state_k, env_k = _fresh(num_agents, seed)
    state_o, env_o = _fresh(num_agents, seed)
    kaggle_interpreter(state_k, env_k)
    ours_interpreter(state_o, env_o)
    diff = _diff(state_k, state_o)
    if diff:
        return False, f"init: {diff}", -1, 0

    action_rng = random.Random(seed * 7919 + 1)

    for step_idx in range(500):
        actions = _rand_actions(state_k, action_rng, num_agents)
        for i, a in enumerate(actions):
            state_k[i].action = a
            state_o[i].action = list(a)
        kaggle_interpreter(state_k, env_k)
        ours_interpreter(state_o, env_o)
        _bookkeep(state_k, env_k)
        _bookkeep(state_o, env_o)

        d = _diff(state_k, state_o)
        if d:
            return False, d, step_idx, step_idx + 1

        if env_k.done or env_o.done:
            if env_k.done != env_o.done:
                return False, f"done mismatch: kaggle={env_k.done} ours={env_o.done}", step_idx, step_idx + 1
            return True, "", step_idx, step_idx + 1

    return True, "", 499, 500


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-2p", type=int, default=100)
    ap.add_argument("--seeds-4p", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=0)
    args = ap.parse_args(argv)

    failures = []
    total_steps = 0
    t0 = time.perf_counter()

    for num_agents, n_seeds in [(2, args.seeds_2p), (4, args.seeds_4p)]:
        for k in range(n_seeds):
            seed = args.start_seed + k
            ok, msg, last_step, n_steps = run_episode(seed, num_agents)
            total_steps += n_steps
            if not ok:
                failures.append((num_agents, seed, last_step, msg))
                print(
                    f"DIVERGE  n={num_agents}  seed={seed}  step={last_step}  {msg}"
                )
            elif k % 10 == 0:
                print(f"OK       n={num_agents}  seed={seed}  episode_len={n_steps}")

    elapsed = time.perf_counter() - t0
    total_episodes = args.seeds_2p + args.seeds_4p
    print()
    print(
        f"=== sweep: {total_episodes} episodes, {total_steps} step pairs, "
        f"{elapsed:.1f} s ({elapsed*1000/total_steps:.2f} ms/step-pair) ==="
    )
    if failures:
        print(f"FAILED: {len(failures)} episodes diverged")
        return 1
    print("PASSED: byte-exact parity across all episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
