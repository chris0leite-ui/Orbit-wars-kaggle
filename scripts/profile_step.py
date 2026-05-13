"""Microbenchmark + cProfile harness for the orbit_wars simulator stack.

Use this to measure the impact of optimisations to lib/game/interpreter.py
or lib/fast_sim.py. Reports per-step wallclock and a cProfile top-25 for
the interpreter and fast_sim paths.

Usage:
    python -m scripts.profile_step                # 2000 steps each
    python -m scripts.profile_step --steps 5000   # longer sample
    python -m scripts.profile_step --no-profile   # just wallclock
"""

from __future__ import annotations

import argparse
import cProfile
import io
import math
import pstats
import random
import time

from kaggle_environments import make
from kaggle_environments.utils import Struct

from lib.fast_sim import _FakeEnv, clone as fs_clone, from_obs, step as fs_step
from lib.game.interpreter import interpreter as ours_interp


def _fresh_state(seed: int):
    state = [
        Struct(observation=Struct(), action=[], status="ACTIVE", reward=0, info={})
        for _ in range(2)
    ]
    conf = Struct(
        episodeSteps=500, shipSpeed=6.0, cometSpeed=4.0, actTimeout=1.0,
        agentTimeout=60.0, runTimeout=1200.0, seed=seed,
    )
    fe = _FakeEnv(configuration=conf, episode_seed=seed)
    fe.info = {}
    fe.done = False
    return state, fe


def _rand_act(obs, pid, rng):
    moves = []
    for p in obs.planets:
        if p[1] == pid and p[5] > 0 and rng.random() < 0.3:
            angle = rng.uniform(0, 2 * math.pi)
            ships = max(1, int(p[5] * rng.uniform(0.1, 0.7)))
            if 0 < ships <= p[5]:
                moves.append([p[0], angle, ships])
    return moves


def _bookkeep(state, fe):
    obs0 = state[0].observation
    obs0.step = int(obs0.get("step", 0)) + 1
    for i in range(1, len(state)):
        state[i].observation.step = obs0.step
    if any(s.status == "DONE" for s in state):
        fe.done = True


def run_interp_only(n: int):
    rng = random.Random(0)
    state, fe = _fresh_state(42)
    ours_interp(state, fe)
    _bookkeep(state, fe)
    steps = 0
    while steps < n:
        if fe.done:
            state, fe = _fresh_state(rng.randint(0, 1 << 30))
            ours_interp(state, fe)
            _bookkeep(state, fe)
            continue
        state[0].action = _rand_act(state[0].observation, 0, rng)
        state[1].action = _rand_act(state[1].observation, 1, rng)
        ours_interp(state, fe)
        _bookkeep(state, fe)
        steps += 1


def run_fast_sim(n: int):
    rng = random.Random(0)
    env = make("orbit_wars", configuration={"seed": 42})
    env.reset(num_agents=2)
    snap = from_obs(
        env.state[0].observation, env.configuration,
        episode_seed=env.info["seed"], num_seats=2,
    )
    steps = 0
    while steps < n:
        if snap.fake_env.done:
            env = make("orbit_wars", configuration={"seed": rng.randint(0, 1 << 30)})
            env.reset(num_agents=2)
            snap = from_obs(
                env.state[0].observation, env.configuration,
                episode_seed=env.info["seed"], num_seats=2,
            )
            continue
        a0 = _rand_act(snap.state[0].observation, 0, rng)
        a1 = _rand_act(snap.state[1].observation, 1, rng)
        snap = fs_step(snap, [a0, a1])
        steps += 1


def _profile(fn, label, top=25):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(top)
    print(f"\n========== {label} ==========")
    print(s.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    # Warmup
    run_interp_only(50)
    run_fast_sim(50)

    t0 = time.perf_counter(); run_interp_only(args.steps); t_i = time.perf_counter() - t0
    t0 = time.perf_counter(); run_fast_sim(args.steps);    t_f = time.perf_counter() - t0

    print(f"\nWallclock for {args.steps} steps:")
    print(f"  interpreter only: {t_i*1000:.1f} ms ({t_i/args.steps*1e6:.0f} us/step)")
    print(f"  fast_sim.step():  {t_f*1000:.1f} ms ({t_f/args.steps*1e6:.0f} us/step)")
    print(f"  clone overhead:   {(t_f - t_i)/args.steps*1e6:.0f} us/step\n")

    if not args.no_profile:
        _profile(lambda: run_interp_only(args.steps), "interpreter only")
        _profile(lambda: run_fast_sim(args.steps), "fast_sim.step()")


if __name__ == "__main__":
    main()
