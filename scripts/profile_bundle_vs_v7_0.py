"""One-game instrumented run: bundle (lite) vs v7_0, seed 42.

Streams per-turn timing + capture/elimination events on stdout (one
line per turn for Monitor). cProfile output written to
audit/2026-05-18-bundle-lite-profile.prof for hotspot review.

Usage:
    BUNDLE_ME_FOLLOWUP=lite python scripts/profile_bundle_vs_v7_0.py
"""
from __future__ import annotations

import cProfile
import importlib.util
import os
import pstats
import sys
import time
import traceback
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make
from agents.bundle.main import agent as bundle_agent


def load_v7_0():
    spec = importlib.util.spec_from_file_location(
        "v7_0_loaded", REPO / "submissions" / "v7_0_drop_one.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v7_0_loaded"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def planet_count(obs, me):
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    my = opp = neu = 0
    for p in planets:
        owner = int(p[1])
        if owner == me:
            my += 1
        elif owner == -1:
            neu += 1
        else:
            opp += 1
    return my, opp, neu


def main():
    seed = int(os.environ.get("SEED", "42"))
    v7_0_agent = load_v7_0()

    bundle_ms_log: list[float] = []
    v7_ms_log: list[float] = []
    err_count = {"bundle": 0, "v7_0": 0}

    def wrap(agent, label, sink):
        def wrapped(obs, cfg=None):
            t0 = time.perf_counter()
            try:
                res = agent(obs, cfg)
            except Exception as e:
                err_count[label] += 1
                print(f"  ERR {label}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                res = []
            dt = (time.perf_counter() - t0) * 1000.0
            sink.append(dt)
            return res
        wrapped.__code__ = agent.__code__ if hasattr(agent, "__code__") else None
        return wrapped

    bundle_wrapped = wrap(bundle_agent, "bundle", bundle_ms_log)
    v7_wrapped = wrap(v7_0_agent, "v7_0", v7_ms_log)

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    profiler = cProfile.Profile()
    profiler.enable()

    t_game_start = time.perf_counter()
    last_my = last_opp = -1
    print(f"START seed={seed} mode={os.environ.get('BUNDLE_ME_FOLLOWUP', 'off')}", flush=True)

    try:
        # env.run drives the game internally; we stream timing AFTER it finishes
        # since we can't intercept per-turn easily without rewriting env.step.
        # Use env.step manually instead so we get per-turn streaming.
        turn = 0
        max_turns = 400
        while not env.done and turn < max_turns:
            actions = []
            for player_idx, w in enumerate([bundle_wrapped, v7_wrapped]):
                obs = env.state[player_idx].observation
                cfg = env.configuration
                actions.append(w(obs, cfg))
            env.step(actions)
            turn += 1
            # Stream every 20 turns + first 5 + last 5 turns leading to done.
            obs0 = env.state[0].observation
            my, opp, neu = planet_count(obs0, 0)
            if (turn <= 5 or turn % 20 == 0 or env.done
                    or my != last_my or opp != last_opp):
                last_my, last_opp = my, opp
                msg = (
                    f"t={turn:3d} p={my}-{opp}-{neu} "
                    f"b={bundle_ms_log[-1]:6.1f}ms v={v7_ms_log[-1]:6.1f}ms"
                )
                if bundle_ms_log[-1] > 800:
                    msg += " BUNDLE_SLOW"
                if v7_ms_log[-1] > 800:
                    msg += " V7_SLOW"
                print(msg, flush=True)
    finally:
        profiler.disable()

    elapsed = time.perf_counter() - t_game_start
    rewards = [s.reward for s in env.state]
    final_my, final_opp, final_neu = planet_count(env.state[0].observation, 0)

    print(f"DONE turns={turn} elapsed={elapsed:.1f}s rewards={rewards} "
          f"final_p={final_my}-{final_opp}-{final_neu}", flush=True)

    import statistics
    def stats(name, log):
        if not log:
            return
        s = sorted(log)
        p50 = s[len(s)//2]
        p95 = s[int(len(s)*0.95)]
        p99 = s[int(len(s)*0.99)] if len(s) >= 100 else s[-1]
        slow800 = sum(1 for t in log if t > 800)
        slow1000 = sum(1 for t in log if t > 1000)
        print(f"{name}: n={len(log)} p50={p50:.1f} p95={p95:.1f} "
              f"p99={p99:.1f} max={max(log):.1f}ms "
              f">800ms:{slow800} >1000ms:{slow1000}", flush=True)

    stats("bundle", bundle_ms_log)
    stats("v7_0", v7_ms_log)
    print(f"errors: {err_count}", flush=True)

    prof_path = REPO / "audit" / "2026-05-18-bundle-lite-profile.prof"
    profiler.dump_stats(str(prof_path))
    print(f"profile dumped: {prof_path}", flush=True)

    # Top 30 functions by cumulative time, restricted to lib/ and agents/.
    buf = StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    ps.print_stats(r"(lib/|agents/)", 30)
    print("--- TOP-30 HOTSPOTS (lib/ + agents/, by cumulative time) ---", flush=True)
    print(buf.getvalue(), flush=True)


if __name__ == "__main__":
    main()
