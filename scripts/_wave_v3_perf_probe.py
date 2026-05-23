"""Latency diagnosis for V3 wave under high-load conditions.

Two runs:
  1. 2P: V3 (leaf-validate ON) vs baseline_full, seed 5199 (a known long
     game from the probe). Captures per-turn end-to-end timing, per-wave
     timing from the existing telemetry, and cProfile top functions.
  2. 4P: V3 vs three random opponents, seed 42. Same telemetry, plus a
     latency check (4P expands the action space; per-turn cost should
     scale).

Usage: python scripts/_wave_v3_perf_probe.py [2p_seed] [4p_seed]
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["BASELINE_CONVERGENCE_WAVE"] = "1"
os.environ["BASELINE_WAVE_LEAF_VALIDATE"] = "1"
os.environ["BASELINE_WAVE_INSTRUMENT"] = "1"

import importlib.util
from kaggle_environments import make
import agents.baseline.main as bm

bm.CONVERGENCE_WAVE_ENABLED = True
bm.WAVE_LEAF_VALIDATE = True
bm.WAVE_INSTRUMENT = True


# Per-turn end-to-end timing — wrap bm.agent so each call is measured.
_TURN_TIMES: list[float] = []
_orig_agent = bm.agent


def _timed_agent(obs, configuration=None):
    t0 = time.perf_counter()
    out = _orig_agent(obs, configuration)
    _TURN_TIMES.append((time.perf_counter() - t0) * 1000)
    return out


bm.agent = _timed_agent


def load(path: str):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"_m_{p.stem}", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m.agent


def summarize(label: str):
    """Print per-turn timing summary + wave-fire breakdown."""
    print(f"\n  --- {label} ---")
    if _TURN_TIMES:
        sorted_t = sorted(_TURN_TIMES)
        p50 = sorted_t[len(sorted_t) // 2]
        p95 = sorted_t[int(0.95 * len(sorted_t))]
        p99 = sorted_t[min(len(sorted_t) - 1, int(0.99 * len(sorted_t)))]
        print(f"  per-turn ms: n={len(_TURN_TIMES)}  "
              f"p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  "
              f"max={max(_TURN_TIMES):.0f}")
        # Show the 5 worst turns.
        worst = sorted(
            range(len(_TURN_TIMES)),
            key=lambda i: -_TURN_TIMES[i],
        )[:5]
        print(f"  5 worst turns (idx -> ms):")
        for i in worst:
            print(f"    turn #{i}: {_TURN_TIMES[i]:.0f}ms")

    walls = [r.get("wall_ms", 0) for r in bm._WAVE_TELEMETRY
             if r.get("wall_ms")]
    fires = [r for r in bm._WAVE_TELEMETRY if r.get("actual_fire")]
    rejected = [r for r in bm._WAVE_TELEMETRY
                if r.get("gate_rejected_reason")]
    candidates = [r for r in bm._WAVE_TELEMETRY
                  if r.get("gate_delta") is not None]
    print(f"  wave invocations: {len(bm._WAVE_TELEMETRY)}")
    print(f"  gate candidates: {len(candidates)}  "
          f"rejected: {len(rejected)}  fires: {len(fires)}")
    if walls:
        wsort = sorted(walls)
        print(f"  wave wall_ms: p50={wsort[len(wsort)//2]:.2f}"
              f"  p95={wsort[int(0.95*len(wsort))]:.2f}"
              f"  max={max(walls):.2f}")
    # Per-turn for the slowest turns: was the wave the culprit?
    if _TURN_TIMES and bm._WAVE_TELEMETRY:
        slow_threshold = max(800.0, sorted_t[int(0.9 * len(sorted_t))])
        slow_turns = [(i, t) for i, t in enumerate(_TURN_TIMES)
                      if t > slow_threshold]
        print(f"\n  slow turns ( > {slow_threshold:.0f}ms ) "
              f"and wave wall_ms contribution:")
        for (i, t) in slow_turns[:10]:
            # Find wave telemetry record for this step (best-effort).
            # Telemetry has 'step' from the env obs; turn idx may differ.
            # Use position in telemetry list as a proxy (one record per
            # invocation = one record per turn for the focal seat).
            if i < len(bm._WAVE_TELEMETRY):
                rec = bm._WAVE_TELEMETRY[i]
                wave_ms = rec.get("wall_ms", 0) or 0
                gate_ms = rec.get("gate_ms")
                act = rec.get("actual_fire", False)
                rej = rec.get("gate_rejected_reason")
                tag = "FIRE" if act else (f"REJ:{rej}" if rej else "noop")
                print(f"    turn #{i}: total={t:.0f}ms  "
                      f"wave={wave_ms:.1f}ms  "
                      f"gate={gate_ms or 0:.1f}ms  [{tag}]")


def reset():
    _TURN_TIMES.clear()
    bm._WAVE_TELEMETRY.clear()
    bm._OWNERSHIP_TRACE.clear()


def run_2p(seed: int):
    print(f"\n=== 2P run: V3 vs baseline_full seed={seed} ===")
    reset()
    opp = load("submissions/baseline_full.py")
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    pr = cProfile.Profile()
    pr.enable()
    env.run([bm.agent, opp])
    pr.disable()
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "P0" if r0 > r1 else "P1" if r1 > r0 else "DRAW"
    print(f"  outcome={outcome}  n_steps={len(env.steps)}")
    summarize("2P timing")
    print(f"\n  --- cProfile top 15 functions by cumulative time ---")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    # Filter out garbage lines.
    for line in s.getvalue().splitlines():
        if any(tag in line for tag in ("/agents/", "/lib/", "cumtime",
                                       "function calls", "ncalls")):
            print(f"    {line}")


def run_4p(seed: int):
    print(f"\n=== 4P run: V3 vs 3 random opps seed={seed} ===")
    reset()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    pr = cProfile.Profile()
    pr.enable()
    env.run([bm.agent, "random", "random", "random"])
    pr.disable()
    final = env.steps[-1]
    rewards = [f.reward for f in final]
    print(f"  rewards: {rewards}  n_steps={len(env.steps)}")
    winner = rewards.index(max(rewards))
    print(f"  winner seat: {winner}  "
          f"(P0=focal {'WON' if winner == 0 else 'LOST'})")
    summarize("4P timing")
    print(f"\n  --- cProfile top 10 functions by cumulative time ---")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    for line in s.getvalue().splitlines():
        if any(tag in line for tag in ("/agents/", "/lib/", "cumtime",
                                       "function calls", "ncalls")):
            print(f"    {line}")


if __name__ == "__main__":
    seed_2p = int(sys.argv[1]) if len(sys.argv) > 1 else 5199
    seed_4p = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_2p(seed_2p)
    run_4p(seed_4p)
