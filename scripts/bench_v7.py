"""Per-variant p95-turn-ms gate for v7 ablations.

Runs each v7 variant agent on a fixed set of warm board states and
reports the median + p95 wallclock per `agent(obs)` call. Gate:
**p95 < 800 ms** (Rule 2 actTimeout safety, 200 ms margin below the
1 s ladder limit).

Usage:
    python scripts/bench_v7.py [--seeds 42,7,100] [--warmup 30]

Output: prints a table; also returns nonzero exit code if any variant
fails the p95 gate (so the gate is CI-friendly).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402


VARIANTS = [
    ("v7_0_drop_one",   "agents/v7_ablations/v7_0_drop_one/main.py"),
    ("v7_1_target_swap", "agents/v7_ablations/v7_1_target_swap/main.py"),
    ("v7_2_ship_sweep", "agents/v7_ablations/v7_2_ship_sweep/main.py"),
    ("v7_3_archetype",  "agents/v7_ablations/v7_3_archetype/main.py"),
    ("v7_4_hungarian",  "agents/v7_ablations/v7_4_hungarian/main.py"),
    ("v7_combined",     "agents/v7_combined/main.py"),
]

P95_GATE_MS = 800.0


def _load_agent(path: str):
    p = REPO / path
    spec = importlib.util.spec_from_file_location(f"_v7_bench_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _warm_obs_list(seeds: list[int], warmup: int) -> list:
    """Return a list of (obs, configuration) pairs from warmed games."""
    out = []
    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        rng = random.Random(seed * 31)
        for _ in range(warmup):
            acts = []
            for p in range(2):
                launches = [
                    [pl[0], rng.uniform(0.0, 6.283), int(pl[5] // 3)]
                    for pl in env.state[0].observation["planets"]
                    if pl[1] == p and pl[5] > 8 and rng.random() < 0.35
                ]
                acts.append(launches)
            env.step(acts)
        # Take the obs from both seats — gives us 2 samples per seed.
        out.append((env.state[0].observation, env.configuration))
        out.append((env.state[1].observation, env.configuration))
    return out


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", default="42,7,100,314,2026",
        help="comma-separated seeds; each yields 2 obs samples (both seats)",
    )
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Building {len(seeds)} warm board states (warmup={args.warmup})...",
          flush=True)
    samples = _warm_obs_list(seeds, args.warmup)
    print(f"  {len(samples)} samples ready\n")

    print(f"{'variant':<22} {'median':>10} {'mean':>10} {'p95':>10} {'max':>10}  gate")
    print("-" * 78)

    fails: list[str] = []
    for name, rel_path in VARIANTS:
        try:
            agent_fn = _load_agent(rel_path)
        except Exception as exc:
            print(f"{name:<22} LOAD ERROR: {exc}")
            fails.append(name)
            continue

        times: list[float] = []
        for obs, cfg in samples:
            t0 = time.perf_counter()
            _ = agent_fn(obs, cfg)
            times.append((time.perf_counter() - t0) * 1000.0)

        med = statistics.median(times)
        mean = statistics.mean(times)
        p95 = _percentile(times, 0.95)
        mx = max(times)
        gate = "PASS" if p95 < P95_GATE_MS else "FAIL"
        if gate == "FAIL":
            fails.append(name)
        print(f"{name:<22} {med:>8.0f}ms {mean:>8.0f}ms {p95:>8.0f}ms {mx:>8.0f}ms  {gate}")

    print()
    if fails:
        print(f"FAIL: {len(fails)} variant(s) exceeded p95 < {P95_GATE_MS} ms: {fails}")
        sys.exit(1)
    print(f"PASS: all variants under p95 = {P95_GATE_MS} ms gate.")


if __name__ == "__main__":
    main()
