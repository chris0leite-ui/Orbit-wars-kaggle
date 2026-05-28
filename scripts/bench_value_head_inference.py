"""Inference-latency bench for the learned value head.

The chooser invokes the leaf eval `N_candidates * K_horizon` times per
turn (see `agents/baseline/chooser.py`). For a `~500-call` budget inside
`600 ms` per turn we need median < 200 µs and p99 < 500 µs.

Bench drives `agents/baseline/value_learned.favor_learned` against a
real game observation pulled from a fresh `kaggle_environments` rollout.
Reports median, p95, p99, max over N calls.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402

from agents.baseline.value_learned import (  # noqa: E402
    favor_learned, weights_loaded, weights_summary,
)
from lib.opp_model import lite_greedy_policy  # noqa: E402

N_CALLS_DEFAULT = 1_000
MEDIAN_DEADLINE_US = 200.0
P99_DEADLINE_US = 500.0


def _fetch_real_obs(turn: int = 100) -> dict:
    """Get a non-degenerate game observation at the requested turn."""
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)

    def lg(obs, cfg=None):
        return lite_greedy_policy(obs)

    env.run([lg, lg])
    if turn >= len(env.steps):
        turn = len(env.steps) - 1
    return env.steps[turn][0].observation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_CALLS_DEFAULT)
    ap.add_argument("--turn", type=int, default=100)
    args = ap.parse_args()

    print(weights_summary(), flush=True)
    obs = _fetch_real_obs(args.turn)
    print(f"obs at turn {args.turn}: planets={len(obs.get('planets', []))} "
          f"fleets={len(obs.get('fleets', []))}", flush=True)

    # Warmup so JIT / page faults don't pollute timings.
    for _ in range(50):
        favor_learned(obs, me=0, num_seats=2)

    times_us = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        favor_learned(obs, me=0, num_seats=2)
        times_us.append((time.perf_counter() - t0) * 1e6)

    times_us.sort()
    median = statistics.median(times_us)
    p95 = times_us[int(0.95 * len(times_us))]
    p99 = times_us[int(0.99 * len(times_us))]
    mx = times_us[-1]
    print(
        f"\nlatency (N={args.n}): "
        f"median={median:.1f}µs  p95={p95:.1f}µs  "
        f"p99={p99:.1f}µs  max={mx:.1f}µs",
        flush=True,
    )

    ok = True
    if median > MEDIAN_DEADLINE_US:
        print(
            f"FAIL: median {median:.1f}µs > {MEDIAN_DEADLINE_US:.0f}µs",
            flush=True,
        )
        ok = False
    if p99 > P99_DEADLINE_US:
        print(
            f"WARN: p99 {p99:.1f}µs > {P99_DEADLINE_US:.0f}µs", flush=True,
        )
    if not weights_loaded():
        print(
            "NOTE: running on ZERO_FALLBACK weights — re-bench after "
            "embedding trained weights via "
            "`scripts/embed_value_head_weights.py`",
            flush=True,
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
