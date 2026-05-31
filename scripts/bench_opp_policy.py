"""Bench the per-call wallclock of an opp policy on real replay obs.

Useful for Phase 4 verification of the distilled-ladder policy. The
plan's speed target is ≤ 1 ms median, ≤ 3 ms p95, vs lite_greedy
~0.5 ms and top_tier_mirror ~5-10 ms.

Usage:
    python scripts/bench_opp_policy.py --tier 2 --n 200
    python scripts/bench_opp_policy.py --tier 1 --n 50   # compare Tier 1
    python scripts/bench_opp_policy.py --tier 0 --n 200  # lite_greedy floor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from random import Random
from statistics import median

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", type=int, choices=[0, 1, 2], default=2)
    p.add_argument(
        "--replays-dir", default="/tmp/ow_replays",
        help="Directory of replay JSONs to sample obs from",
    )
    p.add_argument("--n", type=int, default=200, help="Number of obs to time")
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    from lib.opp_model import (  # noqa: E402
        lite_greedy_policy,
        top_tier_mirror_policy,
        trained_logreg_policy,
    )

    policies = {
        0: ("lite_greedy", lite_greedy_policy),
        1: ("top_tier_mirror", top_tier_mirror_policy),
        2: ("trained_logreg (distilled-ladder v2)", trained_logreg_policy),
    }
    name, policy = policies[args.tier]

    replays_dir = Path(args.replays_dir)
    eps = sorted(replays_dir.glob("*.json"))
    if not eps:
        print(f"no replays in {replays_dir}", file=sys.stderr)
        return 1

    # Sample obs from random (episode, step, seat) tuples
    rng = Random(0)
    sampled: list[dict] = []
    while len(sampled) < args.n + args.warmup:
        ep_path = rng.choice(eps)
        try:
            ep = json.loads(ep_path.read_text())
        except Exception:
            continue
        steps = ep.get("steps") or []
        if len(steps) < 5 or len(steps[0]) != 2:
            continue
        # Skip step 0 (empty actions); skip end-of-game.
        t = rng.randint(1, len(steps) - 1)
        seat = rng.randint(0, 1)
        seat_view = steps[t][seat]
        if seat_view.get("status") != "ACTIVE":
            continue
        obs = dict(seat_view.get("observation") or {})
        if not obs.get("planets"):
            continue
        if "step" not in obs:
            obs["step"] = t
        sampled.append(obs)

    print(f"benching {name} on {args.n} obs ({args.warmup} warmup) ...")
    # Warmup
    for obs in sampled[: args.warmup]:
        try:
            policy(obs)
        except Exception:
            pass

    times_ms: list[float] = []
    emit_counts: list[int] = []
    for obs in sampled[args.warmup : args.warmup + args.n]:
        t0 = time.perf_counter()
        try:
            out = policy(obs)
        except Exception:
            out = []
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        emit_counts.append(len(out or []))

    times_ms.sort()
    n = len(times_ms)
    p50 = median(times_ms)
    p95 = times_ms[int(n * 0.95)] if n >= 20 else times_ms[-1]
    p99 = times_ms[int(n * 0.99)] if n >= 100 else times_ms[-1]
    pmax = times_ms[-1]
    pmean = sum(times_ms) / n

    print(f"\n=== {name} ===")
    print(f"  n        = {n}")
    print(f"  mean     = {pmean:.3f} ms")
    print(f"  median   = {p50:.3f} ms")
    print(f"  p95      = {p95:.3f} ms")
    print(f"  p99      = {p99:.3f} ms")
    print(f"  max      = {pmax:.3f} ms")
    print(f"  emits/call: "
          f"mean={sum(emit_counts)/n:.2f}, "
          f"min={min(emit_counts)}, "
          f"max={max(emit_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
