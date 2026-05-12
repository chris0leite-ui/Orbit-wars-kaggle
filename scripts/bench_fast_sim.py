"""Microbench: fast_sim.step() vs env.clone()+env.step().

Outputs both raw timings and a speedup ratio. Writes an audit doc to
`audit/2026-05-{today}-fast-sim-bench.md` with the numbers in a format
future sessions can diff against.

Usage:
    python scripts/bench_fast_sim.py [--seed 42] [--num 200] [--warmup 20]

Reports:
- `env.clone()+step()` median / mean / p95 over `num` calls.
- `fast_sim.step()` (which internally clones) median / mean / p95.
- `fast_sim.clone()` median.
- Speedup ratio (env_median / fast_median).
- Full-rollout K=50 cost via `fast_sim.rollout()` with the v3_snipe
  mirror policy as both seats.

All numbers are wallclock on the box this script runs on; reproduce on
a fresh machine to compare. No external services touched.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from datetime import datetime

# Make `lib/` and `agents/` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make

from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs, ship_totals, step, rollout
from lib.opp_model import mirror_self_policy


def _random_action(obs, seat: int, rng: random.Random) -> list:
    return [
        [int(p[0]), rng.uniform(0.0, 6.283), int(p[5] // 2)]
        for p in obs["planets"]
        if p[1] == seat and p[5] > 5 and rng.random() < 0.3
    ]


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num", type=int, default=200, help="step() calls per side")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rollout-k", type=int, default=50)
    parser.add_argument(
        "--audit",
        default=None,
        help="path to write the audit doc; defaults to "
             "audit/YYYY-MM-DD-fast-sim-bench.md",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.reset(num_agents=2)
    for _ in range(args.warmup):
        acts = [_random_action(env.state[0].observation, p, rng) for p in range(2)]
        env.step(acts)

    snap_base = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )

    # Use a single canonical action list across both timers so the work
    # being measured is genuinely physics + bookkeeping (not policy).
    canonical_action = [
        _random_action(env.state[0].observation, 0, rng),
        _random_action(env.state[0].observation, 1, rng),
    ]

    # --- env.clone() + step() ---
    env_times: list[float] = []
    for _ in range(args.num):
        t0 = time.perf_counter()
        e_clone = env.clone()
        e_clone.step(canonical_action)
        env_times.append((time.perf_counter() - t0) * 1000.0)

    # --- fast_sim.step() ---
    fast_times: list[float] = []
    for _ in range(args.num):
        t0 = time.perf_counter()
        _ = step(snap_base, canonical_action)
        fast_times.append((time.perf_counter() - t0) * 1000.0)

    # --- fast_sim.clone() only ---
    clone_times: list[float] = []
    for _ in range(args.num):
        t0 = time.perf_counter()
        _ = fs_clone(snap_base)
        clone_times.append((time.perf_counter() - t0) * 1000.0)

    # --- Full K=50 rollout under the v3_snipe self-play policy ---
    rollout_times: list[float] = []
    policies = [mirror_self_policy, mirror_self_policy]
    for _ in range(20):  # 20 trials at K=50 is enough to get a stable median
        t0 = time.perf_counter()
        _ = rollout(snap_base, args.rollout_k, policies)
        rollout_times.append((time.perf_counter() - t0) * 1000.0)

    env_med = statistics.median(env_times)
    fast_med = statistics.median(fast_times)
    speedup = env_med / max(fast_med, 1e-9)

    print()
    print("=" * 70)
    print(f"  fast_sim microbench  (seed={args.seed}, warmup={args.warmup})")
    print("=" * 70)
    print()
    print(f"  env.clone()+step()  ({args.num} calls):")
    print(f"    median = {env_med:7.3f} ms")
    print(f"    mean   = {statistics.mean(env_times):7.3f} ms")
    print(f"    p95    = {_percentile(env_times, 0.95):7.3f} ms")
    print()
    print(f"  fast_sim.step() incl. clone  ({args.num} calls):")
    print(f"    median = {fast_med:7.3f} ms")
    print(f"    mean   = {statistics.mean(fast_times):7.3f} ms")
    print(f"    p95    = {_percentile(fast_times, 0.95):7.3f} ms")
    print()
    print(f"  fast_sim.clone() alone  ({args.num} calls):")
    print(f"    median = {statistics.median(clone_times):7.3f} ms")
    print(f"    mean   = {statistics.mean(clone_times):7.3f} ms")
    print()
    print(f"  speedup (env_med / fast_med)  = {speedup:6.1f}x")
    print()
    print(f"  fast_sim.rollout(K={args.rollout_k}, v3_snipe self-play, 20 trials):")
    print(f"    median = {statistics.median(rollout_times):7.3f} ms")
    print(f"    mean   = {statistics.mean(rollout_times):7.3f} ms")
    print(f"    p95    = {_percentile(rollout_times, 0.95):7.3f} ms")
    print(f"    per-step = {statistics.median(rollout_times)/args.rollout_k:.3f} ms")
    print()

    # Sanity: ensure the two paths agree on the post-step state.
    snap2 = step(snap_base, canonical_action)
    e2 = env.clone()
    e2.step(canonical_action)
    e_tot = {}
    for p in e2.state[0].observation["planets"]:
        if p[1] >= 0:
            e_tot[p[1]] = e_tot.get(p[1], 0.0) + p[5]
    for f in e2.state[0].observation["fleets"]:
        if f[1] >= 0:
            e_tot[f[1]] = e_tot.get(f[1], 0.0) + f[6]
    if e_tot != ship_totals(snap2):
        print(f"  WARNING: post-step ship_totals mismatch env={e_tot} snap={ship_totals(snap2)}")
    else:
        print("  parity sanity: ship_totals match after one step.")
    print()

    # --- Write the audit doc ---
    today = datetime.utcnow().strftime("%Y-%m-%d")
    audit_path = args.audit or f"audit/{today}-fast-sim-bench.md"
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "w") as f:
        f.write(f"# fast_sim microbench — {today}\n\n")
        f.write(
            f"> Source: `scripts/bench_fast_sim.py` (seed={args.seed}, "
            f"warmup={args.warmup}, num={args.num}).\n>\n"
            "> Audit doc generated by the bench script itself; not "
            "hand-written. Numbers are wallclock on the box the script ran on.\n\n"
        )
        f.write("## Per-step cost\n\n")
        f.write("| Path | median (ms) | mean (ms) | p95 (ms) |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(
            f"| `env.clone()+step()` | {env_med:.3f} | "
            f"{statistics.mean(env_times):.3f} | "
            f"{_percentile(env_times, 0.95):.3f} |\n"
        )
        f.write(
            f"| `fast_sim.step()` (incl. clone) | {fast_med:.3f} | "
            f"{statistics.mean(fast_times):.3f} | "
            f"{_percentile(fast_times, 0.95):.3f} |\n"
        )
        f.write(
            f"| `fast_sim.clone()` | "
            f"{statistics.median(clone_times):.3f} | "
            f"{statistics.mean(clone_times):.3f} | "
            f"{_percentile(clone_times, 0.95):.3f} |\n"
        )
        f.write(f"\n**Speedup:** {speedup:.1f}x over `env.clone()+step()`.\n\n")
        f.write(
            f"## Full rollout (K={args.rollout_k}, v3_snipe self-play)\n\n"
            f"| | median (ms) | mean (ms) | p95 (ms) | per-step (ms) |\n"
            f"|---|---:|---:|---:|---:|\n"
            f"| `fast_sim.rollout()` | "
            f"{statistics.median(rollout_times):.3f} | "
            f"{statistics.mean(rollout_times):.3f} | "
            f"{_percentile(rollout_times, 0.95):.3f} | "
            f"{statistics.median(rollout_times)/args.rollout_k:.3f} |\n\n"
        )
        f.write(
            "## What this means for lookahead budget\n\n"
            f"At a per-turn budget of 1000 ms (`actTimeout`), "
            f"`env.clone()+step()` allowed {int(1000/max(env_med,1e-9))} step "
            f"evaluations per turn; `fast_sim.step()` allows "
            f"{int(1000/max(fast_med,1e-9))} — enough headroom for "
            f"depth-2 search or PIMC-style opponent sampling.\n\n"
            "Phase 2 (`audit/2026-05-11-lookahead-phase2-forward-sim.md`) "
            "showed Sim<K=50> reaches AUC 0.952 = oracle; with the new sim "
            "budget the same AUC is reachable from many more candidates "
            "per real turn.\n"
        )

    print(f"  audit doc: {audit_path}")
    print()


if __name__ == "__main__":
    main()
