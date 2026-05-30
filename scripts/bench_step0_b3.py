"""B.3 Step 0 — fast_sim verification gate.

Four small benches decide whether `lib.fast_sim` can host the live
champion `baseline_pv_eta` as a policy callback at the speed + parity
needed for the B.3 CRN-paired advantage labelling cycle.

V0.0 pv_eta determinism on identical obs — essential for CRN to cancel
V0.1 state-leakage probe (rollout determinism with/without reset)
V0.2 timing (K=10 rollout p50/p95/max)
V0.3 parity vs env.step (action-replay; isolates fast_sim STEP correctness)

PASS on all four → stage 2 corpus cost ~7.5 h CPU local (feasible).
FAIL → fall back to env.clone+step at ~150 h CPU (Kaggle parallel).

Run: python scripts/bench_step0_b3.py [--wallclock-ms 100] [--seeds 10]
                                       [--n-timing 20] [--audit-out PATH]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("BASELINE_WALLCLOCK_MS", "100")

from kaggle_environments import make  # noqa: E402

from lib.fast_sim import clone, from_obs, rollout, ship_totals, step  # noqa: E402

import submissions._imported.baseline_pv_eta as pve  # noqa: E402


def make_pv_eta_policy(configuration):
    def policy(obs):
        return pve.agent(obs, configuration)

    return policy


def reset_pv_eta_state() -> None:
    """Reset every module-level mutable in baseline_pv_eta we know about."""
    pve._reset_state_for_tests()
    pve._PENDING_LAUNCHES.clear()


def random_warmup_action(obs0, num_seats: int, rng: random.Random) -> list[list]:
    """Mirror of tests/test_fast_sim_parity._make_actions; used only for warmup."""
    actions: list[list] = [[] for _ in range(num_seats)]
    for p in obs0["planets"]:
        owner = p[1]
        if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
            actions[owner].append([p[0], rng.uniform(0.0, 6.283), int(p[5] // 2)])
    return actions


def env_totals(env) -> dict[int, float]:
    obs = env.state[0].observation
    totals: dict[int, float] = {}
    for p in obs["planets"]:
        if p[1] >= 0:
            totals[p[1]] = totals.get(p[1], 0.0) + p[5]
    for f in obs["fleets"]:
        if f[1] >= 0:
            totals[f[1]] = totals.get(f[1], 0.0) + f[6]
    return totals


def fresh_env_and_snap(seed: int, warmup_steps: int = 20):
    """Build env at seed, advance `warmup_steps` ticks with random actions,
    return (env, snap) where snap mirrors env's state."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    rng = random.Random(seed * 7 + 1)
    for _ in range(warmup_steps):
        acts = random_warmup_action(env.state[0].observation, 2, rng)
        env.step(acts)
    snap = from_obs(
        env.state[0].observation,
        env.configuration,
        episode_seed=env.info["seed"],
        num_seats=2,
    )
    return env, snap


def bench_v00(args) -> dict:
    """Is pv_eta deterministic on the same obs from a clean state?

    CRN-paired advantage requires this — if pv_eta jitters across calls,
    `margin(action) − margin(idle)` is contaminated by policy noise that
    DOESN'T cancel.
    """
    env, _ = fresh_env_and_snap(seed=42)
    obs = env.state[0].observation
    cfg = env.configuration
    reset_pv_eta_state(); a1 = pve.agent(obs, cfg)
    reset_pv_eta_state(); a2 = pve.agent(obs, cfg)
    reset_pv_eta_state(); a3 = pve.agent(obs, cfg)
    same12 = a1 == a2
    same13 = a1 == a3
    return {
        "name": "V0.0 pv_eta determinism on identical obs",
        "pass": same12 and same13,
        "detail": f"call1==call2: {same12}, call1==call3: {same13}; len(a1)={len(a1)}",
    }


def bench_v01(args) -> dict:
    """Two K=10 rollouts on the same snap.

    WITH reset between rollouts → must be identical (substrate determinism).
    WITHOUT reset → diverges iff state leaks across rollouts; informational.
    """
    env, snap_warm = fresh_env_and_snap(seed=42)
    cfg = env.configuration
    policy = make_pv_eta_policy(cfg)

    reset_pv_eta_state()
    snap_a = rollout(snap_warm, 10, [policy, policy])
    tot_a = ship_totals(snap_a)

    reset_pv_eta_state()
    snap_b = rollout(snap_warm, 10, [policy, policy])
    tot_b = ship_totals(snap_b)
    with_reset_match = tot_a == tot_b

    # Now run a third rollout WITHOUT resetting — state from rollout 2 persists.
    snap_c = rollout(snap_warm, 10, [policy, policy])
    tot_c = ship_totals(snap_c)
    without_reset_match = tot_a == tot_c

    leak_note = (
        "(no leakage signal)"
        if without_reset_match
        else "(LEAKAGE DETECTED — without reset, rollout differs from with-reset baseline; with-reset is the safe path)"
    )
    return {
        "name": "V0.1 K=10 rollout determinism (with/without state reset)",
        "pass": with_reset_match,
        "detail": (
            f"with-reset      rollout-1 totals={tot_a}\n"
            f"                   rollout-2 totals={tot_b}  match={with_reset_match}\n"
            f"                without-reset rollout-3 totals={tot_c}  match-to-1={without_reset_match} {leak_note}"
        ),
    }


def bench_v02(args) -> dict:
    """Time N K=10 rollouts after 2 warmup rollouts. Pass: p95 < 100ms."""
    env, snap_warm = fresh_env_and_snap(seed=42)
    cfg = env.configuration
    policy = make_pv_eta_policy(cfg)

    for _ in range(2):
        reset_pv_eta_state()
        rollout(snap_warm, 10, [policy, policy])

    times_ms: list[float] = []
    for _ in range(args.n_timing):
        reset_pv_eta_state()
        t0 = time.perf_counter()
        rollout(snap_warm, 10, [policy, policy])
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[max(0, int(len(times_ms) * 0.95) - 1)]
    p_max = max(times_ms)
    wallclock = os.environ.get("BASELINE_WALLCLOCK_MS", "<default>")
    return {
        "name": f"V0.2 K=10 rollout timing (N={args.n_timing}, wallclock_ms={wallclock})",
        "pass": p95 < 100.0,
        "detail": f"p50={p50:.1f}ms  p95={p95:.1f}ms  max={p_max:.1f}ms  (target p95 < 100ms)",
    }


def bench_v03(args) -> dict:
    """Action-replay parity: env.step driven by pv_eta vs fast_sim.step on the
    same recorded actions. Isolates fast_sim STEP correctness from pv_eta
    non-determinism (which V0.0 probes separately)."""
    candidate_seeds = [7, 42, 100, 314, 2026, 1, 11, 31, 99, 555, 41, 137]
    seeds = candidate_seeds[: args.seeds]
    rows = []
    for s in seeds:
        env, snap_warm = fresh_env_and_snap(seed=s)
        ref_env = env.clone()
        cfg = ref_env.configuration

        # Drive env with pv_eta for K=10 ticks, recording actions.
        reset_pv_eta_state()
        action_trace: list[list[list]] = []
        for _ in range(10):
            acts = [
                pve.agent(ref_env.state[seat].observation, cfg) for seat in range(2)
            ]
            action_trace.append(acts)
            ref_env.step(acts)
        env_tot = env_totals(ref_env)

        # Replay the recorded action trace through fast_sim.
        snap = clone(snap_warm)
        for acts in action_trace:
            snap = step(snap, acts, in_place=True)
        fs_tot = ship_totals(snap)

        match = env_tot == fs_tot
        rows.append((s, match, env_tot, fs_tot))

    n_pass = sum(1 for _, m, _, _ in rows if m)
    seed_lines = [
        f"seed={s} match={m} env={e} fs={f}"
        for s, m, e, f in rows
    ]
    return {
        "name": f"V0.3 parity (action-replay, K=10, seeds={len(seeds)})",
        "pass": n_pass == len(seeds),
        "detail": f"{n_pass}/{len(seeds)} seeds matched\n                " + "\n                ".join(seed_lines),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallclock-ms", type=int, default=None,
        help="Override BASELINE_WALLCLOCK_MS for this run (default: 100ms via setdefault)",
    )
    parser.add_argument("--seeds", type=int, default=10, help="V0.3 seed count (max 12)")
    parser.add_argument("--n-timing", type=int, default=20, help="V0.2 rollout count")
    parser.add_argument("--audit-out", type=str, default=None,
                        help="Optional path to write a verdict markdown summary")
    args = parser.parse_args()
    if args.wallclock_ms is not None:
        os.environ["BASELINE_WALLCLOCK_MS"] = str(args.wallclock_ms)

    print("=== B.3 Step 0 — fast_sim verification (pv_eta as policy) ===")
    print(f"wallclock_ms = {os.environ.get('BASELINE_WALLCLOCK_MS')}")
    print()

    benches = [bench_v00, bench_v01, bench_v02, bench_v03]
    results: list[dict] = []
    for b in benches:
        t0 = time.perf_counter()
        r = b(args)
        r["wallclock_s"] = time.perf_counter() - t0
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['name']}  ({r['wallclock_s']:.1f}s)")
        print(f"        {r['detail']}")
        print()

    all_pass = all(r["pass"] for r in results)
    print("=" * 72)
    verdict = "PASS" if all_pass else "FAIL"
    suit = "SUITABLE" if all_pass else "NOT SUITABLE"
    print(f"VERDICT: {verdict} — fast_sim is {suit} as B.3 stage-2 substrate")
    if all_pass:
        print("Stage 2 cost estimate: ~7.5 h CPU local (100 games × top-N=10 × K=10).")
    else:
        print("Fallback: env.clone+step path (stage 2 ~150 h CPU local; needs Kaggle parallel).")

    if args.audit_out:
        out = Path(args.audit_out)
        lines = [
            f"# B.3 Step 0 verification — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
            "",
            f"`BASELINE_WALLCLOCK_MS = {os.environ.get('BASELINE_WALLCLOCK_MS')}`",
            "",
            "| Bench | Verdict | Wallclock (s) |",
            "|---|---|---:|",
        ]
        for r in results:
            lines.append(
                f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['wallclock_s']:.1f} |"
            )
        lines.append("")
        lines.append(f"**Overall verdict: {verdict}**")
        lines.append("")
        for r in results:
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append("```")
            lines.append(r["detail"])
            lines.append("```")
            lines.append("")
        out.write_text("\n".join(lines))
        print(f"\nAudit summary written to {out}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
