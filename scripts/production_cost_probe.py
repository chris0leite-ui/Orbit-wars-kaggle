"""Production-cost probe — what does ONE TURN actually cost the production
agent, broken out by simulator vs policy vs clone?

Why this exists: `scripts/bench_fast_sim.py` benches `fast_sim.step()`
on an idle reused Snapshot and reports ~0.1 ms/step. In a real game the
per-step cost in `fast_sim.rollout(K=50, v3_snipe self-play)` is ~12.5
ms — 100x worse than the headline bench. The gap is policy + clone +
state-traffic. Without measuring the real workload we end up optimising
against fictional numbers (Rule 38: fix-verification reproduces the
failure).

What this probe measures: one full kaggle_environments game with the
production baseline as both seats. Per turn it records:
  - agent_ms     — total time spent inside agent(obs, config).
  - sim_steps    — number of fast_sim.step calls the agent made.
  - sim_ms       — wallclock summed across those step calls.
  - clone_count  — number of fast_sim.clone calls.
  - clone_ms     — wallclock summed across clone calls.

It then buckets turns into early / mid / late game and prints both raw
medians and a derived headroom calculation: how many simulated steps
could fit in the remaining ~950 ms budget if we removed the policy cost.

Usage:
    python -m scripts.production_cost_probe                   # one game, seed 42
    python -m scripts.production_cost_probe --seed 7 --turns 500
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make

import lib.fast_sim as fast_sim
from agents.baseline.main import agent as baseline_agent

# Loader for non-baseline opponents (bundled .py files in submissions/).
import importlib.util


def _load_callable_from_path(path: str):
    """Mirror fast.py's _load_callable for bundled .py agents.

    Register the loaded module in sys.modules under a unique name so
    bundles using @dataclass work (dataclass walks sys.modules to
    resolve forward references).
    """
    mod_name = f"_opp_bundle_{os.path.basename(path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    for name in ("agent", "main", "act"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(f"no callable agent/main/act in {path}")


# Per-turn counters. The monkey-patched step/clone functions append into
# the lists keyed by the CURRENT turn — set by the agent wrapper.
_per_turn: list[dict] = []
_current_turn_record: dict | None = None
_FOCAL_SEAT = 0  # set in main(); only timing for the focal agent is reported in mixed-opp games


def _install_monkeypatches() -> None:
    """Wrap fast_sim.step and fast_sim.clone to record call count + ms."""
    orig_step = fast_sim.step
    orig_clone = fast_sim.clone

    def timed_step(snap, actions_per_seat, *, in_place: bool = False):
        t0 = time.perf_counter()
        result = orig_step(snap, actions_per_seat, in_place=in_place)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if _current_turn_record is not None:
            _current_turn_record["sim_steps"] += 1
            _current_turn_record["sim_ms"] += dt_ms
        return result

    def timed_clone(snap):
        t0 = time.perf_counter()
        result = orig_clone(snap)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if _current_turn_record is not None:
            _current_turn_record["clone_count"] += 1
            _current_turn_record["clone_ms"] += dt_ms
        return result

    fast_sim.step = timed_step
    fast_sim.clone = timed_clone

    # Also patch the names the chooser imported. Bundler-flavoured
    # imports (`from lib.fast_sim import step as fs_step`) bind at
    # import time, so replacing the attribute on fast_sim doesn't reach
    # already-imported callers. Reach in and rebind.
    import agents.baseline.chooser as _ch
    import agents.baseline.chooser_trajectory as _cht
    _ch.fs_step = timed_step
    _ch.fs_clone = timed_clone
    _cht.fs_step = timed_step
    _cht.fs_clone = timed_clone


def _instrumented_agent(obs, configuration=None):
    """Wrap baseline.agent with per-turn timing + counter setup."""
    global _current_turn_record
    rec = {
        "turn": int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0)),
        "seat": int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0)),
        "agent_ms": 0.0,
        "sim_steps": 0,
        "sim_ms": 0.0,
        "clone_count": 0,
        "clone_ms": 0.0,
    }
    _current_turn_record = rec
    t0 = time.perf_counter()
    try:
        action = baseline_agent(obs, configuration)
    finally:
        rec["agent_ms"] = (time.perf_counter() - t0) * 1000.0
        _per_turn.append(rec)
        _current_turn_record = None
    return action


def _bucket(rec: dict, turns: int) -> str:
    t = rec["turn"]
    early = turns // 3
    late = 2 * turns // 3
    if t < early:
        return "early"
    if t < late:
        return "mid"
    return "late"


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _report(rows: list[dict], label: str) -> str:
    if not rows:
        return f"  [{label}] empty\n"
    agent = [r["agent_ms"] for r in rows]
    sim = [r["sim_ms"] for r in rows]
    clone = [r["clone_ms"] for r in rows]
    sim_steps = [r["sim_steps"] for r in rows]
    sim_step_ms = [(r["sim_ms"] / r["sim_steps"]) for r in rows if r["sim_steps"] > 0]
    clone_per = [(r["clone_ms"] / r["clone_count"]) for r in rows if r["clone_count"] > 0]
    interp_ms = [r["agent_ms"] - r["sim_ms"] for r in rows]  # everything outside fast_sim.step
    lines = [
        f"  [{label}]  n_turns = {len(rows)}",
        f"    agent_ms     median {statistics.median(agent):7.2f}  mean {statistics.mean(agent):7.2f}  p95 {_pct(agent, 0.95):7.2f}",
        f"    sim_ms       median {statistics.median(sim):7.2f}  mean {statistics.mean(sim):7.2f}  p95 {_pct(sim, 0.95):7.2f}",
        f"    non-sim_ms   median {statistics.median(interp_ms):7.2f}  mean {statistics.mean(interp_ms):7.2f}  (policy + proposer + bookkeeping)",
        f"    clone_ms     median {statistics.median(clone):7.2f}  mean {statistics.mean(clone):7.2f}",
        f"    sim_steps    median {statistics.median(sim_steps):7.1f}  mean {statistics.mean(sim_steps):7.1f}",
        f"    ms/sim_step  median {statistics.median(sim_step_ms) if sim_step_ms else float('nan'):7.3f}",
        f"    ms/clone     median {statistics.median(clone_per) if clone_per else float('nan'):7.3f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="comma list, overrides --seed (e.g. 7,13,99,42,3)")
    ap.add_argument("--opp", default=None, help="opponent path or builtin (e.g. submissions/v7_0_drop_one.py)")
    ap.add_argument("--focal-seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--turns", type=int, default=500, help="episodeSteps cap")
    ap.add_argument("--audit", default=None, help="audit doc output path")
    args = ap.parse_args()

    global _FOCAL_SEAT, _per_turn
    _FOCAL_SEAT = args.focal_seat
    _install_monkeypatches()

    if args.opp is not None:
        opp_agent = _load_callable_from_path(args.opp)
        agents_pair = [None, None]
        agents_pair[args.focal_seat] = _instrumented_agent
        agents_pair[1 - args.focal_seat] = opp_agent
        setup_label = f"focal=baseline (seat {args.focal_seat}) vs opp={args.opp}"
    else:
        agents_pair = [_instrumented_agent, _instrumented_agent]
        setup_label = "baseline-vs-baseline (both seats timed)"

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    print(f"Setup: {setup_label}")
    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": args.turns})
        env.reset(num_agents=2)
        before = len(_per_turn)
        print(f"  seed={seed} ...", end=" ", flush=True)
        t_game = time.perf_counter()
        env.run(agents_pair)
        t_game = time.perf_counter() - t_game
        added = len(_per_turn) - before
        print(f"{t_game:.1f}s ({added} agent calls)")
    print(f"All games complete ({len(_per_turn)} agent calls total).\n")

    # Filter to focal seat only when in a mixed-opponent game; if it's
    # baseline-vs-baseline we want both seats.
    if args.opp is not None:
        _per_turn = [r for r in _per_turn if r["seat"] == args.focal_seat]

    # Phase buckets
    by_bucket: dict[str, list[dict]] = {"early": [], "mid": [], "late": []}
    for r in _per_turn:
        by_bucket[_bucket(r, args.turns)].append(r)

    # ---- Reporting ----
    out_lines = []
    out_lines.append(f"# production-cost probe — {datetime.utcnow().strftime('%Y-%m-%d')}")
    out_lines.append("")
    out_lines.append(
        f"> Source: `scripts/production_cost_probe.py` (seed={args.seed}, "
        f"episodeSteps={args.turns}, baseline-vs-baseline).\n"
        "> One full game; both seats are `agents/baseline/main.py`. Counters monkey-patch "
        "`lib.fast_sim.step` / `clone` and bind into `agents.baseline.chooser{,_trajectory}` "
        "to catch already-imported references.\n"
    )
    out_lines.append("## Per-turn cost (all seats merged)")
    out_lines.append("")

    print("=" * 78)
    print("  Per-turn cost  (all seats merged)")
    print("=" * 78)

    txt_all = _report(_per_turn, "ALL")
    print(txt_all)
    out_lines.append("```")
    out_lines.append(txt_all.rstrip())
    out_lines.append("```\n")
    for label in ("early", "mid", "late"):
        txt = _report(by_bucket[label], label)
        print(txt)
        out_lines.append("```")
        out_lines.append(txt.rstrip())
        out_lines.append("```\n")

    # ---- Headroom derivation ----
    print("=" * 78)
    print("  Headroom analysis")
    print("=" * 78)

    all_agent = [r["agent_ms"] for r in _per_turn]
    all_sim = [r["sim_ms"] for r in _per_turn]
    all_sim_steps = [r["sim_steps"] for r in _per_turn]
    nonsim = [r["agent_ms"] - r["sim_ms"] for r in _per_turn]
    med_agent = statistics.median(all_agent)
    med_sim = statistics.median(all_sim)
    med_nonsim = statistics.median(nonsim)
    med_sim_steps = statistics.median(all_sim_steps)
    sim_per_step_ms = (sum(all_sim) / max(sum(all_sim_steps), 1))

    lines = [
        f"  Per-turn medians:  agent {med_agent:.1f} ms  =  sim {med_sim:.1f} ms  +  non-sim {med_nonsim:.1f} ms",
        f"  Median sim-step count per turn:  {med_sim_steps:.0f}",
        f"  Mean ms per sim step (in production traffic):  {sim_per_step_ms:.3f} ms",
        f"  Remaining of 950 ms budget after current chooser:  {950 - med_agent:.0f} ms",
        f"",
        f"  Headroom counterfactuals (median turn):",
        f"    if sim were 2x cheaper:   turn = {med_nonsim + med_sim/2:.0f} ms  →  +{med_sim/2:.0f} ms free",
        f"    if sim were 10x cheaper:  turn = {med_nonsim + med_sim/10:.0f} ms  →  +{med_sim*9/10:.0f} ms free",
        f"    if non-sim (policy) were 2x cheaper:  turn = {med_nonsim/2 + med_sim:.0f} ms  →  +{med_nonsim/2:.0f} ms free",
        f"    if non-sim (policy) were 10x cheaper: turn = {med_nonsim/10 + med_sim:.0f} ms  →  +{med_nonsim*9/10:.0f} ms free",
        f"",
        f"  At current cost {sim_per_step_ms:.3f} ms/sim_step, 950 ms budget = "
        f"{int(950 / max(sim_per_step_ms, 1e-9))} sim steps theoretical max.",
        f"  Today we use {med_sim_steps:.0f} (={med_sim_steps * sim_per_step_ms:.0f} ms) — leaving headroom only "
        f"after non-sim costs settle.",
    ]
    headroom_txt = "\n".join(lines)
    print(headroom_txt)
    print()
    out_lines.append("## Headroom analysis")
    out_lines.append("")
    out_lines.append("```")
    out_lines.append(headroom_txt)
    out_lines.append("```")
    out_lines.append("")

    # ---- Write audit ----
    today = datetime.utcnow().strftime("%Y-%m-%d")
    audit_path = args.audit or f"audit/{today}-production-cost-probe.md"
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"  audit doc: {audit_path}")


if __name__ == "__main__":
    main()
