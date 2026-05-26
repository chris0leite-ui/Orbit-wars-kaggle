"""One-off diagnostic: sa_online vs simple/roi, step-by-step for first N turns.

Logs per turn: P0 + P1 action counts, P0 + P1 planet/ship totals, agent
wall time, and whether sa_online emitted anything from its cached plan.

Goal: find out WHY sa_online gets eliminated against ROI on seed 7542
when opp_model=ROI matches the actual opp.
"""
from __future__ import annotations

import os
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_agent_module(path):
    spec = spec_from_file_location(f"a_{path.name}", str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_step(obs):
    if isinstance(obs, dict):
        return int(obs.get("step", 0))
    return int(getattr(obs, "step", 0))


def main():
    seed = int(os.environ.get("DIAG_SEED", "7542"))
    steps = int(os.environ.get("DIAG_STEPS", "200"))
    trace_n = int(os.environ.get("DIAG_TRACE_N", "50"))

    # Set sa_online's module-load env vars
    os.environ["SA_SEED"] = str(seed)
    os.environ["SA_EPISODE_STEPS"] = str(steps)
    os.environ.setdefault("SA_ITER_INIT", "50")
    os.environ.setdefault("SA_ITER_STEP", "5")
    os.environ.setdefault("SA_OPP_AGENT", "agents/simple/roi.py")

    from kaggle_environments import make
    env = make("orbit_wars",
               configuration={"seed": seed, "episodeSteps": steps},
               debug=False)
    env.reset(num_agents=2)

    t_load = time.perf_counter()
    sa_mod = _load_agent_module(REPO / "agents" / "sa_online" / "main.py")
    load_wall = time.perf_counter() - t_load
    print(f"[diag] sa_online module load: {load_wall:.1f}s "
          f"(_PLAN_READY-equivalent: plan_turns={len(sa_mod._PLAN_BY_TURN)})",
          file=sys.stderr)

    # Inspect the cached plan's emission count + spread
    plan = sa_mod._PLAN_BY_TURN
    n_emit_first50 = sum(len(acts) for t, acts in plan.items() if t < trace_n)
    n_emit_total = sum(len(acts) for acts in plan.values())
    print(f"[diag] cached plan: {n_emit_total} total emissions, "
          f"{n_emit_first50} in first {trace_n} turns",
          file=sys.stderr)
    if n_emit_total > 0:
        emit_turns = sorted(plan.keys())
        print(f"[diag] plan turns: min={emit_turns[0]} "
              f"max={emit_turns[-1]} count={len(emit_turns)}",
              file=sys.stderr)
        # Sample first 5 emissions
        for t in emit_turns[:5]:
            print(f"  t={t}: {plan[t]}", file=sys.stderr)

    sa_agent = sa_mod.agent
    roi_mod = _load_agent_module(REPO / "agents" / "simple" / "roi.py")
    roi_agent = roi_mod.agent

    print(f"\n=== diag sa_online vs roi seed={seed} steps={steps} "
          f"trace_n={trace_n} ===", file=sys.stderr)
    print(f"\n  t | sa_w(ms) sa_acts roi_acts | P0[pl/sh/fl] P1[pl/sh/fl] "
          f"neut | cached_t_in_plan", file=sys.stderr)
    print("----+-----------------------------+----------------------------"
          "-------+-----------------", file=sys.stderr)

    state = env.steps[0]
    for step in range(trace_n):
        obs0 = state[0]["observation"] if isinstance(state[0], dict) else state[0].observation
        obs1 = state[1]["observation"] if isinstance(state[1], dict) else state[1].observation

        t0 = time.perf_counter()
        a0 = sa_agent(obs0, env.configuration)
        sa_wall_ms = (time.perf_counter() - t0) * 1000.0
        a1 = roi_agent(obs1)

        # Stats from obs0 (same planets list for both)
        od0 = obs0 if isinstance(obs0, dict) else dict(obs0)
        planets = od0.get("planets") or []
        fleets = od0.get("fleets") or []
        p0_pl = sum(1 for p in planets if int(p[1]) == 0)
        p1_pl = sum(1 for p in planets if int(p[1]) == 1)
        nu = sum(1 for p in planets if int(p[1]) == -1)
        p0_sh = sum(int(p[5]) for p in planets if int(p[1]) == 0)
        p1_sh = sum(int(p[5]) for p in planets if int(p[1]) == 1)
        p0_fl = sum(int(f[6]) for f in fleets if int(f[1]) == 0)
        p1_fl = sum(int(f[6]) for f in fleets if int(f[1]) == 1)
        in_plan = step in sa_mod._PLAN_BY_TURN
        cached_n = len(sa_mod._PLAN_BY_TURN.get(step, []))

        print(f"{step:>4d} |{sa_wall_ms:>8.0f} {len(a0):>7d} {len(a1):>8d} | "
              f"{p0_pl:>2d}/{p0_sh:>4d}/{p0_fl:>4d} "
              f"{p1_pl:>2d}/{p1_sh:>4d}/{p1_fl:>4d} "
              f"{nu:>3d}  | {'Y' if in_plan else 'N'}({cached_n})",
              file=sys.stderr)

        # Also print specific actions when non-empty
        if a0 or a1:
            if a0:
                print(f"    sa_online emits: {a0[:4]}{' …' if len(a0) > 4 else ''}",
                      file=sys.stderr)
            if a1:
                print(f"    roi      emits: {a1[:4]}{' …' if len(a1) > 4 else ''}",
                      file=sys.stderr)

        state = env.step([a0, a1])
        s0 = state[0]
        status0 = s0.get("status") if isinstance(s0, dict) else getattr(s0, "status", "ACTIVE")
        if status0 != "ACTIVE":
            print(f"\n[diag] game ended at step {step + 1}: status={status0}",
                  file=sys.stderr)
            break

    # Final summary
    final = env.steps[-1] if hasattr(env, "steps") else state
    obs = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
    od = obs if isinstance(obs, dict) else dict(obs)
    planets = od.get("planets") or []
    fleets = od.get("fleets") or []
    p0t = sum(int(p[5]) for p in planets if int(p[1]) == 0) + sum(int(f[6]) for f in fleets if int(f[1]) == 0)
    p1t = sum(int(p[5]) for p in planets if int(p[1]) == 1) + sum(int(f[6]) for f in fleets if int(f[1]) == 1)
    print(f"\n[diag] after {trace_n} steps (or end): P0={p0t} P1={p1t}",
          file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
