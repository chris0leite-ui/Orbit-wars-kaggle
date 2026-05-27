"""Subprocess-equivalent timing trace: run lagrange_simple vs random in the
SAME way the gate does (fresh subprocess via env.run with the agent path),
but instrument per-turn wall time + cands count via env-var-controlled
logging inside the agent.

Trick: import the agent module, monkey-patch enumerate_candidates +
solve to record per-call timing into a file, then env.run as usual.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("BASELINE_ORBITAL_SAFETY", "1")
os.environ.setdefault("KINEMATIC_TABLE_ENABLED", "1")

from kaggle_environments import make
from agents.lagrange_simple import main as ls_main
from agents.lagrange_simple import score as ls_score
from agents.lagrange_simple import dual as ls_dual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--focal-seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--act-timeout", type=float, default=1.0,
                    help="actTimeout in seconds (default 1.0, matches Kaggle)")
    args = ap.parse_args()

    log_path = Path(f"/tmp/perturn_seed{args.seed}_seat{args.focal_seat}.jsonl")
    log_path.unlink(missing_ok=True)

    orig_enum = ls_score.enumerate_candidates
    orig_solve = ls_dual.solve

    state = {"step_idx": 0}

    def timed_enum(world, model, my_id, omega, comet_ids=None):
        t0 = time.perf_counter()
        out = orig_enum(world, model, my_id, omega, comet_ids)
        state["enum_ms"] = (time.perf_counter() - t0) * 1000.0
        state["n_cands"] = len(out)
        state["n_solo"] = sum(1 for c in out if not c.is_partial)
        state["my_p"] = len([p for p in world.planets_by_id.values()
                             if int(p.owner) == int(my_id)])
        state["opp_p"] = len([p for p in world.planets_by_id.values()
                              if int(p.owner) != int(my_id) and int(p.owner) >= 0])
        return out

    def timed_solve(candidates, source_budgets, source_prods=None, **kw):
        t0 = time.perf_counter()
        out = orig_solve(candidates, source_budgets, source_prods, **kw)
        state["dual_ms"] = (time.perf_counter() - t0) * 1000.0
        state["n_picks"] = len(out)
        return out

    # main.py bound the names at import; patch in main's namespace too.
    ls_score.enumerate_candidates = timed_enum
    ls_dual.solve = timed_solve
    ls_main.enumerate_candidates = timed_enum
    ls_main.solve_dual = timed_solve

    def focal(obs, configuration=None):
        t0 = time.perf_counter()
        state["enum_ms"] = 0.0
        state["dual_ms"] = 0.0
        state["n_cands"] = 0
        state["n_solo"] = 0
        state["n_picks"] = 0
        state["my_p"] = 0
        state["opp_p"] = 0
        out = ls_main.agent(obs, configuration)
        wall = (time.perf_counter() - t0) * 1000.0
        step = int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0))
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step,
                "wall_ms": wall,
                "enum_ms": state["enum_ms"],
                "dual_ms": state["dual_ms"],
                "n_cands": state["n_cands"],
                "n_solo": state["n_solo"],
                "n_picks": state["n_picks"],
                "my_p": state["my_p"],
                "opp_p": state["opp_p"],
            }) + "\n")
        return out

    env = make(
        "orbit_wars",
        configuration={
            "seed": args.seed,
            "actTimeout": float(args.act_timeout),
            "agentTimeout": 600,    # don't kill agent permanently
            "runTimeout": 3600,
        },
        debug=False,
    )
    env.reset(2)
    if args.focal_seat == 0:
        env.run([focal, "random"])
    else:
        env.run(["random", focal])

    final = env.steps[-1]
    obs_final = final[0]["observation"]
    planets = obs_final.get("planets", []) if isinstance(obs_final, dict) \
        else getattr(obs_final, "planets", [])
    n_p0 = sum(1 for p in planets if int(p[1]) == 0)
    n_p1 = sum(1 for p in planets if int(p[1]) == 1)
    n_steps = len(env.steps)
    print(f"seed={args.seed} focal_seat={args.focal_seat} actTimeout={args.act_timeout}  "
          f"final: P0={n_p0}p P1={n_p1}p steps={n_steps}", flush=True)

    # Per-turn analysis
    import collections
    rows = [json.loads(l) for l in open(log_path)]
    over = [r for r in rows if r["wall_ms"] > args.act_timeout * 1000.0]
    print(f"  rows={len(rows)}  over_actTimeout={len(over)}  "
          f"max_wall_ms={max((r['wall_ms'] for r in rows), default=0):.0f}  "
          f"p95_wall_ms={sorted(r['wall_ms'] for r in rows)[max(0,int(len(rows)*0.95)-1)] if rows else 0:.0f}")
    # Histogram of wall_ms buckets
    buckets = collections.Counter()
    for r in rows:
        ms = r["wall_ms"]
        if ms < 100: b = "<100"
        elif ms < 250: b = "100-250"
        elif ms < 500: b = "250-500"
        elif ms < 1000: b = "500-1000"
        elif ms < 2000: b = "1000-2000"
        else: b = ">=2000"
        buckets[b] += 1
    for b in ["<100", "100-250", "250-500", "500-1000", "1000-2000", ">=2000"]:
        print(f"    {b:>10}ms  {buckets[b]:>3}")
    # First 5 overruns:
    if over:
        print("  first 5 overruns:")
        for r in over[:5]:
            print(f"    step={r['step']:>3}  wall={r['wall_ms']:.0f}ms  "
                  f"enum={r['enum_ms']:.0f}ms  dual={r['dual_ms']:.0f}ms  "
                  f"my={r['my_p']} opp={r['opp_p']}  "
                  f"cands={r['n_cands']}(solo={r['n_solo']}) picks={r['n_picks']}")


if __name__ == "__main__":
    main()
