"""Inspect the opening MILP's plan directly (no opp).

Builds a TurnContext from a seed at step 0, calls `opening_plan`, and
prints each ScheduleEntry with target geometry and feasibility breakdown.

Usage:
    python scripts/inspect_opening_plan.py --seed 384458460
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402

from lib.intent import World  # noqa: E402
from lib.joint_solver.opening_planner import (  # noqa: E402
    opening_plan, _build_candidates,
)
from lib.world_model import WorldModel, build_arrival_ledger  # noqa: E402


def _dist(a, b) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--at_step", type=int, default=0,
                    help="Replay env to this step before inspecting plan.")
    args = ap.parse_args()

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.reset(num_agents=2)
    # Run a few turns of no-op to advance.
    for _ in range(args.at_step):
        env.step([[], []])
    obs = env.state[0]["observation"]
    omega = obs.get("angular_velocity", 0.0)

    world = World.from_obs(obs)
    fleets = obs.get("fleets", []) or []
    planets_obj = list(world.planets_by_id.values())
    ledger = build_arrival_ledger(fleets, planets_obj, omega)
    model = WorldModel(ledger=ledger, timelines={}, horizon=200)

    print(f"# Seed {args.seed}, step {obs.get('step', 0)}, omega={omega:.5f}")
    my_id = 0
    my_planets = [p for p in obs["planets"] if int(p[1]) == my_id]
    opp_planets = [p for p in obs["planets"] if int(p[1]) == 1]
    neutrals = [p for p in obs["planets"] if int(p[1]) == -1]
    print(f"# my_planets: {[int(p[0]) for p in my_planets]}")
    print(f"# opp_planets: {[int(p[0]) for p in opp_planets]}")
    print(f"# neutrals: {len(neutrals)}")
    print()

    # Show ranked targets per source.
    for src in my_planets:
        sid = int(src[0])
        print(f"# Source p{sid} at ({src[2]:.1f},{src[3]:.1f}) "
              f"ships={src[5]} prod={src[6]}")
        scored = []
        for t in obs["planets"]:
            tid = int(t[0])
            if tid == sid or int(t[1]) == my_id:
                continue
            d = _dist(src, t)
            score = float(t[6]) / (d + 1.0)
            scored.append((score, d, t))
        scored.sort(reverse=True)
        print(f"  Top 8 targets by prod/(dist+1):")
        for score, d, t in scored[:8]:
            print(f"     p{int(t[0]):>2}  d={d:5.1f}  prod={t[6]}  "
                  f"ships={t[5]:>3}  owner={'OPP' if int(t[1])==1 else 'N'}  "
                  f"score={score:.3f}")

    # Show all candidates the MILP would see.
    print()
    cands, _wf = _build_candidates(world, model, my_id, 2)
    print(f"# {len(cands)} candidates fed to MILP:")
    print(f"  {'col':>3} {'fire':>5} {'src':>4} {'tgt':>4} {'ships':>5} "
          f"{'eta':>4} {'arr':>4} {'angle':>7} {'value':>8}")
    for c in sorted(cands, key=lambda c: (c.fire_step, c.tgt_id)):
        print(f"  {c.column_id:>3} {c.fire_step:>5} {c.src_id:>4} {c.tgt_id:>4} "
              f"{c.ships:>5} {c.eta:>4} {c.arrival:>4} {c.angle:>7.2f} "
              f"{c.value:>8.1f}")

    # Run the opening planner.
    print()
    plan = opening_plan(world, model, my_id, 2)
    print(f"# OpeningPlan: status={plan.status} n_vars={plan.n_vars} "
          f"n_constraints={plan.n_constraints} obj={plan.objective:.1f}")
    print(f"#   pruning waterfall: {plan.pruning_waterfall}")
    print(f"#   schedule has {len(plan.schedule)} entries:")
    print()
    print(f"  {'fire':>5} {'src':>4} {'tgt':>4} {'ships':>5} {'eta':>4} "
          f"{'arr':>4} {'angle':>7} {'value':>8}  desc")
    for e in plan.schedule:
        src = next(p for p in obs["planets"] if int(p[0]) == e.src_id)
        tgt = next(p for p in obs["planets"] if int(p[0]) == e.tgt_id)
        d = math.hypot(float(src[2]) - float(tgt[2]),
                       float(src[3]) - float(tgt[3]))
        print(f"  {e.fire_step:>5} {e.src_id:>4} {e.tgt_id:>4} {e.ships:>5} "
              f"{e.eta:>4} {e.fire_step + e.eta:>4} {e.angle:>7.2f} {e.value:>8.1f}  "
              f"src→tgt dist={d:.1f} tgt_prod={tgt[6]} tgt_ships={tgt[5]} "
              f"tgt_owner={'OPP' if int(tgt[1])==1 else 'N'}")


if __name__ == "__main__":
    main()
