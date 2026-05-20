"""Audit analytical agent emits: count sun/oob launches and idle turns.

Reproduces the PI-observed failure modes for the analytical agent:
  1. fleets into sun / out of bounds
  2. endgame idling when we could close out

Wraps lib.joint_solver.mpc.solve_turn so we can:
  - check each emitted launch's fate via predict_fleet_fate
  - count solver_status == "endgame_winning_idle" turns
  - tag each idle turn with whether opp has captureable planets

Usage:
    python scripts/audit_analytical_emits.py --seed 42 --opp baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make

from lib.intent import World
from lib.joint_solver.mpc import solve_turn
from lib.trajectory import predict_fleet_fate


def _planet_by_id(world: World, pid: int):
    return world.planets_by_id.get(int(pid))


def _build_wrapped(emit_log: list, idle_log: list):
    def wrapped(obs, configuration=None):
        moves, diag = solve_turn(obs, configuration, return_diagnostics=True)
        if diag.solver_status == "endgame_winning_idle":
            # Snapshot opp planet count for context.
            obs_d = obs if isinstance(obs, dict) else obs.observation
            me = int(obs_d.get("player", obs_d.get("playerIndex", 0)) or 0)
            world = World.from_obs(obs_d)
            opp_planets = [p for p in world.planets_by_id.values()
                           if int(p.owner) != me and int(p.owner) != -1]
            idle_log.append({
                "step": diag.step,
                "opp_planets": len(opp_planets),
                "opp_planet_ids": [int(p.id) for p in opp_planets],
            })
        for m in moves:
            src_id, angle, ships = int(m[0]), float(m[1]), int(m[2])
            obs_d = obs if isinstance(obs, dict) else obs.observation
            me = int(obs_d.get("player", obs_d.get("playerIndex", 0)) or 0)
            world = World.from_obs(obs_d)
            src = _planet_by_id(world, src_id)
            if src is None:
                continue
            # The emitted move tells us src + angle + ships. Target isn't
            # in the move tuple, so pass src as a dummy target — fate
            # outcome "target" then means "we'd hit src itself" which the
            # spawn-step skip in predict_fleet_fate disregards; outcomes
            # "sun" / "oob" / "planet" / "timeout" are all interesting.
            fate = predict_fleet_fate(src, src, angle, ships, world)
            emit_log.append({
                "step": diag.step,
                "src_id": src_id,
                "angle_deg": round(angle * 180 / 3.14159265, 1),
                "ships": ships,
                "outcome": fate.outcome,
                "hit_planet_id": fate.hit_planet_id,
                "step_of_hit": fate.step,
            })
        return moves
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", default="baseline",
                    choices=["baseline", "v7_0", "random", "lite_greedy"])
    args = ap.parse_args()

    emit_log: list = []
    idle_log: list = []
    me_agent = _build_wrapped(emit_log, idle_log)

    if args.opp == "baseline":
        from agents.baseline.main import agent as opp
    elif args.opp == "lite_greedy":
        from lib.opp_model import lite_greedy_policy as opp
    elif args.opp == "v7_0":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v7_0_bundle", str(REPO / "submissions" / "v7_0_drop_one.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        opp = mod.agent
    else:
        opp = "random"

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([me_agent, opp])
    last = env.steps[-1]

    # Fate histogram.
    fate_hist: dict[str, int] = {}
    for e in emit_log:
        fate_hist[e["outcome"]] = fate_hist.get(e["outcome"], 0) + 1

    print(f"\n=== seed={args.seed}  vs {args.opp}  game ended at step "
          f"{len(env.steps)}  rewards={[s.reward for s in last]} ===")
    print(f"total emits: {len(emit_log)}")
    print("emit fate histogram:")
    for outcome, n in sorted(fate_hist.items(), key=lambda kv: -kv[1]):
        pct = 100 * n / max(1, len(emit_log))
        flag = " <-- BUG" if outcome in ("sun", "oob") else ""
        print(f"  {outcome:<10s} {n:>3d}  ({pct:>5.1f}%){flag}")

    sun_emits = [e for e in emit_log if e["outcome"] == "sun"]
    oob_emits = [e for e in emit_log if e["outcome"] == "oob"]
    if sun_emits:
        print(f"\nfirst 3 sun-kill emits:")
        for e in sun_emits[:3]:
            print(f"  step={e['step']:>3} src={e['src_id']} angle={e['angle_deg']}deg "
                  f"ships={e['ships']} hit_step={e['step_of_hit']}")
    if oob_emits:
        print(f"\nfirst 3 OOB emits:")
        for e in oob_emits[:3]:
            print(f"  step={e['step']:>3} src={e['src_id']} angle={e['angle_deg']}deg "
                  f"ships={e['ships']} hit_step={e['step_of_hit']}")

    print(f"\nendgame_winning_idle turns: {len(idle_log)}")
    if idle_log:
        first = idle_log[0]
        last_idle = idle_log[-1]
        print(f"  first at step {first['step']}: opp had "
              f"{first['opp_planets']} planet(s) {first['opp_planet_ids']}")
        print(f"  last at step {last_idle['step']}: opp had "
              f"{last_idle['opp_planets']} planet(s) {last_idle['opp_planet_ids']}")
        # How many had capturable opp planets?
        with_opp_planets = [i for i in idle_log if i["opp_planets"] > 0]
        print(f"  idle turns where opp still had ≥1 planet: "
              f"{len(with_opp_planets)}/{len(idle_log)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
