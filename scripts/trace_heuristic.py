"""Trace agents/heuristic decisions across one game for behavior inspection.

Drives the env step-by-step (not env.run) so per-turn prints are visible.

Usage:
    python scripts/trace_heuristic.py [--seed N] [--opp data/main.py] [--steps 500]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

from kaggle_environments import make

from agents.heuristic.main import agent as heuristic
from lib.intent import World
from lib.world_model import WorldModel, fleet_target_planet


def load_opp(path: str):
    if path.endswith(".py"):
        name = f"_traceopp_{os.path.splitext(os.path.basename(path))[0]}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # register before exec for @dataclass bundles
        spec.loader.exec_module(mod)
        return mod.agent
    raise ValueError("only .py opponents")


def trace(obs, my_id):
    world = World.from_obs(obs)
    wm = WorldModel.from_world(world)
    step = world.step

    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    enemy_planets = [p for p in world.planets_by_id.values() if p.owner not in (my_id, -1)]
    neutrals = [p for p in world.planets_by_id.values() if p.owner == -1]

    my_total = sum(p.ships for p in my_planets)
    enemy_total = sum(p.ships for p in enemy_planets)

    raw_obs = obs if isinstance(obs, dict) else obs.__dict__
    fleets_raw = raw_obs.get("fleets", []) if isinstance(raw_obs, dict) else getattr(raw_obs, "fleets", [])
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
    all_fleets = [Fleet(*f) for f in fleets_raw]
    enemy_fleets = [f for f in all_fleets if f.owner != my_id]

    planets_list = list(world.planets_by_id.values())
    my_threats_inbound = []
    for f in enemy_fleets:
        tgt, eta = fleet_target_planet(f, planets_list, world.omega)
        if tgt is None:
            continue
        if tgt.owner == my_id:
            my_threats_inbound.append((tgt.id, f.ships, eta))

    return (step, my_planets, enemy_planets, neutrals, my_total, enemy_total,
            my_threats_inbound, planets_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opp", default="data/main.py")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    opp = load_opp(args.opp)

    env = make("orbit_wars", configuration={"episodeSteps": args.steps},
               info={"seed": args.seed})
    env.reset()

    # We are player 0
    while not env.done:
        obs_us = env.state[0]["observation"]
        obs_opp = env.state[1]["observation"]
        moves = heuristic(obs_us)
        opp_moves = opp(obs_opp)

        step, my_planets, enemy_planets, neutrals, my_total, enemy_total, threats, all_p = trace(obs_us, 0)

        # Reverse-resolve emit targets
        emit_lines = []
        for m in moves:
            src_id, angle, ships = m
            tgt_id, tgt_owner = None, None
            src = next((p for p in all_p if p.id == src_id), None)
            if src is not None:
                best_d = float("inf")
                for t in all_p:
                    if t.id == src_id:
                        continue
                    ang = math.atan2(t.y - src.y, t.x - src.x)
                    d = abs((ang - angle + math.pi) % (2 * math.pi) - math.pi)
                    if d < best_d and d < 0.5:
                        best_d = d
                        tgt_id, tgt_owner = t.id, t.owner
            emit_lines.append(f"src={src_id}->tgt={tgt_id}(owner={tgt_owner}) ships={ships}")

        if moves or threats or step % 20 == 0:
            print(f"[t={step:3d}] me planets={len(my_planets)} ships={my_total} | "
                  f"opp planets={len(enemy_planets)} ships={enemy_total} | "
                  f"neutrals={len(neutrals)}")
            for line in emit_lines:
                print(f"        EMIT  {line}")
            for tid, ships, eta in threats:
                print(f"        THREAT  inbound to planet={tid}  opp-ships={ships} eta={eta}")

        env.step([moves, opp_moves])

    last = env.steps[-1]
    print(f"\nfinal: us={last[0]['reward']}  opp={last[1]['reward']}  steps={len(env.steps)}")


if __name__ == "__main__":
    main()
