"""Reproduce seed-0 game to t=98 and dump the agent's candidate enumeration.

Goal: find out WHY 6 of 7 sources stayed idle when 95 ships sat on them
against a board that should have positive-ROI neutral targets.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

from kaggle_environments import make

from agents.heuristic.main import (
    DEFEND_HORIZON, GARRISON_BUFFER, MIN_SHIPS_TO_LAUNCH, ENEMY_DENIAL_BONUS,
    EPISODE_STEPS, SOURCE_RESERVE_HORIZON, _aim, _defense_candidate,
    _earliest_flip, _hold_margin, _max_sendable, _roi, _ships_for_capture,
    agent as heuristic,
)
from lib.intent import World
from lib.trajectory import predict_fleet_fate
from lib.world_model import WorldModel


def load(path):
    name = f"_dbgopp_{os.path.basename(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def dump_candidates(obs, target_step=98):
    world = World.from_obs(obs)
    if world.step != target_step:
        return False

    wm = WorldModel.from_world(world)
    my_id = world.my_id
    my_planets = [p for p in world.planets_by_id.values() if p.owner == my_id]
    all_planets = list(world.planets_by_id.values())
    threatened = [p for p in my_planets
                  if _earliest_flip(p.id, wm, my_id, DEFEND_HORIZON) is not None]
    sendable = {p.id: _max_sendable(p, wm, my_id) for p in my_planets}

    print(f"\n=== t={target_step} debug dump ===")
    print(f"my_planets={len(my_planets)}  threatened={[p.id for p in threatened]}")
    print(f"sendable=", {p.id: sendable[p.id] for p in my_planets},
          f"  raw_ships=", {p.id: int(p.ships) for p in my_planets})

    for src in my_planets:
        cap = sendable[src.id]
        print(f"\nSRC {src.id}: ships={int(src.ships)} sendable={cap} prod={src.production}")
        if cap < MIN_SHIPS_TO_LAUNCH:
            print("   --- skipped (sendable < MIN_SHIPS_TO_LAUNCH) ---")
            continue
        n_aim_fail = 0
        n_will_be_ours = 0
        n_too_costly = 0
        n_physics_fail = 0
        n_roi_zero = 0
        n_candidates = 0
        roi_list = []
        for tgt in all_planets:
            if tgt.id == src.id or tgt.owner == my_id:
                continue
            guess = max(MIN_SHIPS_TO_LAUNCH, int(tgt.ships) + GARRISON_BUFFER)
            aim = _aim(src, tgt, guess, world)
            if aim is None:
                n_aim_fail += 1
                continue
            _, _, eta_float = aim
            eta_int = max(1, int(math.ceil(eta_float)))
            pred_owner = wm.owner_at(tgt.id, eta_int)
            pred_garrison = wm.ships_at(tgt.id, eta_int)
            if pred_owner == my_id:
                n_will_be_ours += 1
                continue
            ships = _ships_for_capture(pred_garrison if pred_garrison is not None else tgt.ships)
            ships += _hold_margin(tgt.id, tgt.production, eta_int, wm, my_id)
            if ships > cap:
                n_too_costly += 1
                continue
            aim2 = _aim(src, tgt, ships, world)
            if aim2 is None:
                n_aim_fail += 1
                continue
            angle, _, eta_float = aim2
            fate = predict_fleet_fate(src, tgt, angle, ships, world)
            if fate.outcome != "target" or fate.hit_planet_id != tgt.id:
                n_physics_fail += 1
                continue
            roi = _roi(src, tgt, eta_float, ships, world, my_id)
            if roi is None or roi <= 0:
                n_roi_zero += 1
                continue
            n_candidates += 1
            roi_list.append((roi, tgt.id, tgt.owner, ships, eta_int))
        roi_list.sort(reverse=True)
        print(f"   candidates={n_candidates}  aim_fail={n_aim_fail}  "
              f"already_ours={n_will_be_ours}  too_costly={n_too_costly}  "
              f"physics_fail={n_physics_fail}  roi<=0={n_roi_zero}")
        for r, tid, o, sh, et in roi_list[:5]:
            print(f"      ROI={r:.2f}  tgt={tid}(owner={o})  ships={sh}  eta={et}")
    return True


def main():
    opp = load("submissions/v7_0_drop_one.py")
    env = make("orbit_wars", configuration={"episodeSteps": 200},
               info={"seed": 0})
    env.reset()
    dumped = False
    while not env.done and not dumped:
        obs_us = env.state[0]["observation"]
        obs_opp = env.state[1]["observation"]
        if not dumped:
            dumped = dump_candidates(obs_us, target_step=98)
        moves = heuristic(obs_us)
        opp_moves = opp(obs_opp)
        env.step([moves, opp_moves])


if __name__ == "__main__":
    main()
