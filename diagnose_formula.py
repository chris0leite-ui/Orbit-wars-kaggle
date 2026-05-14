"""diagnose_formula.py — compare score_action (v1 F2) vs analytic
F3-aware Δfavor for each candidate.

If we replaced score_action's Δfavor calc with one that uses the same
F3 formula favor uses at the leaf, would those same candidates still
score positive?

Usage:  python diagnose_formula.py [seed] [target_turn]
"""
from __future__ import annotations

import copy
import math
import sys
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

import main
from favor import favor


def _obs_after_capture(obs, src_id, tgt_id, ships, me, arrival_turns):
    """Analytically apply a single capture action; return new dict obs.
    Assumes capture succeeds (garrison_at_arrival < ships). Advances
    step by arrival_turns and adds production to every owned planet.
    """
    new = copy.deepcopy(dict(obs)) if not isinstance(obs, dict) else copy.deepcopy(obs)
    new["step"] = int(new.get("step", 0)) + arrival_turns
    new_planets = []
    for p in new["planets"]:
        p = list(p)
        if p[0] == src_id:
            p[5] = int(p[5]) - ships
            p[5] += int(p[6]) * arrival_turns   # source production while fleet flies
        elif p[0] == tgt_id:
            garrison = int(p[5])
            if p[1] != -1:                       # if enemy/owned, grow
                garrison += int(p[6]) * arrival_turns
            surplus = ships - garrison
            p[1] = me
            p[5] = max(0, surplus)
        else:
            if p[1] >= 0:                        # other owned planets grow
                p[5] = int(p[5]) + int(p[6]) * arrival_turns
        new_planets.append(p)
    new["planets"] = new_planets
    return new


def dump(seed, target_turn, my_slot=0):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = ["baselines/v7_0.py", "baselines/v7_0.py"]
    agents[my_slot] = "main.py"
    env.run(agents)
    obs = env.steps[target_turn][my_slot].observation
    obs = dict(obs)

    player = my_slot
    raw_planets = obs["planets"]
    raw_fleets = obs["fleets"]
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    print(f"=== seed={seed} turn={target_turn} player={player} ===")
    print(f"favor_now = {favor(obs, player):.1f}\n")
    cands = main._enumerate_candidates(my_planets, targets, fleets, target_turn, player)
    if not cands:
        print("no analytic candidates")
        return

    favor_now = favor(obs, player)
    print(f"{'cand':35s} {'analytic_v1':>12s} {'analytic_v2':>12s} {'arrival':>7s}")
    print("-" * 75)
    for score_v1, src, tgt, ships in cands[:15]:
        # Analytic v2 Δfavor: apply the action, evaluate favor with F3.
        dist = max(0.0, math.hypot(src.x - tgt.x, src.y - tgt.y) - src.radius - tgt.radius)
        speed = main._speed(ships)
        arrival = math.ceil(dist / speed) if speed > 0 else 1
        # Check capture
        if tgt.owner == -1:
            garrison_at_arrival = tgt.ships
        else:
            garrison_at_arrival = tgt.ships + tgt.production * arrival
        if ships <= garrison_at_arrival:
            score_v2 = float("nan")  # would fail capture
        else:
            new_obs = _obs_after_capture(obs, src.id, tgt.id, ships, player, arrival)
            favor_new = favor(new_obs, player)
            score_v2 = favor_new - favor_now
        print(f"  P{src.id:2d}→P{tgt.id:2d} ×{ships:3d} (tgt prod={tgt.production}, ships={tgt.ships})  "
              f"{score_v1:>+12.1f} {score_v2:>+12.1f} {arrival:>7d}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1003
    target_turn = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    dump(seed, target_turn)
