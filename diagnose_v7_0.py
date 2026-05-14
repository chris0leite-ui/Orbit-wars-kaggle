"""diagnose_v7_0.py — single-game post-mortem dump.

Runs my agent vs v7_0 on a fixed seed, walks env.steps, and prints a
turn-by-turn ledger:

    turn  | me_planets me_ships me_prod | opp_planets opp_ships opp_prod | me_action | opp_action

Captures the divergence point (when v7_0 pulls clearly ahead) so we
can eyeball what v7_0 does that we don't.

Usage:  python diagnose_v7_0.py [seed]   # default seed 1000, slot 0
"""
from __future__ import annotations

import math
import sys
from kaggle_environments import make


def _ships_total(obs, player):
    n = sum(int(p[5]) for p in obs["planets"] if int(p[1]) == player)
    n += sum(int(f[6]) for f in obs["fleets"] if int(f[1]) == player)
    return n


def _planet_count(obs, player):
    return sum(1 for p in obs["planets"] if int(p[1]) == player)


def _prod_total(obs, player):
    return sum(int(p[6]) for p in obs["planets"] if int(p[1]) == player)


def _format_action(action, obs):
    """[[src_id, angle, ships], ...] → readable string."""
    if not action:
        return "hold"
    parts = []
    planet_pos = {int(p[0]): (float(p[2]), float(p[3])) for p in obs["planets"]}
    for mv in action:
        src_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
        # find the nearest planet on the bearing line as the inferred target
        sx, sy = planet_pos.get(src_id, (0, 0))
        best_pid, best_d = None, float("inf")
        for pid, (px, py) in planet_pos.items():
            if pid == src_id:
                continue
            bearing = math.atan2(py - sy, px - sx)
            d_ang = abs(math.atan2(math.sin(angle - bearing), math.cos(angle - bearing)))
            if d_ang < 0.15:
                d = math.hypot(px - sx, py - sy)
                if d < best_d:
                    best_d, best_pid = d, pid
        tgt_repr = f"P{best_pid}" if best_pid is not None else f"θ={math.degrees(angle):.0f}°"
        parts.append(f"P{src_id}→{tgt_repr} ×{ships}")
    return " | ".join(parts)


def main(seed=1000, my_slot=0, max_turns=30):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = ["baselines/v7_0.py", "baselines/v7_0.py"]
    agents[my_slot] = "main.py"
    env.run(agents)
    opp_slot = 1 - my_slot

    print(f"seed={seed}  my_slot={my_slot}  (me=p{my_slot}, opp=p{opp_slot}=v7_0)")
    print(f"total_turns={len(env.steps)}  outcome: rewards={[s.reward for s in env.steps[-1]]}")
    print()
    print(f"{'turn':>4} | {'meP':>3} {'meS':>5} {'meR':>3} | {'opP':>3} {'opS':>5} {'opR':>3} | me_action / opp_action")
    print("-" * 100)
    for t, step_state in enumerate(env.steps):
        if t > max_turns:
            break
        obs = step_state[0].observation
        meP, meS, meR = _planet_count(obs, my_slot), _ships_total(obs, my_slot), _prod_total(obs, my_slot)
        opP, opS, opR = _planet_count(obs, opp_slot), _ships_total(obs, opp_slot), _prod_total(obs, opp_slot)

        # The action assigned to each player BEFORE this turn was processed
        # is recorded in step_state[i].action.
        my_act = step_state[my_slot].action or []
        op_act = step_state[opp_slot].action or []
        me_str = _format_action(my_act, obs)
        op_str = _format_action(op_act, obs)
        print(f"{t:>4} | {meP:>3} {meS:>5} {meR:>3} | {opP:>3} {opS:>5} {opR:>3} | {me_str:30s} / {op_str}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(seed=seed, max_turns=max_turns)
