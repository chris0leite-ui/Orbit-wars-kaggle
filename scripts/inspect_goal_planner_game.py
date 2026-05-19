"""Turn-by-turn trace of goal_planner for one game.

For each turn captures:
  - mine vs opp planet counts, ship totals
  - predicate state (winning, prod_advantage, opp_pool, remaining)
  - portfolio chosen by P2
  - defense reservations from P4
  - acquisition schedule from P3
  - agent's emitted launches (turn_offset=0 subset)

Run: python scripts/inspect_goal_planner_game.py [--seed N] [--vs path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle_environments import make

from agents.goal_planner.main import agent as goal_agent
from lib.goal_planner.defense import defense_actions
from lib.goal_planner.portfolio import smallest_winning_portfolio
from lib.goal_planner.predicate import (
    is_winning_state, opp_pool, prod_advantage, remaining_turns,
)
from lib.goal_planner.sequencer import backwards_acquisition_plan
from lib.trajectory_layer import World


def trace_turn(obs, cfg, turn_idx):
    world = World.from_obs(obs, cfg)
    my_id = world.my_id
    others = [p.owner for p in world.planets
              if p.owner not in (-1, my_id)]
    if not others:
        return None
    opp_id = max(set(others), key=others.count)

    mine = [p for p in world.planets if p.owner == my_id]
    opp = [p for p in world.planets if p.owner == opp_id]
    my_fleets = [f for f in world.fleets if f.owner == my_id]
    opp_fleets = [f for f in world.fleets if f.owner == opp_id]

    win = is_winning_state(world, my_id, opp_id)
    adv = prod_advantage(world, my_id, opp_id)
    op = opp_pool(world, opp_id)
    rem = remaining_turns(world)

    portfolio = smallest_winning_portfolio(world, my_id, opp_id)
    reservations = {}
    defense = defense_actions(world, my_id, opp_id, reservations)
    acquisition = (backwards_acquisition_plan(world, my_id, portfolio)
                   if not win and portfolio else [])

    emits = goal_agent(obs, cfg)

    return {
        "turn": turn_idx,
        "my_id": my_id,
        "opp_id": opp_id,
        "mine_count": len(mine),
        "mine_ships": int(sum(p.ships for p in mine)),
        "mine_prod": int(sum(p.production for p in mine)),
        "opp_count": len(opp),
        "opp_ships": int(sum(p.ships for p in opp)),
        "opp_prod": int(sum(p.production for p in opp)),
        "my_fleets_total": int(sum(f.ships for f in my_fleets)),
        "opp_fleets_total": int(sum(f.ships for f in opp_fleets)),
        "winning": win,
        "adv": adv,
        "opp_pool": op,
        "rem": rem,
        "portfolio": list(portfolio[:6]),
        "portfolio_len": len(portfolio),
        "defense_count": len(defense),
        "acquisition_count": len(acquisition),
        "acq_immediate": sum(1 for L in acquisition if L.turn_offset == 0),
        "acq_delayed": sum(1 for L in acquisition if L.turn_offset > 0),
        "emits": emits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vs", type=str,
                    default="/home/user/Orbit-wars-kaggle/agents/simple/nearest.py")
    ap.add_argument("--print-every", type=int, default=1,
                    help="print every Nth turn (1=every turn)")
    ap.add_argument("--max-turns", type=int, default=200)
    args = ap.parse_args()

    env = make("orbit_wars", configuration={"seed": args.seed})
    trainer = env.train([None, args.vs])
    obs = trainer.reset()
    cfg = env.configuration

    print(f"=== seed={args.seed} focal=goal_planner vs {Path(args.vs).name} ===")
    print(f"{'turn':>4} {'mine':>5} {'mShips':>6} {'mProd':>5} "
          f"{'oppP':>5} {'oShips':>6} {'oProd':>5} "
          f"{'win':>5} {'adv':>4} {'oppPool':>7} "
          f"{'portLen':>7} {'def':>3} {'acq0/W':>6} {'emits':>5}")

    turn = 0
    last_print = -1
    while turn < args.max_turns:
        snap = trace_turn(obs, cfg, turn)
        if snap is None:
            break
        if turn == 0 or turn - last_print >= args.print_every or len(snap["emits"]) > 0:
            print(f"{snap['turn']:>4} {snap['mine_count']:>5} "
                  f"{snap['mine_ships']:>6} {snap['mine_prod']:>5} "
                  f"{snap['opp_count']:>5} {snap['opp_ships']:>6} {snap['opp_prod']:>5} "
                  f"{str(snap['winning'])[:5]:>5} {snap['adv']:>4} {snap['opp_pool']:>7} "
                  f"{snap['portfolio_len']:>7} {snap['defense_count']:>3} "
                  f"{snap['acq_immediate']}/{snap['acq_delayed']:<3} "
                  f"{len(snap['emits']):>5}")
            last_print = turn
        obs, reward, done, info = trainer.step(snap["emits"])
        turn += 1
        if done:
            print(f"=== game over at turn {turn}: reward={reward} ===")
            break


if __name__ == "__main__":
    main()
