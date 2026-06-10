"""protoflow_game_trace — single-game DECISION narrative for fast evaluator iteration.

Runs ONE game (protoflow vs one opponent, one seed) and prints a turn-by-turn story
built from the agent's own launch trace merged with both players' planet/ship curves:

  - every 10 turns: a board line (my planets/ships vs theirs) so the arc (peak,
    collapse, comeback) is readable at a glance;
  - every turn we launched: the launches (kind, src->tgt, ships, target owner) so a
    single bad decision is visible WITH the board state it was made in;
  - a phase summary at the end (peak planets, when lost, net ship curve extremes).

This is the instrument for the observation-driven loop: spot good/bad decision
making from one game instead of waiting on a panel.

Usage:
    python scripts/protoflow_game_trace.py --opponent agents/producer/producer_agent.py \
        --seed 0 --flowdiff
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import agents.protoflow.main as proto  # noqa: E402
from lib.intent import World  # noqa: E402
from fast import _load_callable  # noqa: E402


def board_state(obs_struct, me: int):
    """(my_planets, my_ships, opp_planets, opp_ships) from a raw step observation."""
    world = World.from_obs(obs_struct)
    mine = [p for p in world.planets_by_id.values() if int(p.owner) == me]
    theirs = [p for p in world.planets_by_id.values() if int(p.owner) not in (-1, me)]
    return (len(mine), int(sum(p.ships for p in mine)),
            len(theirs), int(sum(p.ships for p in theirs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="agents/producer/producer_agent.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seat", type=int, default=0, choices=(0, 1))
    ap.add_argument("--every", type=int, default=10, help="board line every N turns")
    ap.add_argument("--simulate-value", action="store_true")
    ap.add_argument("--drain-cost", action="store_true")
    ap.add_argument("--drain-anticipatory", action="store_true")
    ap.add_argument("--flowdiff", action="store_true")
    ap.add_argument("--flowdiff-tail", action="store_true")
    ap.add_argument("--flowdiff-reaction", action="store_true")
    args = ap.parse_args()

    proto.SIMULATE_VALUE = bool(args.simulate_value)
    proto.SIMVALUE_DRAIN_COST = bool(args.drain_cost)
    proto.SIMVALUE_DRAIN_ANTICIPATORY = bool(args.drain_anticipatory)
    proto.FLOWDIFF_VALUE = bool(args.flowdiff)
    proto.FLOWDIFF_TAIL = bool(args.flowdiff_tail)
    proto.FLOWDIFF_REACTION = bool(args.flowdiff_reaction)

    from kaggle_environments import make

    opp = _load_callable(str(REPO / args.opponent))
    proto.reset_trace()
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    line_up = [proto.agent, opp] if args.seat == 0 else [opp, proto.agent]
    env.run(line_up)

    me = args.seat
    trace = {t["step"]: t for t in proto.get_trace()}
    peak = (0, 0)   # (planets, step)
    print(f"seed={args.seed} seat=p{me} opponent={Path(args.opponent).stem} "
          f"flowdiff={proto.FLOWDIFF_VALUE} sim={proto.SIMULATE_VALUE}")
    for i, step_frames in enumerate(env.steps):
        obs_struct = step_frames[0].observation
        mp, ms, op_, os_ = board_state(obs_struct, me)
        if mp > peak[0]:
            peak = (mp, i)
        tr = trace.get(i)
        launches = tr["launches"] if tr else []
        if i % args.every == 0 or launches:
            line = f"t{i:>3}  me {mp:>2}p/{ms:>4}s   opp {op_:>2}p/{os_:>4}s"
            if launches:
                legs = "  ".join(
                    f"{lc['kind']}:{lc['src']}->{lc['tgt']}"
                    f"({lc['ships']}s,{'N' if lc['tgt_owner'] == -1 else 'E' if lc['tgt_owner'] != me else 'M'})"
                    for lc in launches)
                line += "   " + legs
            print(line)
    # WAVE OUTCOMES: for every offense wave, was the target OURS shortly after the priced
    # arrival, and was it STILL ours 15 turns later? Measures the do-nothing illusion directly:
    # a high captured-but-not-held rate means the evaluator buys planets the opponent takes back.
    owner_curve: dict[int, dict[int, int]] = {}
    for i, step_frames in enumerate(env.steps):
        world = World.from_obs(step_frames[0].observation)
        for p in world.planets_by_id.values():
            owner_curve.setdefault(int(p.id), {})[i] = int(p.owner)

    def owner_at(pid: int, turn: int):
        curve = owner_curve.get(pid, {})
        return curve.get(min(turn, max(curve) if curve else 0))

    n_waves = n_captured = n_held = 0
    for tr_step, tr_rec in sorted(trace.items()):
        for lc in tr_rec["launches"]:
            if lc["kind"] != "wave" or lc["tgt_owner"] == me:
                continue
            land = tr_step + lc["arrive_turn"]
            n_waves += 1
            if owner_at(lc["tgt"], land + 2) == me:
                n_captured += 1
                if owner_at(lc["tgt"], land + 15) == me:
                    n_held += 1
    final = env.steps[-1]
    r = (final[0].reward, final[1].reward)
    mine_r = r[me] if r[me] is not None else -99
    opp_r = r[1 - me] if r[1 - me] is not None else -99
    print(f"\nRESULT: {'WIN' if mine_r > opp_r else 'LOSS' if mine_r < opp_r else 'TIE'}  "
          f"peak={peak[0]} planets at t{peak[1]}")
    print(f"WAVES: {n_waves} offense legs -> captured(+2t)={n_captured} -> still ours(+15t)={n_held}"
          + (f"   capture_rate={n_captured/n_waves:.2f} hold_rate={n_held/max(1,n_captured):.2f}"
             if n_waves else ""))


if __name__ == "__main__":
    main()
