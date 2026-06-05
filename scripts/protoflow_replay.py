"""protoflow_replay — render a game as the interactive Kaggle HTML player.

Produces the SAME self-contained interactive replay you see on Kaggle (scrub the
timeline, watch fleets launch and combat resolve). Open the written .html in any
browser -- it embeds everything, no server needed.

Run:
    python scripts/protoflow_replay.py --opponent agents/producer/producer_agent.py --seed 0 --seat p0
    python scripts/protoflow_replay.py --opponent agents/simple/nearest.py --seed 3 --seat p1 --out /tmp/win.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle_environments import make

import agents.protoflow.main as proto
from scripts.protoflow_probe import load_callable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", default="agents/simple/nearest.py",
                    help="opponent agent path")
    ap.add_argument("--seed", type=int, default=0, help="board seed")
    ap.add_argument("--seat", choices=["p0", "p1"], default="p0",
                    help="which seat protoflow plays (p0 = first/top, p1 = second)")
    ap.add_argument("--out", default=None, help="output .html path")
    ap.add_argument("--simulate-value", action="store_true",
                    help="use the simulation-based evaluator (proto.SIMULATE_VALUE = True)")
    ap.add_argument("--drain-cost", action="store_true",
                    help="price the source-drain cost in offense values (proto.SIMVALUE_DRAIN_COST = True)")
    args = ap.parse_args()

    proto.SIMULATE_VALUE = bool(args.simulate_value)
    proto.SIMVALUE_DRAIN_COST = bool(args.drain_cost)

    opp = load_callable(args.opponent)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    lineup = [proto.agent, opp] if args.seat == "p0" else [opp, proto.agent]
    env.run(lineup)

    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    me, them = (r0, r1) if args.seat == "p0" else (r1, r0)
    result = ("WIN" if (me is not None and them is not None and me > them)
              else ("LOSS" if (me is not None and them is not None and me < them) else "DRAW"))

    out = args.out or (f"/tmp/replay_protoflow_vs_{Path(args.opponent).stem}"
                       f"_seed{args.seed}_{args.seat}.html")
    Path(out).write_text(env.render(mode="html"))
    print(f"{result}  protoflow({args.seat}) score={me} vs {Path(args.opponent).stem} score={them}")
    print(f"  -> {out}  (open in a browser)")


if __name__ == "__main__":
    main()
