"""Liveness-guarded head-to-head battery for the oracle agent.

Plays sequential SOLO games (no parallel workers — CPU contention changes
torch agents' behavior, see HANDOVER binding lessons), asserts opponent
liveness every game (launches > 0 and steps > 30, else the row is flagged
DEAD and excluded from the tally), and logs per-game rows to JSONL.

Usage:
  python scripts/oracle_battery.py --opp producer --seeds 300-315
  python scripts/oracle_battery.py --opp v7_0_drop_one --seeds 100-131 \
      --focal agents/oracle/main.py --out /tmp/bat.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from _agent_paths import resolve_agent_path  # noqa: E402


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def count_launches(steps, idx):
    n = 0
    for st in steps:
        if idx < len(st):
            a = st[idx].get("action") or []
            n += len(a)
    return n


def play(focal, opp, seed, swap):
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    agents = [opp, focal] if swap else [focal, opp]
    t0 = time.time()
    env.run(agents)
    wall = time.time() - t0
    steps = env.steps
    fi = 1 if swap else 0
    rewards = [steps[-1][k].reward for k in range(2)]
    statuses = [steps[-1][k].status for k in range(2)]
    return {
        "seed": seed, "swap": swap, "n_steps": len(steps),
        "reward_focal": rewards[fi], "reward_opp": rewards[1 - fi],
        "status_focal": str(statuses[fi]), "status_opp": str(statuses[1 - fi]),
        "launches_focal": count_launches(steps, fi),
        "launches_opp": count_launches(steps, 1 - fi),
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="agents/oracle/main.py")
    ap.add_argument("--opp", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--swap", action="store_true",
                    help="also play the seat-swapped game per seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    focal = resolve_agent_path(args.focal)
    opp = resolve_agent_path(args.opp)
    seeds = parse_seeds(args.seeds)
    out_path = args.out or f"/tmp/oracle_bat_{Path(opp).stem}_{int(time.time())}.jsonl"

    wins = losses = draws = dead = 0
    with open(out_path, "w") as f:
        for seed in seeds:
            for swap in ([False, True] if args.swap else [False]):
                row = play(focal, opp, seed, swap)
                alive = (row["launches_opp"] > 0 and row["n_steps"] > 30
                         and row["launches_focal"] > 0)
                row["alive"] = alive
                f.write(json.dumps(row) + "\n")
                f.flush()
                if not alive:
                    dead += 1
                    tag = "DEAD"
                elif row["reward_focal"] > row["reward_opp"]:
                    wins += 1
                    tag = "WIN"
                elif row["reward_focal"] < row["reward_opp"]:
                    losses += 1
                    tag = "LOSS"
                else:
                    draws += 1
                    tag = "DRAW"
                print(f"seed {seed}{' sw' if swap else '   '} {tag:5s} "
                      f"steps {row['n_steps']:3d} "
                      f"launches {row['launches_focal']}/{row['launches_opp']} "
                      f"({row['wall_s']}s)", flush=True)
    n = wins + losses + draws
    print(f"\n{wins}W-{losses}L-{draws}D of {n} live games "
          f"({dead} DEAD games excluded) -> {out_path}")
    if n:
        p = wins / n
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
        print(f"win rate {p:.3f}, Wilson 95% [{center-half:.3f}, {center+half:.3f}]")


if __name__ == "__main__":
    main()
