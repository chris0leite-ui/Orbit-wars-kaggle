"""Probe: can the trained value net rank expert actions above null/random?

For states in top-band replays where the stronger player launched, inject
(a) the expert's actual launches, (b) nothing, (c) random legal launches
into the exact ledger via the planner's own leaf path, and compare V.

If V(expert) > V(null) only at chance level, the value gradient is too flat
to steer plan selection and a policy prior must carry candidate ranking.

Usage: python scripts/oracle_value_probe.py [--n 60]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agents.oracle.engine import World                      # noqa: E402
from agents.oracle.features import FeatureContext           # noqa: E402
from agents.oracle.value import ValueNet                    # noqa: E402

REPLAY_DIR = REPO / "data" / "external" / "replays"


def leaf_for_action(world, ctx, moves, me):
    """Replay-format moves [[pid, angle, ships], ...] -> leaf features."""
    deltas, extras, flights = {}, {}, []
    for mv in moves:
        pid, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
        if pid not in world.idx_of or ships < 1:
            continue
        src = world.idx_of[pid]
        ships = min(ships, world.ships0[src])
        sx, sy = world.px[src], world.py[src]
        lx = sx + math.cos(angle) * (world.pr[src] + 0.1)
        ly = sy + math.sin(angle) * (world.pr[src] + 0.1)
        hit, dt = world.fly(lx, ly, angle, ships, 1)
        deltas[src] = deltas.get(src, 0) - ships
        if hit is not None:
            extras.setdefault(hit, []).append((dt, me, ships))
        flights.append((me, ships, dt, hit))
    overrides = {i: (deltas.get(i, 0), extras.get(i, []))
                 for i in set(deltas) | set(extras)}
    return ctx.leaf(overrides, flights)


def random_action(world, me, rng):
    mine = [i for i in range(world.n_planets)
            if world.owner0[i] == me and world.ships0[i] >= 4]
    if not mine:
        return []
    moves = []
    for src in rng.sample(mine, k=min(len(mine), rng.randint(1, 2))):
        ships = rng.randint(2, max(2, world.ships0[src]))
        tgt = rng.randrange(world.n_planets)
        if tgt == src:
            continue
        aim = world.aim_at(src, tgt, ships)
        if aim is None:
            continue
        moves.append([world.pid[src], aim[0], ships])
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--min-step", type=int, default=20)
    ap.add_argument("--max-step", type=int, default=260)
    args = ap.parse_args()

    net = ValueNet()
    assert net.loaded, "train weights first"
    rng = random.Random(0)
    paths = sorted(REPLAY_DIR.glob("episode-*-replay.json"))
    rng.shuffle(paths)

    wins_vs_null = 0
    wins_vs_rand = 0
    gaps = []
    n_done = 0
    for path in paths:
        if n_done >= args.n:
            break
        try:
            d = json.load(open(path))
        except Exception:
            continue
        steps = d.get("steps", [])
        if len(steps) < 80:
            continue
        rewards = [s.get("reward") for s in steps[-1]]
        if None in rewards:
            continue
        winner = max(range(len(rewards)), key=lambda p: rewards[p])
        # states where the eventual winner acted. Replay convention:
        # the action stored at index t+1 was decided FROM obs[t].
        cand_ts = [t for t in range(args.min_step,
                                    min(args.max_step, len(steps) - 2))
                   if steps[t + 1][winner].get("action")]
        if not cand_ts:
            continue
        t = rng.choice(cand_ts)
        obs = dict(steps[t][0]["observation"])
        obs["player"] = winner
        obs["step"] = t
        world = World(obs, horizon=48)
        world.build_ledger()
        ctx = FeatureContext(world)
        moves = steps[t + 1][winner]["action"]
        v_expert = float(net.batch([leaf_for_action(world, ctx, moves, winner)])[0])
        v_null = float(net.batch([ctx.leaf(None, None)])[0])
        v_rands = [float(net.batch(
            [leaf_for_action(world, ctx, random_action(world, winner, rng),
                             winner)])[0]) for _ in range(5)]
        wins_vs_null += int(v_expert > v_null)
        wins_vs_rand += sum(int(v_expert > vr) for vr in v_rands)
        gaps.append(v_expert - v_null)
        n_done += 1

    print(f"states probed: {n_done}")
    print(f"V(expert) > V(null):   {wins_vs_null}/{n_done} "
          f"({100*wins_vs_null/max(1,n_done):.0f}%)")
    print(f"V(expert) > V(random): {wins_vs_rand}/{5*n_done} "
          f"({100*wins_vs_rand/max(1,5*n_done):.0f}%)")
    gaps.sort()
    if gaps:
        print(f"gap median {gaps[len(gaps)//2]:+.4f}, "
              f"mean {sum(gaps)/len(gaps):+.4f}")


if __name__ == "__main__":
    main()
