"""Analyze what the value-horizon lever actually changed in a focal-vs-champion
replay, to explain the negative A/B.

For the focal (value-horizon) agent's committed captures, split into:
  - LEVER-ADMITTED: arrival eta > the per-target baseline K (capture_horizon_k)
    — these launches exist ONLY because BASELINE_VALUE_HORIZON admitted them.
  - NORMAL: eta <= K (the champion would have made these too).

For each, measure hold@HOLD_WINDOW and whether the SOURCE planet was lost
within SRC_WINDOW turns (over-drain / over-extension signal). If lever-admitted
captures waste / over-drain far more than normal ones, the lever over-extends.

Runs with the champion state-driven-K env so capture_horizon_k matches the
agent's baseline ceiling. Read-only.
"""
from __future__ import annotations

import glob
import json
import os
import sys

os.environ.update(
    BASELINE_LAUNCH_RULES="1", BASELINE_STATE_DRIVEN_K="1",
    BASELINE_STATE_K_CEIL="30", BASELINE_CAPTURE_HORIZON_K="10",
)

from agents.baseline.launch_rules import capture_horizon_k, resolve_launch_target  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402

HOLD_WINDOW = 15
SRC_WINDOW = 12


def focal_seat(d):
    names = (d.get("info", {}) or {}).get("TeamNames", [])
    for i, n in enumerate(names):
        if "focal" in str(n).lower():
            return i
    return 0


def analyze(path):
    d = json.load(open(path))
    seat = focal_seat(d)
    steps = d["steps"]
    won = steps[-1][seat].get("reward", 0) == 1
    buckets = {
        "admitted": {"n": 0, "held": 0, "src_lost": 0},
        "normal": {"n": 0, "held": 0, "src_lost": 0},
    }
    for si, st in enumerate(steps):
        action = st[seat].get("action") or []
        if not action:
            continue
        obs = st[seat]["observation"]
        try:
            world = World.from_obs(dict(obs))
            model = WorldModel.from_world(world)
        except Exception:
            continue
        for mv in action:
            src = world.planets_by_id.get(int(mv[0]))
            if src is None:
                continue
            hit_pid, step, _ = resolve_launch_target(src, mv[1], int(mv[2]), world)
            if hit_pid is None:
                continue
            tgt = world.planets_by_id.get(int(hit_pid))
            if tgt is None or int(tgt.owner) == seat:
                continue
            if int(mv[2]) <= int(tgt.ships):
                continue  # under-strength poke
            k = capture_horizon_k(si, tgt_id=int(hit_pid), world=world,
                                  model=model, me=seat)
            b = buckets["admitted"] if int(step) > k else buckets["normal"]
            b["n"] += 1
            # hold@HOLD_WINDOW
            check = si + int(step) + HOLD_WINDOW
            if check < len(steps):
                fut = steps[check][seat]["observation"]["planets"]
                if next((int(p[1]) for p in fut if int(p[0]) == int(hit_pid)), None) == seat:
                    b["held"] += 1
            # source lost within SRC_WINDOW of launch?
            scheck = si + SRC_WINDOW
            if scheck < len(steps):
                fut = steps[scheck][seat]["observation"]["planets"]
                so = next((int(p[1]) for p in fut if int(p[0]) == int(src.id)), seat)
                if so != seat:
                    b["src_lost"] += 1

    ep = os.path.basename(path)
    print(f"\n{ep}  focal seat={seat}  result={'WIN' if won else 'LOSS'}")
    for name, b in buckets.items():
        if not b["n"]:
            print(f"  {name:9s}: 0 captures")
            continue
        print(f"  {name:9s}: n={b['n']:3d}  hold@{HOLD_WINDOW}={100*b['held']/b['n']:3.0f}%  "
              f"src_lost@{SRC_WINDOW}={100*b['src_lost']/b['n']:3.0f}%")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("/tmp/ab_replays/*.json"))
    for p in paths:
        analyze(p)
