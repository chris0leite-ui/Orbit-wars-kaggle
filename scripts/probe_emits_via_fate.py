"""Replay-probe: count how many historical emits would have been
suppressed by the physics-validation gate.

For each replay episode, walk every turn and re-run trajectory_roi /
goal_planner on the obs. For every emit they produce, project the
fleet via predict_fleet_fate. Count the fraction whose fate is NOT
'target' — those are the launches that physically can't reach where
the chooser thinks they go.

This quantifies the magnitude of the bug across the experimental line.

Usage: python scripts/probe_emits_via_fate.py [--agent NAME]
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.trajectory import predict_fleet_fate
from lib.trajectory_layer import World
from lib.goal_planner.validate import _PlanetShim, _WorldShim


REPLAY_DIR = Path("audit/live-episodes/52784853")


def _classify_emit(emit, world: World) -> str:
    """Return the FleetFate outcome for the emit's trajectory.

    Emit shape: [src_id, aim_angle, ships]. predict_fleet_fate needs
    a target, so we walk the trajectory to whatever it hits first."""
    src_id = int(emit[0])
    aim_angle = float(emit[1])
    ships = int(emit[2])
    wshim = _WorldShim(world)
    src_shim = wshim.planets_by_id.get(src_id)
    if src_shim is None:
        return "no-source"
    # Pick any planet as "target" — predict_fleet_fate's outcome
    # doesn't depend on the target identity (only on which planet
    # is hit first). Pick the SECOND planet just for shim purposes.
    others = [p for pid, p in wshim.planets_by_id.items() if pid != src_id]
    if not others:
        return "no-others"
    target_shim = others[0]
    fate = predict_fleet_fate(src_shim, target_shim, aim_angle, ships, wshim)
    return fate.outcome  # "target" | "planet" | "sun" | "oob" | "timeout"


def probe_agent(agent_name: str, replay_paths: list[Path]) -> None:
    mod = importlib.import_module(f"agents.{agent_name}.main")
    agent_fn = mod.agent

    total_emits = 0
    outcome_counter: Counter[str] = Counter()
    games = 0
    for rp in replay_paths:
        with rp.open() as f:
            replay = json.load(f)
        cfg = replay["configuration"]
        games += 1
        for step_idx, step_states in enumerate(replay["steps"]):
            for seat_idx, state in enumerate(step_states):
                obs = state["observation"]
                if obs.get("step") != step_idx:
                    obs = dict(obs)
                    obs["step"] = step_idx
                try:
                    emits = agent_fn(obs, cfg)
                except Exception:
                    continue
                if not emits:
                    continue
                world = World.from_obs(obs, cfg)
                for e in emits:
                    total_emits += 1
                    outcome_counter[_classify_emit(e, world)] += 1
    print(f"\n--- {agent_name} across {games} episodes ---")
    print(f"total emits: {total_emits}")
    if total_emits == 0:
        return
    for outcome, n in outcome_counter.most_common():
        pct = 100 * n / total_emits
        print(f"  {outcome:8s}: {n:5d} ({pct:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="trajectory_roi,goal_planner",
                    help="comma-separated agent names")
    ap.add_argument("--n-episodes", type=int, default=3)
    args = ap.parse_args()

    replays = sorted(REPLAY_DIR.glob("episode-*.json"))[:args.n_episodes]
    if not replays:
        print(f"No replays under {REPLAY_DIR}")
        return
    print(f"Sampling {len(replays)} episodes: {[p.name for p in replays]}")

    for name in args.agent.split(","):
        probe_agent(name.strip(), replays)


if __name__ == "__main__":
    main()
