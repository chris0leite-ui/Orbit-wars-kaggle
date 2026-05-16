"""Instrument fleet outcomes from a sequence of seeds.

For each seed, run agent_a vs agent_b and walk the actions log step by
step. For every fleet that was launched, classify its eventual outcome
via `lib.trajectory.predict_fleet_fate` against the world AT LAUNCH
time. Aggregate per-agent counts.

This is the explicit Rule-38 verification: reproduce the failure state
(non-zero "sun" count for v20 or v15) and confirm the fix (zero for v21).

Usage:
    python -m scripts.outcome_histogram \\
        --agent-a agents/v21_compound/main.py \\
        --agent-b agents/v20/main.py \\
        --seeds 1-16

Output: per-agent histogram printed to stdout, with a non-zero "sun"
count flagged with a "*** SUN BUG ***" marker.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running as a script without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.trajectory import predict_fleet_fate


def _parse_seeds(spec: str) -> list[int]:
    """'1-16' → [1..16]; '42' → [42]; '1,2,3' → [1,2,3]."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(s) for s in spec.split(",")]
    return [int(spec)]


def _run_and_classify(agent_a: str, agent_b: str, seed: int) -> dict[int, Counter]:
    """Run one game, classify every launched fleet's outcome.

    Returns {seat_idx: Counter({outcome: count, ...})}.

    Implementation: env.steps[i+1][0].action contains the moves agent 0
    submitted at step i; env.steps[i+1][1].action for agent 1. We look
    at the obs AT step i (before move execution) to compute the fleet's
    trajectory.
    """
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent_a, agent_b])

    out: dict[int, Counter] = {0: Counter(), 1: Counter()}

    for i in range(len(env.steps) - 1):
        # State at step i (BEFORE moves). Each state[i] is a list of one
        # entry per seat with observation/action.
        state_i = env.steps[i]
        for seat in (0, 1):
            seat_state = env.steps[i + 1][seat]
            actions = seat_state.action or []
            if not actions:
                continue
            obs_d = dict(state_i[seat].observation)
            try:
                world = World.from_obs(obs_d)
            except Exception:
                continue
            planets_by_id = world.planets_by_id
            for mv in actions:
                if not isinstance(mv, (list, tuple)) or len(mv) != 3:
                    continue
                try:
                    src_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
                except Exception:
                    continue
                src = planets_by_id.get(src_id)
                if src is None:
                    out[seat]["invalid_src"] += 1
                    continue
                # Use the source planet as our "target" placeholder —
                # predict_fleet_fate cares about the target only for the
                # "target" outcome label, not for collision detection.
                # We label as "target" only if the fleet hits the source
                # (which shouldn't happen — would be "planet" instead).
                fate = predict_fleet_fate(src, src, angle, ships, world)
                # Reclassify "target" (which is impossible here since
                # target==src) into "planet" — what we actually want to
                # know is the outcome bucket.
                outcome = fate.outcome
                if outcome == "target":
                    outcome = "planet"  # the fleet hit some planet (not the source)
                out[seat][outcome] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-a", required=True, help="path to agent A main.py")
    ap.add_argument("--agent-b", required=True, help="path to agent B main.py")
    ap.add_argument("--seeds", default="1-8", help="e.g. 1-16 or 42 or 1,2,3")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    total: dict[int, Counter] = {0: Counter(), 1: Counter()}
    print(f"Running {len(seeds)} games: {args.agent_a} vs {args.agent_b}")
    for s in seeds:
        per_game = _run_and_classify(args.agent_a, args.agent_b, s)
        for seat in (0, 1):
            total[seat].update(per_game[seat])
        sun_a = per_game[0]["sun"]
        sun_b = per_game[1]["sun"]
        flag = " *** SUN BUG ***" if (sun_a + sun_b) > 0 else ""
        print(f"  seed {s:>3}: A={dict(per_game[0])} | B={dict(per_game[1])}{flag}")

    print("\n=== TOTAL ===")
    for seat, label in [(0, args.agent_a), (1, args.agent_b)]:
        n_total = sum(total[seat].values())
        sun = total[seat]["sun"]
        oob = total[seat]["oob"]
        planet = total[seat]["planet"]
        timeout = total[seat]["timeout"]
        sun_pct = (100.0 * sun / n_total) if n_total else 0.0
        print(f"  {label}:")
        print(f"    total_launches: {n_total}")
        print(f"    planet (hit):   {planet}")
        print(f"    sun (lost):     {sun}  ({sun_pct:.2f}%)" +
              ("  *** NON-ZERO ***" if sun > 0 else ""))
        print(f"    oob (lost):     {oob}")
        print(f"    timeout:        {timeout}")


if __name__ == "__main__":
    main()
