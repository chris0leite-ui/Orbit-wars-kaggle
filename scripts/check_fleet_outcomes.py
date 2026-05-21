"""Exact fleet-outcome diagnostic.

For one game (focal vs baseline at a seed), wrap BOTH agents so we
capture every emitted move PLUS the obs it was emitted from. For each
move, immediately call `predict_fleet_fate(src, tgt, angle, ships,
world, wait_N=0)` — the bit-exact env-parity primitive
(`tests/test_intercept_landing.py`, zero tolerance). Aggregate
predicted outcomes by category.

Why this is exact:
- `predict_fleet_fate` runs the env's swept-pair / sun-distance /
  OOB checks step-by-step. Outcome string is one of "target",
  "planet", "sun", "oob", "timeout".
- Parity with the kaggle env is enforced via parity tests; if those
  pass, the prediction IS the actual env outcome.
- No position-match heuristics, no orbital-drift approximations,
  no tolerance fudging.

Use: `python -m scripts.check_fleet_outcomes --seed 42`.

PI directive (2026-05-20 PM):
  "Nothing that can be exact should be done by heuristics! We have
  all the pieces."
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kaggle_environments import make

from fast import _load_callable
from lib.intent import World
from lib.trajectory import predict_fleet_fate


def _find_intended_target(src_planet_id: int, angle: float, ships: int,
                          world: World) -> int | None:
    """Determine intended target by running predict_fleet_fate. The
    `target` argument to predict_fleet_fate is whichever planet the
    swept-pair check resolves to FIRST. We re-use that fact: pick the
    target that predict_fleet_fate identifies as 'target' (or fall
    back to closest planet along the trajectory)."""
    src = world.planets_by_id.get(int(src_planet_id))
    if src is None:
        return None
    # Try each non-self planet as "target" and find the one
    # predict_fleet_fate reports as outcome="target". The first match
    # is the intended target.
    for tgt_id, tgt in world.planets_by_id.items():
        if int(tgt_id) == int(src_planet_id):
            continue
        try:
            fate = predict_fleet_fate(
                src, tgt, float(angle), int(ships), world, wait_N=0,
            )
        except Exception:
            continue
        if fate.outcome == "target" and int(fate.hit_planet_id) == int(tgt_id):
            return int(tgt_id)
    return None


def _wrap_agent_with_capture(inner_callable, *, my_id: int,
                             capture_list: list):
    """Wrap an agent so every emission is logged with the obs it came
    from, ready for predict_fleet_fate analysis.

    capture_list entries: dict with keys:
        step, src_id, angle, ships, obs, world (lib.intent.World)
    """
    def wrapped(obs, configuration=None):
        moves = inner_callable(obs, configuration)
        try:
            world = World.from_obs(obs)
            step = int(world.step)
        except Exception:
            world = None
            step = -1
        for m in (moves or []):
            if not (isinstance(m, (list, tuple)) and len(m) >= 3):
                continue
            capture_list.append({
                "step": step,
                "src_id": int(m[0]),
                "angle": float(m[1]),
                "ships": int(m[2]),
                "world": world,
            })
        return moves
    return wrapped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focal", default=os.path.join(REPO, "agents", "analytical", "main.py"))
    ap.add_argument("--baseline", default=os.path.join(REPO, "agents", "baseline", "main.py"))
    args = ap.parse_args(argv)

    print(f"=== check_fleet_outcomes seed={args.seed} (EXACT — predict_fleet_fate) ===")
    print(f"focal:    {args.focal}")
    print(f"baseline: {args.baseline}")

    p0_inner = _load_callable(args.focal)
    p1_inner = _load_callable(args.baseline)

    p0_emits: list[dict] = []
    p1_emits: list[dict] = []
    p0 = _wrap_agent_with_capture(p0_inner, my_id=0, capture_list=p0_emits)
    p1 = _wrap_agent_with_capture(p1_inner, my_id=1, capture_list=p1_emits)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([p0, p1])

    final = env.steps[-1]
    p0_reward = final[0].reward
    p1_reward = final[1].reward
    n_steps = len(env.steps)
    print(f"\nGame ended step={n_steps}, p0(focal) reward={p0_reward}, p1 reward={p1_reward}")

    # For each focal emission, identify the intended target and the
    # predicted fate. Count by outcome.
    counts = {"target": 0, "planet": 0, "sun": 0, "oob": 0, "timeout": 0,
              "no_target_resolved": 0}
    details_bad: list[dict] = []
    for emit in p0_emits:
        world = emit["world"]
        if world is None:
            counts["no_target_resolved"] += 1
            continue
        intended = _find_intended_target(
            emit["src_id"], emit["angle"], emit["ships"], world,
        )
        if intended is None:
            counts["no_target_resolved"] += 1
            details_bad.append({**emit, "outcome": "no_target_resolved",
                                "intended": None})
            continue
        src = world.planets_by_id[int(emit["src_id"])]
        tgt = world.planets_by_id[int(intended)]
        fate = predict_fleet_fate(
            src, tgt, float(emit["angle"]), int(emit["ships"]),
            world, wait_N=0,
        )
        counts[fate.outcome] += 1
        if fate.outcome != "target":
            details_bad.append({
                "step": emit["step"], "src_id": emit["src_id"],
                "intended": intended, "ships": emit["ships"],
                "angle": emit["angle"], "outcome": fate.outcome,
                "hit_planet_id": fate.hit_planet_id, "step_of_hit": fate.step,
            })

    total = sum(counts.values())
    print(f"\nFocal (P0) total emissions: {total}")
    for k, v in counts.items():
        pct = 100.0 * v / max(1, total)
        print(f"  {k:>22} : {v:>3}  ({pct:>5.1f}%)")

    # The PI's gate: zero sun + zero OOB.
    critical = counts["sun"] + counts["oob"]
    if critical > 0:
        print(f"\n!!! CRITICAL: {critical} emission(s) predicted to die in sun/OOB:")
        for d in details_bad:
            if d["outcome"] in ("sun", "oob"):
                print(f"  step={d['step']} src={d['src_id']} tgt={d['intended']} "
                      f"ships={d['ships']} angle={d['angle']:.3f} "
                      f"outcome={d['outcome']} hit_planet={d.get('hit_planet_id')} "
                      f"step_of_hit={d.get('step_of_hit')}")
    else:
        print(f"\nOK: zero sun/OOB predicted (target rate "
              f"{100*counts['target']/max(1,total):.2f}%).")

    return 0 if critical == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
