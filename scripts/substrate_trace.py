"""Rule 47 substrate trace for the F6 path_fate feature.

Runs one game between two agents (default: baseline vs baseline) and for
every emit, casts predict_fleet_fate to tally the outcome distribution
{target, planet, sun, oob, timeout}. The Phase 2 v2 plan requires
sun + oob + timeout < 2 % of total emits; otherwise the F6 feature's
signal is muddied and substrate needs investigation before retraining
the validator.

Usage:
    python -m scripts.substrate_trace
    python -m scripts.substrate_trace --a agents/baseline/main.py \\
        --b agents/baseline/main.py --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_agent_callable(path: str):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"_agent_{p.stem}_{abs(hash(path))}", str(p)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", default="agents/baseline/main.py")
    p.add_argument("--b", default="agents/baseline/main.py")
    p.add_argument("--seed", type=int, default=10_001)
    args = p.parse_args(argv)

    from kaggle_environments import make
    from lib.intent import World
    from lib.trajectory import predict_fleet_fate
    from lib.shot_features import fleet_speed, infer_target_pid

    agent_a = _load_agent_callable(args.a)
    agent_b = _load_agent_callable(args.b)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([agent_a, agent_b])
    payload = env.toJSON()
    steps = payload.get("steps", [])
    n_steps = len(steps)
    print(f"game finished: seed={args.seed} steps={n_steps} "
          f"outcome={payload.get('rewards')}")

    outcomes: Counter = Counter()
    seat_outcomes: dict[int, Counter] = {0: Counter(), 1: Counter()}
    n_emits = 0
    n_unresolved = 0  # emits where ray-cast couldn't find a target planet

    for step_idx, step in enumerate(steps):
        obs0 = step[0].get("observation", {}) or {}
        if not obs0.get("planets"):
            continue
        try:
            world = World.from_obs(obs0)
        except Exception:
            continue

        for seat in range(len(step)):
            action = step[seat].get("action") or []
            if not action:
                continue
            obs = step[seat].get("observation", {}) or {}
            planets = obs.get("planets", []) or []
            by_id = {int(p[0]): p for p in planets}

            for a in action:
                if not a or len(a) < 3:
                    continue
                try:
                    src_pid = int(a[0])
                    angle = float(a[1])
                    ships = float(a[2])
                except (TypeError, ValueError):
                    continue
                src = by_id.get(src_pid)
                if src is None:
                    continue
                if ships <= 0:
                    continue
                tgt_pid = infer_target_pid(
                    (float(src[2]), float(src[3])), angle, planets,
                )
                if tgt_pid is None:
                    n_unresolved += 1
                    continue
                target = by_id.get(tgt_pid)
                if target is None:
                    n_unresolved += 1
                    continue

                src_obj = world.planets_by_id.get(src_pid)
                tgt_obj = world.planets_by_id.get(tgt_pid)
                if src_obj is None or tgt_obj is None:
                    n_unresolved += 1
                    continue

                d = math.hypot(float(target[2]) - float(src[2]),
                               float(target[3]) - float(src[3]))
                v = fleet_speed(ships)
                eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
                # Same max_steps cap that lib.shot_features uses.
                max_steps_cap = max(20, int(eta) + 20)
                # Recompute aim angle from the resolved geometry (matches the
                # encoder's F6 computation; the caller's `angle` may not point
                # exactly at the centre of the resolved target).
                angle_geom = math.atan2(
                    float(tgt_obj.y) - float(src_obj.y),
                    float(tgt_obj.x) - float(src_obj.x),
                )
                fate = predict_fleet_fate(
                    src_obj, tgt_obj, angle_geom,
                    int(round(ships)), world,
                    max_steps=max_steps_cap,
                )
                outcomes[fate.outcome] += 1
                seat_outcomes[seat][fate.outcome] += 1
                n_emits += 1

    print()
    print(f"=== Rule 47 substrate trace ===  emits={n_emits}  "
          f"ray-cast-unresolved={n_unresolved}")
    if n_emits == 0:
        print("no emits — nothing to trace")
        return 1
    print()
    print("outcome distribution:")
    for k in ("target", "planet", "sun", "oob", "timeout"):
        c = outcomes.get(k, 0)
        pct = 100.0 * c / n_emits
        flag = "  <-- WASTE" if k in ("sun", "oob", "timeout") else ""
        print(f"  {k:<8} {c:>5}  ({pct:5.2f} %){flag}")
    waste = sum(outcomes.get(k, 0) for k in ("sun", "oob", "timeout"))
    waste_pct = 100.0 * waste / n_emits
    print()
    print(f"  WASTE total (sun+oob+timeout): {waste} / {n_emits} = {waste_pct:.2f} %")
    gate_pct = 2.0
    if waste_pct < gate_pct:
        print(f"  PASS — waste {waste_pct:.2f} % below the {gate_pct:.1f} % gate")
        return 0
    else:
        print(f"  FAIL — waste {waste_pct:.2f} % at/above the {gate_pct:.1f} % gate; "
              f"F6 signal is muddied (Rule 47 — investigate before retraining)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
