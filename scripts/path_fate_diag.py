"""Diagnostic dump for path-fate substrate disagreement.

Investigates the 2026-05-28 finding from scripts/substrate_trace.py:
predict_fleet_fate buckets 25/238 emits as "oob" but 18 of those 25
captured the target in the real env. This script dumps the first N
oob-but-captured emits with full geometry so we can see WHY the
trajectory prediction diverges from env reality.

Per-emit dump fields:
  - launch obs: step, seat, src(id,xy,radius), target(id,xy,orbit_phase)
  - emit: angle, ships, fleet_speed, distance, eta
  - predict_fleet_fate says: outcome, step-of-oob, final fleet position
  - env says (at step + eta + 10): target.owner — captured by seat or not
  - check: re-aim straight-line at LAUNCH-time target position; does
    pff still say oob?
  - check: re-aim straight-line at FUTURE target position; does pff
    say target now?
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
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
    p.add_argument("--n-samples", type=int, default=5,
                   help="number of oob-but-captured emits to dump")
    args = p.parse_args(argv)

    from kaggle_environments import make
    from lib.intent import World
    from lib.trajectory import predict_fleet_fate
    from lib.orbit import is_orbiting, predict_relative
    from lib.shot_features import fleet_speed, infer_target_pid

    agent_a = _load_agent_callable(args.a)
    agent_b = _load_agent_callable(args.b)
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([agent_a, agent_b])
    steps = env.toJSON().get("steps", [])
    n_steps = len(steps)
    print(f"game finished: seed={args.seed} steps={n_steps}")
    print(f"world omega will be read from per-step obs\n")
    LABEL_BUFFER = 10
    dumped = 0

    for step_idx, step in enumerate(steps):
        if dumped >= args.n_samples:
            break
        obs0 = step[0].get("observation", {}) or {}
        if not obs0.get("planets"):
            continue
        try:
            world = World.from_obs(obs0)
        except Exception:
            continue

        for seat in range(len(step)):
            if dumped >= args.n_samples:
                break
            action = step[seat].get("action") or []
            if not action:
                continue
            obs = step[seat].get("observation", {}) or {}
            planets = obs.get("planets", []) or []
            by_id = {int(p[0]): p for p in planets}

            for a in action:
                if dumped >= args.n_samples:
                    break
                if not a or len(a) < 3:
                    continue
                try:
                    src_pid = int(a[0])
                    angle = float(a[1])
                    ships = float(a[2])
                except (TypeError, ValueError):
                    continue
                src = by_id.get(src_pid)
                if src is None or ships <= 0:
                    continue
                tgt_pid = infer_target_pid(
                    (float(src[2]), float(src[3])), angle, planets,
                )
                if tgt_pid is None:
                    continue
                target = by_id.get(tgt_pid)
                if target is None:
                    continue
                src_obj = world.planets_by_id.get(src_pid)
                tgt_obj = world.planets_by_id.get(tgt_pid)
                if src_obj is None or tgt_obj is None:
                    continue

                d = math.hypot(float(target[2]) - float(src[2]),
                               float(target[3]) - float(src[3]))
                v = fleet_speed(ships)
                if v <= 0:
                    continue
                eta = int(math.ceil(d / max(v, 1e-6)))
                max_steps_cap = max(20, int(eta) + 20)

                fate = predict_fleet_fate(
                    src_obj, tgt_obj, angle,
                    int(round(ships)), world,
                    max_steps=max_steps_cap,
                )
                if fate.outcome != "oob":
                    continue

                # Label lookup
                check_step = min(step_idx + eta + LABEL_BUFFER, n_steps - 1)
                if check_step >= n_steps:
                    continue
                check_obs = steps[check_step][seat].get("observation", {}) or {}
                check_planets = check_obs.get("planets", []) or []
                check_by_id = {int(p[0]): p for p in check_planets}
                target_check = check_by_id.get(tgt_pid)
                if target_check is None:
                    continue
                label = 1 if int(target_check[1]) == seat else 0
                if label != 1:
                    continue  # we only want oob-but-captured

                # === DUMP ===
                dumped += 1
                print(f"=== sample {dumped} ===  step={step_idx}  seat={seat}")
                print(f"  src   id={src_pid:>2}  owner={int(src[1]):>2}  "
                      f"xy=({float(src[2]):6.2f},{float(src[3]):6.2f})  "
                      f"r={float(src[4]):.2f}  ships={float(src[5]):.0f}")
                print(f"  tgt   id={tgt_pid:>2}  owner={int(target[1]):>2}  "
                      f"xy=({float(target[2]):6.2f},{float(target[3]):6.2f})  "
                      f"r={float(target[4]):.2f}  ships={float(target[5]):.0f}")
                print(f"  emit  angle={angle:+.4f}rad  "
                      f"({math.degrees(angle):+7.2f}°)  ships={ships:.0f}  "
                      f"v={v:.3f}  d={d:.2f}  eta={eta}")
                print(f"  pff   outcome={fate.outcome}  step={fate.step}  "
                      f"hit_planet={fate.hit_planet_id}")
                # Where did pff say the fleet ended up?
                spawn_x = float(src[2]) + math.cos(angle) * (float(src[4]) + 0.1)
                spawn_y = float(src[3]) + math.sin(angle) * (float(src[4]) + 0.1)
                end_x = spawn_x + math.cos(angle) * v * fate.step
                end_y = spawn_y + math.sin(angle) * v * fate.step
                print(f"  fleet  spawn=({spawn_x:6.2f},{spawn_y:6.2f})  "
                      f"end=({end_x:6.2f},{end_y:6.2f}) at pff-step={fate.step}")
                # Where did the env say target ended up at arrival?
                arr_step = min(step_idx + eta, n_steps - 1)
                arr_obs = steps[arr_step][seat].get("observation", {}) or {}
                arr_planets = arr_obs.get("planets", []) or []
                arr_by_id = {int(p[0]): p for p in arr_planets}
                tgt_at_arr = arr_by_id.get(tgt_pid)
                if tgt_at_arr is not None:
                    print(f"  env-target-at-arrival  step={arr_step}  "
                          f"xy=({float(tgt_at_arr[2]):6.2f},"
                          f"{float(tgt_at_arr[3]):6.2f})  "
                          f"owner={int(tgt_at_arr[1])}  "
                          f"ships={float(tgt_at_arr[5]):.0f}")
                # Where does predict_relative say target ended up?
                tgt_tuple = [tgt_pid, int(target[1]), float(target[2]),
                             float(target[3]), float(target[4]),
                             float(target[5]), float(target[6])]
                omega = world.omega
                tgt_orbiting = is_orbiting(tgt_tuple)
                src_tuple = [src_pid, int(src[1]), float(src[2]),
                             float(src[3]), float(src[4]),
                             float(src[5]), float(src[6])]
                src_orbiting = is_orbiting(src_tuple)
                print(f"  omega={omega:+.6f}  tgt_orbiting={tgt_orbiting}  "
                      f"src_orbiting={src_orbiting}")
                if tgt_orbiting and omega != 0:
                    pred_tgt_xy = predict_relative(tgt_tuple, omega, eta)
                    print(f"  pff-target-at-arrival (predict_relative): "
                          f"xy=({pred_tgt_xy[0]:6.2f},{pred_tgt_xy[1]:6.2f})")
                # Where would a centre-to-centre angle land?
                ang_geom = math.atan2(
                    float(target[3]) - float(src[3]),
                    float(target[2]) - float(src[2]),
                )
                fate_geom = predict_fleet_fate(
                    src_obj, tgt_obj, ang_geom,
                    int(round(ships)), world,
                    max_steps=max_steps_cap,
                )
                print(f"  centre-to-centre re-aim: angle={ang_geom:+.4f}rad  "
                      f"({math.degrees(ang_geom):+7.2f}°)  "
                      f"outcome={fate_geom.outcome}  step={fate_geom.step}")
                print()

    print(f"\ndumped {dumped} oob-but-captured samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
