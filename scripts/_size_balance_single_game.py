"""Single-game behavioral diagnosis of BASELINE_SIZE_BALANCE (A+D fix).

Plays ONE game (fix ON) vs a cheap opponent, then walks every realized
turn and, for each (owned source -> nearest non-owned target) decision
point, compares what enumerate_ship_counts emits with the flag OFF vs ON.
Reports where the fix changes a decision:

  SUPPRESS : ON emits [] where OFF emits a launch  -> doomed launch avoided (D)
  UPSIZE   : ON's lean column > OFF's lean cap      -> arrival-correct size (D)
  CLAMP    : ON's max send  < OFF's budget          -> source-keep respected (A)

This is the qualitative "why" companion to the win-rate A/B (the "whether").
Run vs a cheap opp so it doesn't starve a concurrent A/B (Rule 31).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Champion production config (shared) — match the A/B's config.
for k, v in {
    "BASELINE_JOINT_AGGR": "1", "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60", "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1", "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5", "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1", "BASELINE_PV_ETA": "1",
    "BASELINE_LAUNCH_RULES": "1", "BASELINE_CAPTURE_HORIZON_K": "10",
    "BASELINE_VALUE_HEAD": "hybrid", "BASELINE_CHOOSER": "trajectory",
    "BASELINE_JOINT": "1",
}.items():
    os.environ.setdefault(k, v)

from kaggle_environments import make  # noqa: E402

from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402
from agents.baseline.proposer import (  # noqa: E402
    NUM_TARGETS_PER_SOURCE, MIN_FLEET_SIZE,
    capture_size, capture_floor_arrival, source_keep_floor,
    enumerate_ship_counts, nearest_k,
)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OPP = sys.argv[2] if len(sys.argv) > 2 else "submissions/v7_0_drop_one.py"
ME = 0


def emit_off_on(src, tgt, model, omega, world):
    os.environ.pop("BASELINE_SIZE_BALANCE", None)
    off = enumerate_ship_counts(src, tgt, model, omega, ME, world)
    os.environ["BASELINE_SIZE_BALANCE"] = "1"
    on = enumerate_ship_counts(src, tgt, model, omega, ME, world)
    os.environ.pop("BASELINE_SIZE_BALANCE", None)
    return off, on


def main():
    # Play one game with the fix ON so the realized trajectory reflects it.
    os.environ["BASELINE_SIZE_BALANCE"] = "1"
    env = make("orbit_wars", configuration={"seed": SEED}, debug=False)
    env.run([str(REPO / "agents/baseline/main.py"), str(REPO / OPP)])
    os.environ.pop("BASELINE_SIZE_BALANCE", None)

    final = env.steps[-1]
    r0, r1 = final[0]["reward"], final[1]["reward"]
    outcome = "WIN" if (r0 or 0) > (r1 or 0) else ("LOSS" if (r1 or 0) > (r0 or 0) else "DRAW")
    print(f"== single-game size-balance diag  seed={SEED}  vs {Path(OPP).name} ==")
    print(f"   outcome (focal P0): {outcome}  reward P0={r0} P1={r1}  steps={len(env.steps)}\n")

    n_points = n_suppress = n_upsize = n_clamp = 0
    upsize_gain = clamp_cut = 0
    examples = []

    for t, step in enumerate(env.steps):
        obs = step[ME].get("observation")
        if not obs or not obs.get("planets"):
            continue
        try:
            world = World.from_obs(obs)
            model = WorldModel.from_world(world)
        except Exception:
            continue
        omega = float(obs.get("angular_velocity", 0.0) or 0.0)
        my_srcs = [p for p in world.planets_by_id.values()
                   if int(p.owner) == ME and int(p.ships) >= MIN_FLEET_SIZE]
        non_owned = [p for p in world.planets_by_id.values() if int(p.owner) != ME]
        for src in my_srcs:
            for tgt in nearest_k(non_owned, src, NUM_TARGETS_PER_SOURCE):
                off, on = emit_off_on(src, tgt, model, omega, world)
                if not off:
                    continue  # OFF emitted nothing -> nothing for the fix to change
                n_points += 1
                if not on:
                    n_suppress += 1
                    if len(examples) < 6:
                        cap_arr = capture_floor_arrival(src, tgt, model, omega, ME, world)
                        keep = source_keep_floor(src, 0, world, model, ME)
                        examples.append(
                            f"   t{t:>3} SUPPRESS S{src.id}->T{tgt.id} "
                            f"(own={int(tgt.owner)}): OFF would send {off}, but "
                            f"arrival-need={cap_arr} > sendable={int(src.ships)-int(keep)} "
                            f"(src={int(src.ships)},keep={int(keep)}) -> doomed, dropped")
                    continue
                lean_off, lean_on = min(off), min(on)
                max_off, max_on = max(off), max(on)
                changed = False
                if lean_on > lean_off:
                    n_upsize += 1; upsize_gain += lean_on - lean_off; changed = True
                if max_on < max_off:
                    n_clamp += 1; clamp_cut += max_off - max_on; changed = True
                if changed and len([e for e in examples if "SUPPRESS" not in e]) < 4:
                    cap = capture_size(src, tgt, model, omega, ME, world)
                    cap_arr = capture_floor_arrival(src, tgt, model, omega, ME, world)
                    examples.append(
                        f"   t{t:>3} CHANGE   S{src.id}->T{tgt.id} (own={int(tgt.owner)}): "
                        f"OFF={off} ON={on}  [old cap={cap}, arrival cap={cap_arr}]")

    print(f"   decision points examined (OFF emitted >=1 launch): {n_points}")
    print(f"   SUPPRESS (doomed launch dropped, mode D): {n_suppress}")
    print(f"   UPSIZE   (lean column raised, mode D):    {n_upsize}"
          + (f"  avg +{upsize_gain/n_upsize:.1f} ships" if n_upsize else ""))
    print(f"   CLAMP    (full-send capped, mode A):      {n_clamp}"
          + (f"  avg -{clamp_cut/n_clamp:.1f} ships" if n_clamp else ""))
    unchanged = n_points - n_suppress - max(n_upsize, n_clamp)
    print(f"   (note: UPSIZE and CLAMP can co-occur on one point)\n")
    print("   examples:")
    for e in examples:
        print(e)


if __name__ == "__main__":
    main()
