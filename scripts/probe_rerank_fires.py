"""Instrumentation probe: how many turns per game does the marco-EAM
rerank actually fire?

We trace a focal agent (baseline with both flags ON) vs a chosen opponent
for `--seeds` games, and per turn record:
  - was the focal seat's rerank gate satisfied? (BASELINE_OPP_MARCO,
    BASELINE_ADVERSARIAL_RERANK, step < 50, num_seats==2)
  - did predict_marco_plan return non-None for any opp seat?
  - opp's planet count at this step

Output a per-game histogram of "rerank-fires" turns vs "no-op" turns,
plus the breakdown of why the gate fell through.

Usage:
    python scripts/probe_rerank_fires.py --seeds 3 --opp <agent.py>
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KAGGLE_ENV_QUIET", "1")
# Force the flags ON so the focal agent we load below runs the rerank.
os.environ["BASELINE_OPP_MARCO"] = "1"
os.environ["BASELINE_ADVERSARIAL_RERANK"] = "1"
os.environ["BASELINE_ADV_RERANK_MARCO_BUDGET_MS"] = "150.0"


def _load(path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--opp", required=True,
                    help="opponent agent path (e.g. romantamrazov.py)")
    ap.add_argument("--max-step", type=int, default=80,
                    help="stop trace at this step (default 80 — enough to see EAM window+early-mid)")
    args = ap.parse_args()

    from kaggle_environments import make
    from lib.opp_marco import predict_marco_plan, EAM_OPENING_LIMIT, EAM_MAX_MY_PLANETS

    # The focal agent IS the baseline at the source level — we want to
    # observe the chooser path. We don't need to load the BUNDLED
    # baseline_marco_on.py; loading agents.baseline.main keeps the
    # observability inside our process.
    from agents.baseline.main import agent as focal_agent
    opp_mod = _load(args.opp, "_opp_probe")

    total_active_turns = 0
    total_gate_passed = 0
    total_predict_non_none = 0
    per_game_summary: list[dict] = []

    for offset in range(args.seeds):
        seed = args.start_seed + offset
        env = make("orbit_wars",
                   configuration={"episodeSteps": args.max_step + 5,
                                  "actTimeout": 1.0, "seed": seed},
                   debug=False)
        env.reset()
        state = env.state

        active_turns = 0; gate_passed = 0; predict_non_none = 0
        gate_skip_step = 0; gate_skip_planets = 0; gate_skip_4p = 0
        per_turn_log: list[tuple[int, str]] = []

        for step in range(args.max_step + 1):
            obs0 = state[0]["observation"]
            obs1 = state[1]["observation"]
            cfg = env.configuration

            # We only instrument focal=P0; this is enough to characterise
            # the rerank window for the symmetric agent.
            planets0 = obs0.get("planets", []) if isinstance(obs0, dict) \
                else getattr(obs0, "planets", [])
            my_planets_count = sum(1 for p in planets0 if int(p[1]) == 0)
            opp_planets_count = sum(1 for p in planets0 if int(p[1]) == 1)
            num_seats = 2  # 2P game

            active_turns += 1

            # Reproduce the gate logic from chooser_trajectory.py without
            # actually running the chooser — just count.
            gate_step_ok = step < 50  # ADV_RERANK_LIMIT
            gate_4p_ok = (num_seats <= 2)
            if not gate_step_ok:
                gate_skip_step += 1
            elif not gate_4p_ok:
                gate_skip_4p += 1
            else:
                # Call predict_marco_plan from focal=P0's perspective on
                # opp_id=1's obs (this is what _build_opp_marco_plans does).
                opp_obs = obs1
                t0 = time.perf_counter()
                plan = predict_marco_plan(opp_obs, opp_seat=1,
                                          time_budget_ms=150.0)
                planner_ms = 1000.0 * (time.perf_counter() - t0)
                if plan is not None:
                    gate_passed += 1
                    predict_non_none += 1
                    per_turn_log.append((step, f"FIRE n_commits={len(plan)} opp_planets={opp_planets_count} planner={planner_ms:.0f}ms"))
                else:
                    # Distinguish why marco's gate failed (opp owns > 6, fall_turn risk).
                    reason = "marco_gate_fail"
                    if opp_planets_count > EAM_MAX_MY_PLANETS:
                        reason = f"opp_owns>{EAM_MAX_MY_PLANETS}"
                        gate_skip_planets += 1
                    per_turn_log.append((step, f"NOFIRE {reason} opp_planets={opp_planets_count}"))

            # Step env using actual agent moves.
            try:
                act0 = focal_agent(obs0, cfg)
            except Exception as e:
                print(f"  seed={seed} step={step}: focal crashed: {e!r}", file=sys.stderr)
                break
            try:
                act1 = opp_mod.agent(obs1, cfg)
            except Exception:
                act1 = []
            env.step([act0, act1])
            state = env.state
            done = all(s.get("status") in ("DONE", "INACTIVE", "ERROR")
                       for s in state)
            if done:
                break

        per_game_summary.append({
            "seed": seed, "active": active_turns, "fires": gate_passed,
            "skip_step": gate_skip_step, "skip_planets": gate_skip_planets,
            "skip_4p": gate_skip_4p,
        })
        total_active_turns += active_turns
        total_gate_passed += gate_passed
        total_predict_non_none += predict_non_none

        # Per-game turn-by-turn brief
        print(f"\n=== seed {seed}: {gate_passed}/{active_turns} turns fired "
              f"(skip step>=50: {gate_skip_step}, skip opp>6 planets: "
              f"{gate_skip_planets}) ===")
        # Print first 20 turns of trace and any FIRE event
        printed = 0
        for s, e in per_turn_log:
            if "FIRE" in e or s < 20:
                print(f"  step={s:2d} {e}")
                printed += 1
                if printed > 40:
                    break

    print()
    print("=" * 60)
    print("RERANK-FIRES SUMMARY")
    print(f"  Games:            {args.seeds}")
    print(f"  Total turns:      {total_active_turns}")
    print(f"  Rerank fires:     {total_gate_passed} "
          f"({100*total_gate_passed/max(1,total_active_turns):.1f}%)")
    print(f"  Mean fires/game:  {total_gate_passed/max(1,args.seeds):.1f}")
    print()
    for r in per_game_summary:
        print(f"  seed={r['seed']:>3} "
              f"fires={r['fires']:>2}/{r['active']:>3} active "
              f"skip(step)={r['skip_step']:>2} skip(opp>6)={r['skip_planets']:>2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
