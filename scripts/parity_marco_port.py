"""Parity gate for the lib/opp_marco port (PLAN.md phase 1).

Runs marco-dg-v3-3.py in a kaggle game vs an opponent for a handful of
seeds, captures each turn's obs, then runs:
  - marco's own `eam_choose_moves` internals (via `_plan_beam_search`) to
    get the (src_id, tgt_id) pairs of marco's best plan's fire-now commits;
  - our `predict_marco_plan` to get the predicted fire-now (src_id, tgt_id)
    pairs.

Compares (src_id, tgt_id) tuples directly — bypasses the env-side
orbital-aim correction and the env's own ray-cast attribution, which
would confound a launch-angle parity test.

Pass gate (PLAN.md): per-turn first-3-launch match >= 80% means PASS.
< 60% means ABORT.

Usage:
    python scripts/parity_marco_port.py --seeds 5 --max-step 15
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.opp_marco import predict_marco_plan


def _load_module(path: str, alias: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(alias, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _marco_fire_now_pairs(marco_mod, obs, time_budget_s: float = 0.8) -> tuple[set[tuple[int, int]], bool]:
    """Run marco's own beam search on `obs` and return:
      - the set of (src_id, tgt_id) pairs for fire-now commits (t_launch == 0)
      - True iff the EAM gate fell through (returned None) — meaning marco
        DID NOT use EAM this turn (would have used plan_moves instead).
    """
    world = marco_mod.build_world(obs)
    # Mirror eam_choose_moves' gate checks first.
    if not marco_mod.USE_EAM_OPENING:
        return set(), True
    if not world.my_planets:
        return set(), True
    if world.step >= marco_mod.EAM_OPENING_LIMIT:
        return set(), True
    if world.is_four_player:
        return set(), True
    if len(world.my_planets) > marco_mod.EAM_MAX_MY_PLANETS:
        return set(), True
    for planet in world.my_planets:
        fall = world.fall_turn_map.get(planet.id)
        if fall is not None and fall < marco_mod.EAM_DEFENSE_LOOKAHEAD:
            return set(), True

    n = len(world.my_planets)
    if n == 1:
        depth = 5
    elif n == 2:
        depth = 4
    elif n <= 4:
        depth = 3
    else:
        depth = 2

    deadline = time.perf_counter() + time_budget_s
    best = marco_mod._plan_beam_search(world, depth=depth,
                                       beam_width=marco_mod.PLAN_BEAM_WIDTH,
                                       deadline=deadline)
    if best is None or not best.get("moves"):
        return set(), True

    pairs: set[tuple[int, int]] = set()
    for commit in best["moves"]:
        if int(commit["t_launch"]) == 0:
            pairs.add((int(commit["src_id"]), int(commit["tgt_id"])))
    return pairs, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument(
        "--marco",
        default="audit/2026-06-02-marco-lineage-reference/kernels/marco-dg-v3-3.py",
    )
    ap.add_argument(
        "--opp",
        default="submissions/baseline_pv_eta_anchor_1163.py",
    )
    ap.add_argument("--max-step", type=int, default=15)
    ap.add_argument("--budget-ms", type=float, default=300.0,
                    help="port time budget per call (default 300ms)")
    ap.add_argument("--marco-budget-ms", type=float, default=800.0,
                    help="marco time budget per call (matches marco kernel default)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("KAGGLE_ENV_QUIET", "1")
    from kaggle_environments import make

    marco_mod = _load_module(args.marco, "_parity_marco")
    opp_mod = _load_module(args.opp, "_parity_opp")

    total_matched = 0
    total_predicted = 0
    total_marco_eam = 0
    eam_active_turns = 0
    fall_through_turns = 0
    per_seed: list[dict] = []

    for offset in range(args.seeds):
        seed = args.start_seed + offset
        env = make("orbit_wars", configuration={"episodeSteps": 60,
                                                "actTimeout": 1.0,
                                                "seed": seed},
                   debug=False)
        env.reset()
        state = env.state
        seed_match = 0; seed_pred = 0; seed_actual = 0; seed_eam = 0
        for step in range(args.max_step + 1):
            obs0 = state[0]["observation"]
            obs1 = state[1]["observation"]
            cfg = env.configuration

            # Step the game using both agents' real actions.
            marco_action = marco_mod.agent(obs0, cfg) or []
            opp_action = opp_mod.agent(obs1, cfg) or []

            # Pure parity: marco's beam-search internals vs port.
            marco_pairs, marco_skipped = _marco_fire_now_pairs(
                marco_mod, obs0, time_budget_s=args.marco_budget_ms / 1000.0,
            )
            t0 = time.perf_counter()
            port_plan = predict_marco_plan(obs0, opp_seat=0,
                                           time_budget_ms=args.budget_ms)
            port_ms = 1000.0 * (time.perf_counter() - t0)
            port_pairs: set[tuple[int, int]] = set()
            if port_plan is not None:
                for c in port_plan[:3]:
                    if c.t_launch == 0:
                        port_pairs.add((c.src_id, c.tgt_id))

            if marco_skipped:
                fall_through_turns += 1
                if not args.quiet:
                    print(f"  seed={seed} step={step:2d} | marco-EAM-gate-skip "
                          f"| port={'None' if port_plan is None else len(port_pairs)} "
                          f"port_ms={port_ms:.0f}")
            else:
                eam_active_turns += 1; seed_eam += 1
                marco_first3 = list(marco_pairs)[:3] if len(marco_pairs) > 3 else list(marco_pairs)
                marco_first3_set = set(marco_first3)
                port_first3 = list(port_pairs)[:3] if len(port_pairs) > 3 else list(port_pairs)
                port_first3_set = set(port_first3)
                matched = len(port_first3_set & marco_first3_set)
                total_matched += matched; seed_match += matched
                total_predicted += min(len(port_first3_set), 3)
                total_marco_eam += min(len(marco_first3_set), 3)
                seed_pred += min(len(port_first3_set), 3)
                seed_actual += min(len(marco_first3_set), 3)
                if not args.quiet:
                    print(
                        f"  seed={seed} step={step:2d} "
                        f"| marco={sorted(marco_first3_set)} "
                        f"| port={sorted(port_first3_set)} "
                        f"| match={matched} "
                        f"| port_ms={port_ms:.0f}"
                    )

            env.step([marco_action, opp_action])
            state = env.state
            done = all(s.get("status") in ("DONE", "INACTIVE", "ERROR")
                       for s in state)
            if done:
                break

        per_seed.append({
            "seed": seed, "eam_turns": seed_eam,
            "matched": seed_match,
            "predicted": seed_pred,
            "marco": seed_actual,
        })

    union = max(total_predicted, total_marco_eam)
    pct = (100.0 * total_matched / union) if union > 0 else 0.0
    print()
    print("=" * 60)
    print(f"PARITY SUMMARY over {args.seeds} seeds, steps 0-{args.max_step}")
    print(f"  EAM-active turns          : {eam_active_turns}")
    print(f"  Gate-skip (fall-through)  : {fall_through_turns}")
    print(f"  Total matched pairs       : {total_matched}")
    print(f"  Total port (first-3)      : {total_predicted}")
    print(f"  Total marco (first-3)     : {total_marco_eam}")
    print(f"  Match rate (matched/max)  : {pct:.1f}%")
    print()
    print("Per-seed breakdown:")
    for row in per_seed:
        u = max(row["predicted"], row["marco"])
        p = (100.0 * row["matched"] / u) if u > 0 else 0.0
        print(f"  seed={row['seed']:>3} eam_turns={row['eam_turns']:>2} "
              f"matched={row['matched']:>3} port={row['predicted']:>3} "
              f"marco={row['marco']:>3}  -> {p:5.1f}%")
    print()
    print("PLAN.md gate: pass >= 80%, abort < 60%.")
    if pct >= 80.0:
        print(f"  RESULT: PASS ({pct:.1f}%)")
        return 0
    if pct < 60.0:
        print(f"  RESULT: ABORT ({pct:.1f}%) — port not faithful enough.")
        return 2
    print(f"  RESULT: INCONCLUSIVE ({pct:.1f}%) — between gate thresholds.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
