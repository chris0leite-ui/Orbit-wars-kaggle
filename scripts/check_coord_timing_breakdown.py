"""Instrumentation probe — measures per-turn timing + bundle-count
breakdown for coord, to diagnose "agent idles in crowded games."

Wraps the public stages (`enumerate_attack_bundles`, `enumerate_defend_bundles`,
`cheap_filter_bundles`, `tier2_score_bundles`, `lagrangian_clear`,
`emit_bundle_actions`) and records:
  - wallclock per stage (ms)
  - bundle count at each stage
  - moves emitted

Identifies idle turns (0 moves emitted) and reports which stage starved
the pipeline.

Usage:
    python scripts/check_coord_timing_breakdown.py [--seeds 0,1] [--max-turns 200]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Ensure default coord behavior (DELTA_W=1, leaf_floor=0.0).
os.environ.setdefault("COORD_DELTA_W", "1")

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet  # noqa: E402

from agents.coord.main import (  # noqa: E402
    CHEAP_FILTER_TOP_K,
    WALLCLOCK_BUDGET_MS,
    cheap_filter_bundles,
    emit_bundle_actions,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
    lagrangian_clear,
    tier2_score_bundles,
    _as_dict,
    _num_seats,
)
from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


def instrumented_turn(obs):
    """Run coord's pipeline on `obs` with per-stage timing."""
    timings = {}
    counts = {}
    t0 = time.perf_counter()

    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    num_seats = _num_seats(planets, fleets)
    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    snap_base = fs_from_obs(obs, num_seats=num_seats)
    timings["parse"] = (time.perf_counter() - t0) * 1000.0
    counts["my_planets"] = len(my_planets)
    counts["other_planets"] = len(other_planets)
    counts["in_flight"] = len(raw_fleets)

    if not my_planets:
        return {"timings": timings, "counts": counts, "moves": 0,
                "abort_reason": "no_own_planets"}

    t1 = time.perf_counter()
    attacks = enumerate_attack_bundles(
        my_planets, other_planets, world, model, me, omega,
    )
    defends = enumerate_defend_bundles(
        my_planets, world, model, me, omega,
    )
    timings["enumerate"] = (time.perf_counter() - t1) * 1000.0
    counts["raw_attack"] = len(attacks)
    counts["raw_defend"] = len(defends)
    all_bundles = attacks + defends

    if not all_bundles:
        return {"timings": timings, "counts": counts, "moves": 0,
                "abort_reason": "no_bundles_enumerated"}

    t2 = time.perf_counter()
    cheap = cheap_filter_bundles(
        all_bundles, world, model, me, num_seats, K=CHEAP_FILTER_TOP_K,
    )
    timings["cheap_filter"] = (time.perf_counter() - t2) * 1000.0
    counts["cheap"] = len(cheap)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    tier2_budget_ms = max(50.0, float(WALLCLOCK_BUDGET_MS) - elapsed_ms - 60.0)

    t3 = time.perf_counter()
    scored = tier2_score_bundles(
        cheap, snap_base, me, num_seats, world, model,
        wallclock_ms=tier2_budget_ms,
    )
    timings["tier2"] = (time.perf_counter() - t3) * 1000.0
    counts["scored"] = len(scored)
    # Did Tier-2 pre-bail drop bundles?
    counts["tier2_dropped"] = len(cheap) - len(scored)

    # How many bundles had positive composite vs negative?
    positive_composite = sum(
        1 for b in scored if (b.tier2_score + b.endgame_bonus) > 0
    )
    counts["positive_composite"] = positive_composite
    # How many would pass the leaf-floor (default 0.0)?
    above_leaf_floor = sum(1 for b in scored if b.tier2_score >= 0.0)
    counts["above_leaf_floor"] = above_leaf_floor

    t4 = time.perf_counter()
    selected = lagrangian_clear(scored, my_planets=my_planets)
    timings["lagrangian"] = (time.perf_counter() - t4) * 1000.0
    counts["selected"] = len(selected)

    t5 = time.perf_counter()
    moves = emit_bundle_actions(selected, world, model, me)
    timings["emit"] = (time.perf_counter() - t5) * 1000.0
    counts["moves"] = len(moves)
    timings["total"] = (time.perf_counter() - t0) * 1000.0

    abort_reason = None
    if not moves:
        if not selected:
            abort_reason = (
                "lagrangian_chose_nothing"
                if scored else "no_scored_bundles"
            )
        else:
            abort_reason = "all_selected_legs_were_wait_or_stranded"

    return {"timings": timings, "counts": counts, "moves": len(moves),
            "abort_reason": abort_reason}


def play_one(seed: int, max_turns: int = 200):
    from agents.minimal.main import agent as minimal_agent
    env = make("orbit_wars", configuration={"seed": int(seed)})
    env.reset(num_agents=2)
    per_turn = []
    for turn in range(max_turns):
        if env.done:
            break
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        # Coord at P0 with instrumentation.
        info = instrumented_turn(obs0)
        info["turn"] = turn
        per_turn.append(info)
        # Apply the actual coord move, opponent plays minimal.
        from agents.coord.main import agent as coord_agent
        a0 = coord_agent(obs0)
        a1 = minimal_agent(obs1)
        env.step([a0, a1])
    return per_turn


def summarize(per_turn):
    n = len(per_turn)
    idles = [t for t in per_turn if t["moves"] == 0]
    print(f"\n=== {n} turns played ===")
    print(f"  idle turns (0 moves emitted): {len(idles)} ({len(idles)/n:.1%})")

    if idles:
        reasons = {}
        for t in idles:
            r = t.get("abort_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        print("  idle reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}")

    # Per-stage wallclock distribution
    print("\n  per-stage wallclock (ms, p50/p95/max):")
    stages = ("parse", "enumerate", "cheap_filter", "tier2",
              "lagrangian", "emit", "total")
    for s in stages:
        vals = [t["timings"].get(s, 0.0) for t in per_turn
                if "timings" in t and s in t["timings"]]
        if not vals:
            continue
        vals.sort()
        p50 = vals[len(vals) // 2]
        p95 = vals[int(len(vals) * 0.95)] if len(vals) > 1 else vals[0]
        print(f"    {s:>14s}: p50={p50:6.1f}  p95={p95:6.1f}  max={max(vals):6.1f}")

    # Bundle counts distribution
    print("\n  bundle counts (median over turns):")
    keys = ("my_planets", "other_planets", "in_flight",
            "raw_attack", "raw_defend", "cheap", "scored",
            "tier2_dropped", "positive_composite",
            "above_leaf_floor", "selected", "moves")
    for k in keys:
        vals = [t["counts"].get(k, 0) for t in per_turn if "counts" in t]
        if not vals:
            continue
        vals.sort()
        med = vals[len(vals) // 2]
        mx = max(vals)
        print(f"    {k:>20s}: median={med}  max={mx}")

    # Crowded turn flag: turns where in_flight is in the top quartile
    in_flight_vals = sorted(t["counts"].get("in_flight", 0)
                            for t in per_turn if "counts" in t)
    if in_flight_vals:
        q75 = in_flight_vals[int(len(in_flight_vals) * 0.75)]
        crowded = [t for t in per_turn if t["counts"].get("in_flight", 0) > q75]
        crowded_idles = [t for t in crowded if t["moves"] == 0]
        if crowded:
            print(f"\n  crowded turns (in_flight > q75={q75}): "
                  f"{len(crowded)}, of which idle: "
                  f"{len(crowded_idles)} ({len(crowded_idles)/len(crowded):.1%})")
            crowded_total_ms = [t["timings"].get("total", 0)
                                for t in crowded if "timings" in t]
            normal_total_ms = [t["timings"].get("total", 0)
                               for t in per_turn
                               if "timings" in t
                               and t["counts"].get("in_flight", 0) <= q75]
            if crowded_total_ms and normal_total_ms:
                print(f"  crowded p50 wallclock: "
                      f"{sorted(crowded_total_ms)[len(crowded_total_ms)//2]:.1f}ms")
                print(f"  normal  p50 wallclock: "
                      f"{sorted(normal_total_ms)[len(normal_total_ms)//2]:.1f}ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0",
                        help="comma-separated seed list (1-2 recommended)")
    parser.add_argument("--max-turns", type=int, default=200)
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for s in seeds:
        print(f"\n========================================")
        print(f"=== seed {s} (max {args.max_turns} turns)")
        print(f"========================================")
        per_turn = play_one(s, args.max_turns)
        summarize(per_turn)

        # Also list the first 5 idle turns with context
        idles = [t for t in per_turn if t["moves"] == 0][:5]
        if idles:
            print("\n  first 5 idle turns:")
            for t in idles:
                print(f"    turn {t['turn']}: reason={t.get('abort_reason')}, "
                      f"counts={t['counts']}, "
                      f"timings={'/'.join(f'{k}={v:.0f}' for k, v in t['timings'].items())}")


if __name__ == "__main__":
    main()
