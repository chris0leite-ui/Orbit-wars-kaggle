"""Phase 0 (Phase E plan) — Bundle-vs-Baseline failure-mode classifier.

Validates PI's efficiency-is-the-bottleneck hypothesis with data BEFORE
designing the coordinated-ROI scorer. For each of N games of bundle vs
baseline:
  1. Run the game via kaggle_environments.make.
  2. Walk every fleet bundle launched.
  3. Classify each fleet's outcome via existing
     `scripts.episode_postmortem.attribute_fleets` (the live-replay
     classifier from the 2026-05-17 replay-mine work).

Aggregate over all games: % of bundle-launched ships per bucket.

PI gate (from /root/.claude/plans/no-go-forward-test-fluttering-token.md):
  - >=30% wasted (bounced_* + arrived_but_lost + sun/oob/vanished):
    efficiency is the bottleneck → proceed to Phase 1.
  - <15% wasted: target-selection is the bottleneck → ROI is the wrong axis.
  - 15-30%: gray; PI ratifies with the breakdown.

Usage:
    python scripts/diag_bundle_baseline_failures.py [--seeds 8] [--workers 4]

Writes: audit/2026-05-18-phase-e-phase0-failures.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402
from scripts.episode_postmortem import attribute_fleets  # noqa: E402

SEEDS = [42, 1, 7, 13, 31, 100, 17, 23]


def _load_agent(submission_path: Path):
    """Load a submission file's `agent` callable."""
    mod_name = f"_diag_{submission_path.stem}_{id(submission_path)}"
    spec = importlib.util.spec_from_file_location(mod_name, submission_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _run_one_game(seed: int, bundle_seat: int) -> dict:
    """Run one bundle-vs-baseline game with bundle in `bundle_seat` (0 or 1).

    Returns a dict with:
      - seed, bundle_seat
      - reward_bundle (+1 win / -1 loss / 0 draw)
      - n_steps
      - fleets: list of attribute_fleets outputs for bundle's launches
    """
    bundle = _load_agent(REPO / "submissions" / "bundle.py")
    baseline = _load_agent(REPO / "submissions" / "baseline.py")

    if bundle_seat == 0:
        agents = [bundle, baseline]
    else:
        agents = [baseline, bundle]

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    t0 = time.time()
    env.run(agents)
    elapsed = time.time() - t0

    replay = env.toJSON()
    final = replay["steps"][-1]
    rewards = [s.get("reward") or 0 for s in final]
    reward_bundle = rewards[bundle_seat]

    # Determine bundle's player_id from obs (typically == seat, but read it).
    obs_first = replay["steps"][0][bundle_seat]["observation"]
    our_player_id = obs_first.get("player", bundle_seat)

    fleets = attribute_fleets(replay, bundle_seat, our_player_id)

    return {
        "seed": seed,
        "bundle_seat": bundle_seat,
        "reward_bundle": reward_bundle,
        "n_steps": len(replay["steps"]),
        "elapsed_s": round(elapsed, 1),
        "fleets": fleets,
    }


def _run_one_task(task):
    seed, bundle_seat = task
    return _run_one_game(seed, bundle_seat)


WASTED_BUCKETS = {
    "bounced_neutral", "bounced_enemy", "arrived_but_lost",
    "sun", "oob", "vanished_in_space", "comet_collision",
    "hit_planet_unknown_flip",
}
PRODUCTIVE_BUCKETS = {"captured", "reinforced_self"}
INFLIGHT_BUCKETS = {"alive_at_end"}
UNKNOWN_BUCKETS = {"unknown"}


def aggregate(games: list[dict]) -> dict:
    """Roll up per-outcome ship totals and counts across N games."""
    bucket_ships: dict[str, int] = defaultdict(int)
    bucket_count: dict[str, int] = defaultdict(int)
    total_ships = 0
    total_fleets = 0
    wins = 0
    for g in games:
        if g["reward_bundle"] > 0:
            wins += 1
        for f in g["fleets"]:
            ships = int(f.get("ships", 0))
            outcome = f.get("outcome", "unknown")
            bucket_ships[outcome] += ships
            bucket_count[outcome] += 1
            total_ships += ships
            total_fleets += 1

    pct_ships = {b: bucket_ships[b] / total_ships * 100 if total_ships else 0.0
                 for b in bucket_ships}
    pct_count = {b: bucket_count[b] / total_fleets * 100 if total_fleets else 0.0
                 for b in bucket_count}

    wasted_ships = sum(bucket_ships[b] for b in WASTED_BUCKETS)
    productive_ships = sum(bucket_ships[b] for b in PRODUCTIVE_BUCKETS)
    inflight_ships = sum(bucket_ships[b] for b in INFLIGHT_BUCKETS)

    return {
        "n_games": len(games),
        "bundle_wins": wins,
        "total_fleets": total_fleets,
        "total_ships_launched": total_ships,
        "bucket_ships": dict(bucket_ships),
        "bucket_count": dict(bucket_count),
        "pct_ships_by_bucket": pct_ships,
        "pct_count_by_bucket": pct_count,
        "pct_wasted_ships": (wasted_ships / total_ships * 100) if total_ships else 0.0,
        "pct_productive_ships": (productive_ships / total_ships * 100) if total_ships else 0.0,
        "pct_inflight_ships": (inflight_ships / total_ships * 100) if total_ships else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8, help="Number of seeds (each runs both sides)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="audit/2026-05-18-phase-e-phase0-failures.json")
    args = ap.parse_args()

    seeds = SEEDS[:args.seeds]
    tasks = [(s, 0) for s in seeds] + [(s, 1) for s in seeds]
    print(f"Running {len(tasks)} games ({len(seeds)} seeds × 2 sides) with {args.workers} workers...")
    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        games = []
        for i, g in enumerate(pool.imap_unordered(_run_one_task, tasks)):
            games.append(g)
            print(f"  [{i+1}/{len(tasks)}] seed={g['seed']} bundle_seat={g['bundle_seat']} "
                  f"reward={g['reward_bundle']:+d} n_steps={g['n_steps']} "
                  f"fleets={len(g['fleets'])} elapsed={g['elapsed_s']}s",
                  flush=True)

    agg = aggregate(games)
    elapsed_total = time.time() - t0

    out = {
        "schema_version": 1,
        "config": {
            "bundle": "submissions/bundle.py",
            "baseline": "submissions/baseline.py",
            "seeds": seeds,
            "n_games": len(games),
            "workers": args.workers,
        },
        "elapsed_s": round(elapsed_total, 1),
        "aggregate": agg,
        "per_game": [
            {k: v for k, v in g.items() if k != "fleets"} | {"n_fleets": len(g["fleets"])}
            for g in games
        ],
    }

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n========== Aggregate ({agg['n_games']} games, {agg['total_fleets']} fleets, "
          f"{agg['total_ships_launched']} ships) ==========")
    print(f"  Bundle wins: {agg['bundle_wins']}/{agg['n_games']} = "
          f"{agg['bundle_wins']/agg['n_games']*100:.1f}%")
    print(f"  Ships PRODUCTIVE (captured + reinforced):  {agg['pct_productive_ships']:5.1f}%")
    print(f"  Ships WASTED (bounce/recapture/transit):   {agg['pct_wasted_ships']:5.1f}%")
    print(f"  Ships IN-FLIGHT at end:                    {agg['pct_inflight_ships']:5.1f}%")
    print("\n  Per-bucket ship share:")
    for bucket in sorted(agg["pct_ships_by_bucket"], key=lambda b: -agg["pct_ships_by_bucket"][b]):
        pct_s = agg["pct_ships_by_bucket"][bucket]
        pct_c = agg["pct_count_by_bucket"][bucket]
        ships = agg["bucket_ships"][bucket]
        count = agg["bucket_count"][bucket]
        print(f"    {bucket:30s}  ships={ships:6d} ({pct_s:5.1f}%)  fleets={count:4d} ({pct_c:5.1f}%)")

    print(f"\nElapsed: {elapsed_total:.0f}s.  Detail dump: {out_path}")

    # PI gate check
    wasted = agg["pct_wasted_ships"]
    if wasted >= 30:
        print(f"\nGate: PASS for ROI axis (wasted={wasted:.1f}% ≥ 30%). Efficiency IS the bottleneck.")
    elif wasted < 15:
        print(f"\nGate: FAIL for ROI axis (wasted={wasted:.1f}% < 15%). Target-selection is the bottleneck.")
    else:
        print(f"\nGate: GRAY (wasted={wasted:.1f}%, in 15-30% band). PI to ratify direction with this breakdown.")


if __name__ == "__main__":
    main()
