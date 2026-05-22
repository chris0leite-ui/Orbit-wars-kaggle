"""check_coord_3source_ablation — Gate 3 multi-source falsifier.

Asks: does 3-source coordination EVER produce strictly better bundles
than 2-source on the same target? If essentially never, the 3-source
machinery is computational dead weight and we should ship at
MAX_BUNDLE_SIZE=2 (matching minimal's joint-pair pass coverage).

Per-turn protocol on n=8 seeds × ~30 turns × both player perspectives:

1. Drive games via minimal-vs-minimal.
2. At each turn, enumerate ALL bundles via enumerate_attack_bundles
   with max_bundle_size=3 (strict superset of 2-source enumeration).
3. Run cheap_filter + Tier-2 once.
4. Partition scored bundles by len(legs):
   - best_2_per_target[t] = max tier2_score among 1-or-2-leg bundles
   - best_3_per_target[t] = max tier2_score among exactly-3-leg bundles
5. lift[t] = best_3[t] - best_2[t] (when both exist for target t).
6. Aggregate over all turns.

Acceptance:
- frac_turns_3wins >= 5% AND mean_lift_when_winning >= +2.0
  → 3-source validated, ship at MAX_BUNDLE_SIZE=3.
- frac_turns_3wins >= 5% BUT mean_lift_when_winning < +2.0
  → marginal, ship as MAX_BUNDLE_SIZE=2 for compute savings.
- frac_turns_3wins < 5%
  → 3-source falsified, MAX_BUNDLE_SIZE=2 default.

Usage:
    python scripts/check_coord_3source_ablation.py             # default
    python scripts/check_coord_3source_ablation.py --seeds 4 --turns 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet  # noqa: E402

from agents.coord.main import (  # noqa: E402
    CHEAP_FILTER_TOP_K,
    cheap_filter_bundles,
    enumerate_attack_bundles,
    tier2_score_bundles,
)
from agents.minimal.main import (  # noqa: E402
    _as_dict,
    _num_seats,
    agent as minimal_agent,
)
from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


def _probe_turn(obs, me: int) -> dict | None:
    """Enumerate at max_bundle_size=3, partition by leg count, compute
    per-target lift = best_3 - best_2 across all targets.
    """
    obs_d = _as_dict(obs)
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return None
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return None

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    snap_base = fs_from_obs(obs, num_seats=num_seats)

    attacks = enumerate_attack_bundles(
        my_planets, other_planets, world, model, me, omega,
        max_bundle_size=3,
    )
    if not attacks:
        return None
    cheap = cheap_filter_bundles(
        attacks, world, model, me, num_seats, K=CHEAP_FILTER_TOP_K,
    )
    scored = tier2_score_bundles(cheap, snap_base, me, num_seats, world, model)
    if not scored:
        return None

    best_2: dict[int, float] = {}
    best_3: dict[int, float] = {}
    for b in scored:
        n_legs = len(b.legs)
        if n_legs <= 2:
            cur = best_2.get(int(b.target_id), float("-inf"))
            if b.tier2_score > cur:
                best_2[int(b.target_id)] = float(b.tier2_score)
        elif n_legs == 3:
            cur = best_3.get(int(b.target_id), float("-inf"))
            if b.tier2_score > cur:
                best_3[int(b.target_id)] = float(b.tier2_score)

    lifts: list[float] = []
    three_only_count = 0
    for tid, score_3 in best_3.items():
        if tid in best_2:
            lifts.append(score_3 - best_2[tid])
        else:
            three_only_count += 1

    return {
        "n_targets_2": len(best_2),
        "n_targets_3": len(best_3),
        "n_targets_with_both": len(lifts),
        "n_targets_3_only": three_only_count,
        "lifts": lifts,
        "max_lift": max(lifts) if lifts else 0.0,
        "any_strict_win": any(L > 0 for L in lifts),
    }


def run_probe(seeds: int, turns: int) -> dict:
    per_turn: list[dict] = []
    t_start = time.perf_counter()
    for s in range(seeds):
        env = make("orbit_wars", configuration={"seed": int(s)})
        env.reset(num_agents=2)
        for t in range(turns):
            for me in (0, 1):
                obs = env.state[me].observation
                metrics = _probe_turn(obs, me)
                if metrics:
                    metrics["seed"] = s
                    metrics["turn"] = t
                    metrics["player"] = me
                    per_turn.append(metrics)
            a0 = minimal_agent(env.state[0].observation)
            a1 = minimal_agent(env.state[1].observation)
            env.step([a0, a1])
            if env.done:
                break
        elapsed = time.perf_counter() - t_start
        print(f"  [seed {s}] {len(per_turn)} samples, "
              f"elapsed {elapsed:.1f}s", flush=True)

    n = len(per_turn)
    if n == 0:
        return {"error": "no metrics collected"}

    turns_with_strict_3win = sum(1 for m in per_turn if m["any_strict_win"])
    all_positive_lifts = [
        L for m in per_turn for L in m["lifts"] if L > 0
    ]
    total_three_only = sum(m["n_targets_3_only"] for m in per_turn)
    total_targets_3 = sum(m["n_targets_3"] for m in per_turn)
    max_lift_seen = max(m["max_lift"] for m in per_turn) if per_turn else 0.0

    return {
        "samples": n,
        "turns_with_strict_3win": turns_with_strict_3win,
        "frac_turns_3wins": turns_with_strict_3win / n,
        "n_positive_lifts": len(all_positive_lifts),
        "mean_lift_when_winning": (
            statistics.mean(all_positive_lifts) if all_positive_lifts else 0.0
        ),
        "max_lift_observed": max_lift_seen,
        "total_targets_3_only": total_three_only,
        "total_targets_3": total_targets_3,
        "frac_targets_3_only": (
            total_three_only / total_targets_3 if total_targets_3 > 0 else 0.0
        ),
        "elapsed_seconds": time.perf_counter() - t_start,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--turns", type=int, default=30)
    args = ap.parse_args()

    print(f"[gate3] 3-source ablation on {args.seeds} seeds x "
          f"{args.turns} turns x 2 perspectives", flush=True)
    summary = run_probe(args.seeds, args.turns)

    print()
    print("=" * 62)
    print("GATE 3 — 3-SOURCE ABLATION RESULT")
    print("=" * 62)
    print(f"  Samples:                       {summary['samples']}")
    print(f"  Turns with any 3-source win:   {summary['turns_with_strict_3win']}")
    print(f"  Fraction:                      {summary['frac_turns_3wins']:.3f}")
    print(f"  Positive lifts observed:       {summary['n_positive_lifts']}")
    print(f"  Mean lift when 3 wins:         {summary['mean_lift_when_winning']:+.2f}")
    print(f"  Max lift observed:             {summary['max_lift_observed']:+.2f}")
    print(f"  3-only targets (not 2-reachable): {summary['total_targets_3_only']}")
    print(f"  Frac 3-only of all 3-source:   {summary['frac_targets_3_only']:.3f}")

    frac = summary["frac_turns_3wins"]
    mean_lift = summary["mean_lift_when_winning"]

    if frac >= 0.05 and mean_lift >= 2.0:
        verdict = "PASS — ship at MAX_BUNDLE_SIZE=3"
        rc = 0
    elif frac >= 0.05:
        verdict = (f"MARGINAL — 3-source wins {frac:.1%} but lift only "
                   f"{mean_lift:+.2f} (< 2.0). Consider MAX_BUNDLE_SIZE=2.")
        rc = 0
    else:
        verdict = (f"FALSIFIED — 3-source wins only {frac:.1%} of turns. "
                   f"Ship at MAX_BUNDLE_SIZE=2.")
        rc = 1

    print()
    print(f"  VERDICT: {verdict}")
    print(f"  Elapsed: {summary['elapsed_seconds']:.1f}s")

    audit_dir = REPO / "audit"
    audit_dir.mkdir(exist_ok=True)
    out_path = audit_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"gate3-3source-ablation.json"
    )
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON: {out_path}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
