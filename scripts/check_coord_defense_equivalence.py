"""check_coord_defense_equivalence — Gate 2 defense-equivalence probe.

Measures whether coord's unified-market defense fires at least as often
as minimal's post-hoc emit_threat_reinforcements pass. Day 4's
cheap-filter probe confirmed DEFEND retention at 100% in cheap_top_K,
so defense bundles reach Tier-2. The question now is whether they
win in the Lagrangian or get dropped by competing ATTACK bundles
with higher tier2_score.

Per-turn protocol on n=2 seeds × 60 turns:
1. Drive the game minimal-vs-minimal (state evolution we care about).
2. At each turn, query both agents on player-0's obs:
   - Minimal: propose → choose_trajectory → emit_threat_reinforcements.
     Defensive launches = the extras emit_threat_reinforcements adds.
   - Coord: explicit pipeline (enumerate + cheap_filter + tier2_score
     + lagrangian_clear) with MAX_BUNDLE_SIZE=1 for parity with
     minimal's singleton-shape behaviour. Defensive launches = count
     of selected bundles with kind == BundleKind.DEFEND.

Acceptance:
- Pragmatic: coord_def / minimal_def >= 0.8 → PASS (allow 20% shortfall
  for legitimate counter-attack substitution).
- Hard fail: ratio < 0.5 → calibrate DEFEND_PRIORITY_BOOST.

Usage:
    python scripts/check_coord_defense_equivalence.py             # default
    python scripts/check_coord_defense_equivalence.py --seeds 4
    python scripts/check_coord_defense_equivalence.py --defend-boost 1.5
"""
from __future__ import annotations

import argparse
import json
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
    BundleKind,
    CHEAP_FILTER_TOP_K,
    cheap_filter_bundles,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
    lagrangian_clear,
    tier2_score_bundles,
)
from agents.minimal.main import (  # noqa: E402
    MAX_HORIZON,
    WALLCLOCK_BUDGET_MS,
    _as_dict,
    _num_seats,
    agent as minimal_agent,
    choose_trajectory,
    emit_threat_reinforcements,
    propose,
)
from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


def _minimal_defense_count(obs, me: int) -> int:
    """Run minimal's pipeline up to choose_trajectory, then through
    emit_threat_reinforcements. Defensive launches = added extras.
    """
    obs_d = _as_dict(obs)
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return 0
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return 0
    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine
    snap_base = fs_from_obs(obs, num_seats=num_seats)
    prerank = propose(my_planets, target_pool, world, model, me, omega,
                      baseline_len=MAX_HORIZON + 1)
    moves_pre = choose_trajectory(
        snap_base, prerank, me, num_seats, WALLCLOCK_BUDGET_MS, world,
    )
    moves_post = emit_threat_reinforcements(
        moves_pre, planets, me, world, model, omega,
    )
    return max(0, len(moves_post) - len(moves_pre))


def _coord_defense_count(obs, me: int, defend_boost: float) -> dict:
    """Run coord's explicit pipeline (singleton mode), count selected
    DEFEND bundles. defend_boost multiplies DEFEND tier2_scores before
    Lagrangian.

    Returns dict with: defense_count, attack_count, total_selected.
    """
    obs_d = _as_dict(obs)
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return {"defense_count": 0, "attack_count": 0, "total_selected": 0}
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return {"defense_count": 0, "attack_count": 0, "total_selected": 0}
    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)
    snap_base = fs_from_obs(obs, num_seats=num_seats)

    # Singleton mode + defense enabled — matches the Gate 2 protocol.
    attacks = enumerate_attack_bundles(
        my_planets, other_planets, world, model, me, omega,
        max_bundle_size=1,
    )
    defends = enumerate_defend_bundles(
        my_planets, world, model, me, omega, max_bundle_size=1,
    )
    all_bundles = attacks + defends
    if not all_bundles:
        return {"defense_count": 0, "attack_count": 0, "total_selected": 0}
    cheap = cheap_filter_bundles(
        all_bundles, world, model, me, num_seats, K=CHEAP_FILTER_TOP_K,
    )
    scored = tier2_score_bundles(cheap, snap_base, me, num_seats, world)

    # Apply DEFEND_PRIORITY_BOOST if requested.
    if defend_boost != 1.0:
        from dataclasses import replace
        scored = [
            replace(b, tier2_score=b.tier2_score * defend_boost)
            if b.kind == BundleKind.DEFEND else b
            for b in scored
        ]

    selected = lagrangian_clear(scored, my_planets=my_planets)
    defense_count = sum(1 for b in selected if b.kind == BundleKind.DEFEND)
    attack_count = sum(1 for b in selected if b.kind == BundleKind.ATTACK)
    return {
        "defense_count": defense_count,
        "attack_count": attack_count,
        "total_selected": len(selected),
    }


def run_probe(seeds: int, turns: int, defend_boost: float) -> dict:
    per_turn: list[dict] = []
    t_start = time.perf_counter()
    for s in range(seeds):
        env = make("orbit_wars", configuration={"seed": int(s)})
        env.reset(num_agents=2)
        for t in range(turns):
            # Sample BOTH player perspectives — defense fires asymmetrically
            # for whichever player is currently under threat.
            for me in (0, 1):
                obs = env.state[me].observation
                mc = _minimal_defense_count(obs, me=me)
                cc = _coord_defense_count(obs, me=me, defend_boost=defend_boost)
                per_turn.append({
                    "seed": s, "turn": t, "player": me,
                    "minimal_def": mc,
                    "coord_def": cc["defense_count"],
                    "coord_atk": cc["attack_count"],
                })
            a0 = minimal_agent(env.state[0].observation)
            a1 = minimal_agent(env.state[1].observation)
            env.step([a0, a1])
            if env.done:
                break
        elapsed = time.perf_counter() - t_start
        print(f"  [seed {s}] {len(per_turn)} samples so far, "
              f"elapsed {elapsed:.1f}s", flush=True)

    minimal_total = sum(r["minimal_def"] for r in per_turn)
    coord_def_total = sum(r["coord_def"] for r in per_turn)
    coord_atk_total = sum(r["coord_atk"] for r in per_turn)
    ratio = (coord_def_total / minimal_total) if minimal_total > 0 else float("inf")
    turns_minimal_defended = sum(1 for r in per_turn if r["minimal_def"] > 0)
    turns_coord_defended = sum(1 for r in per_turn if r["coord_def"] > 0)
    return {
        "defend_boost": defend_boost,
        "samples": len(per_turn),
        "minimal_def_total": minimal_total,
        "coord_def_total": coord_def_total,
        "coord_atk_total": coord_atk_total,
        "ratio_coord_over_minimal": ratio,
        "turns_minimal_defended": turns_minimal_defended,
        "turns_coord_defended": turns_coord_defended,
        "elapsed_seconds": time.perf_counter() - t_start,
        "per_turn_sample": per_turn[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--defend-boost", type=float, default=1.0)
    args = ap.parse_args()

    print(f"[gate2] defense-equivalence on {args.seeds} seeds x "
          f"{args.turns} turns (defend_boost={args.defend_boost})", flush=True)
    summary = run_probe(args.seeds, args.turns, args.defend_boost)

    print()
    print("=" * 62)
    print("GATE 2 — DEFENSE-EQUIVALENCE RESULT")
    print("=" * 62)
    print(f"  Samples:                  {summary['samples']}")
    print(f"  Minimal def launches:     {summary['minimal_def_total']}")
    print(f"  Coord def launches:       {summary['coord_def_total']}")
    print(f"  Coord atk launches:       {summary['coord_atk_total']}")
    print(f"  Turns minimal defended:   {summary['turns_minimal_defended']}")
    print(f"  Turns coord defended:     {summary['turns_coord_defended']}")
    print(f"  Ratio coord/minimal:      {summary['ratio_coord_over_minimal']:.3f}")
    print(f"  Defend boost applied:     {summary['defend_boost']:.2f}x")

    ratio = summary["ratio_coord_over_minimal"]
    if ratio >= 0.8:
        verdict = "PASS (>= 0.8)"
        rc = 0
    elif ratio >= 0.5:
        verdict = "MARGINAL (>= 0.5, < 0.8) — consider DEFEND_PRIORITY_BOOST"
        rc = 1
    else:
        verdict = "FAIL (< 0.5) — DEFEND_PRIORITY_BOOST calibration required"
        rc = 1

    print()
    print(f"  VERDICT: {verdict}")
    print(f"  Elapsed: {summary['elapsed_seconds']:.1f}s")

    audit_dir = REPO / "audit"
    audit_dir.mkdir(exist_ok=True)
    out_path = audit_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"gate2-defense-equivalence-boost{int(args.defend_boost * 100)}.json"
    )
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON: {out_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
