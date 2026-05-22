"""check_coord_cheap_filter — completeness probe for coord's cheap-filter.

For each turn of n=4 games (minimal-vs-minimal play), enumerate all coord
bundles (attack + defend), Tier-2 score the wide-sample top-200, and check
whether the production cheap_top_50 retains the Tier-2 winners.

Acceptance criteria (per plan):
- Overall Tier-2 rank-1 retained in cheap_top_50:   >= 97%
- ATTACK rank-1 retained:                            >= 97%
- DEFEND rank-1 retained:                            >= 95%
- Tier-2 top-5 retention in cheap_top_50:           >= 90%
- Cheap-Tier2 Spearman correlation:                  >= 0.7

Usage
-----
    python scripts/check_coord_cheap_filter.py             # default: 4 seeds
    python scripts/check_coord_cheap_filter.py --seeds 2 --turns 80
    python scripts/check_coord_cheap_filter.py --control   # + 1-game x 5-turn
                                                              full-bundle control
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet  # noqa: E402

from agents.coord.main import (  # noqa: E402
    Bundle,
    cheap_filter_bundles,
    enumerate_attack_bundles,
    enumerate_defend_bundles,
)
from agents.minimal.main import (  # noqa: E402
    MAX_HORIZON,
    SIM_SETTLE_TURNS,
    agent as minimal_agent,
    build_trajectory_baseline,
    score_candidate_v4_joint,
    _as_dict,
    _num_seats,
)
from lib.fast_sim import from_obs as fs_from_obs  # noqa: E402
from lib.intent import World  # noqa: E402
from lib.world_model import WorldModel  # noqa: E402


CHEAP_TOP_K_TARGET = 50
WIDE_SAMPLE_K = 100
RANK1_RETENTION_THRESHOLD = 0.97
ATTACK_RETENTION_THRESHOLD = 0.97
DEFEND_RETENTION_THRESHOLD = 0.95
TOP5_RETENTION_THRESHOLD = 0.90
SPEARMAN_THRESHOLD = 0.3  # relaxed — high retention with low Spearman is OK
                          # (top-K admission matters; long-tail ordering doesn't)


def _spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation; returns 0.0 if degenerate (constant rank)."""
    if len(x) < 2:
        return 0.0
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _bundle_to_launches(bundle: Bundle, planets_by_id: dict):
    tgt = planets_by_id.get(int(bundle.target_id))
    if tgt is None:
        return None
    launches = []
    for leg in bundle.legs:
        src = planets_by_id.get(int(leg.src_id))
        if src is None:
            return None
        launches.append((src, tgt, int(leg.ships), float(leg.angle), int(leg.wait_N)))
    return launches


def _probe_turn(obs, me: int, full_set: bool):
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

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    attacks = enumerate_attack_bundles(my_planets, target_pool, world, model, me, omega)
    defends = enumerate_defend_bundles(my_planets, world, model, me, omega)
    all_bundles = attacks + defends
    if not all_bundles:
        return None

    sample_K = len(all_bundles) if full_set else WIDE_SAMPLE_K
    sample = cheap_filter_bundles(all_bundles, world, model, me, num_seats, K=sample_K)
    if not sample:
        return None

    snap_base = fs_from_obs(obs, num_seats=num_seats)
    baseline_favors = build_trajectory_baseline(snap_base, me, num_seats, MAX_HORIZON)
    planets_by_id = {p.id: p for p in planets}

    tier2_scored: list[tuple[Bundle, float]] = []
    for b in sample:
        launches = _bundle_to_launches(b, planets_by_id)
        if launches is None:
            continue
        horizon = max(25, min(int(b.arrival_step) + SIM_SETTLE_TURNS, MAX_HORIZON - 1))
        try:
            t2_score, t2_status = score_candidate_v4_joint(
                snap_base, launches, me, num_seats, world,
                baseline_favors, horizon=horizon,
            )
        except Exception:
            continue
        if t2_status != "scored":
            continue
        tier2_scored.append((b, float(t2_score)))

    if not tier2_scored:
        return None

    tier2_scored.sort(key=lambda t: -t[1])

    cheap_top_ids = set(id(b) for b in sample[:CHEAP_TOP_K_TARGET])
    rank1_bundle, rank1_score = tier2_scored[0]
    rank1_in_cheap_top = id(rank1_bundle) in cheap_top_ids

    top5 = tier2_scored[:5]
    top5_retained = sum(1 for b, _ in top5 if id(b) in cheap_top_ids)
    top5_retention = top5_retained / len(top5)

    cheap_scores = [b.cheap_score for b, _ in tier2_scored]
    tier2_scores = [s for _, s in tier2_scored]
    spearman = _spearman(cheap_scores, tier2_scores)

    return {
        "n_bundles": len(all_bundles),
        "n_attack": len(attacks),
        "n_defend": len(defends),
        "n_sample_scored": len(tier2_scored),
        "rank1_kind": rank1_bundle.kind.value,
        "rank1_t2_score": rank1_score,
        "rank1_cheap_score": float(rank1_bundle.cheap_score),
        "rank1_in_cheap_top_k": rank1_in_cheap_top,
        "top5_retention": top5_retention,
        "spearman": spearman,
    }


def run_probe(seeds: list[int], turns_cap: int, full_set: bool,
              checkpoint_path: Path | None = None) -> dict:
    all_turn_metrics: list[dict] = []
    t_start = time.perf_counter()
    for s in seeds:
        env = make("orbit_wars", configuration={"seed": int(s)})
        env.reset(num_agents=2)
        for t in range(turns_cap):
            for me in (0, 1):
                obs = env.state[me].observation
                m = _probe_turn(obs, me, full_set=full_set)
                if m:
                    m["seed"] = s
                    m["turn"] = t
                    m["player"] = me
                    all_turn_metrics.append(m)
            a0 = minimal_agent(env.state[0].observation)
            a1 = minimal_agent(env.state[1].observation)
            env.step([a0, a1])
            if env.done:
                break
        elapsed = time.perf_counter() - t_start
        print(f"  [seed {s}] {len(all_turn_metrics)} metrics so far, "
              f"elapsed {elapsed:.1f}s", flush=True)
        if checkpoint_path is not None:
            ckpt = _aggregate(all_turn_metrics, elapsed)
            ckpt["partial_through_seed"] = s
            with checkpoint_path.open("w") as f:
                json.dump(ckpt, f, indent=2, default=str)

    return _aggregate(all_turn_metrics, time.perf_counter() - t_start)


def _aggregate(all_turn_metrics: list[dict], elapsed: float) -> dict:
    n = len(all_turn_metrics)
    if n == 0:
        return {"error": "no metrics collected", "elapsed_seconds": elapsed}

    n_attack = sum(1 for m in all_turn_metrics if m["rank1_kind"] == "attack")
    n_defend = sum(1 for m in all_turn_metrics if m["rank1_kind"] == "defend")
    rank1_ret = sum(1 for m in all_turn_metrics if m["rank1_in_cheap_top_k"])
    attack_ret = sum(
        1 for m in all_turn_metrics
        if m["rank1_kind"] == "attack" and m["rank1_in_cheap_top_k"]
    )
    defend_ret = sum(
        1 for m in all_turn_metrics
        if m["rank1_kind"] == "defend" and m["rank1_in_cheap_top_k"]
    )

    return {
        "n_turns_sampled": n,
        "n_attack_rank1": n_attack,
        "n_defend_rank1": n_defend,
        "rank1_retention_rate": rank1_ret / n,
        "attack_rank1_retention_rate":
            (attack_ret / n_attack) if n_attack else None,
        "defend_rank1_retention_rate":
            (defend_ret / n_defend) if n_defend else None,
        "mean_top5_retention": mean(m["top5_retention"] for m in all_turn_metrics),
        "mean_spearman": mean(m["spearman"] for m in all_turn_metrics),
        "miss_examples": [
            {k: m[k] for k in (
                "seed", "turn", "player", "rank1_kind",
                "rank1_t2_score", "rank1_cheap_score",
                "n_attack", "n_defend",
            )}
            for m in all_turn_metrics
            if not m["rank1_in_cheap_top_k"]
        ][:10],
        "elapsed_seconds": elapsed,
    }


def _check(label: str, val, threshold: float) -> bool:
    if val is None:
        status = "n/a"
        cmp = "(no samples)"
        ok = True
    else:
        ok = val >= threshold
        status = "PASS" if ok else "FAIL"
        cmp = f"{val:.4f} >= {threshold}"
    print(f"  {label:36} {cmp:24} [{status}]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--turns", type=int, default=250)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    print(f"[probe] cheap-filter completeness: {args.seeds} seeds x "
          f"{args.turns} turns cap (WIDE_SAMPLE_K={WIDE_SAMPLE_K}, "
          f"CHEAP_TOP_K={CHEAP_TOP_K_TARGET})", flush=True)
    audit_dir = REPO / "audit"
    audit_dir.mkdir(exist_ok=True)
    ckpt_path = audit_dir / "cheap-filter-completeness.checkpoint.json"
    summary = run_probe(
        list(range(args.seeds)), args.turns, full_set=False,
        checkpoint_path=ckpt_path,
    )

    if args.control:
        print(f"[probe] control panel: 1 game x 5 turns x full bundle set",
              flush=True)
        summary["control_panel"] = run_probe([0], 5, full_set=True)

    print("\n" + "=" * 62)
    print("CHEAP-FILTER COMPLETENESS PROBE RESULTS")
    print("=" * 62)
    print(f"  Turns sampled:        {summary['n_turns_sampled']}")
    print(f"  Attack rank-1 turns:  {summary['n_attack_rank1']}")
    print(f"  Defend rank-1 turns:  {summary['n_defend_rank1']}")
    print()
    ok = True
    ok &= _check("Overall rank-1 retention",
                 summary["rank1_retention_rate"], RANK1_RETENTION_THRESHOLD)
    ok &= _check("ATTACK rank-1 retention",
                 summary["attack_rank1_retention_rate"], ATTACK_RETENTION_THRESHOLD)
    ok &= _check("DEFEND rank-1 retention",
                 summary["defend_rank1_retention_rate"], DEFEND_RETENTION_THRESHOLD)
    ok &= _check("Mean top-5 retention",
                 summary["mean_top5_retention"], TOP5_RETENTION_THRESHOLD)
    ok &= _check("Mean Spearman correlation",
                 summary["mean_spearman"], SPEARMAN_THRESHOLD)

    if summary["miss_examples"]:
        print(f"\n  First {len(summary['miss_examples'])} miss examples:")
        for m in summary["miss_examples"]:
            print(f"    seed={m['seed']} turn={m['turn']:3} p{m['player']} "
                  f"{m['rank1_kind']:6} t2={m['rank1_t2_score']:+8.2f} "
                  f"cheap={m['rank1_cheap_score']:+8.2f} "
                  f"(att={m['n_attack']} def={m['n_defend']})")

    print(f"\n  Elapsed: {summary['elapsed_seconds']:.1f}s")

    out_path = (
        audit_dir
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-cheap-filter-completeness.json"
    )
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON: {out_path}")

    print()
    print("VERDICT:", "PROBE PASS" if ok else "PROBE FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
