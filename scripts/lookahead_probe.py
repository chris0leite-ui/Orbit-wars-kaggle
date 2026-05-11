"""Phase 1a — does a WorldModel-horizon predictor forecast the winner better
than the trivial "current ship count" predictor?

The strategic hypothesis (PI 2026-05-11): rich global ROI scoring needs a
*timeline-integrated* signal — sum of our predicted ships at some future
horizon H, NOT the snapshot of current ship totals. If that horizon-integrated
score correlates more tightly with the eventual game outcome than the naive
current-ship-delta does, the lookahead substrate is worth building on; if it
doesn't (because games are decided by adversarial actions our static
simulator can't anticipate), the lever is elsewhere.

Probe design:

    For each (seed, midgame step, player_pov) sample we record:

      naive   = sum(our ships now) - sum(their ships now)
                  (planets owner==me + fleets owner==me)
      horizon_H = sum(predicted our ships at step_now+H from WorldModel)
                  - sum(predicted their ships at step_now+H)
                    for H in {50, 100, 200, 300, 500}.

    Ground truth = winner at step 500 (or earlier termination), labelled
    {+1 us-won, -1 us-lost} from each player POV. Multi-way ties (env
    rewards [1,1]) drop the sample.

    For each (step_now, H) pair we compute the AUC of `score >
    threshold -> predicts us-won`. AUC = 0.5 is no information; 1.0 is
    perfect. AUC delta over `naive` is what we care about.

Generator choice: we deliberately do NOT use v2 self-play — both sides
play identically and converge to a step-500 reward-tied draw, which the
probe drops as zero-signal. Asymmetric pairs (v2 vs roi_baseline,
v2 vs weakest) produce clear winners and the predictor power is
measurable. The probe's `pov` is always the first listed agent
(--p0-agent).

CLI:
    python -m scripts.lookahead_probe --seeds 16 --steps 25,50,75,100,150,200 \\
        --p0-agent agents/v2/main.py --p1-agent agents/simple/roi_baseline.py

The probe writes audit/lookahead/<utc>.json with one row per
(seed, step, pov, horizon) plus the AUC summary table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kaggle_environments import make

from lib.intent import World
from lib.world_model import WorldModel

DEFAULT_HORIZONS = [50, 100, 200, 300, 500]
DEFAULT_PROBE_STEPS = [25, 50, 75, 100, 150, 200, 300]
DEFAULT_SEEDS = [42, 1, 7, 13, 31, 100, 17, 23, 53, 71, 91, 113, 137, 149, 167, 181]


def _load_agent(path: str):
    spec = importlib.util.spec_from_file_location(f"_probe_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _current_ship_total_by_player(world: World) -> dict[int, float]:
    """Ships on owned planets + ships in fleets per owner. Snapshot, not
    timeline-projected."""
    totals: dict[int, float] = {}
    for p in world.planets_by_id.values():
        if p.owner >= 0:
            totals[p.owner] = totals.get(p.owner, 0.0) + float(p.ships)
    # Fleets — sum from raw obs (World doesn't materialise them).
    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict)
        else getattr(raw, "fleets", [])
    )
    for f in fleets_raw:
        # Tuple shape: (id, owner, x, y, angle, from_planet_id, ships)
        owner = int(f[1])
        ships = float(f[6])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + ships
    return totals


def _predicted_ship_total_by_player(
    world: World, model: WorldModel, horizon: int
) -> dict[int, float]:
    """At step_now + horizon, sum predicted ships on each owner's planets.

    Reads from the per-planet timelines built by WorldModel. Ignores
    in-flight fleets at the horizon (they're not captured ships yet).
    """
    totals: dict[int, float] = {}
    H = min(horizon, model.horizon)
    for pid, tl in model.timelines.items():
        owner = tl["owner_at"][H]
        ships = tl["ships_at"][H]
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(ships)
    return totals


def _winner_label_from_rewards(rewards: list[float], pov_player: int) -> int | None:
    """Map env final rewards to {+1, -1, None}.

    None = tie (multiple players at max reward); we drop those samples
    to avoid contaminating the AUC.
    """
    if not rewards or any(r is None for r in rewards):
        return None
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r]
    if len(winners) != 1:
        return None
    return 1 if winners[0] == pov_player else -1


def _auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U normalised → AUC. labels ∈ {+1, -1}."""
    pos = [s for s, lab in zip(scores, labels) if lab == 1]
    neg = [s for s, lab in zip(scores, labels) if lab == -1]
    if not pos or not neg:
        return float("nan")
    # Wilcoxon rank-sum equivalent via pairwise comparison; O(|pos|·|neg|).
    wins = 0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / total if total else float("nan")


def run_probe(
    p0_agent: str = str(REPO / "agents" / "v2" / "main.py"),
    p1_agent: str = str(REPO / "agents" / "simple" / "roi_baseline.py"),
    seeds: list[int] = DEFAULT_SEEDS,
    probe_steps: list[int] = DEFAULT_PROBE_STEPS,
    horizons: list[int] = DEFAULT_HORIZONS,
) -> dict:
    p0_fn = _load_agent(p0_agent)
    p1_fn = _load_agent(p1_agent)
    samples: list[dict] = []
    horizon_build_times: list[float] = []
    n_decisive = 0

    for seed in seeds:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([p0_fn, p1_fn])
        final = env.steps[-1]
        final_rewards = [s.reward for s in final]
        n_steps = len(env.steps)
        # Probe only from P0's POV — that's the "us" we'd be making decisions for.
        pov = 0
        label = _winner_label_from_rewards(final_rewards, pov)
        if label is None:
            continue
        n_decisive += 1
        for step_now in probe_steps:
            if step_now >= n_steps - 1:
                continue
            state = env.steps[step_now]
            obs = state[pov].observation
            world = World.from_obs(obs)
            if not world.planets_by_id:
                continue
            wm_horizon = max(horizons)
            t0 = time.perf_counter()
            wm = WorldModel.from_world(world, horizon=wm_horizon)
            horizon_build_times.append((time.perf_counter() - t0) * 1000.0)

            now_totals = _current_ship_total_by_player(world)
            naive = (now_totals.get(pov, 0.0)
                     - now_totals.get(1 - pov, 0.0))
            horizon_scores = {}
            oracle_scores = {}
            for H in horizons:
                pred = _predicted_ship_total_by_player(world, wm, H)
                horizon_scores[H] = (
                    pred.get(pov, 0.0) - pred.get(1 - pov, 0.0)
                )
                # PERFECT ORACLE: read the actual ship delta at step+H from
                # the replay. Upper bound on what any single-scalar predictor
                # of "delta at H" could achieve; anything below this is
                # information left on the table by the static projector.
                future_idx = min(step_now + H, n_steps - 1)
                future_state = env.steps[future_idx]
                future_obs = future_state[pov].observation
                future_world = World.from_obs(future_obs)
                future_totals = _current_ship_total_by_player(future_world)
                oracle_scores[H] = (
                    future_totals.get(pov, 0.0)
                    - future_totals.get(1 - pov, 0.0)
                )

            samples.append({
                "seed": seed,
                "step_now": step_now,
                "pov": pov,
                "naive_delta": naive,
                **{f"horizon_{H}_delta": horizon_scores[H] for H in horizons},
                **{f"oracle_{H}_delta": oracle_scores[H] for H in horizons},
                "actual_winner_pov_label": label,
            })

    aucs_by_step: dict[int, dict[str, float]] = {}
    for step_now in probe_steps:
        rows = [s for s in samples if s["step_now"] == step_now]
        if not rows:
            continue
        labels = [r["actual_winner_pov_label"] for r in rows]
        row = {"naive": _auc([r["naive_delta"] for r in rows], labels)}
        for H in horizons:
            row[f"H{H}"] = _auc([r[f"horizon_{H}_delta"] for r in rows], labels)
            row[f"O{H}"] = _auc([r[f"oracle_{H}_delta"] for r in rows], labels)
        aucs_by_step[step_now] = row

    return {
        "p0_agent": p0_agent,
        "p1_agent": p1_agent,
        "seeds": seeds,
        "probe_steps": probe_steps,
        "horizons": horizons,
        "n_decisive_games": n_decisive,
        "n_samples": len(samples),
        "horizon_build_ms_median": (
            sorted(horizon_build_times)[len(horizon_build_times)//2]
            if horizon_build_times else None
        ),
        "horizon_build_ms_max": (
            max(horizon_build_times) if horizon_build_times else None
        ),
        "auc_by_step": aucs_by_step,
        "samples": samples,
    }


def _print_auc_table(aucs_by_step: dict, horizons: list[int]) -> None:
    print("\n=== AUC by (probe step, horizon) — pred. of game winner from POV ===")
    print("  H<n> = WorldModel static projection.  O<n> = perfect oracle (future ship delta read from replay).")
    cols = ["step", "naive"]
    for H in horizons:
        cols += [f"H{H}", f"O{H}"]
    print("  " + "  ".join(f"{c:>7s}" for c in cols))
    for step_now in sorted(aucs_by_step):
        row = aucs_by_step[step_now]
        vals = [f"{step_now:>7d}"] + [
            f"{row.get(c, float('nan')):>7.3f}" for c in cols[1:]
        ]
        print("  " + "  ".join(vals))
    print("\n(AUC 0.5 = no signal; 1.0 = perfect; for each row higher is better)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p0-agent", default=str(REPO / "agents" / "v2" / "main.py"),
        help="P0 (us). Default agents/v2/main.py.",
    )
    parser.add_argument(
        "--p1-agent", default=str(REPO / "agents" / "simple" / "roi_baseline.py"),
        help="P1 (opp). Default agents/simple/roi_baseline.py — pairs that "
             "actually produce decisive games (v2 vs v2 step-500-ties).",
    )
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--steps", default=",".join(map(str, DEFAULT_PROBE_STEPS)),
        help="Comma-separated probe-step indices (mid-game samples).",
    )
    parser.add_argument(
        "--horizons", default=",".join(map(str, DEFAULT_HORIZONS)),
        help="Comma-separated lookahead horizons in steps.",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Skip JSON output. Just print the AUC table.",
    )
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    probe_steps = [int(s) for s in args.steps.split(",") if s.strip()]
    horizons = [int(s) for s in args.horizons.split(",") if s.strip()]

    result = run_probe(
        p0_agent=args.p0_agent, p1_agent=args.p1_agent,
        seeds=seeds, probe_steps=probe_steps, horizons=horizons,
    )

    print(f"P0 (us):  {result['p0_agent']}")
    print(f"P1 (opp): {result['p1_agent']}")
    print(
        f"games: {len(seeds)}, decisive (P0 won or lost cleanly): "
        f"{result['n_decisive_games']}, probe steps: {probe_steps}, "
        f"horizons: {horizons}"
    )
    print(
        f"samples: {result['n_samples']} (decisive games × probe step)"
    )
    print(
        f"WorldModel.from_world build wallclock @ horizon={max(horizons)}: "
        f"median {result['horizon_build_ms_median']:.2f} ms, "
        f"max {result['horizon_build_ms_max']:.2f} ms"
    )
    _print_auc_table(result["auc_by_step"], horizons)

    if not args.no_write:
        out_dir = REPO / "audit" / "lookahead"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
