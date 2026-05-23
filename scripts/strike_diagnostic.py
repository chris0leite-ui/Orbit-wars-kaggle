"""Step 3b diagnostic — runs strike-ON games sequentially with per-seed
elect + strike logs, then aggregates by win/loss cohort to surface WHICH
modeling-correctness hypothesis fits the Step-3 negative lift.

The Step-3 ablation A/B (n=16) showed strike-ON 25.0% vs strike-OFF 43.8%
vs `baseline_joint_aggr_consolidated` — a −18.7 pp lift. Three hypotheses
the diagnostic distinguishes between (per plan file):
  (A) high emit-rate but captured planets reclaimed quickly →
      `is_winning_state_if_owned` needs an in-flight-threat term.
  (B) defensive failures during strike turn → defense-only consolidation
      alongside strike (the option ruled out at planning time).
  (C) frequent atomic-drops on budget overflow / physics fail →
      predicate over-counts; fix at source in `evaluate_inflection`.

Per Rule 38/40 the diagnostic precedes any fix attempt.

CLI:
    # Run a 16-seed sweep, log per-seed to <out>/seed{N}_{elect,strike}.jsonl,
    # write a one-row summary per seed in <out>/summary.jsonl:
    python scripts/strike_diagnostic.py \
        --seeds 0..15 \
        --vs baseline_joint_aggr_consolidated \
        --out audit/2026-05-23-strike-loss-diagnostic/

    # Re-read the per-seed logs + summary and print the per-cohort means:
    python scripts/strike_diagnostic.py \
        --analyze audit/2026-05-23-strike-loss-diagnostic/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _parse_seeds(spec: str) -> list[int]:
    """`0..15` or `0,3,7` → list[int]."""
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def _summarise_logs(
    elect_path: Path, strike_path: Path,
) -> dict:
    """Read the per-seed logs and roll up to one-line stats."""
    elect_count = 0
    plan_count = 0
    skip_4p = 0
    elect_predicate_ms: list[float] = []
    if elect_path.is_file():
        with open(elect_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("skipped") == "4p":
                    skip_4p += 1
                    continue
                if "error" in d:
                    continue
                elect_count += 1
                if d.get("plan_found"):
                    plan_count += 1
                ms = d.get("elapsed_ms")
                if isinstance(ms, (int, float)):
                    elect_predicate_ms.append(float(ms))

    drops = defaultdict(int)
    emit_count = 0
    if strike_path.is_file():
        with open(strike_path) as f:
            for line in f:
                d = json.loads(line)
                outcome = d.get("outcome", "?")
                if outcome == "emit":
                    emit_count += 1
                elif outcome in ("budget_overflow", "physics_fail", "empty"):
                    drops[outcome] += 1
                else:
                    drops[f"other:{outcome}"] += 1

    p95 = (
        sorted(elect_predicate_ms)[int(len(elect_predicate_ms) * 0.95)]
        if elect_predicate_ms else 0.0
    )

    return {
        "elect_turns": elect_count,
        "skip_4p_turns": skip_4p,
        "plan_count": plan_count,
        "emit_count": emit_count,
        "drop_budget_overflow": drops.get("budget_overflow", 0),
        "drop_physics_fail": drops.get("physics_fail", 0),
        "drop_empty": drops.get("empty", 0),
        "predicate_ms_p95": p95,
    }


def cmd_run(args: argparse.Namespace) -> int:
    """Sequential per-seed runner. Each game writes its own log files
    keyed by seed; summary.jsonl rolls up one line per seed."""
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Late import — kaggle_environments init is slow; only pay it once.
    from fast import play_one, resolve_agent_spec

    focal_name, focal_path = resolve_agent_spec(args.focal)
    opp_name, opp_path = resolve_agent_spec(args.vs)

    seeds = _parse_seeds(args.seeds)
    summary_path = out_dir / "summary.jsonl"
    # Overwrite summary so re-runs don't compound.
    summary_path.write_text("")

    print(f"[diag] focal={focal_name} opp={opp_name} seeds={seeds} out={out_dir}")
    t0_all = time.perf_counter()

    for seed in seeds:
        elect_path = out_dir / f"seed{seed}_elect.jsonl"
        strike_path = out_dir / f"seed{seed}_strike.jsonl"
        # Clean per-seed logs so re-runs don't compound.
        elect_path.unlink(missing_ok=True)
        strike_path.unlink(missing_ok=True)

        # Strike enabled + both logs point at this seed's files.
        os.environ["BUILDUP_PLANNER_STRIKE_ENABLED"] = "1"
        os.environ["BUILDUP_PLANNER_ELECT_LOG"] = str(elect_path)
        os.environ["BUILDUP_PLANNER_STRIKE_LOG"] = str(strike_path)

        t0 = time.perf_counter()
        # Run BOTH seat orderings to mirror the A/B's full-panel behaviour
        # (focal as p0 then as p1). Each side contributes one entry.
        for swap in (False, True):
            p0, p1 = (
                (focal_path, opp_path) if not swap else (opp_path, focal_path)
            )
            focal_seat = 0 if not swap else 1
            res = play_one(seed, p0, p1)
            focal_rew = res.rewards[focal_seat]
            focal_won = (focal_rew is not None and focal_rew > 0)
            t1 = time.perf_counter()
            focal_times = (
                res.p0_turn_ms if focal_seat == 0 else res.p1_turn_ms
            )
            focal_p95 = (
                sorted(focal_times)[int(len(focal_times) * 0.95)]
                if focal_times else 0.0
            )

            roll = _summarise_logs(elect_path, strike_path)

            row = {
                "seed": seed,
                "focal_seat": focal_seat,
                "focal_won": focal_won,
                "outcome": res.outcome,
                "n_steps": res.n_steps,
                "focal_turn_ms_p95": focal_p95,
                "wallclock_s": round(t1 - t0, 1),
                **roll,
            }
            with open(summary_path, "a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(
                f"[diag] seed={seed} seat={focal_seat} won={focal_won} "
                f"steps={res.n_steps} elect={roll['elect_turns']} "
                f"plans={roll['plan_count']} emit={roll['emit_count']} "
                f"drop_bo={roll['drop_budget_overflow']} "
                f"drop_pf={roll['drop_physics_fail']} "
                f"p95={focal_p95:.0f}ms  ({t1-t0:.0f}s)"
            )
            t0 = time.perf_counter()

    print(f"[diag] all seeds done in {time.perf_counter()-t0_all:.0f}s -> {summary_path}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Read summary.jsonl, print per-cohort means, surface the
    hypothesis that fits."""
    out_dir = Path(args.analyze).resolve()
    summary_path = out_dir / "summary.jsonl"
    if not summary_path.is_file():
        print(f"[diag] no summary at {summary_path}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with open(summary_path) as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        print("[diag] summary is empty", file=sys.stderr)
        return 1

    n_total = len(rows)
    n_won = sum(1 for r in rows if r["focal_won"])
    n_lost = n_total - n_won
    winrate = n_won / n_total

    def _cohort_mean(rows: list[dict], key: str) -> float:
        if not rows:
            return 0.0
        return sum(r.get(key, 0) for r in rows) / len(rows)

    won = [r for r in rows if r["focal_won"]]
    lost = [r for r in rows if not r["focal_won"]]

    metrics = [
        "elect_turns", "plan_count", "emit_count",
        "drop_budget_overflow", "drop_physics_fail", "drop_empty",
        "n_steps", "focal_turn_ms_p95", "predicate_ms_p95",
    ]

    print(f"=== Step 3b diagnostic — {n_total} trials ({n_won} won, {n_lost} lost; winrate {winrate:.1%}) ===\n")
    print(f"{'metric':<28}{'won (mean)':>14}{'lost (mean)':>14}{'lost - won':>14}")
    print("-" * 70)
    for m in metrics:
        wmean = _cohort_mean(won, m)
        lmean = _cohort_mean(lost, m)
        delta = lmean - wmean
        print(f"{m:<28}{wmean:>14.2f}{lmean:>14.2f}{delta:>+14.2f}")
    print()

    # Hypothesis fits — pick the dominant one by inspecting the deltas.
    # (Heuristic only; the eyeball-the-numbers is the real diagnostic.)
    total_plans = sum(r["plan_count"] for r in rows)
    total_emits = sum(r["emit_count"] for r in rows)
    total_drops_bo = sum(r["drop_budget_overflow"] for r in rows)
    total_drops_pf = sum(r["drop_physics_fail"] for r in rows)
    emit_rate = (total_emits / total_plans) if total_plans else 0.0
    drop_bo_rate = (total_drops_bo / total_plans) if total_plans else 0.0
    drop_pf_rate = (total_drops_pf / total_plans) if total_plans else 0.0

    print(f"Pooled rates over {total_plans} plans:")
    print(f"  emit_rate:       {emit_rate:.1%}  (plans that reached the wave)")
    print(f"  drop_budget:     {drop_bo_rate:.1%}  (predicate over-counted ships)")
    print(f"  drop_physics:    {drop_pf_rate:.1%}  (wave physically impossible)")
    print()

    hypotheses = []
    if (drop_bo_rate + drop_pf_rate) >= 0.30:
        hypotheses.append(
            "(C) Atomic-drops dominate → predicate over-counts / over-promises. "
            "Fix at source in `evaluate_inflection` (per-source budget + per-shot "
            "physics check before recording the plan)."
        )
    if emit_rate >= 0.70 and (_cohort_mean(lost, "emit_count") > _cohort_mean(won, "emit_count")):
        hypotheses.append(
            "(A) Lost games have MORE emits than won games. Plans are landing "
            "but not helping. Closed-form gate `is_winning_state_if_owned` "
            "ignores in-flight enemy fleets / snipe responses. Fix is the "
            "in-flight-threat term in the gate."
        )
    if not hypotheses:
        hypotheses.append(
            "Mixed pattern — none of (A)-(C) clearly dominate. Escalate to PI "
            "with this output."
        )

    print("Hypothesis fit:")
    for h in hypotheses:
        print(f"  - {h}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="strike_diagnostic")
    ap.add_argument(
        "--seeds",
        help="seed range like '0..15' or comma list '0,3,7'",
    )
    ap.add_argument(
        "--focal", default="agents/buildup_planner",
        help="focal agent spec (default: agents/buildup_planner)",
    )
    ap.add_argument(
        "--vs", default="baseline_joint_aggr_consolidated",
        help="opp spec (default: baseline_joint_aggr_consolidated)",
    )
    ap.add_argument("--out", help="output dir for per-seed logs + summary")
    ap.add_argument(
        "--analyze",
        help="read summary.jsonl from this dir and print per-cohort means",
    )
    args = ap.parse_args(argv)

    if args.analyze:
        return cmd_analyze(args)
    if args.seeds and args.out:
        return cmd_run(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
