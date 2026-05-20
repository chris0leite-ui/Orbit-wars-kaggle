"""Phase B foundation diagnostic — per-stage pipeline cardinality + wallclock.

For each turn of a real game, instrument the composed pipeline's
`return_trace=True` mode and record:
  - n_candidates (Stage 2 output)
  - n_columns_before_filter / n_columns_after_filter (Stage 3 output;
    portfolio focus + the implicit value<=0 drop inside Stage 5)
  - n_opp_arrivals (Stage 4 output)
  - n_fired_columns, n_emitted_moves (Stage 5 output)
  - wait_N distribution of fired_columns (where the evaporation lives)
  - per-stage wallclock ms

Outputs JSON to `audit/diagnostics/pipeline-funnel-<UTC>.json` and prints
percentile bands (p50 / p90 / p99) to stdout.

Sizes K_MY / K_OPP for Phase D. Confirms HANDOVER's claim that
"50%+ of firing turns have wait_N>0 columns that never emit."

Usage:
  python -m scripts.diag_pipeline_funnel --seeds 42 7 13 1 --opp baseline
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from lib.pipeline import compose
from lib.pipeline.candidates import candidates_default
from lib.pipeline.commit import commit_stateless
from lib.pipeline.decision import decision_outcome_aware_milp
from lib.pipeline.opening import opening_default
from lib.pipeline.opp_model import opp_greedy_roi
from lib.pipeline.perception import perception_default
from lib.pipeline.prerank import prerank_w1w2_filter


# ---------------------------------------------------------------------------
# Timed stage wrappers
# ---------------------------------------------------------------------------


def _timed(fn, log_key: str, log: dict):
    """Wrap a stage callable to record per-call wallclock into `log`."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.setdefault(log_key, []).append(
                (time.perf_counter() - t0) * 1000.0
            )
    return wrapper


def _make_instrumented_agent(per_turn_log: list[dict]):
    """Wrap the default composition so each turn appends a funnel row."""
    def agent_fn(obs, configuration=None):
        row: dict[str, Any] = {}
        timing: dict[str, list[float]] = {}

        # Wrap stages so we can read counts after each
        composed = compose(
            perception=_timed(perception_default, "ms_perception", timing),
            opening_override=_timed(opening_default, "ms_opening", timing),
            candidates=_timed(candidates_default, "ms_candidates", timing),
            opp_model=_timed(opp_greedy_roi, "ms_opp_model", timing),
            prerank=_timed(prerank_w1w2_filter, "ms_prerank", timing),
            decision=_timed(decision_outcome_aware_milp, "ms_decision", timing),
            commit=_timed(commit_stateless, "ms_commit", timing),
            return_trace=True,
        )

        # Set analytical agent's env-var overrides (mirror agents/analytical/main.py).
        prev_drain = os.environ.get("PROPOSER_DRAIN_FILTER")
        prev_hold = os.environ.get("PROPOSER_HOLD_FEASIBILITY")
        os.environ["PROPOSER_DRAIN_FILTER"] = "off"
        os.environ["PROPOSER_HOLD_FEASIBILITY"] = "off"
        try:
            moves, trace = composed(obs, configuration)
        finally:
            if prev_drain is None:
                os.environ.pop("PROPOSER_DRAIN_FILTER", None)
            else:
                os.environ["PROPOSER_DRAIN_FILTER"] = prev_drain
            if prev_hold is None:
                os.environ.pop("PROPOSER_HOLD_FEASIBILITY", None)
            else:
                os.environ["PROPOSER_HOLD_FEASIBILITY"] = prev_hold

        ctx = trace.get("ctx")
        row["step"] = int(getattr(ctx, "step_now", -1)) if ctx else -1
        row["me"] = int(getattr(ctx, "me", -1)) if ctx else -1
        row["num_seats"] = int(getattr(ctx, "num_seats", -1)) if ctx else -1
        row["short_circuit"] = trace.get("short_circuit")
        opening = trace.get("opening")
        row["opening_committed"] = (
            opening is not None and opening.committed is not None
        )
        if not row["opening_committed"] and not row["short_circuit"]:
            cset = trace.get("candidates")
            opp = trace.get("opp")
            cols = trace.get("cols")
            decision = trace.get("decision")
            row["n_candidates"] = len(cset.prerank) if cset else 0
            row["n_opp_arrivals"] = len(opp.opp_arrivals) if opp else 0
            row["n_columns_before_filter"] = (
                cols.n_before_filter if cols else 0
            )
            row["n_columns_after_filter"] = (
                cols.n_after_filter if cols else 0
            )
            row["portfolio_size"] = len(cols.portfolio) if cols else 0
            row["portfolio_filtered"] = bool(cols.portfolio_filtered) if cols else False
            row["is_winning_state"] = bool(cols.is_winning_state) if cols else False
            row["n_fired_columns"] = (
                len(decision.fired_columns) if decision else 0
            )
            row["n_emitted_moves"] = len(decision.moves) if decision else 0
            # wait_N distribution of fired columns
            wait_dist: dict[int, int] = {}
            if decision:
                for col in decision.fired_columns:
                    w = int(getattr(col, "wait_N", 0))
                    wait_dist[w] = wait_dist.get(w, 0) + 1
            row["fired_wait_dist"] = wait_dist
            row["objective"] = (
                float(decision.objective) if decision else 0.0
            )
            row["solver_status"] = (
                str(decision.status) if decision else ""
            )

        # Timing
        for key, vals in timing.items():
            row[key] = sum(vals)
        row["ms_total"] = sum(sum(v) for v in timing.values())

        per_turn_log.append(row)
        return moves

    return agent_fn


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "n": 0}
    xs_sorted = sorted(xs)
    def pct(p):
        if len(xs_sorted) == 1:
            return xs_sorted[0]
        i = max(0, min(len(xs_sorted) - 1, int(round(p / 100 * (len(xs_sorted) - 1)))))
        return xs_sorted[i]
    return {
        "p50": pct(50), "p90": pct(90), "p99": pct(99),
        "max": xs_sorted[-1], "n": len(xs_sorted),
    }


def run_one_game(seed: int, opp_path: str) -> dict:
    """Run one game; return funnel log + per-game summary."""
    from kaggle_environments import make
    per_turn_log: list[dict] = []
    instr = _make_instrumented_agent(per_turn_log)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run([instr, opp_path])
    except Exception as e:
        return {
            "seed": seed, "opp": opp_path, "error": str(e),
            "per_turn": per_turn_log,
        }

    # Summary
    active_turns = [
        r for r in per_turn_log
        if not r.get("short_circuit") and not r.get("opening_committed")
    ]
    wait_n_pos_turns = sum(
        1 for r in active_turns
        if any(w > 0 for w in r.get("fired_wait_dist", {}))
    )
    fired_turns = sum(1 for r in active_turns if r.get("n_fired_columns", 0) > 0)

    cardinality_keys = [
        "n_candidates", "n_columns_before_filter", "n_columns_after_filter",
        "n_opp_arrivals", "n_fired_columns", "n_emitted_moves",
    ]
    cardinality_stats = {
        key: _percentiles([r[key] for r in active_turns if key in r])
        for key in cardinality_keys
    }

    timing_keys = [
        "ms_perception", "ms_opening", "ms_candidates", "ms_opp_model",
        "ms_prerank", "ms_decision", "ms_commit", "ms_total",
    ]
    timing_stats = {
        key: _percentiles([r[key] for r in per_turn_log if key in r])
        for key in timing_keys
    }

    return {
        "seed": seed,
        "opp": opp_path,
        "n_turns": len(per_turn_log),
        "n_active_turns": len(active_turns),
        "n_opening_turns": sum(1 for r in per_turn_log if r.get("opening_committed")),
        "n_short_circuit_turns": sum(1 for r in per_turn_log if r.get("short_circuit")),
        "n_fired_turns": fired_turns,
        "n_wait_n_pos_turns": wait_n_pos_turns,
        "wait_n_pos_frac_of_fired": (
            wait_n_pos_turns / fired_turns if fired_turns else 0.0
        ),
        "cardinality": cardinality_stats,
        "timing_ms": timing_stats,
        "per_turn": per_turn_log,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 1])
    ap.add_argument("--opp", default="agents/baseline/main.py")
    ap.add_argument("--out", default=None,
                    help="Output JSON path; defaults to "
                         "audit/diagnostics/pipeline-funnel-<utc>.json")
    args = ap.parse_args()

    games = []
    for seed in args.seeds:
        print(f"=== seed={seed} ===")
        result = run_one_game(seed, args.opp)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  turns={result['n_turns']} active={result['n_active_turns']} "
                  f"fired={result['n_fired_turns']} "
                  f"wait_n_pos_frac={result['wait_n_pos_frac_of_fired']:.2f}")
            for key in ("n_fired_columns", "n_emitted_moves", "n_columns_after_filter"):
                stats = result["cardinality"][key]
                print(f"  {key:30s} p50={stats['p50']:5.1f} p90={stats['p90']:5.1f} max={stats['max']:5.1f}")
            ms = result["timing_ms"]["ms_decision"]
            print(f"  ms_decision                   p50={ms['p50']:6.1f} p90={ms['p90']:6.1f} max={ms['max']:6.1f}")
            ms_total = result["timing_ms"]["ms_total"]
            print(f"  ms_total                      p50={ms_total['p50']:6.1f} p90={ms_total['p90']:6.1f} max={ms_total['max']:6.1f}")
        games.append(result)

    out_path = args.out
    if out_path is None:
        out_dir = Path("audit/diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pipeline-funnel-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    Path(out_path).write_text(json.dumps(
        {"games": games}, indent=2, default=str
    ))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
