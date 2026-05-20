"""Local A/B driver for ledger validation — Phases C, D, E of the plan.

Wraps `scripts.tournament.run_tournament` with consistent reporting:
- Aggregates wins across BOTH seat assignments.
- Reports Wilson-LB 95%.
- Reports p95/max per-turn timing.
- Skips self-play.

Not committed as a stable utility — exists only for this work-cycle.

Usage:
    python -m scripts._ledger_ab_driver h2h --n 8
    python -m scripts._ledger_ab_driver panel --target v7_0_drop_one --n 32
    python -m scripts._ledger_ab_driver regression --seeds 1492346051 1844543828

Outputs to audit/ledger-validation/<phase>-<n>-<utc>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.tournament import run_tournament, _wilson_ci  # noqa: E402
from scripts.ffa_panel import SEEDS_64  # noqa: E402


# Resolve panel-target short names to file paths. Mirrors the panel
# alias map in `scripts.play4p` and `scripts.ffa_panel`.
PANEL_TARGETS = {
    "v7_0":         "submissions/v7_0_drop_one.py",
    "v4_planner":   "submissions/v4_planner.py",
    "v3.5.1":       "submissions/v3.5.1.py",
}
LEDGER_ON = "agents/_ledger_on/main.py"
LEDGER_OFF = "agents/_ledger_off/main.py"


def aggregate(res, variant_name: str) -> dict:
    """Aggregate wins for `variant_name` across both seat assignments."""
    wins = 0
    total = 0
    draws = 0
    p95_us_max = 0.0
    p95_them_max = 0.0
    per_direction = []
    for a, sub in res.matrix.items():
        for b, ps in sub.items():
            if a == variant_name:
                wins += ps.p0_wins
                p95_us_max = max(p95_us_max, ps.p0_p95_turn_ms)
                p95_them_max = max(p95_them_max, ps.p1_p95_turn_ms)
                per_direction.append({
                    "seat": "P0", "variant": a, "opp": b,
                    "wins": ps.p0_wins, "losses": ps.p1_wins,
                    "draws": ps.draws, "n": ps.n,
                    "us_p95_ms": ps.p0_p95_turn_ms,
                    "opp_p95_ms": ps.p1_p95_turn_ms,
                })
            elif b == variant_name:
                wins += ps.p1_wins
                p95_us_max = max(p95_us_max, ps.p1_p95_turn_ms)
                p95_them_max = max(p95_them_max, ps.p0_p95_turn_ms)
                per_direction.append({
                    "seat": "P1", "variant": b, "opp": a,
                    "wins": ps.p1_wins, "losses": ps.p0_wins,
                    "draws": ps.draws, "n": ps.n,
                    "us_p95_ms": ps.p1_p95_turn_ms,
                    "opp_p95_ms": ps.p0_p95_turn_ms,
                })
            total += ps.n
            draws += ps.draws
    losses = total - wins - draws
    lo, hi = _wilson_ci(wins, total) if total else (0.0, 0.0)
    return {
        "wins": wins, "losses": losses, "draws": draws, "n": total,
        "winrate": wins / total if total else 0.0,
        "wilson_lo": lo, "wilson_hi": hi,
        "p95_us_max_ms": p95_us_max,
        "p95_opp_max_ms": p95_them_max,
        "per_direction": per_direction,
    }


def cmd_h2h(args) -> int:
    """ledger_on vs ledger_off at n seeds."""
    seeds = SEEDS_64[:args.n]
    agents = {"led_on": LEDGER_ON, "led_off": LEDGER_OFF}
    t0 = time.perf_counter()
    res = run_tournament(agents, seeds, include_self_play=False,
                         workers=args.workers)
    dt = time.perf_counter() - t0
    summary = aggregate(res, "led_on")
    summary["phase"] = "h2h"
    summary["n_seeds"] = args.n
    summary["elapsed_s"] = round(dt, 1)
    print(f"\n=== h2h led_on vs led_off, n={args.n} seeds, {dt:.0f}s ===")
    print(f"  led_on wins: {summary['wins']}/{summary['n']} "
          f"({100*summary['winrate']:.1f}%, "
          f"draws={summary['draws']})")
    print(f"  Wilson 95% CI: [{summary['wilson_lo']:.3f}, "
          f"{summary['wilson_hi']:.3f}]")
    print(f"  p95 turn ms: led_on={summary['p95_us_max_ms']:.0f}  "
          f"led_off={summary['p95_opp_max_ms']:.0f}")
    _write_summary("h2h", summary, args)
    return 0


def cmd_panel(args) -> int:
    """ledger_on vs a panel target."""
    if args.target not in PANEL_TARGETS:
        print(f"Unknown target: {args.target}. Known: {list(PANEL_TARGETS)}")
        return 1
    target_path = PANEL_TARGETS[args.target]
    seeds = SEEDS_64[:args.n]
    agents = {"led_on": LEDGER_ON, args.target: target_path}
    t0 = time.perf_counter()
    res = run_tournament(agents, seeds, include_self_play=False,
                         workers=args.workers)
    dt = time.perf_counter() - t0
    summary = aggregate(res, "led_on")
    summary["phase"] = "panel"
    summary["target"] = args.target
    summary["n_seeds"] = args.n
    summary["elapsed_s"] = round(dt, 1)
    print(f"\n=== panel led_on vs {args.target}, n={args.n}, {dt:.0f}s ===")
    print(f"  led_on wins: {summary['wins']}/{summary['n']} "
          f"({100*summary['winrate']:.1f}%, draws={summary['draws']})")
    print(f"  Wilson 95% CI: [{summary['wilson_lo']:.3f}, "
          f"{summary['wilson_hi']:.3f}]")
    print(f"  p95 turn ms: led_on={summary['p95_us_max_ms']:.0f}  "
          f"opp={summary['p95_opp_max_ms']:.0f}")
    _write_summary(f"panel-{args.target}", summary, args)
    return 0


def cmd_regression(args) -> int:
    """ledger_on vs v7_0 on hard-coded failure seeds."""
    seeds = args.seeds
    if not seeds:
        print("No seeds provided.")
        return 1
    agents = {"led_on": LEDGER_ON, "v7_0": PANEL_TARGETS["v7_0"]}
    t0 = time.perf_counter()
    res = run_tournament(agents, seeds, include_self_play=False,
                         workers=args.workers)
    dt = time.perf_counter() - t0
    summary = aggregate(res, "led_on")
    summary["phase"] = "regression"
    summary["seeds"] = seeds
    summary["elapsed_s"] = round(dt, 1)
    # Also aggregate ledger_off vs v7_0 on the same seeds, for comparison.
    agents_off = {"led_off": LEDGER_OFF, "v7_0": PANEL_TARGETS["v7_0"]}
    t1 = time.perf_counter()
    res_off = run_tournament(agents_off, seeds, include_self_play=False,
                             workers=args.workers)
    dt_off = time.perf_counter() - t1
    summary_off = aggregate(res_off, "led_off")
    summary["led_off_baseline"] = {
        "wins": summary_off["wins"], "n": summary_off["n"],
        "winrate": summary_off["winrate"],
        "wilson_lo": summary_off["wilson_lo"],
    }
    print(f"\n=== regression vs v7_0 on seeds {seeds} ===")
    print(f"  led_on:  {summary['wins']}/{summary['n']} "
          f"({100*summary['winrate']:.1f}%) Wlo={summary['wilson_lo']:.3f}")
    print(f"  led_off: {summary_off['wins']}/{summary_off['n']} "
          f"({100*summary_off['winrate']:.1f}%) Wlo={summary_off['wilson_lo']:.3f}")
    _write_summary("regression", summary, args)
    return 0


def _write_summary(label: str, summary: dict, args) -> None:
    out_dir = REPO / "audit" / "ledger-validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{label}-n{summary.get('n_seeds', 'X')}-{utc}.json"
    (out_dir / fname).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  -> wrote {out_dir / fname}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_h2h = sub.add_parser("h2h")
    p_h2h.add_argument("--n", type=int, default=8)
    p_h2h.add_argument("--workers", type=int, default=4)
    p_h2h.set_defaults(func=cmd_h2h)

    p_pan = sub.add_parser("panel")
    p_pan.add_argument("--target", choices=list(PANEL_TARGETS), required=True)
    p_pan.add_argument("--n", type=int, default=32)
    p_pan.add_argument("--workers", type=int, default=4)
    p_pan.set_defaults(func=cmd_panel)

    p_reg = sub.add_parser("regression")
    p_reg.add_argument("--seeds", type=int, nargs="+", required=True)
    p_reg.add_argument("--workers", type=int, default=4)
    p_reg.set_defaults(func=cmd_regression)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
