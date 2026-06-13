"""4P play CLI — thin wrapper around scripts.ffa_tournament.

Examples:

    # Single seed, rotate seats (4 games per seed)
    python scripts/play4p.py --focal v13 --bg v7_0,v7_0,v7_0 \\
        --seeds 76670184 --rotate-seats

    # Small panel, 4 workers
    python scripts/play4p.py --focal v13 --bg v7_0,v7_0,v7_0 \\
        --seeds 76670184,42,7,1492346051,768065184 --rotate-seats \\
        --workers 4

Outputs per-game lines and a summary block matching `fast.py play`
style (focal turn-ms p50/p95/max, first-place count, Wilson 95% CI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._agent_paths import resolve_agent_path  # noqa: E402
from scripts.ffa_tournament import run_ffa_tournament  # noqa: E402

# Mirror of fast.py:_BASELINES (short-name -> bundled .py).
# Kept inline to avoid dynamic-import-fast.py-as-module hazards
# (dataclasses in fast.py break under importlib.util.module_from_spec).
_BASELINES = {
    "v7_0":       str(REPO / "submissions" / "v7_0_drop_one.py"),
    "v7_1":       str(REPO / "submissions" / "v7_1_open_drop_comets.py"),
    "v4_planner": str(REPO / "submissions" / "v4_planner.py"),
    "v7_minimax": str(REPO / "submissions" / "v7_minimax.py"),
    "v3.5.1":     str(REPO / "submissions" / "v3.5.1.py"),
    "nearest":    str(REPO / "agents" / "simple" / "nearest.py"),
    "roi":        str(REPO / "agents" / "simple" / "roi.py"),
    # Vendored public ProducerLite variants (eval-only panel opponents).
    "producer":   str(REPO / "agents" / "producer" / "producer_agent.py"),
    "panel_smarter": str(REPO / "agents" / "panel_smarter" / "agent_entry.py"),
    "panel_veto":   str(REPO / "agents" / "panel_veto" / "agent_entry.py"),
}


def _resolve(name: str) -> str:
    """Resolve agent name to path.

    Order: bundled-submission short-name -> agents/<name>/main.py ->
    `scripts/_agent_paths.resolve_agent_path` (catches v7_ablations
    / simple flat-file / literal paths).
    """
    if name in _BASELINES:
        return _BASELINES[name]
    dir_main = REPO / "agents" / name / "main.py"
    if dir_main.is_file():
        return str(dir_main)
    return resolve_agent_path(name)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _summarize(turn_ms: list[float]) -> tuple[float, float, float]:
    if not turn_ms:
        return (0.0, 0.0, 0.0)
    sorted_ms = sorted(turn_ms)
    p50 = float(median(sorted_ms))
    n = len(sorted_ms)
    idx95 = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    p95 = float(sorted_ms[idx95])
    return (p50, p95, float(sorted_ms[-1]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--focal", required=True,
                   help="focal agent name (e.g. v13) or path")
    p.add_argument("--bg", required=True,
                   help="3 comma-separated background agents (e.g. v7_0,v7_0,v7_0)")
    p.add_argument("--seeds", required=True,
                   help="comma-separated seed ints")
    p.add_argument("--rotate-seats", action="store_true", default=True,
                   help="rotate focal through each seat 0..3 (default)")
    p.add_argument("--no-rotate-seats", dest="rotate_seats",
                   action="store_false",
                   help="run focal only in seat 0 (faster but seat-biased)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel game workers (default 1)")
    args = p.parse_args(argv)

    bg_names = _parse_csv(args.bg)
    if len(bg_names) != 3:
        p.error(f"--bg must have exactly 3 agents (got {len(bg_names)})")
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        p.error("--seeds must list at least one int")

    focal_path = _resolve(args.focal)
    bg_paths = [_resolve(n) for n in bg_names]

    print(f"  focal: {args.focal} ({focal_path})")
    print(f"  bg:    {bg_names}")
    print(f"  seeds: {seeds}  rotate_seats={args.rotate_seats}  "
          f"workers={args.workers}")
    print()

    res = run_ffa_tournament(
        focal=focal_path,
        background=bg_paths,
        focal_name=args.focal,
        background_names=bg_names,
        seeds=seeds,
        rotate_seats=args.rotate_seats,
        workers=args.workers,
        progress=False,
        out_dir=None,
    )

    print("  per-game results:")
    for g in res.games:
        focal_reward = g.rewards[g.focal_seat]
        win = "WIN " if g.focal_first_place() else "loss"
        p50, p95, mx = _summarize(g.focal_turn_ms)
        print(f"    seed={g.seed:<12} focal_seat={g.focal_seat}  "
              f"{win}  reward={focal_reward}  n_steps={g.n_steps}  "
              f"focal turn-ms p50={p50:.0f} p95={p95:.0f} max={mx:.0f}")

    lo, hi = res.wilson_ci()
    p95_all = res.p95_focal_turn_ms()
    all_max = max((max(g.focal_turn_ms) if g.focal_turn_ms else 0.0)
                  for g in res.games)
    over_1000 = sum(1 for g in res.games for t in g.focal_turn_ms
                    if t >= 1000.0)

    print()
    print(f"  SUMMARY: {res.first_place_count}/{res.n_games} first-place "
          f"({res.first_place_rate:.1%})  Wilson95=[{lo:.1%}, {hi:.1%}]")
    print(f"  focal turn-ms p95={p95_all:.0f}  max={all_max:.0f}  "
          f"games_with_turn_over_1000ms={over_1000}")

    verdict_lines = []
    verdict_lines.append(
        "PASS-floor" if res.first_place_rate >= 0.25 else "FAIL-floor")
    verdict_lines.append(
        "PASS-target" if res.first_place_rate >= 0.35 else "BELOW-target")
    verdict_lines.append(
        "PASS-wallclock" if (p95_all < 800.0 and all_max < 1000.0)
        else "FAIL-wallclock")
    print(f"  verdict: {' / '.join(verdict_lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
