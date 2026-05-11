"""4P FFA panel — compare multiple focal agents against a fixed background.

Sister script to `scripts/strategy_panel.py` (2P round-robin). For each
agent in `--focals`, run `run_ffa_tournament` against the same
`--background` triple, then report a calibration ladder of
first-place rates with Wilson 95% CIs.

Why a fixed background: comparing focal=A vs focal=B is only fair if the
3 background opponents are the SAME — otherwise we're confounding
opponent strength with focal-vs-background mix.

CLI:
    python -m scripts.ffa_panel \
        --focals agents/v3_snipe/main.py agents/v2/main.py roi_baseline \
        --background weakest enemy_first baseline \
        --seeds 32

Output:
    `audit/tournaments/ffa-panel-<utc>.json` plus a console calibration table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Stable name for the multiprocessing.Pool pickle path (mirror of
# strategy_panel's tournament-loading shim).
_spec = importlib.util.spec_from_file_location(
    "ffa_tournament", REPO / "scripts" / "ffa_tournament.py"
)
ffa_tournament = importlib.util.module_from_spec(_spec)
sys.modules["ffa_tournament"] = ffa_tournament
_spec.loader.exec_module(ffa_tournament)

from scripts._agent_paths import resolve_agent_path  # noqa: E402


# Same 64-seed extended bag as scripts/strategy_panel.py SEEDS_32, so 2P
# and 4P panel results are seed-comparable.
SEEDS_64 = [
    42, 1, 7, 13, 31, 100, 17, 23, 53, 71,
    91, 113, 137, 149, 167, 181, 199, 211, 233, 257,
    269, 281, 293, 307, 311, 313, 317, 331, 337, 347,
    349, 353,
    359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
    479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563,
]

DEFAULT_FOCALS = ["agents/v2/main.py", "roi"]
DEFAULT_BACKGROUND = ["weakest", "enemy_first", "baseline"]


def _focal_label(spec: str) -> str:
    """Shorten `agents/<name>/main.py` → `<name>` for the table; pass other names through."""
    p = Path(spec)
    if p.suffix == ".py" and p.parent.name and p.parent.parent.name == "agents":
        return p.parent.name
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--focals", nargs="+", default=DEFAULT_FOCALS,
        help="Agents to compare. Each runs against the same --background.",
    )
    parser.add_argument(
        "--background", nargs="+", default=DEFAULT_BACKGROUND,
        help="Fixed 3-agent opponent panel (must total players-1).",
    )
    parser.add_argument(
        "--players", type=int, default=4,
        help="Total seats (focal + background). Default 4.",
    )
    parser.add_argument(
        "--seeds", type=int, default=32,
        help="Seeds per focal (from SEEDS_64). Each seed × seat = 1 game.",
    )
    parser.add_argument(
        "--no-rotate-seats", action="store_true",
        help="Disable seat-rotation (focal always seat 0). Cuts games 4x.",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count() or 1,
        help="Parallel game workers. Default = os.cpu_count().",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Skip writing the JSON snapshot to audit/tournaments/.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-game progress.",
    )
    args = parser.parse_args(argv)

    if len(args.background) != args.players - 1:
        parser.error(
            f"--background must have exactly --players - 1 = "
            f"{args.players - 1} entries; got {len(args.background)}"
        )

    seeds = SEEDS_64[: args.seeds]
    bg_specs = [resolve_agent_path(name) for name in args.background]
    bg_labels = list(args.background)

    rotate = not args.no_rotate_seats
    games_per_focal = len(seeds) * (args.players if rotate else 1)
    print(
        f"--- ffa_panel: focals={len(args.focals)} background={bg_labels} "
        f"seeds={len(seeds)} rotate_seats={rotate} workers={args.workers} "
        f"→ {games_per_focal} games × {len(args.focals)} focals "
        f"= {games_per_focal * len(args.focals)} games total"
    )

    results = []
    for focal_name in args.focals:
        focal_label = _focal_label(focal_name)
        focal_path = resolve_agent_path(focal_name)
        print(f"\n[{focal_label}] running…")
        res = ffa_tournament.run_ffa_tournament(
            focal=focal_path,
            background=bg_specs,
            focal_name=focal_label,
            background_names=bg_labels,
            seeds=seeds,
            players=args.players,
            rotate_seats=rotate,
            workers=args.workers,
            progress=(not args.quiet),
        )
        lo, hi = res.wilson_ci()
        results.append({
            "focal": focal_label,
            "focal_spec": focal_name,
            "background": bg_labels,
            "n_games": res.n_games,
            "first_place_count": res.first_place_count,
            "first_place_rate": res.first_place_rate,
            "wilson_lo_95": lo,
            "wilson_hi_95": hi,
            "p95_focal_turn_ms": res.p95_focal_turn_ms(),
        })

    print("\n=== ffa_panel calibration ladder (vs fixed background) ===")
    bg_str = "{" + ", ".join(bg_labels) + "}"
    print(f"background: {bg_str}")
    print(f"{'focal':24s}  {'1st-place':>12s}  {'Wilson 95':>16s}  {'p95 ms':>8s}")
    print("-" * 72)
    for r in sorted(results, key=lambda r: r["first_place_rate"], reverse=True):
        ci = f"[{r['wilson_lo_95']*100:4.1f},{r['wilson_hi_95']*100:5.1f}]"
        print(
            f"{r['focal']:24s}  "
            f"{r['first_place_count']:3d}/{r['n_games']:<3d} "
            f"({r['first_place_rate']*100:4.1f}%)  "
            f"{ci:>16s}  {r['p95_focal_turn_ms']:6.1f}"
        )

    if not args.no_write:
        out_dir = REPO / "audit" / "tournaments"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"ffa-panel-{stamp}.json"
        out_path.write_text(json.dumps({
            "generated_utc": stamp,
            "focals": args.focals,
            "background": bg_labels,
            "players": args.players,
            "seeds": seeds,
            "rotate_seats": rotate,
            "results": results,
        }, indent=2) + "\n")
        print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
