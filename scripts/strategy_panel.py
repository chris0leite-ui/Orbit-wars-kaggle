"""Strategy-panel tournament — measure target-selection ablations head-to-head.

Plan:  /root/.claude/plans/read-the-handover-next-imperative-whisper.md
Sister scripts:
    scripts/tournament.py — primitive: run_tournament(...) over agents × seeds
    scripts/eval_v1.py    — v1-vs-baseline gate (legacy, kept for parity)
    scripts/ablation.py   — strategy-fixed × mechanism-ablation
    scripts/strategy_panel.py (this file) — mechanism-fixed × strategy-ablation

What this adds on top of `run_tournament`:
- Multi-strategy default panel (the five simple strategies + baseline + v1).
- Aggregated *both-sides* winrate table:
    aggregated[a][b] = (P0_wins_when_a_is_P0[a][b] + P1_wins_when_a_is_P1[b][a])
                     / (n[a][b] + n[b][a])
  i.e. the marginal winrate of `a` against `b` regardless of seat. The raw
  per-seat numbers are still in the persisted JSON via `tournament.run_tournament`.
- Per-strategy calibration row (mean panel winrate excluding self, p95 turn ms).

Quick iter:
    python -m scripts.strategy_panel --seeds 8        (~5 min CPU)

Confidence:
    python -m scripts.strategy_panel --seeds 32       (~20-30 min CPU)

Filter the panel:
    python -m scripts.strategy_panel --strategies nearest production roi --seeds 8

The JSON snapshot is written under audit/tournaments/ alongside every other
tournament artifact, so the existing audit/tournaments/INDEX patterns hold.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# Load the tournament module under the stable top-level name `tournament` so
# its dataclasses (PairStat, GameRecord) resolve identically across import
# paths — same trick as scripts/eval_v1.py + scripts/ablation.py.
_t_spec = importlib.util.spec_from_file_location(
    "tournament", REPO / "scripts" / "tournament.py"
)
tournament = importlib.util.module_from_spec(_t_spec)
sys.modules["tournament"] = tournament
_t_spec.loader.exec_module(tournament)


# ---------------------------------------------------------------------------
# Shared seed bags. SEEDS_20 mirrors scripts/ablation.py + scripts/eval_v1.py
# so a strategy_panel result is comparable to historical ablation runs.
# ---------------------------------------------------------------------------
SEEDS_32 = [
    42, 1, 7, 13, 31, 100, 17, 23, 53, 71,
    91, 113, 137, 149, 167, 181, 199, 211, 233, 257,
    269, 281, 293, 307, 311, 313, 317, 331, 337, 347,
    349, 353,
    # Extension to 64 seeds for confidence runs (Day-2 A/B testing).
    359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
    479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563,
]

# Default panel — the five simple strategies (target-selection ablations) plus
# the comp-shipped baseline and the live v1.x agent for cross-reference.
DEFAULT_STRATEGIES = ["nearest", "production", "roi", "weakest", "enemy_first"]
DEFAULT_REFS = ["baseline", "v1_orbitfix"]

# Named panel presets — pick with `--panel <name>`. Each preset is the full
# agent list; `--panel hardened` is the post-2026-05-14 pre-submit minimum
# (≥3 opponent classes: v7 search, v3 lookahead, aggressive simple, comp
# reference). Origin: audit/2026-05-14-postmortem-geo-session.md
# ("panel MUST include ≥3 opponent classes") + audit/2026-05-14-loss-mode-mine.md
# (v7_pv's t=100 ship-share gap of 30pp is decided in the opening — any
# pre-submit gate needs an aggressive close-arm opponent on the panel,
# not just v7-family search agents).
PANEL_PRESETS: dict[str, list[str]] = {
    "hardened": ["v7_0_drop_one", "v3.5.1", "roi", "baseline"],
    "default":  DEFAULT_STRATEGIES + DEFAULT_REFS,
}


# Name resolution is shared with scripts/ffa_panel.py via
# scripts/_agent_paths.py so 2P and 4P panels accept the same strategy
# names (e.g. `v2`, `v3_snipe`, `roi`, `baseline`).
from scripts._agent_paths import resolve_agent_path as _resolve_agent_path  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation across both seats
# ---------------------------------------------------------------------------


def aggregate_winrates(result, names: list[str]) -> dict[str, dict[str, dict]]:
    """Square table of `a` vs `b` winrates aggregated across both seats.

    Self-cells (a == b) report the raw self-play P0/P1 split (they have no
    seat-mirror to aggregate against).
    """
    agg: dict[str, dict[str, dict]] = {a: {} for a in names}
    for a in names:
        for b in names:
            if a == b:
                # Self-play cell may be absent if `--no-self-play` was passed.
                stat = result.matrix.get(a, {}).get(b)
                if stat is None:
                    agg[a][b] = {
                        "wins": 0, "losses": 0, "draws": 0, "n": 0,
                        "winrate": 0.0, "self_play": True,
                    }
                else:
                    agg[a][b] = {
                        "wins": stat.p0_wins,
                        "losses": stat.p1_wins,
                        "draws": stat.draws,
                        "n": stat.n,
                        "winrate": stat.p0_winrate if stat.n else 0.0,
                        "self_play": True,
                    }
                continue
            ab = result.matrix[a][b]   # a as P0, b as P1
            ba = result.matrix[b][a]   # b as P0, a as P1
            wins = ab.p0_wins + ba.p1_wins
            losses = ab.p1_wins + ba.p0_wins
            draws = ab.draws + ba.draws
            n = ab.n + ba.n
            agg[a][b] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "n": n,
                "winrate": (wins / n) if n else 0.0,
                "self_play": False,
            }
    return agg


def calibration_rows(result, agg, names: list[str]) -> list[dict]:
    """Per-strategy summary: mean winrate vs panel (excluding self), p95 turn ms."""
    rows: list[dict] = []
    for a in names:
        winrates = [agg[a][b]["winrate"] for b in names if b != a]
        mean_wr = sum(winrates) / len(winrates) if winrates else 0.0
        # p95 turn ms aggregated across all seats this agent played in.
        p95s: list[float] = []
        for b in names:
            ab = result.matrix.get(a, {}).get(b)
            ba = result.matrix.get(b, {}).get(a)
            if ab is not None and ab.p0_p95_turn_ms:
                p95s.append(ab.p0_p95_turn_ms)
            if ba is not None and ba.p1_p95_turn_ms:
                p95s.append(ba.p1_p95_turn_ms)
        p95 = max(p95s) if p95s else 0.0
        rows.append(
            {"name": a, "mean_panel_winrate": mean_wr, "max_p95_turn_ms": p95}
        )
    rows.sort(key=lambda r: r["mean_panel_winrate"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def format_table(agg, names: list[str]) -> str:
    col_w = max(12, max(len(n) for n in names) + 1)
    header = " " * col_w + " | " + " | ".join(n.center(col_w) for n in names)
    sep = "-" * len(header)
    lines = [header, sep]
    for a in names:
        cells = []
        for b in names:
            cell = agg[a][b]
            if cell["self_play"]:
                # Show P0/P1 split for self-play cells; not a single winrate.
                txt = f"sp {cell['wins']}/{cell['losses']}/{cell['draws']}"
            elif cell["n"] == 0:
                txt = "—"
            else:
                txt = f"{cell['winrate']:.0%} ({cell['wins']}/{cell['n']})"
            cells.append(txt.center(col_w))
        lines.append(a.ljust(col_w) + " | " + " | ".join(cells))
    return "\n".join(lines)


def format_calibration(rows: list[dict]) -> str:
    name_w = max(12, max(len(r["name"]) for r in rows) + 1)
    out = [
        f"{'strategy'.ljust(name_w)}  mean_wr  max_p95_ms",
        "-" * (name_w + 22),
    ]
    for r in rows:
        out.append(
            f"{r['name'].ljust(name_w)}  {r['mean_panel_winrate']:.1%}     "
            f"{r['max_p95_turn_ms']:.1f}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=8,
        help="Seed-bag size from SEEDS_32 (default: 8 for quick-iter; "
             "use 32 for confidence runs).",
    )
    parser.add_argument(
        "--strategies", nargs="*", default=None,
        help=f"Override the strategy list (default: {DEFAULT_STRATEGIES + DEFAULT_REFS}).",
    )
    parser.add_argument(
        "--panel", choices=sorted(PANEL_PRESETS),
        help="Named preset (overrides --strategies). 'hardened' = the "
             "post-2026-05-14 pre-submit minimum (v7_0_drop_one + v3.5.1 "
             "+ roi + baseline; ≥3 opponent classes).",
    )
    parser.add_argument(
        "--no-refs", action="store_true",
        help="Skip the comp-shipped baseline + v1_orbitfix reference agents.",
    )
    parser.add_argument(
        "--no-self-play", action="store_true",
        help="Skip a-vs-a self-play cells (faster; loses A.6 sanity check).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-game progress prints.",
    )
    parser.add_argument(
        "--capture-replays", action="store_true",
        help="Persist per-game compact replays under audit/replays/<utc>/. "
             "Required for downstream manifold_check / behavioural fingerprinting.",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count() or 1,
        help="Parallel game workers. Default = os.cpu_count() (use all "
             "cores). Pass --workers 1 for sequential debugging. >1 uses "
             "multiprocessing.Pool; agent specs must be string paths.",
    )
    args = parser.parse_args(argv)

    if args.panel is not None:
        names = list(PANEL_PRESETS[args.panel])
    elif args.strategies is not None:
        names = list(args.strategies)
    else:
        names = list(DEFAULT_STRATEGIES)
        if not args.no_refs:
            names = names + DEFAULT_REFS

    if args.seeds > len(SEEDS_32):
        raise SystemExit(
            f"--seeds {args.seeds} exceeds available {len(SEEDS_32)} seeds; "
            f"extend SEEDS_32 in scripts/strategy_panel.py first."
        )
    seeds = SEEDS_32[: args.seeds]

    agents = {n: _resolve_agent_path(n) for n in names}

    out_dir = REPO / "audit" / "tournaments"
    print(
        f"--- strategy_panel: {len(names)} agents × {len(seeds)} seeds = "
        f"{len(names) * len(names) * len(seeds)} games "
        f"({'no-self-play' if args.no_self_play else 'with self-play'})"
    )
    for n in names:
        print(f"    {n:14s} -> {agents[n]}")

    result = tournament.run_tournament(
        agents=agents,
        seeds=seeds,
        include_self_play=not args.no_self_play,
        out_dir=out_dir,
        progress=not args.quiet,
        capture_replays=args.capture_replays,
        workers=args.workers,
    )

    agg = aggregate_winrates(result, names)
    rows = calibration_rows(result, agg, names)

    print()
    print("=== aggregated winrate (rows = agent A; A's marginal winrate vs each B) ===")
    print(format_table(agg, names))
    print()
    print("=== calibration ladder rows ===")
    print(format_calibration(rows))
    print()
    print(f"JSON: {out_dir}/{result.timestamp_utc.replace(':', '').replace('-', '')}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
