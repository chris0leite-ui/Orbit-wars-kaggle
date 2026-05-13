"""A/B/N tournament across snipe-scoring constant variants of v3_snipe.

Each variant is bundled into a self-contained `.py` so module-state
mutations (`AIRTIME_PENALTY_WEIGHT`, `ENDGAME_NEUTRAL_BONUS`, etc.)
don't leak across agents in a single Python process.

Workflow per variant:
1. Patch `lib/missions/snipe.py` in place with the variant's constant
   overrides (regex replace on `NAME = number` lines).
2. Bundle `agents/v3_snipe` → `submissions/_ab/<variant>.py` via
   `scripts.bundle_agent.bundle()`. Parity gate is skipped because we
   bundle from a transiently-edited tree.
3. Restore `lib/missions/snipe.py` from the in-memory snapshot.

After all variants are bundled, run the tournament (`scripts.tournament.
run_tournament`) round-robin without self-play. Report Wilson CIs and
write the JSON snapshot to `audit/tournaments/ab-<utc>.json`.

CLI:
    python -m scripts.ab_variants \\
        --variant v3_5 AIRTIME_PENALTY_WEIGHT=1.0 ENDGAME_NEUTRAL_BONUS=1.5 \\
        --variant v3_4 AIRTIME_PENALTY_WEIGHT=0.0 ENDGAME_NEUTRAL_BONUS=1.0 \\
        --seeds 32 --workers 8

`--seeds` picks the first N from `SEEDS_64` (defined below; same bag as
`scripts/strategy_panel.py` so 2P A/Bs are seed-comparable). `--workers`
fans games out across processes (1 game ≈ 12 s wallclock).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.bundle_agent as bundle_agent  # noqa: E402
import scripts.tournament as tournament  # noqa: E402


# 64-seed bag, same as `scripts.strategy_panel.SEEDS_32` extended.
SEEDS_64 = [
    42, 1, 7, 13, 31, 100, 17, 23, 53, 71,
    89, 97, 113, 137, 149, 167, 181, 197, 211, 233,
    257, 269, 283, 307, 331, 347, 367, 383, 401, 419,
    433, 449, 463, 487, 503, 521, 541, 557, 569, 587,
    599, 613, 631, 647, 661, 677, 691, 709, 733, 751,
    761, 787, 809, 823, 839, 857, 877, 887, 907, 919,
    937, 953, 967, 977,
]


# Lib files we look in for top-level `NAME = NUMBER` constants. Order
# matters only for shadowed names (rare in this codebase); we error if a
# constant is defined in multiple files.
PATCHABLE_PATHS = [
    REPO / "lib" / "missions" / "snipe.py",
    REPO / "lib" / "missions" / "reinforce.py",
    REPO / "lib" / "missions" / "opening.py",
    REPO / "lib" / "missions" / "drain.py",
    REPO / "lib" / "missions" / "gang_up.py",
    REPO / "lib" / "mechanism.py",
    REPO / "lib" / "planner.py",
    REPO / "lib" / "scoring.py",
]
BUNDLE_OUT = REPO / "submissions" / "_ab"


def _constant_pattern(name: str) -> "re.Pattern[str]":
    # Match `NAME = <number>` allowing a trailing inline comment, e.g.
    # `GANG_UP_ENABLED = 0  # opt-in`. The trailing `\s*(?:#.*)?$` group
    # is captured so the substitution can drop the comment cleanly.
    return re.compile(
        rf"^(?P<lead>\s*){re.escape(name)}\s*=\s*[-+0-9.eE]+(?P<tail>\s*(?:#.*)?)$",
        re.MULTILINE,
    )


def _find_constant_path(name: str) -> Path:
    """Locate which lib file defines `name = <number>` at top level.

    Raises ValueError if not found in any path or found in multiple.
    """
    pattern = _constant_pattern(name)
    hits = [p for p in PATCHABLE_PATHS if pattern.search(p.read_text())]
    if not hits:
        raise ValueError(
            f"variant override `{name}` not found as a top-level "
            f"assignment in any of {[str(p) for p in PATCHABLE_PATHS]}"
        )
    if len(hits) > 1:
        raise ValueError(
            f"variant override `{name}` is defined in multiple files: "
            f"{[str(p) for p in hits]}. Disambiguate by removing one."
        )
    return hits[0]


def _patch_one_file(path: Path, overrides: dict[str, float]) -> str:
    """Apply this file's overrides; return original source for restore."""
    original = path.read_text()
    patched = original
    for name, value in overrides.items():
        pattern = _constant_pattern(name)
        patched = pattern.sub(rf"\g<lead>{name} = {value}\g<tail>", patched)
    path.write_text(patched)
    return original


def _bundle_variant(
    name: str,
    overrides: dict[str, float],
    agent_dir: Path = REPO / "agents" / "v3_snipe",
) -> Path:
    """Patch → bundle → restore across whichever lib files own each
    overridden constant. Returns the bundle path. `agent_dir` defaults
    to v3_snipe for backwards compatibility; pass `--agent <path>` to
    A/B variants of any other agent (e.g. v7_0_drop_one for ROI changes
    that need the drop-one rollout to see the new candidate set).
    """
    # Group overrides by source file.
    by_file: dict[Path, dict[str, float]] = {}
    for k, v in overrides.items():
        owner = _find_constant_path(k)
        by_file.setdefault(owner, {})[k] = v
    originals: dict[Path, str] = {}
    try:
        for path, file_overrides in by_file.items():
            originals[path] = _patch_one_file(path, file_overrides)
        BUNDLE_OUT.mkdir(parents=True, exist_ok=True)
        out_path = bundle_agent.bundle(
            agent_dir,
            bundle_agent.DEFAULT_LIB_ORDER,
            out_dir=BUNDLE_OUT,
        )
        # bundle_agent names the file by agent_dir.name. Rename to the
        # variant name so each --variant gets a distinct file.
        renamed = BUNDLE_OUT / f"{name}.py"
        if renamed.exists():
            renamed.unlink()
        out_path.rename(renamed)
        return renamed
    finally:
        for path, original in originals.items():
            path.write_text(original)


def _parse_variant_args(specs: list[list[str]]) -> dict[str, dict[str, float]]:
    """Turn `[['v3_5', 'AIRTIME_PENALTY_WEIGHT=1.0', ...], ...]`
    into `{'v3_5': {'AIRTIME_PENALTY_WEIGHT': 1.0, ...}, ...}`."""
    out: dict[str, dict[str, float]] = {}
    for spec in specs:
        if not spec:
            raise ValueError("empty --variant spec")
        name, *kvs = spec
        overrides: dict[str, float] = {}
        for kv in kvs:
            if "=" not in kv:
                raise ValueError(f"variant {name!r} override missing '=' in {kv!r}")
            k, v = kv.split("=", 1)
            overrides[k.strip()] = float(v.strip())
        out[name] = overrides
    return out


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarise(result: tournament.TournamentResult, names: list[str]) -> dict:
    """Compute total winrate of each variant across both P0+P1 seats."""
    rows: dict[str, dict] = {}
    for a in names:
        wins = 0
        losses = 0
        draws = 0
        for b in names:
            if a == b:
                continue
            # a-as-P0 vs b-as-P1.
            sp = result.matrix[a].get(b)
            if sp is not None:
                wins += sp.p0_wins
                losses += sp.p1_wins
                draws += sp.draws
            # a-as-P1 vs b-as-P0 (i.e., matrix[b][a]).
            sp = result.matrix[b].get(a)
            if sp is not None:
                wins += sp.p1_wins
                losses += sp.p0_wins
                draws += sp.draws
        n = wins + losses + draws
        lo, hi = _wilson_ci(wins, n)
        rows[a] = {
            "wins": wins, "losses": losses, "draws": draws, "n": n,
            "winrate": wins / n if n else 0.0,
            "wilson_lo": lo, "wilson_hi": hi,
        }
    return rows


def _per_anchor_summarise(
    result: tournament.TournamentResult,
    candidate: str,
    anchors: list[str],
) -> dict[str, dict]:
    """Pool-by-seat head-to-head Wilson CI for `candidate` vs each anchor.

    For each anchor, sums candidate-as-P0 + candidate-as-P1 results
    (matrix[candidate][anchor] and matrix[anchor][candidate]). Returns
    one row per anchor with wins/losses/draws/n/winrate/wilson_lo/hi.

    Use this instead of `_summarise` when you want to detect non-
    transitive A>B>C>A loops: a candidate can be ≥55% pooled across
    three anchors and still regress on one of them.
    """
    rows: dict[str, dict] = {}
    for anchor in anchors:
        wins = losses = draws = 0
        # candidate as P0 vs anchor as P1
        sp = result.matrix.get(candidate, {}).get(anchor)
        if sp is not None:
            wins += sp.p0_wins
            losses += sp.p1_wins
            draws += sp.draws
        # candidate as P1 vs anchor as P0
        sp = result.matrix.get(anchor, {}).get(candidate)
        if sp is not None:
            wins += sp.p1_wins
            losses += sp.p0_wins
            draws += sp.draws
        n = wins + losses + draws
        lo, hi = _wilson_ci(wins, n)
        rows[anchor] = {
            "wins": wins, "losses": losses, "draws": draws, "n": n,
            "winrate": wins / n if n else 0.0,
            "wilson_lo": lo, "wilson_hi": hi,
        }
    return rows


def _anchor_gate(per_anchor: dict[str, dict], threshold: float) -> dict:
    """Verdict on a per-anchor breakdown: PASS iff every anchor's
    Wilson lower bound meets `threshold`. Empty anchor set → PASS.
    """
    failing = [
        name for name, r in per_anchor.items() if r["wilson_lo"] < threshold
    ]
    passing = [
        name for name, r in per_anchor.items() if r["wilson_lo"] >= threshold
    ]
    return {
        "threshold": threshold,
        "pass": len(failing) == 0,
        "passing_anchors": passing,
        "failing_anchors": failing,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", nargs="+", required=True,
        metavar="NAME KEY=VALUE",
        help="Variant name followed by zero or more KEY=VALUE constant "
             "overrides. Repeat for each variant. Example: "
             "--variant v3_5 AIRTIME_PENALTY_WEIGHT=1.0 "
             "ENDGAME_NEUTRAL_BONUS=1.5 --variant v3_4 "
             "AIRTIME_PENALTY_WEIGHT=0.0 ENDGAME_NEUTRAL_BONUS=1.0",
    )
    parser.add_argument(
        "--seeds", type=int, default=32,
        help="Seed-bag size from SEEDS_64 (default: 32).",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count() or 1,
        help="Parallel game workers (default: cpu_count).",
    )
    parser.add_argument(
        "--include-self-play", action="store_true",
        help="Run a-vs-a self-play cells too (sanity baseline).",
    )
    parser.add_argument(
        "--candidate", default=None, metavar="NAME",
        help="Name of the variant under test (must match a --variant). "
             "All other variants are treated as anchors; the gate "
             "requires Wilson-lo >= --gate-threshold against EACH "
             "anchor individually (catches non-transitive A>B>C>A loops). "
             "Exit code is 1 on gate failure when --candidate is set.",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.55,
        help="Per-anchor Wilson-lo gate (default 0.55). Only used when "
             "--candidate is set.",
    )
    parser.add_argument(
        "--agent", default=str(REPO / "agents" / "v3_snipe"),
        help="Path to the agent directory to bundle for every variant. "
             "Default `agents/v3_snipe`. Pass e.g. "
             "`agents/v7_ablations/v7_0_drop_one` to A/B candidate "
             "Missions inside the drop-one rollout.",
    )
    args = parser.parse_args(argv)
    agent_dir = Path(args.agent).resolve()
    if not agent_dir.is_dir():
        raise SystemExit(f"--agent {args.agent}: not a directory")
    if not (agent_dir / "main.py").is_file():
        raise SystemExit(f"--agent {args.agent}: missing main.py")

    variants = _parse_variant_args(args.variant)
    if args.seeds > len(SEEDS_64):
        raise SystemExit(f"--seeds {args.seeds} exceeds SEEDS_64 size {len(SEEDS_64)}")
    seeds = SEEDS_64[: args.seeds]

    print(
        f"--- ab_variants: {len(variants)} variants × {args.seeds} seeds "
        f"(agent: {agent_dir.name})"
    )
    bundles: dict[str, str] = {}
    for name, overrides in variants.items():
        ov_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(no overrides)"
        print(f"  bundling {name} [{ov_str}]")
        bundles[name] = str(_bundle_variant(name, overrides, agent_dir=agent_dir))

    out_dir = REPO / "audit" / "tournaments"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = tournament.run_tournament(
        agents=bundles,
        seeds=seeds,
        include_self_play=args.include_self_play,
        out_dir=out_dir,
        progress=True,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - t0
    rows = _summarise(result, list(variants.keys()))

    print()
    print(f"=== A/B summary ({len(seeds)} seeds × both seats × pairs) ===")
    for name, r in rows.items():
        print(
            f"  {name:<12} winrate={r['winrate']*100:5.1f}%  "
            f"Wilson95=[{r['wilson_lo']*100:5.1f}%, {r['wilson_hi']*100:5.1f}%]  "
            f"W/L/D={r['wins']}/{r['losses']}/{r['draws']}  N={r['n']}"
        )
    print(f"  elapsed: {elapsed:.1f}s")

    # Per-anchor breakdown + gate verdict (only when --candidate is set).
    per_anchor: dict[str, dict] | None = None
    gate: dict | None = None
    if args.candidate is not None:
        if args.candidate not in variants:
            raise SystemExit(
                f"--candidate {args.candidate!r} is not among the "
                f"declared --variant names {list(variants.keys())}"
            )
        anchors = [n for n in variants.keys() if n != args.candidate]
        per_anchor = _per_anchor_summarise(result, args.candidate, anchors)
        gate = _anchor_gate(per_anchor, args.gate_threshold)
        print()
        print(
            f"=== per-anchor gate (candidate={args.candidate}, "
            f"threshold Wilson-lo >= {args.gate_threshold:.2f}) ==="
        )
        for anchor, r in per_anchor.items():
            verdict = "PASS" if r["wilson_lo"] >= args.gate_threshold else "FAIL"
            print(
                f"  vs {anchor:<12} winrate={r['winrate']*100:5.1f}%  "
                f"Wilson95=[{r['wilson_lo']*100:5.1f}%, "
                f"{r['wilson_hi']*100:5.1f}%]  "
                f"W/L/D={r['wins']}/{r['losses']}/{r['draws']}  "
                f"N={r['n']}  [{verdict}]"
            )
        verdict = "PASS" if gate["pass"] else "FAIL"
        print(f"  overall: [{verdict}]")
        if not gate["pass"]:
            print(f"  failing anchors: {gate['failing_anchors']}")

    # Persist summary alongside the tournament JSON.
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"ab-{utc}.json"
    payload: dict = {
        "utc": utc,
        "variants": variants,
        "seeds": seeds,
        "include_self_play": args.include_self_play,
        "summary": rows,
        "tournament_json": str(out_dir / f"{result.timestamp_utc}.json"),
        "elapsed_s": round(elapsed, 1),
    }
    if args.candidate is not None:
        payload["candidate"] = args.candidate
        payload["per_anchor"] = per_anchor
        payload["gate"] = gate
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  summary: {summary_path}")
    return 0 if (gate is None or gate["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
