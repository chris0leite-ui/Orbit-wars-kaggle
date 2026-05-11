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


SNIPE_PATH = REPO / "lib" / "missions" / "snipe.py"
BUNDLE_OUT = REPO / "submissions" / "_ab"


def _patch_constants(source: str, overrides: dict[str, float]) -> str:
    """Apply `NAME = NUMBER` overrides via line-anchored regex.

    Only replaces lines whose first non-whitespace tokens are exactly
    `NAME = <number>` (handles ints + floats). Comment lines, function
    bodies, and assignments to attributes are untouched.
    """
    patched = source
    for name, value in overrides.items():
        pattern = re.compile(
            rf"^(?P<lead>\s*){re.escape(name)}\s*=\s*[-+0-9.eE]+\s*$",
            re.MULTILINE,
        )
        if not pattern.search(patched):
            raise ValueError(
                f"variant override `{name}` not found as a top-level "
                f"assignment in {SNIPE_PATH.name}"
            )
        patched = pattern.sub(rf"\g<lead>{name} = {value}", patched)
    return patched


def _bundle_variant(name: str, overrides: dict[str, float]) -> Path:
    """Patch → bundle → restore. Returns the bundle path."""
    original = SNIPE_PATH.read_text()
    try:
        SNIPE_PATH.write_text(_patch_constants(original, overrides))
        BUNDLE_OUT.mkdir(parents=True, exist_ok=True)
        out_path = bundle_agent.bundle(
            REPO / "agents" / "v3_snipe",
            bundle_agent.DEFAULT_LIB_ORDER,
            out_dir=BUNDLE_OUT,
        )
        # bundle_agent names the file by agent_dir.name = "v3_snipe".
        renamed = BUNDLE_OUT / f"{name}.py"
        out_path.rename(renamed)
        return renamed
    finally:
        SNIPE_PATH.write_text(original)


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
    args = parser.parse_args(argv)

    variants = _parse_variant_args(args.variant)
    if args.seeds > len(SEEDS_64):
        raise SystemExit(f"--seeds {args.seeds} exceeds SEEDS_64 size {len(SEEDS_64)}")
    seeds = SEEDS_64[: args.seeds]

    print(f"--- ab_variants: {len(variants)} variants × {args.seeds} seeds")
    bundles: dict[str, str] = {}
    for name, overrides in variants.items():
        ov_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(no overrides)"
        print(f"  bundling {name} [{ov_str}]")
        bundles[name] = str(_bundle_variant(name, overrides))

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

    # Persist summary alongside the tournament JSON.
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"ab-{utc}.json"
    summary_path.write_text(json.dumps({
        "utc": utc,
        "variants": variants,
        "seeds": seeds,
        "include_self_play": args.include_self_play,
        "summary": rows,
        "tournament_json": str(out_dir / f"{result.timestamp_utc}.json"),
        "elapsed_s": round(elapsed, 1),
    }, indent=2) + "\n")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
