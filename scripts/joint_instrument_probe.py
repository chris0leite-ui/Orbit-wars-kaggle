"""Joint-pair instrumentation probe.

Runs N games of baseline vs a chosen opp with
`BASELINE_JOINT_INSTRUMENT=<path>` set, then aggregates the JSONL output
to answer: "how often does the existing pair-joint enumeration fail to
crack a target — i.e. how often would a 3+ source bundle have helped?"

Usage:
    python scripts/joint_instrument_probe.py --n 8 --opp nearest \\
        --out /tmp/joint_probe.jsonl

Output: aggregate summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def run_one_game(seed: int, focal: str, opp: str, jsonl_path: Path) -> None:
    """Spawn one game in a subprocess so atexit-dump fires per game."""
    env = os.environ.copy()
    env["BASELINE_JOINT_INSTRUMENT"] = str(jsonl_path)
    # Single-game runner — uses kaggle_environments.run() directly to keep
    # it simple and fast. Output silenced; we only care about the JSONL.
    code = f"""
import os, sys
sys.path.insert(0, {str(REPO)!r})
from fast import play_one, resolve_agent_spec
_, focal_path = resolve_agent_spec({focal!r})
_, opp_path = resolve_agent_spec({opp!r})
result = play_one({seed!r}, focal_path, opp_path, record_timing=False)
print('outcome=', result.outcome, 'rewards=', result.rewards, flush=True)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180,
    )


def aggregate(jsonl_path: Path) -> dict:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {"n_rows": 0}

    n_rows = len(rows)
    n_pair_positive = sum(1 for r in rows if r["n_pairs_positive"] > 0)
    n_any_solo = sum(1 for r in rows if r["any_solo_winner"])
    # "Captured this turn" = at least one solo positive OR at least one
    # pair positive. The complement is "target with ≥2 candidates that
    # the chooser could NOT crack as solo or pair this turn".
    n_uncrackable = sum(
        1 for r in rows
        if r["n_pairs_positive"] == 0 and not r["any_solo_winner"]
    )
    # Of the uncrackable, how many had ≥3 candidates (so a 3-source
    # bundle was geometrically possible)?
    n_uncrackable_3plus = sum(
        1 for r in rows
        if r["n_pairs_positive"] == 0 and not r["any_solo_winner"]
        and r["n_cands"] >= 3
    )
    # Of the uncrackable with ≥3 cands, how many had ≥4?
    n_uncrackable_4plus = sum(
        1 for r in rows
        if r["n_pairs_positive"] == 0 and not r["any_solo_winner"]
        and r["n_cands"] >= 4
    )
    # All-solo-gated: pair-enum was bypassed entirely because every pair
    # had both srcs as solo winners. Not the n-source-extension target.
    n_all_solo_gated = sum(
        1 for r in rows
        if r["n_pairs_attempted"] == 0 and r["n_pairs_solo_gated"] > 0
    )
    # Target-defender distribution among uncrackable
    uncrackable_garrisons = [
        r["tgt_ships"] for r in rows
        if r["n_pairs_positive"] == 0 and not r["any_solo_winner"]
    ]
    # By target owner
    uncrackable_by_owner = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    for r in rows:
        if r["n_pairs_positive"] == 0 and not r["any_solo_winner"]:
            o = int(r["tgt_owner"])
            if o in uncrackable_by_owner:
                uncrackable_by_owner[o] += 1

    summary = {
        "n_rows_total": n_rows,
        "n_target_observations_with_pair_winner": n_pair_positive,
        "n_target_observations_with_any_solo_winner": n_any_solo,
        "n_uncrackable_targets": n_uncrackable,
        "n_uncrackable_with_3plus_cands": n_uncrackable_3plus,
        "n_uncrackable_with_4plus_cands": n_uncrackable_4plus,
        "n_all_solo_gated": n_all_solo_gated,
        "uncrackable_pct_of_observations": (
            100.0 * n_uncrackable / max(1, n_rows)
        ),
        "uncrackable_3plus_pct_of_uncrackable": (
            100.0 * n_uncrackable_3plus / max(1, n_uncrackable)
        ),
        "uncrackable_3plus_pct_of_observations": (
            100.0 * n_uncrackable_3plus / max(1, n_rows)
        ),
        "uncrackable_by_owner": uncrackable_by_owner,
        "uncrackable_garrison_median": (
            statistics.median(uncrackable_garrisons)
            if uncrackable_garrisons else 0.0
        ),
        "uncrackable_garrison_p90": (
            sorted(uncrackable_garrisons)[
                int(0.9 * len(uncrackable_garrisons))
            ] if uncrackable_garrisons else 0.0
        ),
        "uncrackable_garrison_max": (
            max(uncrackable_garrisons) if uncrackable_garrisons else 0.0
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--focal", default="agents/baseline")
    parser.add_argument("--opp", default="nearest")
    parser.add_argument(
        "--out", default="/tmp/joint_probe.jsonl", type=Path,
    )
    parser.add_argument("--seed-base", type=int, default=10000)
    args = parser.parse_args()

    # Fresh file
    if args.out.exists():
        args.out.unlink()

    for k in range(args.n):
        seed = args.seed_base + k
        print(f"[probe] game {k + 1}/{args.n} seed={seed}", flush=True)
        run_one_game(seed, args.focal, args.opp, args.out)

    summary = aggregate(args.out)
    print()
    print("=== joint pair-enum diagnostic summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
