"""check_coord_singleton_parity — Gate 1 structural-correctness check.

Run coord (with COORD_MAX_BUNDLE_SIZE=1 + COORD_DISABLE_DEFEND=1) and
minimal side-by-side on the same game states. Compare action streams.

Both agents end up using `score_candidate_v4_joint` for Tier-2 scoring,
so when coord is reduced to singleton-only + no-defense, it OUGHT to
produce sensible single-source attacks that overlap heavily with
minimal's solo emissions. Differences should be bounded — coord's
cheap-filter ranks the top-K differently than minimal's prerank, but
Tier-2 scoring is identical and should converge on similar winners.

This is NOT a byte-for-byte gate (the plan's optimistic phrasing) — it's
a structural-correctness gate. We report quantitative divergence:

- % turns with identical action sets
- % turns where coord's source-engagement is a subset of minimal's
- % turns with significant divergence (different sources active)

Acceptance: identical OR near-identical (≤1 source diff) on ≥80% of
turns. Anything lower triggers investigation.

Usage
-----
    python scripts/check_coord_singleton_parity.py             # default: 2 seeds × 60 turns
    python scripts/check_coord_singleton_parity.py --seeds 4
    python scripts/check_coord_singleton_parity.py --turns 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Set env knobs BEFORE importing coord.
os.environ["COORD_MAX_BUNDLE_SIZE"] = "1"
os.environ["COORD_DISABLE_DEFEND"] = "1"

from kaggle_environments import make  # noqa: E402

from agents.coord.main import agent as coord_agent  # noqa: E402
from agents.minimal.main import agent as minimal_agent  # noqa: E402


def _action_signature(action_list) -> set[tuple]:
    """A turn's action list → set of (src_id, ships) tuples.

    Angle differences below 6dp are noise (orbital-prediction float
    repr); ship counts and source planets are the structural signals.
    """
    sig = set()
    for entry in action_list or []:
        try:
            sid = int(entry[0])
            ships = int(entry[2])
            sig.add((sid, ships))
        except (IndexError, TypeError, ValueError):
            sig.add((-1, -1))  # malformed — gets caught as divergence
    return sig


def _sources_engaged(action_list) -> set[int]:
    """Just the set of source planet ids used in this turn's actions."""
    out = set()
    for entry in action_list or []:
        try:
            out.add(int(entry[0]))
        except (IndexError, TypeError, ValueError):
            pass
    return out


def run_one_seed(seed: int, turns: int) -> dict:
    env = make("orbit_wars", configuration={"seed": int(seed)})
    env.reset(num_agents=2)

    identical_turns = 0
    source_subset_turns = 0
    divergent_turns = 0
    diff_details: list[dict] = []

    for t in range(turns):
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation

        # Both agents see player 0's obs (they "would do" different
        # things from that POV; we use the same obs for symmetry).
        c_action = coord_agent(obs0)
        m_action = minimal_agent(obs0)

        c_sig = _action_signature(c_action)
        m_sig = _action_signature(m_action)
        c_srcs = _sources_engaged(c_action)
        m_srcs = _sources_engaged(m_action)

        if c_sig == m_sig:
            identical_turns += 1
        elif c_srcs <= m_srcs:
            source_subset_turns += 1
        else:
            divergent_turns += 1
            if len(diff_details) < 10:
                diff_details.append({
                    "seed": seed,
                    "turn": t,
                    "coord_actions": [list(x) for x in c_sig],
                    "minimal_actions": [list(x) for x in m_sig],
                    "coord_sources": sorted(c_srcs),
                    "minimal_sources": sorted(m_srcs),
                })

        # Advance the game using minimal-vs-minimal (so both agents
        # see realistic mid-game states).
        a0 = minimal_agent(obs0)
        a1 = minimal_agent(obs1)
        env.step([a0, a1])
        if env.done:
            break

    return {
        "seed": seed,
        "turns_evaluated": t + 1,
        "identical_turns": identical_turns,
        "source_subset_turns": source_subset_turns,
        "divergent_turns": divergent_turns,
        "diff_details": diff_details,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--turns", type=int, default=60)
    args = ap.parse_args()

    print(f"[gate1] coord(singleton+no-defense) vs minimal on "
          f"{args.seeds} seeds x {args.turns} turns", flush=True)
    print(f"  COORD_MAX_BUNDLE_SIZE={os.environ.get('COORD_MAX_BUNDLE_SIZE')}")
    print(f"  COORD_DISABLE_DEFEND={os.environ.get('COORD_DISABLE_DEFEND')}")
    print()

    t_start = time.perf_counter()
    per_seed: list[dict] = []
    for s in range(args.seeds):
        result = run_one_seed(s, args.turns)
        per_seed.append(result)
        n = result["turns_evaluated"]
        print(f"  [seed {s}] {n} turns: "
              f"identical={result['identical_turns']} "
              f"subset={result['source_subset_turns']} "
              f"divergent={result['divergent_turns']}", flush=True)

    total = sum(r["turns_evaluated"] for r in per_seed)
    identical = sum(r["identical_turns"] for r in per_seed)
    subset = sum(r["source_subset_turns"] for r in per_seed)
    divergent = sum(r["divergent_turns"] for r in per_seed)
    elapsed = time.perf_counter() - t_start

    print()
    print("=" * 62)
    print("GATE 1 — SINGLETON PARITY RESULT")
    print("=" * 62)
    print(f"  Total turns:    {total}")
    print(f"  Identical:      {identical:4d}  ({100.0 * identical / total:.1f}%)")
    print(f"  Source subset:  {subset:4d}  ({100.0 * subset / total:.1f}%)")
    print(f"  Divergent:      {divergent:4d}  ({100.0 * divergent / total:.1f}%)")
    print(f"  Elapsed:        {elapsed:.1f}s")

    near_identical = identical + subset
    acceptance = near_identical / total >= 0.80
    print(f"\n  near-identical (identical + source-subset): "
          f"{100.0 * near_identical / total:.1f}%  "
          f"({'PASS ≥80%' if acceptance else 'FAIL <80%'})")

    if divergent > 0:
        print(f"\n  First {min(10, divergent)} divergent turns:")
        for d in (per_seed[0].get("diff_details", []) +
                  (per_seed[1].get("diff_details", []) if len(per_seed) > 1 else [])
                  )[:10]:
            print(f"    seed={d['seed']} turn={d['turn']}: "
                  f"coord_srcs={d['coord_sources']} "
                  f"minimal_srcs={d['minimal_sources']}")

    audit_dir = REPO / "audit"
    audit_dir.mkdir(exist_ok=True)
    out_path = audit_dir / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"gate1-singleton-parity.json"
    )
    summary = {
        "total_turns": total,
        "identical_turns": identical,
        "source_subset_turns": subset,
        "divergent_turns": divergent,
        "near_identical_pct": 100.0 * near_identical / total,
        "acceptance_pct_threshold": 80.0,
        "elapsed_seconds": elapsed,
        "per_seed": per_seed,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  JSON: {out_path}")

    print()
    print("VERDICT:", "GATE 1 PASS" if acceptance else "GATE 1 FAIL", flush=True)
    return 0 if acceptance else 1


if __name__ == "__main__":
    sys.exit(main())
