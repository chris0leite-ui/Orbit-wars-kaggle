"""Ablation harness — measure each mechanism's individual contribution.

Composes the same strategy with different mechanism subsets and runs them
through the existing tournament fixture. Per the plan's Step 3.5 ablation
story: name the agent `<strategy>__<+joined sorted mech names>` so the
audit JSON is greppable.

Usage (current — Step 3.5.B):
    python scripts/ablation.py arrival_size

Runs:
    v1__validate+lead_aim         (parity baseline)
    v1__validate+lead_aim+arrival_size   (with new mechanism)

vs the shipped baseline + each other, over the standard 20-seed bag, both
sides. Persists JSON to audit/tournaments/.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# Tournament module under its top-level name so dataclasses resolve correctly.
_t_spec = importlib.util.spec_from_file_location("tournament", REPO / "scripts" / "tournament.py")
tournament = importlib.util.module_from_spec(_t_spec)
sys.modules["tournament"] = tournament
_t_spec.loader.exec_module(tournament)


SEEDS_20 = [42, 1, 7, 13, 31, 100, 17, 23, 53, 71, 91, 113, 137, 149, 167, 181, 199, 211, 233, 257]


def _build_agent(strategy_propose, mechanisms):
    """Compose strategy + mechanism list into an `agent(obs)` callable."""
    from lib.intent import realize

    def agent(obs):
        return realize(strategy_propose(obs), obs, mechanisms=mechanisms)

    return agent


def _agent_name(strategy_label: str, mechanisms) -> str:
    mech_names = "+".join(sorted(m.__name__ for m in mechanisms))
    return f"{strategy_label}__{mech_names}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mechanism",
        choices=["arrival_size", "comet_aim", "sun_avoid"],
        help="which new mechanism to ablate",
    )
    parser.add_argument(
        "--seeds", type=int, default=20,
        help="how many fixed seeds to use (default: 20 — the eval_v1 bag)",
    )
    args = parser.parse_args(argv)

    from agents.v1_orbitfix.main import propose_intents
    from lib import mechanism as mech_mod

    seeds = SEEDS_20[: args.seeds]

    base_mechs = [mech_mod.validate, mech_mod.lead_aim]
    base = _build_agent(propose_intents, base_mechs)
    base_name = _agent_name("v1", base_mechs)

    new = getattr(mech_mod, args.mechanism)
    # The new mechanism's pipeline position depends on its semantics — we
    # mirror the canonical DEFAULT_MECHANISMS ordering.
    if args.mechanism == "arrival_size":
        full_mechs = [mech_mod.validate, new, mech_mod.lead_aim]
    elif args.mechanism == "comet_aim":
        # comet_aim runs AFTER lead_aim per the doc-string in lib/mechanism.py.
        full_mechs = [mech_mod.validate, mech_mod.lead_aim, new]
    elif args.mechanism == "sun_avoid":
        full_mechs = [mech_mod.validate, mech_mod.lead_aim, new]
    else:
        raise ValueError(args.mechanism)

    full = _build_agent(propose_intents, full_mechs)
    full_name = _agent_name("v1", full_mechs)

    baseline = str(REPO / "data" / "main.py")

    out_dir = REPO / "audit" / "tournaments"
    result = tournament.run_tournament(
        agents={
            full_name: full,
            base_name: base,
            "baseline": baseline,
        },
        seeds=seeds,
        include_self_play=False,
        out_dir=out_dir,
        progress=True,
    )

    print()
    print("=== Ablation summary ===")
    pairs = [
        (full_name, base_name),
        (base_name, full_name),
        (full_name, "baseline"),
        ("baseline", full_name),
        (base_name, "baseline"),
        ("baseline", base_name),
    ]
    for row, col in pairs:
        stat = result.matrix[row][col]
        print(
            f"{row:60s} (P0) vs {col:30s} (P1): "
            f"{stat.p0_wins}/{stat.n} P0 wins  "
            f"(Wilson 95% {stat.wilson_lo:.2f}..{stat.wilson_hi:.2f})  "
            f"p95 turn ms P0={stat.p0_p95_turn_ms:.1f} P1={stat.p1_p95_turn_ms:.1f}"
        )

    # Aggregate full-vs-base across both sides.
    a = result.matrix[full_name][base_name]
    b = result.matrix[base_name][full_name]
    full_wins = a.p0_wins + b.p1_wins
    full_total = a.n + b.n
    print()
    print(f"{full_name} vs {base_name} aggregate: "
          f"{full_wins}/{full_total} = {full_wins / full_total:.1%}  (gate: ≥55%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
