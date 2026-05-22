"""H2H smoke: ledger+opportunistic-opp vs current production.

P0 runs the modeling-fix variant: BASELINE_LEDGER=on +
BASELINE_OPP_TIER=opportunistic. P1 runs current production
(env vars unset, matches baseline_joint_aggr_consolidated_orbitfix).

Both seats use the SAME agents.baseline.main.agent module; the
wrapper scopes env-vars per call. This is the "one game first,
understand it" diagnostic per PI standing instruction.

NOT a submission test. NOT a panel test. Just one self-play game
to see if the variant survives at all before any further n-scaling.

Usage:
  python scripts/h2h_ledger_opp_vs_prod.py [--seeds 42] [--seats 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Match production submission env vars before importing the agent.
_PROD_DEFAULTS = {
    "BASELINE_JOINT_AGGR": "1",
    "BASELINE_JOINT_TOP_K": "5",
    "BASELINE_JOINT_MAX_PAIRS": "60",
    "BASELINE_REINFORCE_EMIT": "1",
    "BASELINE_REINFORCE_ANTICIPATE": "1",
    "BASELINE_NEUTRAL_BONUS": "2.0",
    "BASELINE_NEUTRAL_EARLY_EXTRA": "1.5",
    "BASELINE_NEUTRAL_EARLY_HORIZON": "50",
    "BASELINE_ORBITAL_SAFETY": "1",
}
for k, v in _PROD_DEFAULTS.items():
    os.environ.setdefault(k, v)

# Variant-specific vars that the wrapper toggles per call.
_VARIANT_VARS = {
    "BASELINE_LEDGER": "on",
    "BASELINE_OPP_TIER": "opportunistic",
}

from kaggle_environments import make
from agents.baseline.main import agent as _baseline_agent


def _make_wrapped(use_variant: bool, name: str):
    """Build a per-seat agent wrapper that scopes the variant env-vars.

    When `use_variant=True`, sets the variant's env vars before each
    call and restores prior values after. When False, ensures the
    variant vars are UNSET for the call so production code path runs.
    """
    var_names = list(_VARIANT_VARS.keys())

    def _wrapped(obs, configuration=None):
        prev = {k: os.environ.get(k) for k in var_names}
        if use_variant:
            for k, v in _VARIANT_VARS.items():
                os.environ[k] = v
        else:
            for k in var_names:
                os.environ.pop(k, None)
        try:
            return _baseline_agent(obs, configuration)
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    _wrapped.__name__ = name
    return _wrapped


def _summarize_actions(steps, seat: int) -> dict:
    """Per-game launch stats for this seat."""
    sizes = []
    for step in steps:
        actions = step[seat].get("action") or []
        for a in actions:
            if isinstance(a, (list, tuple)) and len(a) >= 3:
                sizes.append(int(a[2]))
    n = len(sizes) or 1
    sorted_sizes = sorted(sizes)
    tiny = sum(1 for s in sizes if s < 5)
    return {
        "launches": len(sizes),
        "ships_sent": sum(sizes),
        "avg_size": sum(sizes) / n if sizes else 0,
        "median": sorted_sizes[n // 2] if sizes else 0,
        "tiny_pct": 100 * tiny / n if sizes else 0,
    }


def run_one(seed: int, seats: int = 2) -> dict:
    p0_agent = _make_wrapped(use_variant=True, name="VARIANT")
    p1_agent = _make_wrapped(use_variant=False, name="PROD")
    if seats == 4:
        # 4P: P0,P2 = VARIANT; P1,P3 = PROD (balance).
        agents_list = [p0_agent, p1_agent, p0_agent, p1_agent]
    else:
        agents_list = [p0_agent, p1_agent]

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=seats)
    t0 = time.time()
    env.run(agents_list)
    elapsed = time.time() - t0
    steps = env.steps
    n_steps = len(steps)
    final = steps[-1]
    rewards = [final[s].get("reward") for s in range(seats)]
    statuses = [final[s].get("status") for s in range(seats)]

    out = {
        "seed": seed,
        "seats": seats,
        "n_steps": n_steps,
        "wallclock_s": elapsed,
        "rewards": rewards,
        "statuses": statuses,
        "variant_seats": [0, 2] if seats == 4 else [0],
        "prod_seats": [1, 3] if seats == 4 else [1],
        "per_seat": {s: _summarize_actions(steps, s) for s in range(seats)},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42",
                    help="Comma-separated seeds")
    ap.add_argument("--seats", type=int, default=2, choices=[2, 4])
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    for seed in seeds:
        print(f"\n{'='*72}\nseed={seed}  seats={args.seats}")
        out = run_one(seed, args.seats)
        print(f"  steps={out['n_steps']}  wallclock={out['wallclock_s']:.1f}s")
        print(f"  rewards={out['rewards']}  statuses={out['statuses']}")
        print(f"  variant seats: {out['variant_seats']}   prod seats: {out['prod_seats']}")
        print(f"\n  {'seat':>4} {'role':>8} {'launches':>9} {'ships':>7} "
              f"{'avg':>6} {'med':>6} {'tiny%':>7}")
        for s in range(args.seats):
            role = "VARIANT" if s in out['variant_seats'] else "PROD"
            st = out['per_seat'][s]
            print(f"  {s:>4} {role:>8} {st['launches']:>9} {st['ships_sent']:>7} "
                  f"{st['avg_size']:>6.1f} {st['median']:>6} {st['tiny_pct']:>6.1f}%")


if __name__ == "__main__":
    main()
