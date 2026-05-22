"""Head-to-head: CRN-lite (P0) vs legacy-baseline (P1), same match.

Both seats run the SAME `agents.baseline.main.agent` module. The
wrapper temporarily sets / unsets `BASELINE_OPP_TRAJ_TIER` per call,
so P0 takes the CRN code path and P1 takes the legacy reactive path
inside one env.run().

This is the "one game first, understand it, against current production
baseline" diagnostic (2026-05-23 PI request).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agents.baseline.main import agent as _baseline_agent


_TIER_VAR = "BASELINE_OPP_TRAJ_TIER"


def _make_wrapped(tier: str | None):
    """Build a per-seat agent that scoped-sets the tier env var before
    calling into the baseline agent module. Restores prior value after.
    """
    def _wrapped(obs, configuration=None):
        prev = os.environ.get(_TIER_VAR, "")
        if tier is None or tier == "off":
            os.environ.pop(_TIER_VAR, None)
        else:
            os.environ[_TIER_VAR] = tier
        try:
            return _baseline_agent(obs, configuration)
        finally:
            if prev:
                os.environ[_TIER_VAR] = prev
            else:
                os.environ.pop(_TIER_VAR, None)
    _wrapped.__name__ = f"baseline_{tier or 'off'}"
    return _wrapped


def _actions_repr(act) -> str:
    if not isinstance(act, list) or not act:
        return "(idle)"
    parts = []
    for m in act:
        if isinstance(m, list) and len(m) >= 3:
            parts.append(f"src={int(m[0])} ang={float(m[1]):+.2f} ships={int(m[2])}")
    return "; ".join(parts) if parts else "(idle)"


def _ship_totals_per_seat(step) -> dict[int, int]:
    obs = step[0]["observation"]
    by: dict[int, int] = {}
    for p in obs.get("planets", []):
        owner = int(p[1])
        if owner < 0:
            continue
        by[owner] = by.get(owner, 0) + int(p[5])
    for f in obs.get("fleets", []):
        owner = int(f[1])
        by[owner] = by.get(owner, 0) + int(f[5])
    return by


def _planet_counts(step) -> dict[int, int]:
    obs = step[0]["observation"]
    counts: dict[int, int] = {}
    for p in obs.get("planets", []):
        owner = int(p[1])
        counts[owner] = counts.get(owner, 0) + 1
    return counts


def run_h2h(seed: int, swap: bool, verbose_window: int = 30) -> None:
    from kaggle_environments import make

    crn = _make_wrapped("lite")
    legacy = _make_wrapped("off")

    if swap:
        agents = [legacy, crn]
        labels = ["LEGACY", "CRN-lite"]
    else:
        agents = [crn, legacy]
        labels = ["CRN-lite", "LEGACY"]

    print(f"==== seed={seed}   P0={labels[0]}   P1={labels[1]} ====")
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    env.run(agents)
    steps = env.steps

    final = steps[-1]
    r0, r1 = final[0]["reward"], final[1]["reward"]
    winner = 0 if r0 > r1 else (1 if r1 > r0 else -1)
    print(f"\nGame: {len(steps)} turns. Final: P0 reward={r0}, P1 reward={r1}. "
          f"Winner=P{winner} ({labels[winner] if winner >= 0 else 'TIE'}).")

    # Per-seat aggregates.
    emit_count = [0, 0]
    ship_emit = [0, 0]
    captures_by = [0, 0]
    prev_owners: dict[int, int] = {}
    initial_obs = steps[0][0]["observation"]
    for p in initial_obs.get("planets", []):
        prev_owners[int(p[0])] = int(p[1])

    for t in range(1, len(steps)):
        step = steps[t]
        for pid in (0, 1):
            act = step[pid].get("action") if pid < len(step) else None
            if isinstance(act, list):
                emit_count[pid] += len(act)
                for m in act:
                    if isinstance(m, list) and len(m) >= 3:
                        try:
                            ship_emit[pid] += int(m[2])
                        except (TypeError, ValueError):
                            pass
        # Captures.
        curr_obs = step[0]["observation"]
        for p in curr_obs.get("planets", []):
            pid_p = int(p[0])
            owner_now = int(p[1])
            prev = prev_owners.get(pid_p)
            if prev is not None and prev != owner_now and owner_now >= 0:
                captures_by[owner_now] += 1
            prev_owners[pid_p] = owner_now

    print(f"\nLaunches  — P0 ({labels[0]}): {emit_count[0]} launches, {ship_emit[0]} ships")
    print(f"           P1 ({labels[1]}): {emit_count[1]} launches, {ship_emit[1]} ships")
    print(f"Captures  — P0: {captures_by[0]}    P1: {captures_by[1]}")

    final_ships = _ship_totals_per_seat(steps[-1])
    final_planets = _planet_counts(steps[-1])
    print(f"Final     — P0 ships={final_ships.get(0, 0)}, planets={final_planets.get(0, 0)}")
    print(f"           P1 ships={final_ships.get(1, 0)}, planets={final_planets.get(1, 0)}")

    # Verbose window — show first N turns where both seats acted, plus
    # the first divergence-after-symmetric-opening (if any).
    print(f"\n--- First {verbose_window} turns of per-seat actions ---")
    for t in range(min(verbose_window, len(steps) - 1)):
        step = steps[t + 1]
        a0 = step[0].get("action") if 0 < len(step) else None
        a1 = step[1].get("action") if 1 < len(step) else None
        if not (isinstance(a0, list) and a0) and not (isinstance(a1, list) and a1):
            continue  # both idle; skip
        print(f"  turn {t+1:3d}  P0: {_actions_repr(a0)}")
        print(f"          P1: {_actions_repr(a1)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap", action="store_true",
                    help="Put legacy as P0 (CRN as P1)")
    ap.add_argument("--verbose-window", type=int, default=60)
    args = ap.parse_args()
    run_h2h(args.seed, swap=args.swap, verbose_window=args.verbose_window)


if __name__ == "__main__":
    main()
