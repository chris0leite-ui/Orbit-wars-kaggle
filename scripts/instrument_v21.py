"""scripts/instrument_v21.py — per-turn diagnostic for v21 patch flags.

Runs v21 against v20 on a single seed, captures per-turn instrumentation
counters at every step from v21's seat. Prints a roll-up at the end.

Acceptance criteria (per the iteration plan):

- `n_comet_targets_filtered > 0` in any turn with a comet on the board
  (else Patch E1's comet branch is dead code).
- `n_filtered_by_prefilter > 0` in at least 20% of turns
  (else local-force-ratio gate is too lax or never fires).
- `n_rescore_rounds > 0` in at least 30% of turns
  (else Patch A is dead code).
- `n_filtered_by_hold_check > 0` in at least 5% of capture-decisions
  (else Patch E2 is dead code).

Usage:
    python -m scripts.instrument_v21 --seed 1492346051 [--opp v20] [--steps 200]
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1492346051)
    ap.add_argument("--opp", default="v20")
    ap.add_argument("--me-seat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--steps", type=int, default=500,
                    help="hard cap on simulated steps (rarely hit)")
    ap.add_argument("--wallclock-ms", type=float, default=600.0,
                    help="ORBIT_WARS_PARITY_WALLCLOCK_MS override")
    args = ap.parse_args()

    os.environ["ORBIT_WARS_PARITY_WALLCLOCK_MS"] = str(args.wallclock_ms)

    from kaggle_environments import make

    v21 = import_module("agents.v21.main")
    opp = import_module(f"agents.{args.opp}.main")

    per_turn = []  # list[dict] of counters snapshotted post-agent-call
    comet_turn_flags = []  # True if at least 1 comet on board this turn

    def instrumented_v21(obs, configuration=None):
        # Detect comets on the board at this step.
        obs_d = obs if isinstance(obs, dict) else {
            k: getattr(obs, k, None)
            for k in ("planets", "comet_planet_ids", "step")
        }
        has_comet = bool(obs_d.get("comet_planet_ids"))
        action = v21.agent(obs, configuration)
        per_turn.append(dict(v21._INSTRUMENT_COUNTERS))
        comet_turn_flags.append(has_comet)
        return action

    if args.me_seat == 0:
        agents_list = [instrumented_v21, opp.agent]
    else:
        agents_list = [opp.agent, instrumented_v21]

    env = make("orbit_wars", configuration={"seed": args.seed})
    env.run(agents_list)
    final = env.steps[-1]
    my_reward = final[args.me_seat].get("reward")
    print(f"\n=== Game complete: seed={args.seed}, opp={args.opp}, "
          f"me_seat={args.me_seat}, my_reward={my_reward}, "
          f"n_steps={len(env.steps)}")

    n_turns = len(per_turn)
    if n_turns == 0:
        print("v21 never called — abort")
        return 1

    n_with_validated = sum(1 for c in per_turn if c["last_n_validated"] > 0)
    n_with_rescore = sum(1 for c in per_turn if c["last_n_rescore_rounds"] > 0)
    n_with_prefilter = sum(1 for c in per_turn if c["last_n_filtered_by_prefilter"] > 0)
    n_with_hold = sum(1 for c in per_turn if c["last_n_filtered_by_hold_check"] > 0)
    n_with_comet_filtered = sum(1 for c in per_turn if c["last_n_comet_targets_filtered"] > 0)
    n_with_comet_on_board = sum(1 for f in comet_turn_flags if f)

    total_validated = sum(c["last_n_validated"] for c in per_turn)
    total_committed = sum(c["last_n_committed"] for c in per_turn)
    total_rescore = sum(c["last_n_rescore_rounds"] for c in per_turn)
    total_filtered_prefilter = sum(c["last_n_filtered_by_prefilter"] for c in per_turn)
    total_filtered_hold = sum(c["last_n_filtered_by_hold_check"] for c in per_turn)
    total_comet_filtered = sum(c["last_n_comet_targets_filtered"] for c in per_turn)
    total_candidates = sum(c["last_n_candidates"] for c in per_turn)

    print(f"\nTotals across {n_turns} turns:")
    print(f"  candidates enumerated:    {total_candidates}")
    print(f"  filtered_by_prefilter:    {total_filtered_prefilter} "
          f"(in {n_with_prefilter} turns = {n_with_prefilter / n_turns:.0%})")
    print(f"  comet targets filtered:   {total_comet_filtered} "
          f"(comet on board: {n_with_comet_on_board} turns = "
          f"{n_with_comet_on_board / n_turns:.0%})")
    print(f"  validated:                {total_validated} "
          f"(in {n_with_validated} turns)")
    print(f"  committed (emitted):      {total_committed}")
    print(f"  rescore rounds (Patch A): {total_rescore} "
          f"(in {n_with_rescore} turns = {n_with_rescore / n_turns:.0%})")
    print(f"  filtered_by_hold_check:   {total_filtered_hold} "
          f"(in {n_with_hold} turns = {n_with_hold / n_turns:.0%})")

    capture_decisions = max(1, sum(1 for c in per_turn if c["last_n_committed"] > 0))
    pct_hold_check = n_with_hold / capture_decisions

    print("\nAcceptance gates:")
    ok = True

    def gate(name, cond):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}")

    gate("E1 comet branch active (filtered > 0 if comets present)",
         n_with_comet_on_board == 0 or n_with_comet_filtered > 0)
    gate("E1 prefilter ≥ 20% of turns",
         n_with_prefilter / n_turns >= 0.20)
    gate("Patch A rescore rounds ≥ 30% of turns",
         n_with_rescore / n_turns >= 0.30)
    gate("E2 hold-check ≥ 5% of capture-decisions",
         pct_hold_check >= 0.05)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
