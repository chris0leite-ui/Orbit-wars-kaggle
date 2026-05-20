"""Per-turn introspect for the analytical (joint_solver) agent.

Runs a single game (analytical agent vs lite_greedy opp by default) and
dumps per-turn LP diagnostics:
  - column counts (total, positive-value)
  - fired columns (broken down by wait_N)
  - emitted moves (the LP's t=0 commits)
  - solver objective and status

Output format: one row per turn to stdout, plus a summary block at the end.

STOP-gate question (Phase 3): does the LP fire ≥1 wait_N>0 OR ≥1
multi-source-per-target column per 5 turns? If yes, the multi-turn
machinery is doing visible coordination work.

Usage:
  python scripts/joint_solver_introspect.py [--turns N] [--seed S] [--opp NAME]
"""

from __future__ import annotations

import argparse
import sys
import time

from kaggle_environments import make

from lib.joint_solver.mpc import solve_turn


def _build_wrapped_agent(turn_log: list):
    """Wrap solve_turn to capture per-turn diagnostics into `turn_log`."""

    def wrapped(obs, configuration=None):
        moves, diag = solve_turn(obs, configuration, return_diagnostics=True)
        turn_log.append(diag)
        return moves

    return wrapped


def _format_diag(diag) -> str:
    wait_str = ",".join(f"w{w}={n}" for w, n in
                        sorted(diag.fired_wait_distribution.items()))
    return (
        f"step={diag.step:>3} prerank={diag.n_prerank:>3} "
        f"cols={diag.n_columns:>3} pos={diag.n_positive_columns:>3} "
        f"fired={diag.n_fired_columns:>2}({wait_str or '-'}) "
        f"emit={diag.n_emitted_moves:>2} "
        f"obj={diag.objective:>8.2f} status={diag.solver_status}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=120,
                        help="how many turns to log (game still runs to end)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opp", default="lite_greedy",
                        choices=["lite_greedy", "random", "baseline"],
                        help="opponent policy")
    parser.add_argument("--every", type=int, default=1,
                        help="log every Nth turn (default 1 = log all)")
    args = parser.parse_args()

    turn_log: list = []
    me = _build_wrapped_agent(turn_log)

    if args.opp == "lite_greedy":
        from lib.opp_model import lite_greedy_policy as _opp
        opp = _opp
    elif args.opp == "baseline":
        from agents.baseline.main import agent as _opp
        opp = _opp
    else:
        opp = "random"

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    t0 = time.time()
    env.run([me, opp])
    elapsed = time.time() - t0
    last = env.steps[-1]

    # Print per-turn rows.
    print("=== per-turn diagnostics ===")
    for i, diag in enumerate(turn_log):
        if i % args.every != 0 and i != len(turn_log) - 1:
            continue
        print(_format_diag(diag))

    # Summary.
    print()
    print("=== summary ===")
    print(f"game length: {len(env.steps)} steps; wallclock: {elapsed:.1f}s; "
          f"avg ms/turn: {1000 * elapsed / max(1, len(turn_log)):.1f}")
    print(f"final statuses: {[s.status for s in last]}")
    print(f"rewards: {[s.reward for s in last]}")

    # Multi-turn coordination evidence.
    # Distinguish "firing turns" (LP picked something) from "all turns"
    # (most of which return no_positive_columns because the W1/W2 value
    # bounds are conservative — separate concern from multi-turn capability).
    firing_turns = [d for d in turn_log if d.n_fired_columns > 0]
    n_turns_with_multi_wait = sum(
        1 for d in turn_log
        if any(w > 0 for w in d.fired_wait_distribution.keys())
    )
    n_turns_emitting = sum(1 for d in turn_log if d.n_emitted_moves > 0)
    n_turns_with_gang_up = sum(
        1 for d in turn_log if d.n_fired_columns > 1
    )

    multi_wait_rate_among_firing = (
        100.0 * n_turns_with_multi_wait / max(1, len(firing_turns))
    )
    print()
    print("=== multi-turn coordination ===")
    print(f"turns where LP fired anything:       {len(firing_turns)}/{len(turn_log)}")
    print(f"turns emitting ≥1 move (wait_N=0):   {n_turns_emitting}/{len(turn_log)}")
    print(f"turns with ≥1 fired wait_N>0 column: {n_turns_with_multi_wait}/{len(turn_log)} "
          f"({multi_wait_rate_among_firing:.1f}% of firing turns)")
    print(f"turns with ≥2 fired columns (joint): {n_turns_with_gang_up}/{len(turn_log)}")
    # Gate: in 5 consecutive firing turns, at least 1 picks wait_N>0.
    # Equivalent: rate >= 20% among firing turns.
    gate = (
        multi_wait_rate_among_firing >= 20.0 and n_turns_with_multi_wait >= 1
    )
    print(f"STOP gate (multi-turn coord visible at ≥20% of firing turns): "
          f"{'PASS' if gate else 'FAIL'}")

    # Solver status histogram.
    statuses: dict[str, int] = {}
    for d in turn_log:
        statuses[d.solver_status] = statuses.get(d.solver_status, 0) + 1
    print()
    print("=== solver status histogram ===")
    for s, n in sorted(statuses.items(), key=lambda kv: -kv[1]):
        print(f"  {s}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
