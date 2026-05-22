"""Single-game side-by-side trace: BASELINE_OPP_TRAJ_TIER=off vs lite.

Runs the baseline agent twice on the same seed (once with the env var
unset, once with TIER=lite), capturing P0's launches per turn. Reports:

  - First turn where the two runs diverge (chooser picked different
    actions). Before this turn the games are identical; after, every
    subsequent state differs.
  - Per-turn launch counts and ship totals for both runs.
  - Per-turn captures (planets changing hands).
  - Final outcome of each run.

This is the "step through one game and see what changes" diagnostic
(2026-05-23 PI request). It does NOT make claims about lift — just
"does the new code change observable behavior, and if so, where /
in what direction."

Usage:
    python scripts/trace_crn_vs_legacy.py --seed 42 --opp orbitfix
    python scripts/trace_crn_vs_legacy.py --seed 42 --max-turns 200 --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_agent_module(path: Path, modname: str):
    """Load an agent's `agent` callable.

    For the source agents (agents/baseline, agents/v1_orbitfix) we
    import the module by dotted path — the safer option, since they
    rely on importing from `agents.baseline.X` siblings. For other
    files we fall back to spec_from_file_location.
    """
    if path.parent.name and (REPO / path.parent.name / "__init__.py").exists():
        pkg = f"agents.{path.parent.name}.{path.stem}"
        mod = importlib.import_module(pkg)
        return mod.agent
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def _summarise_turn_actions(env_steps, turn: int, seat: int) -> tuple[int, int]:
    """(launch_count, total_ships_emitted) for `seat` at this turn."""
    if turn >= len(env_steps):
        return 0, 0
    step = env_steps[turn]
    act = step[seat].get("action") if seat < len(step) else None
    if not isinstance(act, list):
        return 0, 0
    n = len(act)
    ships = 0
    for m in act:
        if isinstance(m, list) and len(m) >= 3:
            try:
                ships += int(m[2])
            except (TypeError, ValueError):
                pass
    return n, ships


def _actions_repr(env_steps, turn: int, seat: int) -> str:
    if turn >= len(env_steps):
        return "(no turn)"
    step = env_steps[turn]
    act = step[seat].get("action") if seat < len(step) else None
    if not isinstance(act, list) or not act:
        return "(idle)"
    parts = []
    for m in act:
        if isinstance(m, list) and len(m) >= 3:
            parts.append(f"src={int(m[0])} ang={float(m[1]):.2f} ships={int(m[2])}")
    return "; ".join(parts) if parts else "(idle)"


def _planet_owners(step) -> dict[int, int]:
    obs = step[0]["observation"]
    return {int(p[0]): int(p[1]) for p in obs.get("planets", [])}


def _captures_at(env_steps, turn: int) -> list[tuple[int, int, int]]:
    """[(planet_id, prev_owner, new_owner), ...] that changed hands AT turn."""
    if turn == 0 or turn >= len(env_steps):
        return []
    prev = _planet_owners(env_steps[turn - 1])
    curr = _planet_owners(env_steps[turn])
    out = []
    for pid, owner in curr.items():
        po = prev.get(pid)
        if po is not None and po != owner:
            out.append((pid, po, owner))
    return out


def _ship_totals(step) -> dict[int, int]:
    obs = step[0]["observation"]
    by: dict[int, int] = {}
    for p in obs.get("planets", []):
        pid_owner = int(p[1])
        by[pid_owner] = by.get(pid_owner, 0) + int(p[5])
    # Fleets in flight.
    for f in obs.get("fleets", []):
        owner = int(f[1])
        by[owner] = by.get(owner, 0) + int(f[5])
    return by


def run_one(seed: int, agent_path: Path, opp_path: Path,
            tier: str, max_turns: int) -> dict:
    """Run a single game; return per-turn record + final."""
    from kaggle_environments import make

    # Crucial: env var must be set BEFORE the agent module is loaded,
    # because `compute_opp_trajectory` reads `BASELINE_OPP_TRAJ_TIER`
    # inside `agent()` at call time — but the agent module's import-time
    # side effects (os.environ.setdefault) shouldn't matter for this.
    # Belt-and-suspenders: set it now and restore.
    prev = os.environ.get("BASELINE_OPP_TRAJ_TIER", "")
    if tier == "off":
        os.environ.pop("BASELINE_OPP_TRAJ_TIER", None)
    else:
        os.environ["BASELINE_OPP_TRAJ_TIER"] = tier

    try:
        # Reload agent modules to pick up env var on first call.
        agent_a = _load_agent_module(agent_path, f"_under_test_{tier}")
        agent_b = _load_agent_module(opp_path, f"_opp_{tier}")

        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        # Run game to completion (or max_turns).
        env.run([agent_a, agent_b])
    finally:
        if prev:
            os.environ["BASELINE_OPP_TRAJ_TIER"] = prev
        else:
            os.environ.pop("BASELINE_OPP_TRAJ_TIER", None)

    steps = env.steps
    final = steps[-1]
    r0 = final[0]["reward"]
    r1 = final[1]["reward"]
    winner = 0 if r0 > r1 else (1 if r1 > r0 else -1)

    return {
        "tier": tier,
        "seed": seed,
        "steps": steps,
        "n_turns": len(steps),
        "winner": winner,
        "reward_0": r0,
        "reward_1": r1,
    }


def diff_runs(run_a: dict, run_b: dict, max_show: int = 8, verbose: bool = False):
    a_steps = run_a["steps"]
    b_steps = run_b["steps"]
    n = min(len(a_steps), len(b_steps))

    print(f"\n=== Side-by-side trace, seed={run_a['seed']} ===")
    print(f"Run A: tier={run_a['tier']!r}  turns={run_a['n_turns']}  "
          f"winner=P{run_a['winner']} (rewards {run_a['reward_0']:.1f} / {run_a['reward_1']:.1f})")
    print(f"Run B: tier={run_b['tier']!r}  turns={run_b['n_turns']}  "
          f"winner=P{run_b['winner']} (rewards {run_b['reward_0']:.1f} / {run_b['reward_1']:.1f})")

    # Find first divergent turn (P0's action differs).
    first_diff = None
    for t in range(1, n):
        a_act = a_steps[t][0].get("action") if 0 < len(a_steps[t]) else None
        b_act = b_steps[t][0].get("action") if 0 < len(b_steps[t]) else None
        if a_act != b_act:
            first_diff = t
            break
    print(f"\nFirst turn where P0's action diverges: "
          f"{first_diff if first_diff is not None else 'NEVER'}")

    if first_diff is not None:
        print(f"  Turn {first_diff} — Run A picked: {_actions_repr(a_steps, first_diff, 0)}")
        print(f"  Turn {first_diff} — Run B picked: {_actions_repr(b_steps, first_diff, 0)}")

    # Aggregate emit counts and ship totals across the whole game.
    a_emit, a_ships = 0, 0
    b_emit, b_ships = 0, 0
    a_caps, b_caps = 0, 0
    for t in range(1, len(a_steps)):
        n_, s_ = _summarise_turn_actions(a_steps, t, 0)
        a_emit += n_; a_ships += s_
        a_caps += sum(1 for (_, prev, new) in _captures_at(a_steps, t) if new == 0)
    for t in range(1, len(b_steps)):
        n_, s_ = _summarise_turn_actions(b_steps, t, 0)
        b_emit += n_; b_ships += s_
        b_caps += sum(1 for (_, prev, new) in _captures_at(b_steps, t) if new == 0)

    print(f"\nP0 total launches  — Run A: {a_emit} (ships {a_ships})   "
          f"Run B: {b_emit} (ships {b_ships})")
    print(f"P0 total captures  — Run A: {a_caps}   Run B: {b_caps}")

    a_final_ships = _ship_totals(a_steps[-1])
    b_final_ships = _ship_totals(b_steps[-1])
    print(f"\nFinal ship totals — Run A: P0={a_final_ships.get(0, 0)} P1={a_final_ships.get(1, 0)}")
    print(f"Final ship totals — Run B: P0={b_final_ships.get(0, 0)} P1={b_final_ships.get(1, 0)}")

    if verbose:
        print("\n--- Per-turn P0 actions (until divergence + max_show turns after) ---")
        end = min(n, (first_diff or 0) + max_show + 1)
        start = max(0, (first_diff or 0) - 2)
        for t in range(start, end):
            a_n, a_s = _summarise_turn_actions(a_steps, t, 0)
            b_n, b_s = _summarise_turn_actions(b_steps, t, 0)
            marker = " *" if (a_n, a_s) != (b_n, b_s) else "  "
            print(f"  turn {t:3d}{marker}  A: emit={a_n} ships={a_s}    B: emit={b_n} ships={b_s}")
            if a_n != b_n or a_s != b_s:
                print(f"           A act: {_actions_repr(a_steps, t, 0)}")
                print(f"           B act: {_actions_repr(b_steps, t, 0)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agent", type=str, default="agents/baseline/main.py",
                    help="Agent file path (the one we're refactoring)")
    ap.add_argument("--opp", type=str, default="agents/v1_orbitfix/main.py",
                    help="Opponent agent file path (held fixed across both runs)")
    ap.add_argument("--tier", type=str, default="lite",
                    choices=["lite", "topmix", "top"])
    ap.add_argument("--max-turns", type=int, default=500)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    agent_path = (REPO / args.agent).resolve()
    opp_path = (REPO / args.opp).resolve()
    if not agent_path.exists():
        raise SystemExit(f"agent not found: {agent_path}")
    if not opp_path.exists():
        raise SystemExit(f"opp not found: {opp_path}")

    print(f"Agent: {agent_path.name}    Opp: {opp_path.name}    Tier: {args.tier}")
    run_a = run_one(args.seed, agent_path, opp_path, tier="off", max_turns=args.max_turns)
    run_b = run_one(args.seed, agent_path, opp_path, tier=args.tier, max_turns=args.max_turns)
    diff_runs(run_a, run_b, verbose=args.verbose)


if __name__ == "__main__":
    main()
