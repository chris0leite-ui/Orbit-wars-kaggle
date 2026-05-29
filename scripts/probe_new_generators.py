"""Phase-0 trace probe — count what _enumerate_redeploy_candidates and
_enumerate_gang_up_support_candidates emit per turn in one game.

Usage:
    PROPOSER_REDEPLOY=on PROPOSER_GANG_UP_SUPPORT=on \\
        BASELINE_VALUE_HEAD=hybrid_spatial \\
        python scripts/probe_new_generators.py [SEED]

If 0 redeploy candidates emit across the game, loosen
BASELINE_REDEPLOY_DIST_RATIO and re-probe. If 0 gang-up events detected,
re-examine post-bounce detection (model.owner_at).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make repo importable when run from anywhere.
_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Force the env vars ON if the user forgot — this is a probe, not an A/B.
os.environ.setdefault("PROPOSER_REDEPLOY", "on")
os.environ.setdefault("PROPOSER_GANG_UP_SUPPORT", "on")
os.environ.setdefault("BASELINE_VALUE_HEAD", "hybrid_spatial")

import agents.baseline.proposer as P  # noqa: E402
from kaggle_environments import make  # noqa: E402

# Monkey-patch the two generators to count emissions per call.
_orig_redeploy = P._enumerate_redeploy_candidates
_orig_gangup = P._enumerate_gang_up_support_candidates

_per_turn: list[dict] = []
_call_state = {"current_turn": -1}


def _counting_redeploy(*args, **kwargs):
    out = _orig_redeploy(*args, **kwargs)
    if _per_turn and _per_turn[-1]["step"] == _call_state["current_turn"]:
        _per_turn[-1]["redeploy"] += len(out)
    else:
        _per_turn.append({
            "step": _call_state["current_turn"],
            "redeploy": len(out),
            "gangup": 0,
        })
    return out


def _counting_gangup(*args, **kwargs):
    out = _orig_gangup(*args, **kwargs)
    if _per_turn and _per_turn[-1]["step"] == _call_state["current_turn"]:
        _per_turn[-1]["gangup"] += len(out)
    else:
        _per_turn.append({
            "step": _call_state["current_turn"],
            "redeploy": 0,
            "gangup": len(out),
        })
    return out


P._enumerate_redeploy_candidates = _counting_redeploy
P._enumerate_gang_up_support_candidates = _counting_gangup

# Wrap the agent's act() to track the current step.
import agents.baseline.main as M  # noqa: E402
_orig_agent = M.agent


def _tracking_agent(obs, configuration):
    step = int(obs.get("step", 0))
    _call_state["current_turn"] = step
    return _orig_agent(obs, configuration)


M.agent = _tracking_agent


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    print(f"== probe seed={seed} env={dict((k, os.environ.get(k)) for k in ['PROPOSER_REDEPLOY','PROPOSER_GANG_UP_SUPPORT','BASELINE_VALUE_HEAD'])} ==")
    # Use the bundled SEU7P file for opp; same code path otherwise. Just self-play.
    env.run(["agents/baseline/main.py", "agents/baseline/main.py"])
    n_steps = len(env.steps)
    print(f"\n== game finished: {n_steps} steps ==\n")

    total_redeploy = sum(e["redeploy"] for e in _per_turn)
    total_gangup = sum(e["gangup"] for e in _per_turn)
    turns_with_redeploy = sum(1 for e in _per_turn if e["redeploy"] > 0)
    turns_with_gangup = sum(1 for e in _per_turn if e["gangup"] > 0)

    print(f"   TOTAL redeploy candidates emitted: {total_redeploy} (across {turns_with_redeploy} turns)")
    print(f"   TOTAL gang-up support candidates : {total_gangup} (across {turns_with_gangup} turns)")
    print()

    # Print first 10 non-empty turns
    nonempty = [e for e in _per_turn if e["redeploy"] > 0 or e["gangup"] > 0]
    if nonempty:
        print(f"   First {min(10, len(nonempty))} turns with non-zero emissions:")
        for e in nonempty[:10]:
            print(f"      step={e['step']:>3}  redeploy={e['redeploy']:>2}  gangup={e['gangup']:>2}")
    else:
        print("   NO TURNS produced any candidates — investigate eligibility.")

    # Count launches per game from env.steps for orientation.
    n_launches_p0 = 0
    n_launches_p1 = 0
    for i, step_state in enumerate(env.steps):
        if i == 0:
            continue
        # Step's action for each seat
        for seat in (0, 1):
            actions = step_state[seat].get("action", None) or []
            try:
                n = len(actions)
            except TypeError:
                n = 0
            if seat == 0:
                n_launches_p0 += n
            else:
                n_launches_p1 += n
    print(f"\n   Launches P0={n_launches_p0}  P1={n_launches_p1}  total={n_launches_p0 + n_launches_p1}")


if __name__ == "__main__":
    main()
