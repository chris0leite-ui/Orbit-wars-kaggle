"""inv_decision_diff.py — does the opponent model actually change our move?

The inverse-producer (opp_projection ON) ties the static producer in short
head-to-head games. Hypothesis: against a producer the projected opponent
launches rarely change OUR best action in the uncontested phase, so the
mechanism is a near-no-op early and only bites once territories collide.

This isolates the DECISION-level effect from trajectory divergence:

  1. record  — play static-vs-static on a seed, pickle player-0's observation
               at every turn (a realistic obs trajectory).
  2. decide  — replay that SAME obs trajectory through one producer_plus
               runtime (opp_projection ON or OFF, chosen by env) and pickle
               the chosen action each turn. Memory continuity is preserved
               within the process; both runs see identical observations.
  3. diff    — compare the two action streams turn-by-turn: how many turns
               differ, when the first difference appears, and the per-turn
               total-ships-launched delta.

Run via the driver at the bottom (no args) or the three sub-modes.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROD = REPO / "agents" / "producer"
PLUS = REPO / "agents" / "producer_plus"
TRAJ = Path("/tmp/inv/obs_traj.pkl")
ACT_ON = Path("/tmp/inv/act_on.pkl")
ACT_OFF = Path("/tmp/inv/act_off.pkl")


def _load_static_agent(tag: str = "a"):
    """The static producer (bare producer_plus, opp_projection OFF).

    `tag` namespaces the module so two independent instances can coexist —
    producer_plus keeps a module-global runtime/memory, so the two seats of a
    self-play game MUST load under different module names or they share fleet
    tracking and corrupt the game.
    """
    import importlib.util
    name = f"_pp_static_{tag}"
    spec = importlib.util.spec_from_file_location(
        name, str(PLUS / "producer_agent.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m.agent


def mode_record(seed: int, steps: int):
    from kaggle_environments import make
    static_p0 = _load_static_agent("p0")
    static_p1 = _load_static_agent("p1")   # independent instance for seat 1
    obs_stream = []

    def rec(obs, conf=None):
        # snapshot the dict the agent receives (player 0's view)
        obs_stream.append({k: obs[k] for k in obs.keys()} if hasattr(obs, "keys") else dict(obs))
        return static_p0(obs)
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": steps}, debug=False)
    env.run([rec, static_p1])
    TRAJ.write_bytes(pickle.dumps(obs_stream))
    print(f"recorded {len(obs_stream)} turns -> {TRAJ}")


def mode_decide(out_path: str):
    """Replay the recorded obs trajectory through ONE producer_plus runtime.
    opp_projection is controlled by PRODUCER_PLUS_OPP_PROJECTION in the env
    set by the caller BEFORE this process starts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_pp_decide", str(PLUS / "producer_agent.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_pp_decide"] = m
    spec.loader.exec_module(m)
    agent = m.agent
    obs_stream = pickle.loads(TRAJ.read_bytes())
    actions = []
    for obs in obs_stream:
        actions.append(agent(obs))
    Path(out_path).write_bytes(pickle.dumps(actions))
    print(f"decided {len(actions)} turns -> {out_path}")


def _ships_launched(action) -> float:
    if not action:
        return 0.0
    return float(sum(mv[2] for mv in action))


def mode_diff():
    on = pickle.loads(ACT_ON.read_bytes())
    off = pickle.loads(ACT_OFF.read_bytes())
    n = min(len(on), len(off))
    first_diff = None
    n_diff = 0
    rows = []
    for t in range(n):
        a_on, a_off = on[t], off[t]
        same = (a_on == a_off)
        if not same:
            n_diff += 1
            if first_diff is None:
                first_diff = t
        rows.append((t, _ships_launched(a_off), _ships_launched(a_on), same))
    print(f"\n== decision diff: opp_projection ON vs OFF, same obs stream ==")
    print(f"   turns compared: {n}")
    print(f"   turns where the chosen action DIFFERED: {n_diff} ({100*n_diff/n:.0f}%)")
    print(f"   first differing turn: {first_diff}")
    print(f"\n   {'turn':>4s} {'ships_off':>9s} {'ships_on':>9s} {'changed':>8s}")
    for t, soff, son, same in rows:
        if not same or t % 20 == 0:
            print(f"   {t:>4d} {soff:>9.0f} {son:>9.0f} {('' if same else 'DIFF'):>8s}")


def driver(seed: int, steps: int):
    print(f"== inv_decision_diff seed={seed} steps={steps} ==")
    env = dict(os.environ)
    # 1. record
    subprocess.run([sys.executable, __file__, "record", str(seed), str(steps)],
                   check=True, env=env)
    # 2a. decide OFF
    env_off = dict(env); env_off.pop("PRODUCER_PLUS_OPP_PROJECTION", None)
    subprocess.run([sys.executable, __file__, "decide", str(ACT_OFF)],
                   check=True, env=env_off)
    # 2b. decide ON
    env_on = dict(env); env_on["PRODUCER_PLUS_OPP_PROJECTION"] = "1"
    subprocess.run([sys.executable, __file__, "decide", str(ACT_ON)],
                   check=True, env=env_on)
    # 3. diff
    mode_diff()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver(seed=0, steps=200)
    elif sys.argv[1] == "record":
        mode_record(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "decide":
        mode_decide(sys.argv[2])
    elif sys.argv[1] == "diff":
        mode_diff()
    else:
        driver(seed=int(sys.argv[1]), steps=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
