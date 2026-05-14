"""tests/test_smoke.py — env + agent wiring smoke.

Runs in < 30 s. Fails loudly if:
  - kaggle_environments isn't installed
  - orbit_wars env isn't registered
  - main.py's agent doesn't return a valid move list
  - main.py can't beat `random` at least once in a fixed-seed game

Strategy tests live elsewhere as we build the agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_env_imports():
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    assert env is not None


def test_random_vs_random_runs():
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.run(["random", "random"])
    final = env.steps[-1]
    assert len(final) == 2
    assert all(s.status == "DONE" for s in final)


def test_my_agent_beats_random():
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": 7}, debug=False)
    env.run([str(REPO / "main.py"), "random"])
    final = env.steps[-1]
    assert final[0].status == "DONE" and final[1].status == "DONE"
    assert final[0].reward is not None and final[1].reward is not None
    assert final[0].reward > final[1].reward, (
        f"main.py should beat random on seed=7; got "
        f"{final[0].reward} vs {final[1].reward}"
    )
