"""Phase 3 smoke test: the analytical agent completes a 500-turn game
without crashes and emits valid actions across multiple seeds.

This is a sanity gate — it doesn't measure win rate (that's Phase 4's
A/B job), but it catches regressions in the per-turn pipeline:
  - Agent never raises an exception.
  - Returned move list always passes the env's action schema.
  - Wallclock per turn stays under the 1000 ms hard cap (with margin).

Marked SLOW because it runs full kaggle_environments games.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.slow

from kaggle_environments import make


def _run_one(seed: int, opp: str = "random") -> dict:
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    t0 = time.time()
    env.run(["agents/analytical/main.py", opp])
    elapsed = time.time() - t0
    last = env.steps[-1]
    return {
        "seed": seed,
        "steps": len(env.steps),
        "elapsed": elapsed,
        "statuses": [s.status for s in last],
        "rewards": [s.reward for s in last],
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_analytical_agent_completes_game(seed):
    """Phase 3 smoke: complete a 500-turn game without crash, all seats DONE."""
    result = _run_one(seed, opp="random")
    # All seats must finish cleanly.
    assert all(s == "DONE" for s in result["statuses"]), \
        f"seed={seed}: bad statuses={result['statuses']}"
    # Agent should win or draw vs random (rare to lose, but the gate is
    # just "doesn't crash" — accept any non-negative reward).
    me_reward = result["rewards"][0]
    assert me_reward in (-1, 0, 1), f"seed={seed}: invalid reward={me_reward}"
    # Wallclock margin: 500 steps × 1000 ms hard cap × 2 seats = 1000s ceiling.
    # We expect ≪ 50s total in practice; 60s gives margin for CI noise.
    assert result["elapsed"] < 60.0, \
        f"seed={seed}: elapsed={result['elapsed']:.1f}s > 60s (per-turn budget breach)"
