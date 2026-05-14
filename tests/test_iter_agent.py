"""Tests for the iter fast-iteration scaffold.

The iter agent (`agents/iter/main.py`) is a thin fork of v7_pv. Two
invariants we guard:

1. The module exports a callable `agent` and completes a 1-game smoke
   against a stock kaggle_environments opponent without crashing.
2. Importing `agents.iter.main` enforces `lib.scoring.PV_GAMMA == 0.99`,
   which is what makes day-zero iter functionally equivalent to v7_pv.
   v7_pv was never a committed source file (it was a bundled artifact
   produced with `scripts/ab_variants.py --variant pv PV_GAMMA=0.99`),
   so the parity contract lives in this PV_GAMMA invariant plus
   `tests/test_jax_pv_horizon_parity.py` which verifies the PV math.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture()
def iter_agent_module():
    # Force a fresh import: PV_GAMMA override only fires on first load.
    for name in list(sys.modules):
        if name.startswith("agents.iter") or name == "lib.scoring":
            del sys.modules[name]
    mod = importlib.import_module("agents.iter.main")
    return mod


def test_iter_agent_callable(iter_agent_module):
    assert callable(iter_agent_module.agent)


def test_iter_pv_gamma_is_enforced(iter_agent_module):
    import lib.scoring as scoring
    assert scoring.PV_GAMMA == 0.99, (
        f"iter must enforce PV_GAMMA=0.99 (v7_pv parity); got {scoring.PV_GAMMA}"
    )
    # Also assert the agent module's own constant matches.
    assert iter_agent_module.PV_GAMMA == 0.99


def test_iter_knob_constants_present(iter_agent_module):
    # If any knob name disappears, downstream sweep scripts will need
    # updating — surface it here instead of via silent agent regression.
    for knob in ("K", "WALLCLOCK_MS", "ENUMERATOR_MODE", "OPP_TIERS",
                 "PV_GAMMA", "VALUE_FN", "TERRITORY_WEIGHT", "K_4P"):
        assert hasattr(iter_agent_module, knob), f"missing knob: {knob}"


def test_iter_smoke_one_game_completes(iter_agent_module):
    # 1-game self-play smoke (iter vs the kaggle_environments `random`
    # builtin). Cheap — should run in <30 s on local CPU.
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    env.run([iter_agent_module.agent, "random"])
    assert env.done, "env did not reach DONE"
    statuses = [s["status"] for s in env.steps[-1]]
    assert all(s in {"DONE", "INACTIVE"} for s in statuses), (
        f"unexpected terminal statuses: {statuses}"
    )


def test_iter_smoke_4p_game_completes(iter_agent_module):
    # 1-game 4P smoke. Proves iter actually runs in 4P after the choose_with_4p
    # dispatch swap — pre-swap iter would silently fall back to the v3.5.1
    # incumbent. Action-variance check ensures we're not stuck on a single
    # fixed incumbent action every turn.
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"actTimeout": 2.5})
    # Players-4 is the default 4P mode for orbit_wars; passing 4 explicit
    # agents auto-configures the 4-seat board.
    env.run([iter_agent_module.agent, "random", "random", "random"])
    assert env.done, "4P env did not reach DONE"
    # Action-variance check: collect iter's actions across the first 20
    # turns and confirm at least 2 distinct action shapes. If we always
    # returned the same action, we're probably stuck on a fallback.
    actions_p0 = []
    for step_record in env.steps[:20]:
        a = step_record[0].get("action")
        if a is not None:
            actions_p0.append(tuple(tuple(x) for x in a) if a else ())
    distinct = set(actions_p0)
    assert len(distinct) >= 2, (
        f"iter returned only {len(distinct)} distinct action(s) across 20 turns; "
        f"likely fell to incumbent fallback. Actions: {actions_p0[:5]}"
    )


def test_territory_value_runs_under_5ms(iter_agent_module):
    # Bench gate: territory head must not blow the per-call budget. value_fn
    # is invoked once per candidate at rollout leaf; ~50 candidates per turn
    # × 5 ms each = 250 ms, leaving 450 ms headroom inside the 700 ms knob.
    import time
    from kaggle_environments import make
    from lib.value_heads import territory_value

    env = make("orbit_wars", debug=False)
    env.reset(2)
    # Roll forward 30 turns with iter on P0 so the obs has real ownership /
    # in-flight fleet state — neutral starting obs returns 0 trivially.
    for _ in range(30):
        if env.done:
            break
        env.step([iter_agent_module.agent(env.state[0]["observation"], env.configuration),
                  None])
    obs = env.state[0]["observation"]

    t0 = time.perf_counter()
    for _ in range(20):
        territory_value(obs, 0, weight=0.01)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 20
    assert avg_ms < 5.0, f"territory_value too slow: {avg_ms:.2f} ms/call"
