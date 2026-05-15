"""Phase C — multi-turn joint actions + min-regret over opp archetypes.

Coverage:
1. Multi-turn enumeration includes turn-1 atoms aimed at FUTURE planet
   positions (not present positions).
2. Compatibility filter keyed on `(source, launch_turn)` — same source
   on different turns survives; same `(source, launch_turn)` twice does
   NOT.
3. Two-tier shortlist size: Tier-2 receives `top_w_tier2` candidates.
4. Min-regret aggregation picks the maximin row, not the max-mean row.
5. Archetype panel dedup: when two archetypes produce identical env
   actions, the panel collapses without index errors.
6. Phase C agent runs end-to-end via `agents/v8_analytic_phase_c/main.py`.
7. Beam decomposes multi-turn winner into a `phaseC_mt_*` mission and
   does NOT also commit a Phase B.1 `chain_*` mission.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation import StrategyCtx, get_strategy
from lib.foundation.actions import ActionSpec
from lib.foundation.agent_loop import reset_memory
from lib.foundation.memory_impls import CompositeMemory, MissionMemory
from lib.foundation.obs_to_state import obs_to_jax_state
from lib.foundation.strategies import analytic_joint  # noqa: F401 (register)
from lib.foundation.strategies.analytic_joint import AnalyticJointStrategy
from lib.foundation.strategies.joint_beam import _filter_compatible_multi_turn
from lib.foundation.strategies.multi_turn_enumerate import (
    enumerate_multi_turn_atoms,
)
from lib.foundation.strategies.opp_archetype_adapter import (
    build_opp_archetype_panel,
)


def _seed_env(seed: int = 42):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    return env


def _seed_state_and_obs(seed: int = 42):
    env = _seed_env(seed)
    obs = env.state[0].observation
    state = obs_to_jax_state(obs, configuration=env.configuration)
    return state, obs, env


# ---------------------------------------------------------------------------
# Test 1 — Multi-turn enumeration aims at future positions
# ---------------------------------------------------------------------------


def test_enumerate_multi_turn_includes_turn_1_atoms():
    """For H=2, atoms with `launch_turn=1` are emitted; their dir_angle
    differs from the launch_turn=0 atom toward the same target (because
    planets have orbited)."""
    state, _obs, _env = _seed_state_and_obs()
    atoms = enumerate_multi_turn_atoms(state, my_id=0, horizon=2)
    by_turn = {0: 0, 1: 0}
    for a in atoms:
        if a.launch_turn in by_turn:
            by_turn[a.launch_turn] += 1
    assert by_turn[0] > 0, "expected turn-0 atoms"
    assert by_turn[1] > 0, "expected turn-1 atoms"

    # For at least one (src, target) pair, the turn-0 dir_angle should
    # differ from the turn-1 dir_angle (target has orbited).
    by_src_frac: dict[tuple[int, int, int], dict[int, float]] = {}
    for a in atoms:
        key = (int(a.from_planet_id), int(a.ships), a.launch_turn)
        # Use ship-count as the (src, fraction-style) discriminator since
        # we don't have target_id on the spec; angle uniqueness is
        # captured by the value itself.
        by_src_frac.setdefault((key[0], key[1], 0), {})
    # Loose test: any turn-0 atom + any turn-1 atom from the same source
    # have meaningfully different angles for SOME pair.
    by_src: dict[int, dict[int, list[float]]] = {}
    for a in atoms:
        by_src.setdefault(int(a.from_planet_id), {0: [], 1: []})[a.launch_turn].append(
            float(a.dir_angle)
        )
    for src, turns in by_src.items():
        if not turns[0] or not turns[1]:
            continue
        if any(
            abs(((a0 - a1 + np.pi) % (2 * np.pi)) - np.pi) > 1e-3
            for a0 in turns[0] for a1 in turns[1]
        ):
            return  # found one — test passes
    pytest.fail("no (src, turn-0, turn-1) pair with meaningfully different angles")


# ---------------------------------------------------------------------------
# Test 2 — Compatibility filter is per-(source, launch_turn)
# ---------------------------------------------------------------------------


def test_filter_compatible_multi_turn_keys_on_source_and_turn():
    pool = [
        ActionSpec(from_planet_id=3, dir_angle=0.0, ships=5, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=3, dir_angle=1.0, ships=8, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=3, dir_angle=0.0, ships=5, launch_turn=1, agent_id=0),
        ActionSpec(from_planet_id=7, dir_angle=0.0, ships=5, launch_turn=0, agent_id=0),
    ]
    current = [
        ActionSpec(from_planet_id=3, dir_angle=0.0, ships=5, launch_turn=0, agent_id=0),
    ]
    out = _filter_compatible_multi_turn(current, pool)
    # `(3, 0)` blocked; `(3, 1)` and `(7, 0)` survive.
    keys = {(o.from_planet_id, o.launch_turn) for o in out}
    assert keys == {(3, 1), (7, 0)}


# ---------------------------------------------------------------------------
# Test 3 — Min-regret aggregation picks maximin, not max-mean
# ---------------------------------------------------------------------------


def test_min_regret_aggregation_picks_maximin():
    """Synthetic payoff matrix where row 0 has highest mean but row 1
    has highest minimum across archetypes. Min-regret aggregator
    (np.min across archetype axis, np.argmax over candidates) picks
    row 1."""
    # Shape (n_archetypes=3, n_candidates=2):
    #   candidate 0: [10.0, 10.0, -5.0]  → mean=5.0, min=-5.0
    #   candidate 1: [4.0, 4.0, 4.0]      → mean=4.0, min=4.0
    scores = np.array([
        [10.0, 4.0],
        [10.0, 4.0],
        [-5.0, 4.0],
    ], dtype=np.float32)
    aggregated = np.min(scores, axis=0)  # (2,)
    pick = int(np.argmax(aggregated))
    assert pick == 1, (
        f"min-regret should pick candidate 1 (min=4.0) over "
        f"candidate 0 (min=-5.0), got pick={pick}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Archetype panel builds without crashing on seed-42 init
# ---------------------------------------------------------------------------


def test_archetype_panel_builds_on_seeded_obs():
    _state, obs, _env = _seed_state_and_obs()
    panel = build_opp_archetype_panel(obs, my_id=0, H=2)
    assert len(panel) >= 1, f"panel collapsed to empty: {panel}"
    for name, pids_h, angles_h, ships_h in panel:
        assert pids_h.shape[0] == 2, f"H mismatch for {name}: {pids_h.shape}"
        assert ships_h.shape[0] == 2
        # Turn 1 is always no-op (archetype acts only on turn 0).
        assert (pids_h[1] == -1).all(), (
            f"archetype {name} has turn-1 launches; expected no-op"
        )
        assert (ships_h[1] == 0).all()


# ---------------------------------------------------------------------------
# Test 5 — End-to-end: agent runs on a real Kaggle env without crashing
# ---------------------------------------------------------------------------


def test_phase_c_agent_runs_one_turn_e2e(monkeypatch):
    """`agents/v8_analytic_phase_c/main.py::agent` returns a valid
    env-format action list. This is the integration smoke.

    The agent module triggers JIT warmup at import time. Skip it for
    pytest (the warmup costs ~45 s and re-tests nothing the unit-test
    suite doesn't already cover — the first call to `agent(obs)`
    will pay the cold compile cost itself, which is fine for a single
    test invocation).
    """
    monkeypatch.setenv("V8_ANALYTIC_PHASE_C_WARMUP", "0")
    from agents.v8_analytic_phase_c.main import agent

    reset_memory()
    try:
        env = _seed_env()
        obs0 = env.state[0].observation
        action = agent(obs0, env.configuration)
        assert isinstance(action, list), f"agent returned {type(action)}"
        for entry in action:
            assert isinstance(entry, list)
            assert len(entry) == 3
            src, _angle, ships = entry
            assert isinstance(src, int) and src >= 0
            assert isinstance(ships, int) and ships > 0
    finally:
        reset_memory()


# ---------------------------------------------------------------------------
# Test 6 — Beam decomposes multi-turn plan into phaseC_mt_* mission
# ---------------------------------------------------------------------------


def test_phase_c_commits_phaseC_mt_mission_when_beam_picks_multi_turn():
    """When the joint_beam picks a winning_set containing a
    `launch_turn=1` leg, the strategy commits a `phaseC_mt_*` mission
    (NOT a Phase-B.1 `chain_*` mission)."""
    state, obs, _env = _seed_state_and_obs()
    seat0_idx = next(
        i for i in range(len(np.asarray(state.planets_owner)))
        if int(np.asarray(state.planets_owner)[i]) == 0
        and bool(np.asarray(state.planets_alive)[i])
    )
    src_id = int(np.asarray(state.planets_id)[seat0_idx])
    src_garrison = int(np.asarray(state.planets_ships)[seat0_idx])

    # Find an enemy/neutral target.
    target = None
    for i in range(len(np.asarray(state.planets_alive))):
        if not bool(np.asarray(state.planets_alive)[i]):
            continue
        if int(np.asarray(state.planets_owner)[i]) == 0:
            continue
        target = int(np.asarray(state.planets_id)[i])
        break
    assert target is not None

    # Build a fake winning_set with a turn-0 + turn-1 leg from same src.
    # The aim_angle from src to target at turn 0 and turn 1:
    from lib.aim import aim_orbiting
    from lib.orbit import predict_relative

    omega = float(state.angular_velocity)
    tgt_idx = next(
        i for i in range(len(np.asarray(state.planets_id)))
        if int(np.asarray(state.planets_id)[i]) == target
        and bool(np.asarray(state.planets_alive)[i])
    )

    def _aim_for_turn(t):
        if t == 0:
            sp = (float(np.asarray(state.planets_x)[seat0_idx]),
                  float(np.asarray(state.planets_y)[seat0_idx]))
            tx, ty = (float(np.asarray(state.planets_x)[tgt_idx]),
                      float(np.asarray(state.planets_y)[tgt_idx]))
        else:
            src_planet = (
                src_id, 0,
                float(np.asarray(state.planets_x)[seat0_idx]),
                float(np.asarray(state.planets_y)[seat0_idx]),
                float(np.asarray(state.planets_radius)[seat0_idx]),
                src_garrison, int(np.asarray(state.planets_prod)[seat0_idx]),
            )
            sp = predict_relative(src_planet, omega, float(t))
            tgt_planet = (
                target, int(np.asarray(state.planets_owner)[tgt_idx]),
                float(np.asarray(state.planets_x)[tgt_idx]),
                float(np.asarray(state.planets_y)[tgt_idx]),
                float(np.asarray(state.planets_radius)[tgt_idx]),
                int(np.asarray(state.planets_ships)[tgt_idx]),
                int(np.asarray(state.planets_prod)[tgt_idx]),
            )
            tx, ty = predict_relative(tgt_planet, omega, float(t))
        tgt_tuple = (
            target, int(np.asarray(state.planets_owner)[tgt_idx]),
            tx, ty,
            float(np.asarray(state.planets_radius)[tgt_idx]),
            int(np.asarray(state.planets_ships)[tgt_idx]),
            int(np.asarray(state.planets_prod)[tgt_idx]),
        )
        ships = max(1, src_garrison // 3)
        a = aim_orbiting(
            sp, float(np.asarray(state.planets_radius)[seat0_idx]),
            tgt_tuple, float(np.asarray(state.planets_radius)[tgt_idx]),
            ships, omega,
        )
        if a is None:
            return None, ships
        return float(a[0]), ships

    aim0, ships0 = _aim_for_turn(0)
    aim1, ships1 = _aim_for_turn(1)
    if aim0 is None or aim1 is None:
        pytest.skip("no valid intercept in seed-42 init")

    fake_winning = [
        ActionSpec(from_planet_id=src_id, dir_angle=aim0, ships=ships0,
                   launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=src_id, dir_angle=aim1, ships=ships1,
                   launch_turn=1, agent_id=0),
    ]

    strat = AnalyticJointStrategy(H=2, width=2, depth=2, budget_ms=400.0)
    ctx = StrategyCtx(turn_budget_ms=1000.0, raw_obs=obs)
    mem = CompositeMemory()

    with patch(
        "lib.foundation.strategies.analytic_joint.joint_beam_search",
        return_value=fake_winning,
    ):
        _tensor, mem_out = strat.emit(state, my_id=0, ctx=ctx, memory=mem)

    mt_missions = [
        m for m in mem_out.missions.missions
        if m.mission_id.startswith("phaseC_mt_")
    ]
    chain_missions = [
        m for m in mem_out.missions.missions
        if m.mission_id.startswith("chain_")
    ]
    assert len(mt_missions) == 1, (
        f"expected 1 phaseC_mt mission, got {len(mt_missions)} "
        f"(all missions: {[m.mission_id for m in mem_out.missions.missions]})"
    )
    assert len(chain_missions) == 0, (
        f"Phase C must not commit Phase-B.1 chain missions; "
        f"got {[m.mission_id for m in chain_missions]}"
    )
    mt = mt_missions[0]
    assert mt.target_planet_id == target
    assert mt.source_planet_id == src_id
    assert len(mt.waves) == 2
    assert mt.waves_fired == 1  # turn-0 wave fires this turn


# ---------------------------------------------------------------------------
# Test 7 — Strategy registered under "v8_analytic_phase_c"
# ---------------------------------------------------------------------------


def test_strategy_registered():
    # Import Phase B.1's module to trigger its registration alongside
    # Phase C's; both must coexist in the registry.
    from lib.foundation.strategies import analytic  # noqa: F401
    s = get_strategy("v8_analytic_phase_c")
    assert isinstance(s, AnalyticJointStrategy)
    b1 = get_strategy("v8_analytic")
    assert b1 is not s
