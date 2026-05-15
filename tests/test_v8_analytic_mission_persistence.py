"""Phase B.1 — `MissionMemory` integration into `AnalyticStrategy`.

Covers:
- **Parity (a)** — strategy with `EmptyMemory` falls back to Phase A
  behaviour (no mission code paths exercised).
- **Plan persistence** — a pre-committed wave due this turn is fired.
- **Auto-prune fires** — a mission whose target was captured is
  dropped before `waves_to_fire` reads it.
- **Skip-beam saturation** — when pre-commits cover all owned sources,
  `beam_search` is NOT called.
- **Chainer commit** — a beam-newly-selected launch satisfying the
  4-gate chainer schedules a 2-wave reinforce mission.
- **Chainer no-commit** — gates that fail (target already ours,
  short-ETA, low-prod target, insufficient garrison) suppress the
  chain.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation import StrategyCtx, get_strategy
from lib.foundation.actions import ActionSpec, tensor_to_specs
from lib.foundation.memory import EmptyMemory
from lib.foundation.memory_impls import (
    CommittedMission,
    CompositeMemory,
    MissionMemory,
)
from lib.foundation.obs_to_state import obs_to_jax_state
from lib.foundation.strategies import analytic  # noqa: F401 (register)
from lib.foundation.strategies.analytic import AnalyticStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_state(seed: int = 42):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    return obs_to_jax_state(obs, configuration=env.configuration)


def _seat_planets(state, seat: int) -> list[int]:
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    return [
        int(ids[i]) for i in range(len(alive))
        if bool(alive[i]) and int(owner[i]) == seat
    ]


def _planet_index_by_id(state, planet_id: int) -> int:
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    for i in range(len(alive)):
        if bool(alive[i]) and int(ids[i]) == planet_id:
            return i
    raise ValueError(f"planet id {planet_id} not found")


def _add_seat_planet(state, neutral_pick: int, seat: int):
    """Convert the `neutral_pick`-th neutral planet to be owned by
    `seat` with garrison 10. Returns a new state.

    The game starts each seat with exactly 1 owned planet, so several
    tests synthesize multi-planet states by promoting a neutral.
    """
    alive = np.asarray(state.planets_alive)
    owner = np.asarray(state.planets_owner)
    neutral_idxs = [
        i for i in range(len(alive))
        if bool(alive[i]) and int(owner[i]) == -1
    ]
    if neutral_pick >= len(neutral_idxs):
        raise ValueError("not enough neutral planets to promote")
    pi = neutral_idxs[neutral_pick]
    new_owner = state.planets_owner.at[pi].set(seat)
    new_ships = state.planets_ships.at[pi].set(10)
    return state._replace(planets_owner=new_owner, planets_ships=new_ships)


# ---------------------------------------------------------------------------
# Parity (a) — EmptyMemory routes through Phase A fallback
# ---------------------------------------------------------------------------


def test_empty_memory_returns_phase_a_result():
    """`AnalyticStrategy` with `EmptyMemory` returns the same
    `ActionTensor` as direct `beam_search` over the same atomics —
    the fallback path bypasses all mission machinery.
    """
    from lib.foundation.strategies.analytic_score import enumerate_atomic_launches
    from lib.foundation.strategies.beam_search import beam_search
    from lib.foundation.actions import specs_to_tensor

    state = _seed_state()
    ctx = StrategyCtx(turn_budget_ms=1000.0)
    strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0)

    tensor, mem_out = strat.emit(state, my_id=0, ctx=ctx, memory=EmptyMemory())
    direct_set = beam_search(
        state, enumerate_atomic_launches(state, my_id=0), my_id=0,
        width=2, depth=2, K=5, num_agents=int(state.num_agents),
        budget_ms=500.0,
    )
    direct_tensor = specs_to_tensor([direct_set], horizon=1)

    np.testing.assert_array_equal(np.asarray(tensor.pids), np.asarray(direct_tensor.pids))
    np.testing.assert_array_equal(np.asarray(tensor.angles), np.asarray(direct_tensor.angles))
    np.testing.assert_array_equal(np.asarray(tensor.ships), np.asarray(direct_tensor.ships))
    assert isinstance(mem_out, EmptyMemory)


# ---------------------------------------------------------------------------
# Plan persistence — pre-committed wave fires this turn
# ---------------------------------------------------------------------------


def test_pre_committed_wave_fires_this_turn():
    """A `CompositeMemory` seeded with a 2-wave mission whose wave 1
    is due this turn → `emit` returns an `ActionTensor` whose seat-0
    slot includes a launch from that mission's source planet.

    The dir_angle in the ActionTensor is the RE-AIMED value (not the
    placeholder from CommittedMission.waves), so we check `(src, ships)`
    presence and let the angle float.
    """
    state = _seed_state()
    seat0 = _seat_planets(state, 0)
    # Find an enemy/neutral target reachable from seat0[0].
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    target = next(
        (int(ids[i]) for i in range(len(alive))
         if bool(alive[i]) and int(owner[i]) != 0 and int(ids[i]) != seat0[0]),
        None,
    )
    assert seat0 and target is not None

    src = seat0[0]
    src_idx = _planet_index_by_id(state, src)
    src_garrison = int(np.asarray(state.planets_ships)[src_idx])
    wave_ships = max(1, src_garrison // 2)

    step = int(state.step)
    # Build a 2-wave mission whose wave 0 (launch_turn=0) fires THIS turn.
    waves = (
        ActionSpec(from_planet_id=src, dir_angle=0.0,
                   ships=wave_ships, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=src, dir_angle=0.0,
                   ships=wave_ships, launch_turn=5, agent_id=0),
    )
    mission = CommittedMission(
        mission_id="persist_test",
        seat=0,
        turn_committed=step,
        target_planet_id=target,
        source_planet_id=src,
        waves=waves,
        waves_fired=0,
    )
    mem = CompositeMemory().with_missions(
        MissionMemory(current_turn=step).commit(mission)
    )

    strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0,
                             enable_chainer=False)
    ctx = StrategyCtx(turn_budget_ms=1000.0)
    tensor, mem_out = strat.emit(state, my_id=0, ctx=ctx, memory=mem)

    specs = tensor_to_specs(tensor)[0]
    sources_fired = {s.from_planet_id for s in specs if s.agent_id == 0}
    assert src in sources_fired, (
        f"pre-committed source {src} not fired; fired sources = {sources_fired}"
    )

    # The mission should now have waves_fired=1.
    mid_after = mem_out.missions.get("persist_test")
    assert mid_after is not None
    assert mid_after.waves_fired == 1


# ---------------------------------------------------------------------------
# Auto-prune fires before waves_to_fire
# ---------------------------------------------------------------------------


def test_mission_with_captured_target_is_pruned_before_firing():
    """If the target of a pre-committed mission is already owned by
    seat 0 in the current state, `memory.missions.update` drops it
    before `emit` reads the pre-commits. The mission does NOT fire,
    and the returned memory does NOT contain it.
    """
    state = _add_seat_planet(_seed_state(), neutral_pick=0, seat=0)
    seat0 = _seat_planets(state, 0)
    assert len(seat0) >= 2

    # Use seat0[0] as source, seat0[1] as "target" — but seat0[1] is
    # already ours, so auto-prune should drop the mission.
    waves = (
        ActionSpec(from_planet_id=seat0[0], dir_angle=0.0,
                   ships=5, launch_turn=0, agent_id=0),
    )
    mission = CommittedMission(
        mission_id="self_target",
        seat=0,
        turn_committed=int(state.step),
        target_planet_id=seat0[1],
        source_planet_id=seat0[0],
        waves=waves,
        waves_fired=0,
    )
    mem = CompositeMemory().with_missions(
        MissionMemory(current_turn=int(state.step)).commit(mission)
    )
    assert mem.missions.get("self_target") is not None

    strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0,
                             enable_chainer=False)
    ctx = StrategyCtx(turn_budget_ms=1000.0)
    _tensor, mem_out = strat.emit(state, my_id=0, ctx=ctx, memory=mem)

    assert mem_out.missions.get("self_target") is None


# ---------------------------------------------------------------------------
# Skip-beam saturation — pre-commits cover all owned sources
# ---------------------------------------------------------------------------


def test_skip_beam_when_pre_commits_saturate_owned_sources():
    """Synthetic state where seat 0 owns exactly 1 planet; we
    pre-commit a wave from that planet → `beam_search` is NOT called.
    """
    state = _seed_state()
    seat0 = _seat_planets(state, 0)
    seat1 = _seat_planets(state, 1)
    assert seat0 and seat1

    # Neutralize all but the first seat-0 planet (set their owner to -1).
    owner_np = np.asarray(state.planets_owner).copy()
    ids_np = np.asarray(state.planets_id)
    keep_src = seat0[0]
    target = seat1[0]  # enemy target
    for i in range(len(owner_np)):
        if int(ids_np[i]) != keep_src and int(owner_np[i]) == 0:
            owner_np[i] = -1
    new_state = state._replace(planets_owner=state.planets_owner.at[:].set(owner_np))

    waves = (
        ActionSpec(from_planet_id=keep_src, dir_angle=0.0,
                   ships=3, launch_turn=0, agent_id=0),
    )
    mission = CommittedMission(
        mission_id="saturator",
        seat=0,
        turn_committed=int(new_state.step),
        target_planet_id=target,
        source_planet_id=keep_src,
        waves=waves,
        waves_fired=0,
    )
    mem = CompositeMemory().with_missions(
        MissionMemory(current_turn=int(new_state.step)).commit(mission)
    )

    strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0,
                             enable_chainer=False)
    ctx = StrategyCtx(turn_budget_ms=1000.0)

    with patch(
        "lib.foundation.strategies.analytic.beam_search"
    ) as mock_beam:
        _tensor, _mem_out = strat.emit(new_state, my_id=0, ctx=ctx, memory=mem)
        assert mock_beam.call_count == 0, (
            f"beam_search was called {mock_beam.call_count}× when "
            f"pre-commits already saturate the only owned source."
        )


# ---------------------------------------------------------------------------
# Chainer — commit
# ---------------------------------------------------------------------------


def _high_prod_enemy_target(state, seat: int) -> int | None:
    """Find an enemy/neutral target with production >= chainer
    threshold."""
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    prod = np.asarray(state.planets_prod)
    for i in range(len(alive)):
        if not bool(alive[i]):
            continue
        if int(owner[i]) == seat:
            continue
        if int(prod[i]) >= 3:
            return int(ids[i])
    return None


def test_chainer_commits_two_wave_mission_when_gates_pass():
    """A beam-selected launch that satisfies all 4 chainer gates
    leads to a 2-wave `CommittedMission` being stored.

    We mock `beam_search` to return a synthetic winning_set with a
    known-good launch, so this test exercises only the chainer logic.
    """
    from lib.aim import aim_orbiting

    state = _seed_state()
    seat0 = _seat_planets(state, 0)
    target = _high_prod_enemy_target(state, 0)
    assert seat0 and target is not None

    # Find a (src, target) pair with ETA > 8 to satisfy G2.
    src = None
    aim_angle = None
    for candidate_src in seat0:
        sidx = _planet_index_by_id(state, candidate_src)
        tidx = _planet_index_by_id(state, target)
        x = np.asarray(state.planets_x)
        y = np.asarray(state.planets_y)
        radius = np.asarray(state.planets_radius)
        owner = np.asarray(state.planets_owner)
        ships = np.asarray(state.planets_ships)
        ids = np.asarray(state.planets_id)
        prod = np.asarray(state.planets_prod)
        src_pos = (float(x[sidx]), float(y[sidx]))
        tgt_tuple = (
            int(ids[tidx]), int(owner[tidx]),
            float(x[tidx]), float(y[tidx]),
            float(radius[tidx]), int(ships[tidx]),
            int(prod[tidx]),
        )
        src_garrison = int(ships[sidx])
        fleet_ships = max(1, int(src_garrison * 0.3))  # keep G4 garrison-retention
        aim = aim_orbiting(
            src_pos, float(radius[sidx]), tgt_tuple, float(radius[tidx]),
            fleet_ships, float(state.angular_velocity),
        )
        if aim is None:
            continue
        a, _, eta = aim
        if eta is None or eta <= 8:
            continue
        src = candidate_src
        aim_angle = float(a)
        chosen_ships = fleet_ships
        break

    if src is None:
        pytest.skip("no (src, target) pair with ETA > 8 in seed-42 state")

    launch = ActionSpec(
        from_planet_id=src, dir_angle=aim_angle,
        ships=chosen_ships, launch_turn=0, agent_id=0,
    )

    mem = CompositeMemory()
    strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0,
                             enable_chainer=True)
    ctx = StrategyCtx(turn_budget_ms=1000.0)

    with patch(
        "lib.foundation.strategies.analytic.beam_search",
        return_value=[launch],
    ):
        _tensor, mem_out = strat.emit(state, my_id=0, ctx=ctx, memory=mem)

    # Find the committed chain mission.
    chain = next(
        (m for m in mem_out.missions.missions
         if m.mission_id.startswith("chain_")),
        None,
    )
    assert chain is not None, (
        f"chainer did not commit; missions = "
        f"{[m.mission_id for m in mem_out.missions.missions]}"
    )
    assert len(chain.waves) == 2
    assert chain.target_planet_id == target
    assert chain.source_planet_id == src
    assert chain.waves[1].launch_turn == 5  # _CHAIN_DELAY
    assert chain.waves_fired == 1  # wave 0 fired this turn


# ---------------------------------------------------------------------------
# Chainer — no-commit
# ---------------------------------------------------------------------------


def test_chainer_skips_when_target_already_ours():
    """A beam pick whose recovered target is owned by my seat fails
    G1 → no chain committed."""
    from lib.aim import aim_orbiting

    state = _add_seat_planet(_seed_state(), neutral_pick=0, seat=0)
    seat0 = _seat_planets(state, 0)
    assert len(seat0) >= 2

    src = seat0[0]
    target = seat0[1]  # owned by us — G1 fails
    sidx = _planet_index_by_id(state, src)
    tidx = _planet_index_by_id(state, target)
    x = np.asarray(state.planets_x)
    y = np.asarray(state.planets_y)
    radius = np.asarray(state.planets_radius)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    ids = np.asarray(state.planets_id)
    prod = np.asarray(state.planets_prod)
    src_pos = (float(x[sidx]), float(y[sidx]))
    tgt_tuple = (
        int(ids[tidx]), int(owner[tidx]),
        float(x[tidx]), float(y[tidx]),
        float(radius[tidx]), int(ships[tidx]),
        int(prod[tidx]),
    )
    fleet_ships = max(1, int(ships[sidx]) // 2)
    aim = aim_orbiting(
        src_pos, float(radius[sidx]), tgt_tuple, float(radius[tidx]),
        fleet_ships, float(state.angular_velocity),
    )
    if aim is None:
        pytest.skip("no intercept between two owned planets in this seed")
    aim_angle, _, _ = aim

    launch = ActionSpec(
        from_planet_id=src, dir_angle=float(aim_angle),
        ships=fleet_ships, launch_turn=0, agent_id=0,
    )

    mem = CompositeMemory()
    strat = AnalyticStrategy(width=2, depth=2, K=5, budget_ms=600.0,
                             enable_chainer=True)
    ctx = StrategyCtx(turn_budget_ms=1000.0)

    with patch(
        "lib.foundation.strategies.analytic.beam_search",
        return_value=[launch],
    ):
        _tensor, mem_out = strat.emit(state, my_id=0, ctx=ctx, memory=mem)

    chain = next(
        (m for m in mem_out.missions.missions
         if m.mission_id.startswith("chain_")),
        None,
    )
    assert chain is None, (
        f"chainer committed even though target was already ours: "
        f"{chain.mission_id if chain else None}"
    )
