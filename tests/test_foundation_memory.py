"""Step 7 — cross-turn memory tests.

Coverage:
- `JitCacheMemory.get_or_compile` runs the builder once and caches.
- `MissionMemory.commit`, `mark_wave_fired`, `advance_turn`,
  `waves_to_fire`, `update` (state-driven pruning).
- `CompositeMemory.update` threads to sub-memories.
- `agent_loop.get_memory / set_memory / reset_memory /
  maybe_reset_on_new_game` correctly persist & reset state across
  simulated turn-call sequences.

The plan's Step-7d test for a 3-wave snipe persistence is the
end-to-end gold standard; it's exercised here against a synthesised
GameState (`_build_state`) without invoking a full strategy
(strategies land in Step 8).
"""

from __future__ import annotations

import numpy as np
import pytest
from kaggle_environments import make

from lib.foundation.actions import ActionSpec
from lib.foundation import agent_loop
from lib.foundation.memory_impls import (
    CommittedMission,
    CompositeMemory,
    JitCacheMemory,
    MissionMemory,
    _planet_owner,
)
from lib.game.jax.conversions import scalar_to_jax


def _build_state(seed: int = 42, num_agents: int = 2):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_agents)
    return scalar_to_jax(env.state, env.info["seed"])


# ---------------------------------------------------------------------------
# JitCacheMemory
# ---------------------------------------------------------------------------


def test_jit_cache_builds_once():
    """The builder is invoked at most once per key, even across many
    `get_or_compile` calls."""
    mem = JitCacheMemory()
    call_count = {"n": 0}

    def builder():
        call_count["n"] += 1
        return "result"

    assert mem.get_or_compile("k", builder) == "result"
    assert mem.get_or_compile("k", builder) == "result"
    assert mem.get_or_compile("k", builder) == "result"
    assert call_count["n"] == 1


def test_jit_cache_isolates_keys():
    mem = JitCacheMemory()
    assert mem.get_or_compile("a", lambda: 1) == 1
    assert mem.get_or_compile("b", lambda: 2) == 2
    assert mem.get_or_compile("a", lambda: 99) == 1  # cached
    assert len(mem) == 2


def test_jit_cache_update_is_identity():
    """JIT cache is state-independent; update returns self."""
    mem = JitCacheMemory()
    mem.get_or_compile("k", lambda: "v")
    after = mem.update(state=None, action_taken=None)
    assert after is mem


def test_jit_cache_reset_clears():
    mem = JitCacheMemory()
    mem.get_or_compile("k", lambda: "v")
    fresh = mem.reset()
    assert len(fresh) == 0
    assert len(mem) == 1  # original untouched (immutability via reset)


def test_jit_cache_persists_across_turns_real():
    """End-to-end: a builder that uses time as a proxy for compile
    cost — second `get_or_compile` should be faster (or at least
    same) than the first because cached."""
    import time

    mem = JitCacheMemory()

    def slow_builder():
        time.sleep(0.05)  # simulate JIT compile
        return "compiled"

    t0 = time.perf_counter()
    mem.get_or_compile("fn", slow_builder)
    t1 = time.perf_counter()
    mem.get_or_compile("fn", slow_builder)
    t2 = time.perf_counter()
    first_call = t1 - t0
    second_call = t2 - t1
    # Second call must be at LEAST 5× faster (no sleep happened).
    assert second_call * 5 < first_call, (
        f"first={first_call*1000:.1f}ms second={second_call*1000:.1f}ms — "
        f"cache miss?"
    )


# ---------------------------------------------------------------------------
# MissionMemory — basic API
# ---------------------------------------------------------------------------


def _mk_mission(mid: str, seat: int = 0, turn: int = 0,
                target_pid: int = 1, source_pid: int = 0,
                waves: list[ActionSpec] = None) -> CommittedMission:
    if waves is None:
        waves = [
            ActionSpec(from_planet_id=source_pid, dir_angle=0.0,
                       ships=5, launch_turn=0, agent_id=seat),
            ActionSpec(from_planet_id=source_pid, dir_angle=0.0,
                       ships=5, launch_turn=2, agent_id=seat),
            ActionSpec(from_planet_id=source_pid, dir_angle=0.0,
                       ships=5, launch_turn=4, agent_id=seat),
        ]
    return CommittedMission(
        mission_id=mid, seat=seat, turn_committed=turn,
        target_planet_id=target_pid, source_planet_id=source_pid,
        waves=tuple(waves),
    )


def test_mission_memory_starts_empty():
    mm = MissionMemory()
    assert len(mm) == 0
    assert mm.current_turn == 0
    assert mm.missions == ()


def test_mission_memory_commit_adds_immutably():
    mm0 = MissionMemory()
    m = _mk_mission("m1")
    mm1 = mm0.commit(m)
    assert len(mm0) == 0  # original untouched
    assert len(mm1) == 1
    assert mm1.get("m1") is m


def test_mission_memory_commit_duplicate_id_raises():
    mm = MissionMemory().commit(_mk_mission("m1"))
    with pytest.raises(ValueError, match="duplicate mission_id"):
        mm.commit(_mk_mission("m1"))


def test_mission_memory_mark_wave_fired():
    mm = MissionMemory().commit(_mk_mission("m1"))
    assert mm.get("m1").waves_fired == 0
    mm = mm.mark_wave_fired("m1")
    assert mm.get("m1").waves_fired == 1
    mm = mm.mark_wave_fired("m1", n=2)
    assert mm.get("m1").waves_fired == 3
    assert mm.get("m1").is_complete


def test_mission_memory_waves_to_fire_for_seat():
    """Returns ActionSpecs whose absolute launch turn matches and
    seat matches."""
    waves = [
        ActionSpec(from_planet_id=0, dir_angle=0.0, ships=1, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=0, dir_angle=0.0, ships=1, launch_turn=2, agent_id=0),
        ActionSpec(from_planet_id=0, dir_angle=0.0, ships=1, launch_turn=4, agent_id=0),
    ]
    m = _mk_mission("m1", seat=0, turn=10, waves=waves)
    mm = MissionMemory(current_turn=10).commit(m)

    # Turn 10 (turn_committed + launch_turn=0) — wave 0 due now.
    due = mm.waves_to_fire(seat=0)
    assert len(due) == 1
    assert due[0].launch_turn == 0

    # Other seat: nothing.
    assert mm.waves_to_fire(seat=1) == []

    # Advance to turn 12 (turn_committed + 2 = wave 1's launch_turn).
    mm = mm.advance_turn(12).mark_wave_fired("m1")  # fire wave 0
    due = mm.waves_to_fire(seat=0)
    assert len(due) == 1
    assert due[0].launch_turn == 2


def test_mission_memory_advance_turn():
    mm = MissionMemory()
    assert mm.advance_turn().current_turn == 1
    assert mm.advance_turn(50).current_turn == 50


def test_mission_memory_reset():
    mm = MissionMemory(current_turn=42).commit(_mk_mission("m1"))
    fresh = mm.reset()
    assert len(fresh) == 0
    assert fresh.current_turn == 0


# ---------------------------------------------------------------------------
# MissionMemory — state-driven pruning
# ---------------------------------------------------------------------------


def test_mission_memory_prune_completes_missions():
    """A mission with all waves fired is dropped on update()."""
    m = _mk_mission("done", waves=[
        ActionSpec(from_planet_id=0, dir_angle=0.0, ships=1, launch_turn=0, agent_id=0),
    ])
    state = _build_state()
    mm = MissionMemory().commit(m).mark_wave_fired("done")
    assert mm.get("done").is_complete

    mm_after = mm.update(state, action_taken=None)
    assert mm_after.get("done") is None


def test_mission_memory_prune_target_captured():
    """A mission whose target is now owned by the committing seat
    is dropped (mission accomplished)."""
    state = _build_state()

    # Find a planet owned by seat 0 to serve as the "captured target".
    alive = np.asarray(state.planets_alive)
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    owned_by_0 = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == 0),
        None,
    )
    if owned_by_0 is None:
        pytest.skip("no seat-0 planet at init")

    # Commit a mission targeting that planet (since it's already ours,
    # update() should prune immediately).
    m = _mk_mission("captured", seat=0, target_pid=owned_by_0, source_pid=owned_by_0)
    mm = MissionMemory().commit(m)

    mm_after = mm.update(state, action_taken=None)
    assert mm_after.get("captured") is None


def test_mission_memory_prune_unknown_target():
    """A mission whose target planet id doesn't exist in state is
    dropped (target off-board / never existed)."""
    state = _build_state()
    m = _mk_mission("ghost", target_pid=9999, source_pid=0)
    mm = MissionMemory().commit(m)
    mm_after = mm.update(state, action_taken=None)
    assert mm_after.get("ghost") is None


def test_mission_memory_prune_source_lost():
    """A mission whose source planet isn't owned by `seat` anymore
    is dropped (no fuel)."""
    state = _build_state()
    # Find a planet owned by seat 1 (enemy of seat 0), use as source
    # for a seat-0 mission → simulates "we lost the source."
    alive = np.asarray(state.planets_alive)
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    owned_by_1 = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == 1),
        None,
    )
    # Find an enemy target for seat 0.
    enemy_target = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] not in (0, -1)),
        None,
    )
    if owned_by_1 is None or enemy_target is None:
        pytest.skip("setup planets not available")

    m = _mk_mission("no_fuel", seat=0,
                    target_pid=enemy_target, source_pid=owned_by_1)
    mm = MissionMemory().commit(m)
    mm_after = mm.update(state, action_taken=None)
    assert mm_after.get("no_fuel") is None


def test_mission_memory_keeps_valid_mission():
    """A mission with a valid (owned) source and an unowned target
    that's still alive must survive update()."""
    state = _build_state()
    alive = np.asarray(state.planets_alive)
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    owned_by_0 = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == 0),
        None,
    )
    neutral = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == -1),
        None,
    )
    if owned_by_0 is None or neutral is None:
        pytest.skip("setup planets not available")

    m = _mk_mission("valid", seat=0,
                    target_pid=neutral, source_pid=owned_by_0)
    mm = MissionMemory().commit(m)
    mm_after = mm.update(state, action_taken=None)
    assert mm_after.get("valid") is m  # survived


# ---------------------------------------------------------------------------
# End-to-end: 3-wave snipe persistence + stale-intent pruning
# (the plan's Step-7d test)
# ---------------------------------------------------------------------------


def test_three_wave_snipe_persists_until_target_captured():
    """A 3-wave plan with `launch_turn ∈ {0, 2, 4}` fires waves on
    turns N, N+2, N+4 from memory (not re-proposed). Then, when
    update() observes the target is captured, remaining waves do
    NOT fire."""
    state = _build_state()
    alive = np.asarray(state.planets_alive)
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    owned_by_0 = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == 0),
        None,
    )
    enemy_target = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] not in (0, -1)),
        None,
    )
    neutral = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == -1),
        None,
    )
    target = enemy_target if enemy_target is not None else neutral
    if owned_by_0 is None or target is None:
        pytest.skip("setup planets not available")

    waves = [
        ActionSpec(from_planet_id=owned_by_0, dir_angle=0.0,
                   ships=5, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=owned_by_0, dir_angle=0.0,
                   ships=5, launch_turn=2, agent_id=0),
        ActionSpec(from_planet_id=owned_by_0, dir_angle=0.0,
                   ships=5, launch_turn=4, agent_id=0),
    ]
    m = _mk_mission("snipe", seat=0, turn=0,
                    target_pid=target, source_pid=owned_by_0,
                    waves=waves)
    mm = MissionMemory(current_turn=0).commit(m)

    # Turn 0: wave 0 due.
    assert len(mm.waves_to_fire(seat=0)) == 1
    mm = mm.mark_wave_fired("snipe").advance_turn(1)
    # Turn 1: no waves due.
    assert mm.waves_to_fire(seat=0) == []
    mm = mm.advance_turn(2)
    # Turn 2: wave 1 due.
    assert len(mm.waves_to_fire(seat=0)) == 1
    mm = mm.mark_wave_fired("snipe").advance_turn(3)
    assert mm.waves_to_fire(seat=0) == []
    mm = mm.advance_turn(4)
    # Turn 4: wave 2 due.
    assert len(mm.waves_to_fire(seat=0)) == 1
    mm_after_all_fired = mm.mark_wave_fired("snipe")
    assert mm_after_all_fired.get("snipe").is_complete

    # Now simulate "target captured by us" via update() — the mission
    # is dropped, AND if we hadn't fired all waves, the remaining ones
    # would not fire.
    # Synthesize a state where `target` is owned by seat 0.
    state_taken = state._replace(
        planets_owner=state.planets_owner.at[_idx_for(state, target)].set(0)
    )
    mm_pruned = mm_after_all_fired.update(state_taken, action_taken=None)
    assert mm_pruned.get("snipe") is None


def test_three_wave_snipe_pruned_if_target_captured_mid_plan():
    """If the target is captured between wave 1 and wave 2, update()
    removes the mission and the remaining waves do NOT fire."""
    state = _build_state()
    alive = np.asarray(state.planets_alive)
    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    owned_by_0 = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == 0),
        None,
    )
    neutral = next(
        (int(ids[i]) for i in range(len(alive))
         if alive[i] and owner[i] == -1),
        None,
    )
    if owned_by_0 is None or neutral is None:
        pytest.skip("setup planets not available")

    waves = [
        ActionSpec(from_planet_id=owned_by_0, dir_angle=0.0,
                   ships=5, launch_turn=0, agent_id=0),
        ActionSpec(from_planet_id=owned_by_0, dir_angle=0.0,
                   ships=5, launch_turn=2, agent_id=0),
        ActionSpec(from_planet_id=owned_by_0, dir_angle=0.0,
                   ships=5, launch_turn=4, agent_id=0),
    ]
    m = _mk_mission("snipe", seat=0, turn=0,
                    target_pid=neutral, source_pid=owned_by_0,
                    waves=waves)
    mm = MissionMemory(current_turn=0).commit(m)

    # Fire wave 0.
    mm = mm.mark_wave_fired("snipe").advance_turn(2)

    # Target captured by us between turn 0 and turn 2 — synthesise
    # the captured state.
    state_taken = state._replace(
        planets_owner=state.planets_owner.at[_idx_for(state, neutral)].set(0)
    )
    mm_pruned = mm.update(state_taken, action_taken=None)
    assert mm_pruned.get("snipe") is None
    # Wave 1 at turn 2 does NOT fire.
    assert mm_pruned.waves_to_fire(seat=0) == []


def _idx_for(state, planet_id: int) -> int:
    """Find the array index of a planet by id."""
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    for i in range(len(alive)):
        if alive[i] and int(ids[i]) == planet_id:
            return i
    raise ValueError(f"planet id {planet_id} not found")


# ---------------------------------------------------------------------------
# CompositeMemory
# ---------------------------------------------------------------------------


def test_composite_memory_defaults():
    cm = CompositeMemory()
    assert isinstance(cm.jit_cache, JitCacheMemory)
    assert isinstance(cm.missions, MissionMemory)
    assert cm.scratch == {}


def test_composite_memory_update_threads_to_sub_memories():
    state = _build_state()
    cm = CompositeMemory()
    cm.jit_cache.get_or_compile("k", lambda: "v")

    # Commit a mission that will be pruned (unknown target).
    cm = cm.with_missions(cm.missions.commit(_mk_mission("ghost", target_pid=9999)))
    assert cm.missions.get("ghost") is not None

    cm2 = cm.update(state, action_taken=None)
    # JIT cache survives (no state dependency).
    assert cm2.jit_cache.has("k")
    # Mission pruned.
    assert cm2.missions.get("ghost") is None


def test_composite_memory_reset_clears_all():
    cm = CompositeMemory()
    cm.jit_cache.get_or_compile("k", lambda: "v")
    cm = cm.with_missions(cm.missions.commit(_mk_mission("m1")))
    cm.scratch["scratchpad"] = "data"

    fresh = cm.reset()
    assert len(fresh.jit_cache) == 0
    assert len(fresh.missions) == 0
    assert fresh.scratch == {}


def test_composite_memory_with_missions_keeps_jit_cache():
    """Replacing missions doesn't lose the JIT cache."""
    cm = CompositeMemory()
    cm.jit_cache.get_or_compile("k", lambda: "v")
    new_missions = cm.missions.commit(_mk_mission("m1"))
    cm2 = cm.with_missions(new_missions)
    assert cm2.jit_cache.has("k")
    assert cm2.missions.get("m1") is not None


# ---------------------------------------------------------------------------
# agent_loop module-level singleton
# ---------------------------------------------------------------------------


def test_agent_loop_lazy_init():
    """First get_memory() call creates a CompositeMemory."""
    agent_loop.reset_memory()
    mem = agent_loop.get_memory()
    assert isinstance(mem, CompositeMemory)


def test_agent_loop_singleton_persists():
    """Subsequent get_memory() calls return the SAME object until
    set_memory or reset_memory is called."""
    agent_loop.reset_memory()
    m1 = agent_loop.get_memory()
    m2 = agent_loop.get_memory()
    assert m1 is m2


def test_agent_loop_set_memory():
    """set_memory replaces the singleton."""
    agent_loop.reset_memory()
    custom = CompositeMemory()
    custom.scratch["marker"] = "custom"
    agent_loop.set_memory(custom)
    fetched = agent_loop.get_memory()
    assert fetched is custom
    assert fetched.scratch["marker"] == "custom"


def test_agent_loop_reset_memory():
    agent_loop.reset_memory()
    custom = CompositeMemory()
    custom.scratch["marker"] = "before"
    agent_loop.set_memory(custom)
    agent_loop.reset_memory()
    fresh = agent_loop.get_memory()
    assert fresh.scratch == {}


def test_agent_loop_maybe_reset_on_step_zero_dict():
    """A dict obs with step=0 triggers reset; other steps don't."""
    agent_loop.reset_memory()
    custom = CompositeMemory()
    custom.scratch["before_reset"] = True
    agent_loop.set_memory(custom)

    agent_loop.maybe_reset_on_new_game({"step": 5})
    assert agent_loop.get_memory().scratch.get("before_reset") is True  # not reset

    agent_loop.maybe_reset_on_new_game({"step": 0})
    assert agent_loop.get_memory().scratch == {}  # reset


def test_agent_loop_maybe_reset_on_step_zero_struct():
    """Same but with a Struct-style obs (attribute access)."""
    from kaggle_environments.utils import Struct

    agent_loop.reset_memory()
    custom = CompositeMemory()
    custom.scratch["x"] = 1
    agent_loop.set_memory(custom)

    s5 = Struct(step=5)
    agent_loop.maybe_reset_on_new_game(s5)
    assert agent_loop.get_memory().scratch.get("x") == 1

    s0 = Struct(step=0)
    agent_loop.maybe_reset_on_new_game(s0)
    assert agent_loop.get_memory().scratch == {}


def test_agent_loop_simulated_turn_sequence():
    """End-to-end: simulate a sequence of turn calls. Memory persists
    and accumulates state across them; resets only at step=0."""
    agent_loop.reset_memory()

    # Turn 1 (step=0) — new game, reset.
    agent_loop.maybe_reset_on_new_game({"step": 0})
    mem = agent_loop.get_memory()
    mem.scratch["turn_1_marker"] = "yes"
    agent_loop.set_memory(mem)

    # Turn 2 (step=1) — no reset; marker survives.
    agent_loop.maybe_reset_on_new_game({"step": 1})
    mem = agent_loop.get_memory()
    assert mem.scratch.get("turn_1_marker") == "yes"

    # Turn 3 (step=2) — still no reset.
    agent_loop.maybe_reset_on_new_game({"step": 2})
    mem = agent_loop.get_memory()
    assert mem.scratch.get("turn_1_marker") == "yes"

    # New game starts (step=0 again) — reset.
    agent_loop.maybe_reset_on_new_game({"step": 0})
    mem = agent_loop.get_memory()
    assert mem.scratch == {}
