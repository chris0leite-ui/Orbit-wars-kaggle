"""Phase C strategy: multi-turn joint actions + min-regret over opp archetypes.

Wraps:
- `enumerate_multi_turn_atoms` — strategy-agnostic atoms with
  `launch_turn in range(H)`.
- `build_opp_archetype_panel` — converts the env-format archetype
  outputs into per-turn padded JAX arrays.
- `joint_beam_search` — Tier-1 cheap beam shortlist + Tier-2
  min-regret refinement across the archetype panel.
- `MissionMemory` integration — same pattern as Phase B.1: auto-prune,
  re-aim due waves, mark fired, OR decompose multi-turn winning_set
  into a `CommittedMission` directly (the beam-as-chainer
  replacement).

Registers `"v8_analytic_phase_c"` on import. Phase B.1's
`"v8_analytic"` registration remains intact and untouched.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from lib.aim import aim_orbiting
from lib.foundation.actions import ActionSpec, ActionTensor, specs_to_tensor
from lib.foundation.memory import Memory
from lib.foundation.memory_impls import (
    CommittedMission,
    CompositeMemory,
    MissionMemory,
)
from lib.foundation.strategies.joint_beam import joint_beam_search
from lib.foundation.strategies.multi_turn_enumerate import (
    enumerate_multi_turn_atoms,
)
from lib.foundation.strategies.opp_archetype_adapter import (
    DEFAULT_ARCHETYPE_NAMES,
    build_opp_archetype_panel,
)
from lib.foundation.strategy import StrategyCtx, register_strategy
from lib.game.jax.jax_types import GameState


_MAX_LAUNCH_ETA = 80


class AnalyticJointStrategy:
    """Strategy-agnostic multi-turn joint search with min-regret aggregation.

    Defaults:
        H=2, K_opp_set=("no_launch","v351","counter_snipe"),
        beam (W,D)=(4,4), top_w_tier2=8, budget_ms=800.0.

    Falls back to Phase A behaviour (single-turn, opp no-op) when:
    - `memory` is not a CompositeMemory (e.g. EmptyMemory in tests), OR
    - `ctx.raw_obs` is None (can't build archetype panel without obs).
    """

    name = "v8_analytic_phase_c"

    def __init__(
        self,
        *,
        H: int = 2,
        width: int = 4,
        depth: int = 4,
        K: int = 5,
        top_w_tier2: int = 8,
        budget_ms: float = 800.0,
        opp_aggressive: bool = True,
        archetype_names: tuple[str, ...] = DEFAULT_ARCHETYPE_NAMES,
    ) -> None:
        self._H = H
        self._width = width
        self._depth = depth
        self._K = K
        self._top_w_tier2 = top_w_tier2
        self._budget_ms = budget_ms
        self._opp_aggressive = opp_aggressive
        self._archetype_names = archetype_names

    def emit(
        self,
        state: GameState,
        my_id: int,
        ctx: StrategyCtx,
        memory: Memory,
    ) -> tuple[ActionTensor, Memory]:
        step = int(state.step)
        num_agents = int(state.num_agents)
        effective_budget = min(
            self._budget_ms, max(50.0, ctx.turn_budget_ms - 100.0)
        )

        # If memory isn't Composite or no raw_obs available, run a
        # degraded single-turn no-opp path (Phase A fallback).
        if not isinstance(memory, CompositeMemory) or ctx.raw_obs is None:
            atoms = enumerate_multi_turn_atoms(state, my_id, horizon=1)
            winning_set = joint_beam_search(
                state, atoms, my_id, archetype_panel=[],
                width=self._width, depth=self._depth, H=1,
                K=self._K, top_w_tier2=self._top_w_tier2,
                num_agents=num_agents,
                opp_aggressive=self._opp_aggressive,
                budget_ms=effective_budget,
            )
            return specs_to_tensor([winning_set], horizon=1), memory

        # 1. Auto-prune & advance.
        pruned = memory.missions.update(state, action_taken=None)
        pruned = pruned.advance_turn(step)

        # 2. Identify due missions and re-aim each due wave.
        due_pairs: list[tuple[CommittedMission, ActionSpec]] = []
        for m in pruned.missions:
            if m.seat != my_id or m.is_complete:
                continue
            if m.absolute_launch_turn(m.waves_fired) != step:
                continue
            re_aimed = _re_aim_wave(state, m)
            if re_aimed is None:
                continue
            due_pairs.append((m, re_aimed))

        pre_committed = [spec for _, spec in due_pairs]
        used_source_turns = {(w.from_planet_id, w.launch_turn) for w in pre_committed}

        # 3. Build opp archetype panel (turn-budget once-per-emit).
        archetype_panel = build_opp_archetype_panel(
            ctx.raw_obs, my_id, self._H, archetype_names=self._archetype_names,
        )

        # 4. Multi-turn enumeration + beam (or skip if all sources committed).
        owned_sources = set(_owned_planet_ids(state, my_id))
        committed_turn0_sources = {
            s for (s, t) in used_source_turns if t == 0
        }
        if owned_sources and committed_turn0_sources >= owned_sources:
            # All owned planets already firing this turn from pre-commits.
            # Skip beam; honor pre-commits as-is.
            winning_set = list(pre_committed)
        else:
            atoms = enumerate_multi_turn_atoms(state, my_id, horizon=self._H)
            atoms = [
                a for a in atoms
                if (a.from_planet_id, a.launch_turn) not in used_source_turns
            ]
            winning_set = joint_beam_search(
                state, atoms, my_id, archetype_panel=archetype_panel,
                width=self._width, depth=self._depth, H=self._H,
                K=self._K, top_w_tier2=self._top_w_tier2,
                num_agents=num_agents,
                opp_aggressive=self._opp_aggressive,
                budget_ms=effective_budget,
                pre_committed=pre_committed,
            )

        # 5. Mark fired waves.
        new_missions = pruned
        for mission, _spec in due_pairs:
            new_missions = new_missions.mark_wave_fired(mission.mission_id)

        # 6. Decompose multi-turn winning_set into a CommittedMission
        # (replaces Phase B.1's heuristic chainer when the beam itself
        # picks a multi-turn plan).
        new_launches = winning_set[len(pre_committed):]
        new_missions = _decompose_multi_turn_into_missions(
            state, my_id, step, new_launches, new_missions,
        )

        # The ActionTensor returned to the env carries ONLY turn-0
        # launches (Kaggle's `agent()` is called per-turn; turn-1
        # launches of the multi-turn plan are persisted via mission
        # commits above and fired by the NEXT turn's `emit`).
        turn0_only = [s for s in winning_set if int(s.launch_turn) == 0]
        tensor = specs_to_tensor([turn0_only], horizon=1)
        return tensor, memory.with_missions(new_missions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _owned_planet_ids(state: GameState, my_id: int) -> list[int]:
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    return [
        int(ids[i]) for i in range(len(alive))
        if bool(alive[i]) and int(owner[i]) == my_id and int(ships[i]) > 0
    ]


def _planet_index_by_id(state: GameState, planet_id: int) -> Optional[int]:
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    for i in range(len(alive)):
        if bool(alive[i]) and int(ids[i]) == planet_id:
            return i
    return None


def _re_aim_wave(state: GameState, mission: CommittedMission) -> Optional[ActionSpec]:
    """Re-aim a mission's next-due wave using the current state.

    Same logic as Phase B.1's helper. Returns None when the wave can't
    be fired this turn (source lost, target gone, garrison empty, no
    intercept).
    """
    intended = mission.next_wave()
    if intended is None:
        return None

    src_idx = _planet_index_by_id(state, mission.source_planet_id)
    tgt_idx = _planet_index_by_id(state, mission.target_planet_id)
    if src_idx is None or tgt_idx is None:
        return None

    owner = np.asarray(state.planets_owner)
    if int(owner[src_idx]) != mission.seat:
        return None

    x = np.asarray(state.planets_x)
    y = np.asarray(state.planets_y)
    radius = np.asarray(state.planets_radius)
    ships = np.asarray(state.planets_ships)
    ids = np.asarray(state.planets_id)
    prod = np.asarray(state.planets_prod)
    omega = float(state.angular_velocity)

    src_garrison = int(ships[src_idx])
    if src_garrison <= 0:
        return None

    fleet_ships = max(1, min(int(intended.ships), src_garrison))

    src_pos = (float(x[src_idx]), float(y[src_idx]))
    src_radius = float(radius[src_idx])
    tgt_tuple = (
        int(ids[tgt_idx]), int(owner[tgt_idx]),
        float(x[tgt_idx]), float(y[tgt_idx]),
        float(radius[tgt_idx]), int(ships[tgt_idx]),
        int(prod[tgt_idx]),
    )
    tgt_radius = float(radius[tgt_idx])

    aim = aim_orbiting(
        src_pos, src_radius, tgt_tuple, tgt_radius, fleet_ships, omega,
    )
    if aim is None:
        return None
    aim_angle, _arrival, eta = aim
    if eta is None or eta > _MAX_LAUNCH_ETA:
        return None

    return ActionSpec(
        from_planet_id=mission.source_planet_id,
        dir_angle=float(aim_angle),
        ships=fleet_ships,
        launch_turn=0,
        agent_id=mission.seat,
    )


def _decompose_multi_turn_into_missions(
    state: GameState,
    my_id: int,
    step: int,
    new_launches: list[ActionSpec],
    missions: MissionMemory,
) -> MissionMemory:
    """For each (src) group of new_launches that includes a `launch_turn > 0`
    leg, commit a `CommittedMission` covering all its waves.

    Single-turn legs (`launch_turn == 0` only) carry no future
    commitment, so we don't commit a mission for them (Phase B.1's
    chainer used to commit heuristic 2-wave reinforcements; Phase C's
    beam-as-chainer only commits what the beam *actually* picked).

    Target planet attribution is recovered by finding the planet whose
    aim_orbiting solution from src matches dir_angle (same approach
    as Phase B.1's chainer).
    """
    # Group by source.
    by_src: dict[int, list[ActionSpec]] = {}
    for spec in new_launches:
        by_src.setdefault(int(spec.from_planet_id), []).append(spec)

    for src, waves in by_src.items():
        if all(int(w.launch_turn) == 0 for w in waves):
            continue
        waves_sorted = sorted(waves, key=lambda w: int(w.launch_turn))
        # The first wave is always launch_turn=0 if present; use its
        # target. If no turn-0 wave, use the earliest turn's wave.
        anchor = waves_sorted[0]
        target_id = _recover_target_for_launch(state, anchor)
        if target_id is None:
            continue
        # Adjust waves to have launch_turn RELATIVE to step (mission
        # framework expects relative launch_turn).
        rel_waves: list[ActionSpec] = []
        for w in waves_sorted:
            rel_waves.append(ActionSpec(
                from_planet_id=int(w.from_planet_id),
                dir_angle=float(w.dir_angle),
                ships=int(w.ships),
                launch_turn=int(w.launch_turn),
                agent_id=int(w.agent_id),
            ))
        # waves_fired = number with launch_turn == 0 (firing this turn).
        waves_fired = sum(1 for w in rel_waves if int(w.launch_turn) == 0)
        mission_id = f"phaseC_mt_{step}_{src}_{target_id}"
        mission = CommittedMission(
            mission_id=mission_id,
            seat=my_id,
            turn_committed=step,
            target_planet_id=target_id,
            source_planet_id=src,
            waves=tuple(rel_waves),
            waves_fired=waves_fired,
        )
        try:
            missions = missions.commit(mission)
        except ValueError:
            continue

    return missions


def _recover_target_for_launch(
    state: GameState, launch: ActionSpec,
) -> Optional[int]:
    """Find the planet whose aim_orbiting solution from `launch.from_planet_id`
    matches `launch.dir_angle` within tolerance.

    The launch may have been generated for a future `launch_turn`; in
    that case `aim_orbiting` was called with src/target rotated by
    omega*t. We replay that same rotation here so the angle match
    succeeds.
    """
    src_idx = _planet_index_by_id(state, int(launch.from_planet_id))
    if src_idx is None:
        return None

    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    x = np.asarray(state.planets_x)
    y = np.asarray(state.planets_y)
    radius = np.asarray(state.planets_radius)
    ships = np.asarray(state.planets_ships)
    prod = np.asarray(state.planets_prod)
    omega = float(state.angular_velocity)

    t = float(launch.launch_turn)
    src_planet = (
        int(ids[src_idx]), int(owner[src_idx]),
        float(x[src_idx]), float(y[src_idx]),
        float(radius[src_idx]),
        int(ships[src_idx]), int(prod[src_idx]),
    )
    if t == 0:
        src_pos = (float(x[src_idx]), float(y[src_idx]))
    else:
        from lib.orbit import predict_relative
        src_pos = predict_relative(src_planet, omega, t)
    src_radius = float(radius[src_idx])

    best_match: Optional[int] = None
    best_delta = 1e-3

    for i in range(len(alive)):
        if i == src_idx or not bool(alive[i]):
            continue
        if int(ids[i]) < 0:
            continue
        tgt_planet = (
            int(ids[i]), int(owner[i]),
            float(x[i]), float(y[i]),
            float(radius[i]),
            int(ships[i]), int(prod[i]),
        )
        if t == 0:
            tgt_x, tgt_y = float(x[i]), float(y[i])
        else:
            from lib.orbit import predict_relative
            tgt_x, tgt_y = predict_relative(tgt_planet, omega, t)
        tgt_tuple = (
            int(ids[i]), int(owner[i]),
            tgt_x, tgt_y,
            float(radius[i]), int(ships[i]), int(prod[i]),
        )
        aim = aim_orbiting(
            src_pos, src_radius, tgt_tuple, float(radius[i]),
            int(launch.ships), omega,
        )
        if aim is None:
            continue
        aim_angle, _arrival, eta = aim
        if eta is None:
            continue
        d = (float(aim_angle) - float(launch.dir_angle)) % (2.0 * math.pi)
        if d > math.pi:
            d -= 2.0 * math.pi
        delta = abs(d)
        if delta < best_delta:
            best_delta = delta
            best_match = int(ids[i])

    return best_match


register_strategy("v8_analytic_phase_c", AnalyticJointStrategy())

# Ablation A1: only the `no_launch` archetype. Tests whether the
# joint-scoring layer (3-archetype min-regret) is the regression vs
# B.1. Min over 1 archetype degenerates to "argmax of value-vs-opp-noop",
# so this variant isolates "multi-turn enum + Tier-2 rollout" as the
# only added mechanism over Phase A.
register_strategy(
    "v8_phase_c_no_panel",
    AnalyticJointStrategy(archetype_names=("no_launch",)),
)

# Ablation A2: H=1 (no multi-turn atoms). Tests whether multi-turn
# enumeration is the regression vs B.1. Keeps the 3-archetype panel
# and the joint-scoring layer (now single-turn rollout against the
# panel).
register_strategy(
    "v8_phase_c_h1",
    AnalyticJointStrategy(H=1),
)


def warmup_jits() -> None:
    """Trigger both Tier-1 (Phase A K=5 scorer) and Tier-2 (multi-turn
    H=2 rollout) JIT compilations against a real seed-42 init state.

    Cost: ~30-45 s total (one trace each); subsequent live `emit` calls
    hit the JAX cache and complete in ~300-500 ms warm.

    NOT called at module import — pytest imports of the strategy
    module shouldn't pay the cost. The Kaggle agent main module
    (`agents/v8_analytic_phase_c/main.py`) calls this AT IMPORT time
    so the 45 s lands inside the ~60 s Kaggle agent-init budget,
    not the 1 s per-turn budget.

    Idempotent: subsequent calls within the same process hit the JAX
    cache and return immediately.
    """
    import jax.numpy as jnp
    from kaggle_environments import make

    from lib.foundation.obs_to_state import obs_to_jax_state
    from lib.foundation.strategies.analytic_score import (
        action_specs_to_candidate_arrays,
        score_candidates_vmap_value_prod_jit,
    )
    from lib.foundation.strategies.analytic_score_rollout import (
        score_candidates_multi_turn_rollout_jit,
    )
    from lib.foundation.strategies.joint_beam import (
        FIXED_CANDIDATE_BATCH,
        TIER2_BATCH,
        _action_specs_to_multi_turn_arrays,
    )
    from lib.game.jax.jax_types import MAX_LAUNCH_PER_AGENT

    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    state = obs_to_jax_state(obs, configuration=env.configuration)

    # Tier-1: compile at the exact (FIXED_CANDIDATE_BATCH, ...) shape
    # used during the live beam.
    empty_padded = [[]] * FIXED_CANDIDATE_BATCH
    pids, angles, ships = action_specs_to_candidate_arrays(empty_padded)
    _ = score_candidates_vmap_value_prod_jit(
        state,
        jnp.asarray(pids), jnp.asarray(angles), jnp.asarray(ships),
        K=5, my_id=0, num_agents=2,
        opp_aggressive=True,
    )

    # Tier-2: compile at (TIER2_BATCH, H, MAX_LAUNCH_PER_AGENT).
    empty_sets: list[list] = [[] for _ in range(TIER2_BATCH)]
    my_pids_ch, my_angles_ch, my_ships_ch = _action_specs_to_multi_turn_arrays(
        empty_sets, H=2,
    )
    opp_pids_h = -np.ones((2, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    opp_angles_h = np.zeros((2, MAX_LAUNCH_PER_AGENT), dtype=np.float32)
    opp_ships_h = np.zeros((2, MAX_LAUNCH_PER_AGENT), dtype=np.int32)
    _ = score_candidates_multi_turn_rollout_jit(
        state,
        jnp.asarray(my_pids_ch),
        jnp.asarray(my_angles_ch),
        jnp.asarray(my_ships_ch),
        jnp.asarray(opp_pids_h),
        jnp.asarray(opp_angles_h),
        jnp.asarray(opp_ships_h),
        H=2, my_id=0, num_agents=2,
    )
