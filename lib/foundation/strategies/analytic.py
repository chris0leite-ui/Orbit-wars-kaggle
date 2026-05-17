"""`AnalyticStrategy` — Phase A + Phase B.1.

Strategy-agnostic beam search over `(src, target)` atomic launches,
scored by a Tier-1 mirror K-step rollout followed by a JAX-pure
value head that includes future production (held planets are valued
at `production × (500 - state.step)`, not just current ship count).

Pipeline per `emit` call:
1. Prune stale missions (`memory.missions.update`) and advance the
   internal turn counter to `state.step`.
2. Read pre-committed waves due THIS turn for our seat; re-aim each
   wave from current state (planets orbit, so the dir_angle captured
   at commit-time is stale by fire-time).
3. If pre-commits saturate every owned planet, SKIP the beam and
   return the pre-committed action set directly.
4. Else: enumerate atomic launches (filtered to non-committed
   sources), run `beam_search` SEEDED with the re-aimed pre-commits;
   the beam can only extend by adding launches from sources not
   already in the seed.
5. Mark the pre-committed waves we just fired (`mark_wave_fired`).
6. Conservative chainer: for each NEW launch the beam picked, decide
   whether to commit a 2-wave reinforce mission (wave 0 = the launch
   firing this turn, wave 1 = same source/target at +5 turns,
   re-aimed at fire-time). Gated by 4 conditions; details below.
7. Pack the winning set into an `ActionTensor` and thread the
   updated `CompositeMemory` back.

Phase B.1 deferrals (Phase A's other two omissions remain open):
- Diversified constructors (alternative beam roots per "strategic
  personality") — Phase B.2.
- K-mirror opp rollout (the deferred Phase A scope, busted budget at
  24 s cold compile + 70-200 ms warm at C=128) — Phase B.3.

Registers `"v8_analytic"` on import.
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
from lib.foundation.strategies.analytic_fastsim import (
    score_and_select_via_fastsim,
)
from lib.foundation.strategies.analytic_score import (
    DEFAULT_ATOM_CAP,
    enumerate_capped,
)
from lib.foundation.strategy import StrategyCtx, register_strategy
from lib.game.jax.jax_types import GameState


# Chainer gates — Phase B.1 conservative defaults (see Trade-offs in
# the plan; if a chainer-driven regression appears, vary one at a time
# per Rule 21 / Rule 37).
_CHAIN_ETA_THRESH = 8
_CHAIN_PROD_THRESH = 3
_CHAIN_GARRISON_RETENTION = 0.4
_CHAIN_DELAY = 5
_CHAIN_AIM_TOLERANCE = 1e-3
_MAX_LAUNCH_ETA = 80


class AnalyticStrategy:
    """Strategy-agnostic analytical lookahead with cross-turn missions.

    `width` / `depth` / `K` / `budget_ms` are passed through to
    `beam_search`. Defaults match the Phase A target: 4×4 beam,
    K=5 mirror rollout (currently no-op stub), 800 ms wall-clock cap
    inside the 1000 ms turn budget.
    """

    name = "v8_analytic"

    def __init__(
        self,
        *,
        width: int = 3,
        depth: int = 2,
        K: int = 15,
        budget_ms: float = 800.0,
        opp_aggressive: bool = True,
        enable_chainer: bool = True,
        atom_cap: int = DEFAULT_ATOM_CAP,
        fastsim_top_n: int = 25,
    ) -> None:
        # width/depth/budget_ms/opp_aggressive are now legacy beam knobs
        # kept for API compatibility; the fast_sim scorer in
        # `analytic_fastsim.py` replaced beam_search at session 2026-05-17.
        self._width = width
        self._depth = depth
        self._K = K
        self._budget_ms = budget_ms
        self._opp_aggressive = opp_aggressive
        self._enable_chainer = enable_chainer
        self._atom_cap = atom_cap
        # Top-N cut applied AFTER cheap-rank, BEFORE fast_sim. K=15
        # chosen so the rollout covers most launch ETAs (median 10-30);
        # at K=8 only the closest captures landed, leaving 39/40 atoms
        # bit-equal to no-op. Mid-game per-candidate cost scales as
        # K × in-flight-fleet count: seed-3 p95 measured 643 ms at
        # K=8 N=40 → 760 ms at K=15 N=25 (well in budget). N=25
        # matches v8_scavenge's N_VALIDATE band; see
        # /tmp/v8_scavenge.py:76 for the rationale.
        self._fastsim_top_n = fastsim_top_n

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

        # If memory isn't a CompositeMemory (e.g., tests pass EmptyMemory),
        # fall back to Phase A behaviour: no missions, no chainer.
        if not isinstance(memory, CompositeMemory):
            atomics = enumerate_capped(
                state, my_id,
                world_model=ctx.world_model, raw_obs=ctx.raw_obs,
                max_n=self._atom_cap,
                return_targets=True,
            )
            winning_set = score_and_select_via_fastsim(
                ctx.raw_obs, atomics, my_id,
                pre_committed=[],
                K=self._K, max_n=self._fastsim_top_n,
            )
            return specs_to_tensor([winning_set], horizon=1), memory

        # 1. Auto-prune & advance.
        pruned = memory.missions.update(state, action_taken=None)
        pruned = pruned.advance_turn(step)

        # 2. Identify missions whose next-due wave fires THIS turn and
        # re-aim them from current state.
        due_pairs: list[tuple[CommittedMission, ActionSpec]] = []
        for m in pruned.missions:
            if m.seat != my_id or m.is_complete:
                continue
            if m.absolute_launch_turn(m.waves_fired) != step:
                continue
            re_aimed = _re_aim_wave(state, m)
            if re_aimed is None:
                # Can't re-aim (source lost / target gone / no intercept) —
                # the auto-prune on the NEXT turn will drop this mission.
                # Skip it this turn; do NOT advance waves_fired.
                continue
            due_pairs.append((m, re_aimed))

        pre_committed = [spec for _, spec in due_pairs]
        used_sources = {w.from_planet_id for w in pre_committed}

        # 3. Decide whether to skip beam entirely.
        owned_sources = set(_owned_planet_ids(state, my_id))
        if not owned_sources or used_sources >= owned_sources:
            winning_set = list(pre_committed)
        else:
            # 4. Score atoms via fast_sim, greedy-merge non-dogpile.
            atomics = enumerate_capped(
                state, my_id,
                world_model=ctx.world_model, raw_obs=ctx.raw_obs,
                max_n=self._atom_cap,
                return_targets=True,
            )
            atomics = [(a, t) for (a, t) in atomics
                       if a.from_planet_id not in used_sources]
            winning_set = score_and_select_via_fastsim(
                ctx.raw_obs, atomics, my_id,
                pre_committed=pre_committed,
                K=self._K, max_n=self._fastsim_top_n,
            )

        # 5. Mark fired waves.
        new_missions = pruned
        for mission, _spec in due_pairs:
            new_missions = new_missions.mark_wave_fired(mission.mission_id)

        # 6. Chain new launches.
        if self._enable_chainer:
            new_launches = winning_set[len(pre_committed):]
            new_missions = _maybe_commit_chains(
                state, my_id, step, new_launches, new_missions,
            )

        tensor = specs_to_tensor([winning_set], horizon=1)
        return tensor, memory.with_missions(new_missions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _owned_planet_ids(state: GameState, my_id: int) -> list[int]:
    """Return planet IDs (the public, stable ids; NOT array indices)
    owned by `my_id` with at least 1 ship of garrison.

    Sources with 0 garrison can't launch anything, so they don't
    contribute to the "all sources saturated" check.
    """
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    return [
        int(ids[i]) for i in range(len(alive))
        if bool(alive[i]) and int(owner[i]) == my_id and int(ships[i]) > 0
    ]


def _planet_index_by_id(state: GameState, planet_id: int) -> Optional[int]:
    """Map a planet_id to its array index in state.planets_*, or None
    if the planet is no longer alive."""
    alive = np.asarray(state.planets_alive)
    ids = np.asarray(state.planets_id)
    for i in range(len(alive)):
        if bool(alive[i]) and int(ids[i]) == planet_id:
            return i
    return None


def _re_aim_wave(state: GameState, mission: CommittedMission) -> Optional[ActionSpec]:
    """Re-aim a mission's next-due wave using the current state.

    Returns a fresh ActionSpec with `launch_turn=0` and an updated
    `dir_angle` / `ships` (clamped to current garrison). Returns None
    when the wave can't be fired this turn (source lost, target gone,
    no valid intercept, garrison empty). In that case the strategy
    skips firing — the auto-prune on the next turn picks up the
    target-gone / source-lost cases; non-prunable cases (e.g.,
    transient no-intercept) just defer the wave.
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
        return None  # source no longer ours

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


def _recover_target_and_eta(
    state: GameState, launch: ActionSpec,
) -> Optional[tuple[int, int]]:
    """Given a launch (which stores src + dir_angle + ships but NOT
    the intended target), find the planet whose aim_orbiting solution
    from `launch.from_planet_id` produces the same dir_angle within
    `_CHAIN_AIM_TOLERANCE`. Returns `(target_planet_id, eta)` or None
    if no match.

    Called only at chainer-commit time (~once per new launch per turn),
    so the O(P) cost is fine; the hot beam path doesn't touch this.
    """
    src_idx = _planet_index_by_id(state, launch.from_planet_id)
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

    src_pos = (float(x[src_idx]), float(y[src_idx]))
    src_radius = float(radius[src_idx])

    best_match: Optional[tuple[int, int]] = None
    best_delta = _CHAIN_AIM_TOLERANCE

    for i in range(len(alive)):
        if i == src_idx or not bool(alive[i]):
            continue
        if int(ids[i]) < 0:
            continue
        tgt_tuple = (
            int(ids[i]), int(owner[i]),
            float(x[i]), float(y[i]),
            float(radius[i]), int(ships[i]),
            int(prod[i]),
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
        delta = abs(_circular_angle_diff(float(aim_angle), float(launch.dir_angle)))
        if delta < best_delta:
            best_delta = delta
            best_match = (int(ids[i]), int(eta))

    return best_match


def _circular_angle_diff(a: float, b: float) -> float:
    """Minimal signed angular difference, in `(-π, π]`."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


def _maybe_commit_chains(
    state: GameState,
    my_id: int,
    step: int,
    new_launches: list[ActionSpec],
    missions: MissionMemory,
) -> MissionMemory:
    """Conservative chainer: commits at most one 2-wave reinforce
    mission per new launch.

    Gates (ALL must hold):
        G1. target NOT my seat (we're attacking).
        G2. ETA > `_CHAIN_ETA_THRESH` (long-range attack worth chaining).
        G3. `production(target) >= _CHAIN_PROD_THRESH` (high-value
            target).
        G4. After firing wave 0, source garrison still ≥
            `_CHAIN_GARRISON_RETENTION × original` (retain defense).

    Wave 1's `dir_angle` is captured at commit time but re-aimed at
    fire-time by `_re_aim_wave`, so orbital drift over the chain delay
    is correctly handled.
    """
    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    prod = np.asarray(state.planets_prod)

    for launch in new_launches:
        meta = _recover_target_and_eta(state, launch)
        if meta is None:
            continue
        target_id, eta = meta

        tgt_idx = _planet_index_by_id(state, target_id)
        src_idx = _planet_index_by_id(state, launch.from_planet_id)
        if tgt_idx is None or src_idx is None:
            continue

        # G1: target not ours.
        if int(owner[tgt_idx]) == my_id:
            continue
        # G2: ETA threshold.
        if eta <= _CHAIN_ETA_THRESH:
            continue
        # G3: production threshold.
        if int(prod[tgt_idx]) < _CHAIN_PROD_THRESH:
            continue
        # G4: garrison retention.
        src_pre = int(ships[src_idx])
        if src_pre <= 0:
            continue
        src_post = src_pre - int(launch.ships)
        if src_post < _CHAIN_GARRISON_RETENTION * src_pre:
            continue

        # Build wave 1 placeholder; dir_angle here is the commit-time
        # estimate. `_re_aim_wave` will re-derive it from the state at
        # fire-time. `ships` is a rough estimate (we don't know future
        # garrison); `_re_aim_wave` clamps to actual at fire-time.
        wave_1_ships = max(1, int(launch.ships * 0.6))
        wave_1 = ActionSpec(
            from_planet_id=launch.from_planet_id,
            dir_angle=float(launch.dir_angle),
            ships=wave_1_ships,
            launch_turn=_CHAIN_DELAY,
            agent_id=my_id,
        )
        mission = CommittedMission(
            mission_id=f"chain_{step}_{launch.from_planet_id}_{target_id}",
            seat=my_id,
            turn_committed=step,
            target_planet_id=target_id,
            source_planet_id=launch.from_planet_id,
            waves=(launch, wave_1),
            waves_fired=1,  # wave 0 fires this turn (it IS the beam pick).
        )
        try:
            missions = missions.commit(mission)
        except ValueError:
            # Duplicate id (shouldn't happen — step is monotonic and
            # one launch per source per turn — but be defensive).
            continue

    return missions


register_strategy("v8_analytic", AnalyticStrategy())
