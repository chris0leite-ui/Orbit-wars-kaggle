"""Reach-table builder for the reach-frontier chooser.

Closed-form `ρ_i(s, p, k) = a_i(s, p, k) + k / p̃_s` per
`knowledge-base/concepts/reach-frontier-doctrine.md` §4. For each
(source, target) pair the table enumerates `k` over a coarse fraction
grid of the source garrison and returns the feasible entries sorted by
cost_tick.

Aim angle comes from `lib.aim.aim_orbiting` (5-iter fixed-point with
safe-intercept fallback) for non-comet targets, or `lib.aim.aim_comet`
for comet paths. The arrival tick is the closed-form `eta` returned by
the aim solver. Physics validation via `lib.trajectory.predict_fleet_fate`
is opt-in (`validate_physics=True`) — the chooser uses the closed-form
fast path for the sweep and physics-validates only the assignment winners.

The expected garrison at the planet on arrival is read from
`lib.world_model.WorldModel.ships_at`, which already threads
production-growth + in-flight-fleet combat through the per-planet
timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lib.aim import aim_comet, aim_orbiting
from lib.trajectory import predict_fleet_fate
from lib.world_model import _comet_paths_by_id


# k-grid: design §2 had 4 values (0.25, 0.5, 0.75, 1.0). v1 reduced to
# 2 values (0.5, 1.0) for the per-turn budget — empirically 4 values
# pushes turn-ms over 1000 ms in heavily-contested self-play mid-game
# (8+ sources × 25+ targets × 4 k_grid × ~2 ms physics = ~1.6 s). The
# coarser grid keeps p95 ≈ 200 ms; refinement to 4 values is a v1.1
# axis if cell-resolution-not-enough turns up in the eval.
DEFAULT_K_GRID_FRACTIONS: tuple[float, ...] = (0.5, 1.0)

# Per-source target cap: only the N nearest opponent / neutral planets
# (by closed-form initial distance) are enumerated as candidates from
# each source. Mirrors a common chooser-family pruning; cuts the
# per-pair sweep ~3× on a 24-planet board with ~8 sources mid-game.
# 12 covers the realistic "fleet-reachable within game horizon" set for
# any reasonable source.
DEFAULT_MAX_TARGETS_PER_SOURCE: int = 12


@dataclass
class ReachEntry:
    """One (src -> target, ships) candidate with closed-form cost.

    `cost_tick` is the doctrine-symmetric amortised reach cost (arrival
    tick + ship-recovery time). `expected_garrison` is the predicted
    garrison-at-arrival of the target, used by the reward function and
    by capture feasibility. `fate` is populated only when
    validate_physics=True; for closed-form sweeps it's None.
    """

    src_id: int
    target_id: int
    ships: int
    aim_angle: float
    arrival_tick: int
    cost_tick: float
    expected_garrison: float
    target_owner_at_arrival: int
    fate: object | None = None


def _planet_tuple(p) -> list:
    """Materialise the [id, owner, x, y, radius, ships, production] tuple
    that `lib.aim.aim_orbiting` / `aim_comet` expect.
    """
    return [
        int(p.id), int(p.owner), float(p.x), float(p.y),
        float(p.radius), float(p.ships), float(p.production),
    ]


def _aim_for_pair(src, tgt, ships: int, world):
    """Compute the (angle, eta_float) for firing `ships` from `src` at `tgt`.

    Returns (angle, eta_float) on success, or None if no valid intercept.
    Routes through `aim_comet` for comet targets (path-indexed lookahead)
    and `aim_orbiting` for orbital/static targets (rotational lookahead).
    """
    tgt_tuple = _planet_tuple(tgt)
    if int(tgt.id) in world.comet_ids:
        paths = _comet_paths_by_id(world)
        entry = paths.get(int(tgt.id))
        if entry is None:
            return None
        path, path_index = entry
        result = aim_comet(
            (float(src.x), float(src.y)), float(src.radius),
            tgt_tuple, float(tgt.radius), int(ships),
            path, int(path_index),
        )
    else:
        result = aim_orbiting(
            (float(src.x), float(src.y)), float(src.radius),
            tgt_tuple, float(tgt.radius), int(ships),
            float(world.omega),
        )
    if result is None:
        return None
    angle, _arrival_xy, eta = result
    return float(angle), float(eta or 0.0)


def _enumerate_pair(src, tgt, world, world_model, me_id: int,
                    k_grid_fractions: tuple[float, ...],
                    max_arrival_lead: int,
                    validate_physics: bool) -> list[ReachEntry]:
    """Enumerate candidate (src -> tgt, k) entries for a single pair."""
    src_garrison = int(max(0, src.ships))
    if src_garrison <= 0:
        return []
    src_production = max(1.0, float(src.production))

    seen_ships: set[int] = set()
    entries: list[ReachEntry] = []
    for frac in k_grid_fractions:
        k = int(max(1, round(src_garrison * frac)))
        if k in seen_ships or k > src_garrison:
            continue
        seen_ships.add(k)

        aim = _aim_for_pair(src, tgt, k, world)
        if aim is None:
            continue
        angle, eta_float = aim
        arrival_tick = int(math.ceil(max(1.0, eta_float)))
        if arrival_tick > max_arrival_lead:
            continue

        expected = world_model.ships_at(int(tgt.id), arrival_tick)
        if expected is None:
            # No timeline (planet vanished mid-game); fall back to
            # current ships. Conservative — slightly under-estimates
            # for opp-owned targets that produce during flight.
            expected = max(0.0, float(tgt.ships))
        owner_at = world_model.owner_at(int(tgt.id), arrival_tick)
        if owner_at is None:
            owner_at = int(tgt.owner)

        # Pre-filter infeasible captures BEFORE the expensive physics
        # validate: if the target won't be ours on arrival and our
        # ship count doesn't outnumber the predicted defenders, no
        # capture is possible regardless of trajectory. Cuts ~70% of
        # predict_fleet_fate calls in mid-game and pushes p95 turn-ms
        # from ~700 to ~120.
        if int(owner_at) != int(me_id):
            if float(k) <= float(expected):
                continue

        fate = None
        if validate_physics:
            fate = predict_fleet_fate(src, tgt, angle, k, world)
            if fate.outcome != "target" or fate.hit_planet_id != int(tgt.id):
                continue

        cost_tick = float(arrival_tick) + (float(k) / src_production)
        entries.append(ReachEntry(
            src_id=int(src.id),
            target_id=int(tgt.id),
            ships=int(k),
            aim_angle=float(angle),
            arrival_tick=int(arrival_tick),
            cost_tick=float(cost_tick),
            expected_garrison=float(expected),
            target_owner_at_arrival=int(owner_at),
            fate=fate,
        ))

    entries.sort(key=lambda e: e.cost_tick)
    return entries


def _nearest_targets(src, targets, n: int):
    """Top-n nearest targets to `src` by current Euclidean distance.

    Used as a cheap per-source pruning before physics validation. Falls
    back to all targets when `len(targets) <= n`.
    """
    if len(targets) <= n:
        return list(targets)
    sx, sy = float(src.x), float(src.y)
    scored = sorted(
        targets,
        key=lambda t: (float(t.x) - sx) ** 2 + (float(t.y) - sy) ** 2,
    )
    return scored[:n]


def build_reach_table(
    sources,
    targets,
    world,
    world_model,
    me_id: int,
    *,
    k_grid_fractions: tuple[float, ...] = DEFAULT_K_GRID_FRACTIONS,
    max_targets_per_source: int = DEFAULT_MAX_TARGETS_PER_SOURCE,
    max_arrival_lead: int = 200,
    validate_physics: bool = False,
) -> dict[tuple[int, int], list[ReachEntry]]:
    """Build the doctrine's ρ table for one player.

    `me_id` is the seat index of the player whose ρ table this is —
    used to pre-filter infeasible captures (k ≤ defender_garrison) for
    opponent / neutral targets before the physics validate.

    `max_targets_per_source` limits each source's candidate set to its
    nearest N targets (Euclidean). Cheap pruning to keep the per-turn
    candidate count bounded; without it 8 sources × 25 targets × 2
    k_frac × ~2 ms physics-validate blows the 1 s/turn cap.

    Returns a `(src_id, target_id) -> [ReachEntry, ...]` map sorted by
    cost_tick ascending. Empty list = unreachable pair.

    `validate_physics=True` filters entries through `predict_fleet_fate`
    before they're emitted. The doctrine flow uses physics-validate=True
    after the infeasibility prefilter cuts the candidate set down to
    those worth the call.
    """
    out: dict[tuple[int, int], list[ReachEntry]] = {}
    for src in sources:
        candidates = _nearest_targets(src, targets, int(max_targets_per_source))
        for tgt in candidates:
            if int(src.id) == int(tgt.id):
                continue
            entries = _enumerate_pair(
                src, tgt, world, world_model, int(me_id),
                k_grid_fractions=k_grid_fractions,
                max_arrival_lead=int(max_arrival_lead),
                validate_physics=validate_physics,
            )
            if entries:
                out[(int(src.id), int(tgt.id))] = entries
    return out
