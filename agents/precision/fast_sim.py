"""Event-driven fast forward simulator.

Drop-in replacement for `prediction.rollout` with the same outputs but
~5-10x faster on typical mid-game boards. Instead of walking every step
1..horizon iterating every planet, we leapfrog between actual events
(arrivals + comet exits) and batch production accrual in between.

Correctness is verified bit-for-bit against `prediction.rollout` in
`tests/test_fast_sim_parity.py` — same final score per player.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from agents.precision import intercept, prediction, sim

# Re-export the dataclasses we share with prediction.py for convenience.
Arrival = prediction.Arrival
RolloutResult = prediction.RolloutResult
project_enemy_fleet_arrival = prediction.project_enemy_fleet_arrival


def _inflight_arrivals(world: dict, horizon_steps: int) -> list[Arrival]:
    """In-flight enemy fleets projected to their first impact. Cached per-turn
    on the world dict via key `_inflight_cache` keyed by horizon_steps.

    All rollout calls within a single agent turn share the same in-flight set
    (the world doesn't change between rollouts during one agent invocation).
    Caching here saves the bulk of the per-rollout cost.
    """
    cache = world.setdefault("_inflight_cache", {})
    if horizon_steps in cache:
        return cache[horizon_steps]
    me = world["player"]
    arrivals = []
    for fleet in world["fleets"]:
        if fleet[1] == me or fleet[1] == -1:
            continue
        arr = project_enemy_fleet_arrival(fleet, world, horizon_steps=horizon_steps)
        if arr is not None:
            arrivals.append(arr)
    cache[horizon_steps] = arrivals
    return arrivals


def rollout(
    world: dict,
    planned_shots: list[intercept.Shot],
    horizon_steps: int = 200,
    extra_arrivals: list[Arrival] | None = None,
    track_per_step: bool = False,
) -> RolloutResult:
    """Event-driven forward roll.

    Same signature/semantics as `prediction.rollout`. When `track_per_step`
    is True, fills `planet_timeline` etc. for introspection (slower path).
    """
    cur_step = world["step"]
    end_step = min(cur_step + horizon_steps, sim.EPISODE_STEPS - 2)

    # Flat per-planet state. We keep a dict for O(1) by-id access; the
    # iteration pattern is per-event, not per-step, so dict overhead is fine.
    # Each entry: [owner, ships, production].
    state: dict[int, list] = {
        p.id: [p.owner, p.ships, p.production] for p in world["planets"]
    }

    # Comet exit events: at `exit_step`, planet `pid` is removed.
    comet_exits: dict[int, int] = {}
    for grp in world["comets"]:
        path_index = grp["path_index"]
        for i, pid in enumerate(grp["planet_ids"]):
            remaining = len(grp["paths"][i]) - path_index
            if remaining >= 0:
                comet_exits[pid] = cur_step + remaining

    # Collect arrivals by step. Same logic as prediction.rollout.
    arrivals_by_step: dict[int, list[Arrival]] = {}

    def _schedule(arr: Arrival):
        if cur_step < arr.step <= end_step:
            arrivals_by_step.setdefault(arr.step, []).append(arr)

    # Our planned shots: debit source garrison NOW, queue arrivals.
    for shot in planned_shots:
        if shot.src_id not in state:
            continue
        cur = state[shot.src_id]
        if cur[1] < shot.ship_count:
            continue
        cur[1] -= shot.ship_count
        _schedule(Arrival(
            step=cur_step + shot.eta,
            planet_id=shot.tgt_id,
            owner=world["player"],
            ships=shot.arrival_ships,
        ))

    # In-flight enemy fleets — cached per-turn via _inflight_arrivals.
    for arr in _inflight_arrivals(world, horizon_steps):
        _schedule(arr)

    if extra_arrivals:
        for arr in extra_arrivals:
            _schedule(arr)

    # Build the event timeline: union of arrival steps + comet-exit steps.
    event_steps = sorted(set(arrivals_by_step.keys()) | set(comet_exits.values()))
    event_steps = [s for s in event_steps if cur_step < s <= end_step]

    # Per-step trajectories (only when requested).
    planet_timeline: dict[int, list[tuple[int, int, int]]] = {}
    step_state: dict[int, dict[int, tuple[int, int]]] = {}
    production_per_player: dict[int, dict[int, int]] = {}
    if track_per_step:
        for pid in state:
            planet_timeline[pid] = []

    last_processed = cur_step
    for event_step in event_steps:
        ticks_elapsed = event_step - last_processed
        if ticks_elapsed > 0:
            # Batch production accrual: each owned planet gains ticks * production.
            for pid, row in state.items():
                if row[0] != -1:
                    row[1] += ticks_elapsed * row[2]

            # If tracking, fill in the in-between steps with (owner, ships) trajectory.
            # Production accrues linearly so we can interpolate without re-iterating.
            if track_per_step:
                for s_offset in range(1, ticks_elapsed + 1):
                    s = last_processed + s_offset
                    snap = {}
                    prod_per: dict[int, int] = {}
                    for pid, row in state.items():
                        # At step s (s_offset into the gap), garrison was
                        # row[1] - (ticks_elapsed - s_offset) * row[2] (rewound).
                        if row[0] != -1:
                            interp_ships = row[1] - (ticks_elapsed - s_offset) * row[2]
                            prod_per[row[0]] = prod_per.get(row[0], 0) + row[2]
                        else:
                            interp_ships = row[1]
                        snap[pid] = (row[0], interp_ships)
                        planet_timeline[pid].append((s, row[0], interp_ships))
                    step_state[s] = snap
                    production_per_player[s] = prod_per

        # Handle comet exits at this step (before combat).
        for pid, exit_at in list(comet_exits.items()):
            if exit_at == event_step:
                state.pop(pid, None)
                comet_exits.pop(pid, None)

        # Handle arrivals at this step (combat).
        arrs = arrivals_by_step.get(event_step)
        if arrs:
            by_planet: dict[int, list[tuple[int, int]]] = {}
            for arr in arrs:
                if arr.planet_id not in state:
                    continue
                by_planet.setdefault(arr.planet_id, []).append((arr.owner, arr.ships))
            for pid, atks in by_planet.items():
                gowner, gships, _ = state[pid]
                new_owner, new_ships = sim.combat_resolve(gowner, gships, atks)
                state[pid][0] = new_owner
                state[pid][1] = new_ships

        # If tracking, snapshot the post-event state too.
        if track_per_step:
            snap = {}
            prod_per: dict[int, int] = {}
            for pid, row in state.items():
                snap[pid] = (row[0], row[1])
                if row[0] != -1:
                    prod_per[row[0]] = prod_per.get(row[0], 0) + row[2]
            # Overwrite the per-step entry for event_step with the post-combat state.
            step_state[event_step] = snap
            production_per_player[event_step] = prod_per
            for pid, row in state.items():
                planet_timeline[pid].append((event_step, row[0], row[1]))

        last_processed = event_step

    # Add the tail: production accrual from last event to end_step.
    tail_ticks = end_step - last_processed
    if tail_ticks > 0:
        for pid, row in state.items():
            if row[0] != -1:
                row[1] += tail_ticks * row[2]
        if track_per_step:
            for s_offset in range(1, tail_ticks + 1):
                s = last_processed + s_offset
                snap = {}
                prod_per: dict[int, int] = {}
                for pid, row in state.items():
                    if row[0] != -1:
                        interp_ships = row[1] - (tail_ticks - s_offset) * row[2]
                        prod_per[row[0]] = prod_per.get(row[0], 0) + row[2]
                    else:
                        interp_ships = row[1]
                    snap[pid] = (row[0], interp_ships)
                    planet_timeline[pid].append((s, row[0], interp_ships))
                step_state[s] = snap
                production_per_player[s] = prod_per

    # Final score: sum of ships per owned planet per player.
    score: dict[int, int] = {}
    for pid, (owner, ships, _) in state.items():
        if owner == -1:
            continue
        score[owner] = score.get(owner, 0) + ships

    final_owner = {pid: v[0] for pid, v in state.items()}
    final_ships = {pid: v[1] for pid, v in state.items()}
    return RolloutResult(
        horizon=horizon_steps,
        final_owner=final_owner,
        final_ships=final_ships,
        final_score_per_player=score,
        step_state=step_state,
        arrivals_by_step=arrivals_by_step,
        production_per_player=production_per_player,
        planet_timeline=planet_timeline,
    )


def plan_score(
    world: dict,
    planned_shots: list[intercept.Shot],
    horizon_steps: int = 200,
    extra_arrivals: list[Arrival] | None = None,
) -> float:
    """Higher = better for us. Margin = my_score − max(opp_score)."""
    me = world["player"]
    res = rollout(world, planned_shots, horizon_steps=horizon_steps,
                  extra_arrivals=extra_arrivals)
    my = res.final_score_per_player.get(me, 0)
    opp_max = 0
    for player, sc in res.final_score_per_player.items():
        if player != me and sc > opp_max:
            opp_max = sc
    return float(my - opp_max)


def planet_garrison_projection(
    world: dict,
    planned_shots: list[intercept.Shot],
    planet_id: int,
    horizon_steps: int = 200,
    extra_arrivals: list[Arrival] | None = None,
) -> list[tuple[int, int, int]]:
    """Per-step (step, owner, ships) trajectory for one planet."""
    res = rollout(world, planned_shots, horizon_steps=horizon_steps,
                  track_per_step=True, extra_arrivals=extra_arrivals)
    return res.planet_timeline.get(planet_id, [])
