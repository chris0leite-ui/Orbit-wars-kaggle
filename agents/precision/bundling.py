"""Multi-source wave bundling.

Find the cheapest (in total ships) synchronized 2-source wave that captures a
target. Synchronization means all participating fleets arrive on the SAME engine
step — engine combat rule_1 sums same-owner arrivals into one attacker pool.

Algorithm per target:
  1. List sources within wave-window H = 30 ticks (min-ETA reachable).
  2. For each candidate arrival step T in the source feasibility window:
     - For each source, ask intercept.find_shot_for_arrival(src, tgt, T).
     - Skip sources whose required ships exceed available (after defense reserve).
     - Pick the 1-2 sources with the lowest combined ship cost that together
       exceed defender_at_T (+1 for strict capture).
  3. Keep the cheapest valid wave across all T.

Caller can also still consider single-source shots; bundling.py only generates
multi-source waves (single shots come from intercept.build_shot_menu).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agents.precision import intercept, prediction, scoring, sim


WAVE_WINDOW_TICKS = 30
ARRIVAL_STEP_GRID = 4  # try every 4 ticks within the window


@dataclass(frozen=True)
class Wave:
    target_id: int
    shots: tuple[intercept.Shot, ...]
    arrival_step: int
    total_ships: int
    roi: float


def candidate_waves(
    world: dict,
    defense_reserve: dict[int, int] | None = None,
    extra_arrivals: list[prediction.Arrival] | None = None,
    end_step: int = sim.EPISODE_STEPS,
    max_sources: int = 2,
    deadline: float | None = None,
) -> list[Wave]:
    """Enumerate cheapest 2-source waves per target."""
    import time as _t
    if defense_reserve is None:
        defense_reserve = {}
    me = world["player"]
    obs_step = world["step"]
    omega = world["omega"]
    cache = intercept.SweepCache(omega, obs_step)

    my_planets = [p for p in world["planets"] if p.owner == me and p.ships > 0]
    targets = [p for p in world["planets"] if p.owner != me]

    waves: list[Wave] = []

    for tgt in targets:
        if deadline is not None and _t.perf_counter() >= deadline:
            break
        # Skip if even the strongest single shot can't reach within wave window.
        # First pass: collect min-ETA single shots from each candidate source.
        per_source: list[tuple[intercept.PlanetView, int, int]] = []
        # (src, min_eta, available_after_reserve)
        for src in my_planets:
            avail = src.ships - defense_reserve.get(src.id, 0)
            if avail < 1:
                continue
            shot = intercept.find_shot(src, tgt, min(avail, 200), world, cache=cache)
            if shot is None:
                continue
            if shot.eta > WAVE_WINDOW_TICKS:
                continue
            per_source.append((src, shot.eta, avail))

        if len(per_source) < 2:
            continue

        # Determine arrival-step search range.
        min_eta = min(eta for _, eta, _ in per_source)
        max_eta = max(eta for _, eta, _ in per_source)
        # Don't search before min_eta (nothing can arrive that early) or beyond
        # max_eta + a small slack (slower wave = bigger defender_at_T).
        search_etas = list(range(min_eta, min(max_eta + 5, WAVE_WINDOW_TICKS) + 1, ARRIVAL_STEP_GRID))
        if max_eta not in search_etas:
            search_etas.append(max_eta)
        search_etas = sorted(set(search_etas))

        best_wave: Wave | None = None

        for T_offset in search_etas:
            arrival_step = obs_step + T_offset
            # Predict defender at arrival_step folding in enemy projected arrivals.
            defender = scoring._defender_at(tgt, arrival_step, world, enemy_arrivals=extra_arrivals)

            # Compute each source's required ship count to arrive at T_offset.
            candidates_per_src: list[tuple[intercept.PlanetView, intercept.Shot]] = []
            for src, _min_eta, avail in per_source:
                shot = intercept.find_shot_for_arrival(src, tgt, arrival_step, world, cache=cache)
                if shot is None:
                    continue
                if shot.ship_count > avail:
                    continue
                candidates_per_src.append((src, shot))

            if len(candidates_per_src) < 2:
                continue

            # Sort by ship cost (cheapest first) and try pairs.
            candidates_per_src.sort(key=lambda pair: pair[1].ship_count)
            # Try every 2-source pair, find the cheapest that exceeds defender.
            for i in range(len(candidates_per_src)):
                src_i, shot_i = candidates_per_src[i]
                for j in range(i + 1, len(candidates_per_src)):
                    src_j, shot_j = candidates_per_src[j]
                    total = shot_i.ship_count + shot_j.ship_count
                    if total <= defender:
                        continue
                    if best_wave is None or total < best_wave.total_ships:
                        wave_shots = (shot_i, shot_j)
                        value = scoring._capture_value(tgt, arrival_step, end_step)
                        roi = value / max(1, total)
                        best_wave = Wave(
                            target_id=tgt.id,
                            shots=wave_shots,
                            arrival_step=arrival_step,
                            total_ships=total,
                            roi=roi,
                        )
                    # cheapest-first means later j are pricier; break early if first j fails too.

        if best_wave is not None:
            waves.append(best_wave)

    return waves
