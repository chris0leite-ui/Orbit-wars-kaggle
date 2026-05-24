"""Synchronized-arrival multi-source salvo planner.

Given multiple owned sources and a single target, time each source's
launch so all fleets arrive on the same tick. The slowest source defines
the arrival schedule; faster sources receive `wait_N > 0` so they hold
fire until the geometry catches up.

Why this is novel: existing post-passes (`drain_combat_stack`,
`emit_sniper_strikes`) stack ships onto contested targets but launch
whenever the source has surplus — fleets arrive piecemeal and the
defender resolves each wave individually. A synchronized salvo lands
one combined wave that beats the projected garrison in a single
combat resolve.

Each emitted intent is validated through `predict_fleet_fate`; if the
fate is not `target` (or `planet` with `hit_planet_id == target.id`),
the intent is dropped from the salvo. If fewer than 2 sources survive
validation, the caller is expected to fall back to per-source greedy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lib.fleet import speed as fleet_speed
from lib.intent import Intent
from lib.scoring import s_needed
from lib.trajectory import predict_fleet_fate

# Bounded salvo size — over-saturating a single target wastes ships
# that could capture a second planet.
MAX_SALVO_SOURCES = 6


@dataclass
class SalvoPlan:
    """Bundle of synchronized intents + the arrival tick they share."""
    intents: list[Intent]
    t_arrival: int
    target_id: int
    total_ships: int


def _eta_simple(src, tgt, ships: int) -> int:
    """Geometry-only ETA proxy: distance / fleet_speed(ships), ceil to int.

    Used for the salvo scheduler; we don't need lead-aim precision here
    because each intent is re-validated through `predict_fleet_fate`
    afterwards (which uses the full trajectory ray-cast).
    """
    dx = float(tgt.x) - float(src.x)
    dy = float(tgt.y) - float(src.y)
    flight = max(0.0, math.hypot(dx, dy) - float(src.radius) - float(tgt.radius) - 0.1)
    v = fleet_speed(max(1, int(ships)))
    if v <= 0:
        return 999
    return max(1, int(math.ceil(flight / v)))


def salvo_feasible(sources, target, world, model, my_id, *,
                   min_fleet_size: int = 8, safety: float = 1.15) -> bool:
    """Quick check: can these sources combine to saturate this target?

    Each viable source must have at least `min_fleet_size` ships above
    its dynamic reserve, AND the combined contribution must clear
    `s_needed(target, t_arrival) * safety`. If only 1 source qualifies,
    a synchronized salvo isn't a salvo — caller should use per-source
    greedy instead.
    """
    if len(sources) < 2:
        return False
    contributions: list[int] = []
    etas: list[int] = []
    for src in sources:
        if int(src.ships) < min_fleet_size:
            continue
        ships = int(src.ships) - _reserve(src)
        if ships < min_fleet_size:
            continue
        contributions.append(ships)
        etas.append(_eta_simple(src, target, ships))
    if len(contributions) < 2:
        return False
    t_arrival = max(etas)
    needed = int(math.ceil(s_needed(target, t_arrival) * safety))
    return sum(contributions) >= needed


def plan_synchronized_salvo(
    sources, target, world, model, my_id, omega: float, *,
    min_fleet_size: int = 8, safety: float = 1.15,
    max_sources: int = MAX_SALVO_SOURCES,
) -> SalvoPlan | None:
    """Build a synchronized-arrival salvo against `target`.

    Algorithm:
      1. For each source, compute (raw_ships, raw_eta) using the simple
         distance/speed proxy.
      2. t_arrival = max(raw_eta) over sources (slowest defines schedule).
      3. For each source: wait_N = t_arrival - raw_eta. Source holds fire
         for that many ticks; its fleet arrives at the same tick as the
         slowest source's would have on this turn.
      4. Each intent's aim_angle is computed against the target's
         position at t_arrival (orbital lead via `predict_relative`).
      5. Each intent is validated through `predict_fleet_fate(wait_N=...)`;
         non-target outcomes are dropped.
      6. If total surviving ships at t_arrival ≥ s_needed * safety,
         return the SalvoPlan; else return None.

    Returns None when fewer than 2 sources survive validation or the
    saturation gate isn't met.
    """
    if len(sources) < 2:
        return None

    # Step 1: filter sources by reserve floor, compute raw contributions.
    raw: list[tuple] = []  # (ships, eta, src)
    for src in sources:
        if int(src.ships) < min_fleet_size:
            continue
        ships = int(src.ships) - _reserve(src)
        if ships < min_fleet_size:
            continue
        eta = _eta_simple(src, target, ships)
        raw.append((ships, eta, src))

    if len(raw) < 2:
        return None

    # Trim to the strongest `max_sources` contributors (largest ships
    # first) — large fleets are faster (fleet_speed log-curve) and
    # contribute more to garrison-overcome.
    raw.sort(key=lambda r: -r[0])
    raw = raw[:max_sources]

    t_arrival = max(eta for (_, eta, _) in raw)

    # Step 2: build candidate intents with synchronized arrival.
    from lib.orbit import predict_relative, is_orbiting
    candidates: list[tuple[Intent, int]] = []  # (intent, wait_N)
    for ships, eta, src in raw:
        wait_N = max(0, t_arrival - eta)
        # Aim against the target's predicted position at our arrival.
        tgt_list = [target.id, target.owner, target.x, target.y,
                    target.radius, target.ships, target.production]
        if is_orbiting(tgt_list) and omega != 0.0:
            tx, ty = predict_relative(tgt_list, omega, t_arrival)
        else:
            tx, ty = float(target.x), float(target.y)
        # Source pre-rotation: if src is orbiting and wait_N > 0, its
        # geometry at fire time differs from now.
        src_list = [src.id, src.owner, src.x, src.y, src.radius,
                    src.ships, src.production]
        if wait_N > 0 and is_orbiting(src_list) and omega != 0.0:
            sx, sy = predict_relative(src_list, omega, wait_N)
        else:
            sx, sy = float(src.x), float(src.y)
        angle = math.atan2(ty - sy, tx - sx)
        intent = Intent(
            src_id=int(src.id),
            target_id=int(target.id),
            ships=int(ships),
            aim_angle=float(angle),
            note=f"salvo:t_arrival={t_arrival},wait_N={wait_N}",
        )
        candidates.append((intent, wait_N))

    # Step 3: fate-gate each candidate.
    surviving: list[Intent] = []
    for intent, wait_N in candidates:
        src = world.planets_by_id.get(intent.src_id)
        if src is None:
            continue
        try:
            fate = predict_fleet_fate(
                src, target, intent.aim_angle, intent.ships, world,
                wait_N=wait_N,
            )
        except Exception:
            continue
        ok = (
            fate.outcome == "target"
            or (fate.outcome == "planet" and fate.hit_planet_id == target.id)
        )
        if ok:
            surviving.append(intent)

    if len(surviving) < 2:
        return None

    # Step 4: saturation gate.
    total = sum(int(i.ships) for i in surviving)
    needed = int(math.ceil(s_needed(target, t_arrival) * safety))
    if total < needed:
        return None

    return SalvoPlan(
        intents=surviving,
        t_arrival=int(t_arrival),
        target_id=int(target.id),
        total_ships=total,
    )


def _reserve(planet) -> int:
    """Dynamic per-source reserve: `max(production * 5, 10)`.

    Mirrors the heuristic that `agents/baseline` uses for stagnant-drain
    and sniper paths (production * STAGNANT_RESERVE_MULT,
    STAGNANT_RESERVE_FLOOR). A planet with production=5 holds back 25
    ships as defensive reserve; production=2 holds back 10.
    """
    prod = int(getattr(planet, "production", 0))
    return max(prod * 5, 10)
