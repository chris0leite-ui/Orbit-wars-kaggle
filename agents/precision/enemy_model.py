"""Project enemy launches under two hypotheses.

Both produce a list of Arrival objects (one per projected enemy fleet) that the
rollout can fold in via `extra_arrivals`. They share the same intercept solver
we use ourselves — just from each enemy player's perspective.

- greedy_theirs:    enemy maximizes value / cost from THEIR perspective.
- worst_for_us:     enemy maximizes (our_value_lost) / cost from our perspective.

Per-turn cost is bounded by:
  num_enemies × k_shots_per_player × candidate_pool
with candidate_pool capped at ~6 nearest targets per enemy source.
"""
from __future__ import annotations

import math
from agents.precision import intercept, prediction, sim


def _enemy_world(world: dict, enemy_player: int) -> dict:
    """Return a shallow-copied world dict with player flipped to `enemy_player`."""
    w = dict(world)
    w["player"] = enemy_player
    return w


def _enemy_value_to_them(
    target: intercept.PlanetView, arrival_step: int, end_step: int, enemy_player: int
) -> int:
    """Value the ENEMY assigns to capturing this target (from their perspective)."""
    remaining = max(0, end_step - arrival_step)
    base = target.production * remaining
    if target.owner == -1:
        return base
    if target.owner == enemy_player:
        return 0  # already theirs, no gain
    return 2 * base  # captures from us or another enemy


def _value_lost_to_us(
    target: intercept.PlanetView, arrival_step: int, end_step: int, our_player: int
) -> int:
    """Value the WE lose if this target gets captured."""
    if target.owner != our_player:
        return 0
    remaining = max(0, end_step - arrival_step)
    return 2 * target.production * remaining


def _enemies(world: dict) -> list[int]:
    me = world["player"]
    s = set()
    for p in world["planets"]:
        if p.owner not in (me, -1):
            s.add(p.owner)
    for f in world["fleets"]:
        if f[1] != me and f[1] != -1:
            s.add(f[1])
    return sorted(s)


def _projected_arrivals(
    enemy_world: dict,
    objective: str,  # "greedy_theirs" or "worst_for_us"
    k_shots: int,
    end_step: int,
    our_player: int,
    cache: intercept.SweepCache | None = None,
) -> list[prediction.Arrival]:
    """Find this enemy's top-k shots under the chosen objective; return as Arrivals."""
    me = enemy_world["player"]
    if cache is None:
        cache = intercept.SweepCache(enemy_world["omega"], enemy_world["step"])

    my_planets = [p for p in enemy_world["planets"] if p.owner == me and p.ships > 0]
    all_planets = enemy_world["planets"]
    candidates: list[tuple[float, intercept.Shot, intercept.PlanetView]] = []

    for src in my_planets:
        # Nearest ~6 non-self targets only.
        tgts = sorted(
            (p for p in all_planets if p.id != src.id),
            key=lambda p: (p.x - src.x) ** 2 + (p.y - src.y) ** 2,
        )[:6]
        for tgt in tgts:
            # Try a couple ship counts: min-capture and half-stack.
            # We approximate min-capture by garrison+small-buffer; speed isn't a
            # known function of ETA here, so just use available ships.
            counts = []
            if tgt.owner != me:
                cap_guess = tgt.ships + (tgt.production * 5 if tgt.owner != -1 else 0) + 1
                counts.append(min(src.ships, max(1, cap_guess)))
                counts.append(min(src.ships, max(1, cap_guess + 20)))
            counts.append(max(1, src.ships // 2))
            counts = sorted(set(c for c in counts if 1 <= c <= src.ships))
            for S in counts:
                shot = intercept.find_shot(src, tgt, S, enemy_world, cache=cache)
                if shot is None:
                    continue
                arrival_step = enemy_world["step"] + shot.eta
                if objective == "greedy_theirs":
                    value = _enemy_value_to_them(tgt, arrival_step, end_step, me)
                else:  # worst_for_us
                    value = _value_lost_to_us(tgt, arrival_step, end_step, our_player)
                if value <= 0:
                    continue
                roi = value / max(1, S)
                candidates.append((roi, shot, tgt))

    candidates.sort(key=lambda c: c[0], reverse=True)
    arrivals: list[prediction.Arrival] = []
    used_src_ships: dict[int, int] = {}
    for _roi, shot, _tgt in candidates:
        if len(arrivals) >= k_shots:
            break
        # Don't double-count ships from a source.
        used = used_src_ships.get(shot.src_id, 0)
        # Find the source planet to check budget
        src_planet = next((p for p in enemy_world["planets"] if p.id == shot.src_id), None)
        if src_planet is None:
            continue
        if used + shot.ship_count > src_planet.ships:
            continue
        used_src_ships[shot.src_id] = used + shot.ship_count
        arrivals.append(prediction.Arrival(
            step=enemy_world["step"] + shot.eta,
            planet_id=shot.tgt_id,
            owner=me,
            ships=shot.ship_count,
        ))
    return arrivals


def project_enemy_actions_greedy(
    world: dict,
    k_shots_per_player: int = 1,
    end_step: int = sim.EPISODE_STEPS,
) -> list[prediction.Arrival]:
    """Enemy plays each turn to maximize their own ROI. Returns one batch of
    projected arrivals (this turn's launches by all enemies)."""
    our = world["player"]
    arrivals: list[prediction.Arrival] = []
    for e in _enemies(world):
        ew = _enemy_world(world, e)
        arrivals.extend(_projected_arrivals(
            ew, "greedy_theirs", k_shots_per_player, end_step, our_player=our,
        ))
    return arrivals


def project_enemy_actions_worst_for_us(
    world: dict,
    k_shots_per_player: int = 1,
    end_step: int = sim.EPISODE_STEPS,
) -> list[prediction.Arrival]:
    """Enemy plays each turn to maximize damage to US specifically."""
    our = world["player"]
    arrivals: list[prediction.Arrival] = []
    for e in _enemies(world):
        ew = _enemy_world(world, e)
        arrivals.extend(_projected_arrivals(
            ew, "worst_for_us", k_shots_per_player, end_step, our_player=our,
        ))
    return arrivals
