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


import dataclasses as _dc


def _apply_arrivals(world: dict, arrivals: list[prediction.Arrival]) -> dict:
    """Return a shallow-cloned world with each arrival's combat resolved on its
    target planet. Source garrisons are NOT debited here (the projected fleet
    has already been "launched" — we only model its impact).

    Used by `project_two_turns` to advance the world after the first round of
    enemy launches, so the second round projects from a realistic future state.
    """
    if not arrivals:
        return world
    by_planet: dict[int, list[tuple[int, int]]] = {}
    for arr in arrivals:
        by_planet.setdefault(arr.planet_id, []).append((arr.owner, arr.ships))

    new_planets = []
    new_by_id: dict = {}
    for p in world["planets"]:
        atks = by_planet.get(p.id)
        if atks:
            new_owner, new_ships = sim.combat_resolve(p.owner, p.ships, atks)
            p_new = _dc.replace(p, owner=new_owner, ships=new_ships)
        else:
            p_new = p
        new_planets.append(p_new)
        new_by_id[p_new.id] = p_new

    # Advance step to the latest arrival step (worst-case for "what does
    # the enemy do next?"). Bounded; no further computation depends on it
    # except for end_step ROI horizon.
    new_step = max(world["step"], max(arr.step for arr in arrivals))
    return {
        **world,
        "planets": new_planets,
        "planet_by_id": new_by_id,
        "step": new_step,
    }


def project_two_turns(
    world: dict,
    k_shots_per_player: int = 1,
    end_step: int = sim.EPISODE_STEPS,
) -> list[prediction.Arrival]:
    """Two-turn lookahead worst-for-us enemy projection.

    t1: project enemy's worst response to the CURRENT world.
    t2: project enemy's worst response to the post-t1 world (assuming t1
        landed and resolved combat).

    Returns a list of Arrivals from BOTH t+1 and t+2. The second-turn
    projections catch threats that emerge when our position is weakened by
    the first round (e.g., a fleet we couldn't capture this turn becomes
    cheap to capture after their first-turn strike).
    """
    t1 = project_enemy_actions_worst_for_us(world, k_shots_per_player, end_step)
    if not t1:
        return []  # no first-turn threat → no second-turn cascade either
    world_t1 = _apply_arrivals(world, t1)
    t2 = project_enemy_actions_worst_for_us(world_t1, k_shots_per_player, end_step)
    return list(t1) + list(t2)
