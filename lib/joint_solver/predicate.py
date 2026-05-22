"""Closed-form winning-state predicate.

Ported from origin/claude/ml-competition-strategy-PFhzM:lib/goal_planner/predicate.py.
Adapted to the current branch's World (lib/intent.py), which exposes
`planets_by_id` (dict) rather than `.planets` (list) and has no `.cfg`
attribute — episode length is taken from EPISODE_STEPS (matches
lib/fast_sim.DEFAULT_CONFIG['episodeSteps']).

The predicate: `prod_advantage × remaining_turns > opp_pool`. If True,
even holding current ownership constant, my production accumulation
outpaces the opponent's total recovery capacity by game end. Targets
2P (single opponent); 4P aggregates per-opponent.
"""

from __future__ import annotations

from typing import Any, Iterable

from lib.intent import World


EPISODE_STEPS = 500


def _iter_fleets(world: World) -> Iterable[Any]:
    """Read the fleet list off world.obs_raw (dict or Struct)."""
    obs = world.obs_raw
    if isinstance(obs, dict):
        return obs.get("fleets", []) or []
    return getattr(obs, "fleets", []) or []


def _fleet_owner(fleet) -> int:
    if isinstance(fleet, (list, tuple)):
        return int(fleet[1])
    return int(getattr(fleet, "owner", -1))


def _fleet_ships(fleet) -> int:
    if isinstance(fleet, (list, tuple)):
        return int(fleet[6])
    return int(getattr(fleet, "ships", 0))


def prod_advantage(world: World, my_id: int, opp_id: int) -> int:
    """Sum of my production minus sum of opp production.

    Neutrals contribute 0."""
    my_prod = 0
    opp_prod = 0
    for p in world.planets_by_id.values():
        if p.owner == my_id:
            my_prod += int(p.production)
        elif p.owner == opp_id:
            opp_prod += int(p.production)
    return my_prod - opp_prod


def remaining_turns(world: World, episode_steps: int = EPISODE_STEPS) -> int:
    """Turns left until episode end (>= 0)."""
    return max(0, int(episode_steps) - int(world.step))


def opp_pool(world: World, opp_id: int,
             episode_steps: int = EPISODE_STEPS) -> int:
    """Opp's total recovery capacity through end of game.

    = (ships on opp planets) + (ships in opp in-flight fleets)
      + (opp production × remaining_turns).
    """
    ships_on_planets = 0
    opp_prod = 0
    for p in world.planets_by_id.values():
        if p.owner == opp_id:
            ships_on_planets += int(p.ships)
            opp_prod += int(p.production)
    ships_in_flight = sum(
        _fleet_ships(f) for f in _iter_fleets(world)
        if _fleet_owner(f) == opp_id
    )
    rem = remaining_turns(world, episode_steps)
    return ships_on_planets + ships_in_flight + opp_prod * rem


def is_winning_state(world: World, my_id: int, opp_id: int,
                     episode_steps: int = EPISODE_STEPS) -> bool:
    """`prod_advantage × remaining_turns > opp_pool`.

    Edge cases:
      - prod_advantage <= 0: False (no edge to accumulate).
      - remaining_turns == 0: prod_advantage·0 = 0, so always False
        when opp_pool > 0 (which it usually is).
    """
    adv = prod_advantage(world, my_id, opp_id)
    if adv <= 0:
        return False
    return adv * remaining_turns(world, episode_steps) > opp_pool(
        world, opp_id, episode_steps
    )


def winning_margin(world: World, my_id: int, opp_id: int,
                   episode_steps: int = EPISODE_STEPS) -> int:
    """Signed scalar form of `is_winning_state`.

    Returns `prod_advantage × remaining_turns − opp_pool`. Positive ⇒
    in winning state; magnitude ⇒ how much "ship-equivalent" margin.
    Used by `lp_outcome._endgame_bonus` smooth-ΔW path (Phase α) to
    grade captures by their predicate impact instead of binary tip.

    Property: `is_winning_state(...) == (winning_margin(...) > 0)`.
    """
    return (
        prod_advantage(world, my_id, opp_id)
        * remaining_turns(world, episode_steps)
        - opp_pool(world, opp_id, episode_steps)
    )


def is_winning_state_if_owned(world: World, my_id: int, opp_id: int,
                              extra_planet_ids: set[int],
                              episode_steps: int = EPISODE_STEPS) -> bool:
    """Hypothetical: would `is_winning_state` hold if I ALSO owned
    `extra_planet_ids`?

    Re-attributes each extra planet's production from its current owner
    to me, and removes opp's ships/production from `opp_pool` for any
    extra planet currently held by opp.

    Caller is responsible for passing ids that aren't already mine.
    """
    adv = prod_advantage(world, my_id, opp_id)
    op = opp_pool(world, opp_id, episode_steps)
    rem = remaining_turns(world, episode_steps)
    for pid in extra_planet_ids:
        p = world.planets_by_id.get(pid)
        if p is None or p.owner == my_id:
            continue
        prod = int(p.production)
        if p.owner == opp_id:
            adv += 2 * prod
            op -= int(p.ships) + prod * rem
        else:
            adv += prod
    if adv <= 0:
        return False
    return adv * rem > op


def is_winning_state_if_lost(world: World, my_id: int, opp_id: int,
                             lost_planet_ids: set[int],
                             episode_steps: int = EPISODE_STEPS) -> bool:
    """Hypothetical: would `is_winning_state` hold if I LOST
    `lost_planet_ids` to opp?

    Symmetric to is_winning_state_if_owned. Re-attributes each lost
    planet's production from me to opp; adds opp's recovered production
    over remaining turns to opp_pool. Models the production transfer
    only (NOT the ship-transfer in combat) — combat losses reduce both
    sides, so excluding ship transfer is conservative for a worst-case
    "would losing this planet flip me out of winning state" check.

    Caller is responsible for passing ids that ARE currently mine.
    """
    adv = prod_advantage(world, my_id, opp_id)
    op = opp_pool(world, opp_id, episode_steps)
    rem = remaining_turns(world, episode_steps)
    for pid in lost_planet_ids:
        p = world.planets_by_id.get(pid)
        if p is None or p.owner != my_id:
            continue
        prod = int(p.production)
        adv -= 2 * prod
        op += prod * rem
    if adv <= 0:
        return False
    return adv * rem > op
