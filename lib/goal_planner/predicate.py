"""Closed-form winning-state predicate (PI 2026-05-19 PM pivot).

The predicate: `prod_advantage × remaining_turns > opp_pool`. If True,
even if we just hold current ownership, our production accumulation
outpaces the opponent's total recovery capacity by the end of the game.

All math is closed-form over the current World snapshot — no
forward-projection, no rollout. Targets 2P (single opponent); 4P is a
future extension that aggregates over multiple opponents.
"""

from __future__ import annotations

from lib.trajectory_layer import World


def prod_advantage(world: World, my_id: int, opp_id: int) -> int:
    """Sum of my production minus sum of opp production.

    Neutrals contribute 0. Only the named opponent is counted (for 2P
    this is the unique non-mine non-neutral owner)."""
    my_prod = 0
    opp_prod = 0
    for p in world.planets:
        if p.owner == my_id:
            my_prod += int(p.production)
        elif p.owner == opp_id:
            opp_prod += int(p.production)
    return my_prod - opp_prod


def remaining_turns(world: World) -> int:
    """Turns left until episode end (>= 0)."""
    return max(0, int(world.cfg.episode_steps) - int(world.step))


def opp_pool(world: World, opp_id: int) -> int:
    """Opp's total recovery capacity through end of game.

    = (ships on opp planets) + (ships in opp in-flight fleets)
      + (opp production × remaining_turns).

    This is the ship-volume the opponent can produce or has already
    fielded between now and the end of the game. The predicate
    compares this against our production-accumulation rate."""
    ships_on_planets = 0
    opp_prod = 0
    for p in world.planets:
        if p.owner == opp_id:
            ships_on_planets += int(p.ships)
            opp_prod += int(p.production)
    ships_in_flight = 0
    for f in world.fleets:
        if f.owner == opp_id:
            ships_in_flight += int(f.ships)
    return (ships_on_planets + ships_in_flight
            + opp_prod * remaining_turns(world))


def is_winning_state(world: World, my_id: int, opp_id: int) -> bool:
    """The closed-form predicate from PI 2026-05-19 PM:

        prod_advantage(world) × remaining_turns(world) > opp_pool(world, opp_id)

    Edge cases:
      - remaining_turns == 0: trivially False unless prod_advantage > opp_pool
        with the multiplication collapsing — we test the raw inequality.
      - prod_advantage <= 0: trivially False (we have no production edge to
        accumulate).
    """
    adv = prod_advantage(world, my_id, opp_id)
    if adv <= 0:
        return False
    return adv * remaining_turns(world) > opp_pool(world, opp_id)


def is_winning_state_if_owned(world: World, my_id: int, opp_id: int,
                                extra_planet_ids: set[int]) -> bool:
    """Hypothetical version: would `is_winning_state` hold if we ALSO
    owned the planets in `extra_planet_ids`?

    Used by P2 (portfolio) to evaluate candidate acquisitions without
    materialising a new World. We adjust prod_advantage by re-attributing
    those planets' production from their current owner to us, and
    subtract their ships from `opp_pool` if they currently belong to opp.

    Caller passes ids that are NOT currently mine (no double-counting)."""
    adv = prod_advantage(world, my_id, opp_id)
    op = opp_pool(world, opp_id)
    rem = remaining_turns(world)
    for pid in extra_planet_ids:
        p = world._planet_by_id.get(pid)
        if p is None or p.owner == my_id:
            continue
        prod = int(p.production)
        # Re-attribute production: +prod to us, -prod from opp if applicable.
        if p.owner == opp_id:
            adv += 2 * prod  # +prod for us, +prod for "less opp" → diff +2
            # Opp loses these ships from their pool AND loses prod accumulation.
            op -= int(p.ships) + prod * rem
        else:
            # Neutral: just adds to our prod_advantage.
            adv += prod
    if adv <= 0:
        return False
    return adv * rem > op
