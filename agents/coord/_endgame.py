"""Closed-form winning-margin predicate + per-bundle ΔW helpers.

Ported from `origin/claude/strategy-axis-decision-3437:lib/joint_solver/
predicate.py` (phase-α smooth-ΔW endgame work).

The predicate `winning_margin = prod_advantage × remaining_turns − opp_pool`
is the signed scalar form of `is_winning_state`. Positive ⇒ I will outpace
opp's recovery capacity by game end even if ownership stays constant from
this state onward.

coord uses this as an additive term on each bundle's `tier2_score`:
the `_bundle_endgame_bonus` in `agents/coord/main.py` calls
`bundle_delta_w_attack` or `bundle_delta_w_defend` to compute the
per-bundle ΔW contribution and multiplies by λ_W. The Lagrangian then
clears bundles with the boosted scores.

Design reference: `/root/.claude/plans/eventual-skipping-breeze.md`.
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
    """Sum of my production minus sum of opp production. Neutrals: 0."""
    my_prod = 0
    opp_prod = 0
    for p in world.planets_by_id.values():
        if p.owner == my_id:
            my_prod += int(p.production)
        elif p.owner == opp_id:
            opp_prod += int(p.production)
    return my_prod - opp_prod


def remaining_turns(world: World, episode_steps: int = EPISODE_STEPS) -> int:
    return max(0, int(episode_steps) - int(world.step))


def opp_pool(world: World, opp_id: int,
             episode_steps: int = EPISODE_STEPS) -> int:
    """Opp's total recovery capacity through end of game.

    = (opp ships on planets) + (opp ships in-flight) + opp_prod × rem.
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
    """`prod_advantage × remaining_turns > opp_pool`."""
    adv = prod_advantage(world, my_id, opp_id)
    if adv <= 0:
        return False
    return adv * remaining_turns(world, episode_steps) > opp_pool(
        world, opp_id, episode_steps
    )


def winning_margin(world: World, my_id: int, opp_id: int,
                   episode_steps: int = EPISODE_STEPS) -> int:
    """Signed scalar form. Property: `is_winning_state == (margin > 0)`."""
    return (
        prod_advantage(world, my_id, opp_id)
        * remaining_turns(world, episode_steps)
        - opp_pool(world, opp_id, episode_steps)
    )


# ---------------------------------------------------------------------------
# Per-bundle ΔW — analog of phase-α's `_endgame_bonus_smooth` adapted to
# the bundle-as-action shape. Returns the change in `winning_margin`
# attributable to the bundle's effect on its target planet.
# ---------------------------------------------------------------------------

def bundle_delta_w_attack(target_planet, my_id: int, opp_id: int,
                          rem: int) -> int:
    """ΔW when ATTACK succeeds: target shifts from current owner to me.

    Cases on `target.owner`:
      - me            → 0 (no transition; should not be called for own targets)
      - opp_id        → 3·prod·rem + ships  (we gain prod; opp loses prod
                        AND opp_pool drops by ships+prod·rem)
      - neutral (-1)  → prod·rem            (we gain prod only)
      - other opp     → prod·rem            (we gain prod; attributed opp
                        not directly affected via opp_pool — 4P edge case)
    """
    cur_owner = int(target_planet.owner)
    if cur_owner == my_id:
        return 0
    prod = int(target_planet.production)
    ships = int(target_planet.ships)
    d_adv = prod
    d_op = 0
    if cur_owner == opp_id:
        d_adv += prod
        d_op -= ships + prod * rem
    return d_adv * rem - d_op


def bundle_delta_w_defend(target_planet, my_id: int, opp_threat: int,
                          rem: int) -> int:
    """ΔW for the AVOIDED loss when DEFEND prevents target flipping
    me → opp_threat. Bonus = `−(ΔW of the counterfactual loss)`.

    Counterfactual ΔW for loss (cur=me, pred=opp_threat):
      d_adv = -prod - prod = -2·prod  (we lose prod; opp gains prod)
      d_op  = +prod·rem               (opp gains future prod stream)
      ΔW    = -2·prod·rem - prod·rem  = -3·prod·rem
    Returns `+3·prod·rem` (avoided loss).

    Mirrors `is_winning_state_if_lost`'s conservative approach: NO ship
    transfer modeled (combat reduces both sides; excluding it is
    worst-case for "would the loss flip us out of winning state").

    NOTE: `opp_threat` is currently UNUSED — the formula gives the same
    bonus regardless of which opp threatens the planet. This is by design
    in v1 (conservative; matches the source-branch predicate). The
    parameter is kept as a forward-compat hook for v2 if we scale the
    DEFEND bonus by the threat opp's `opp_pool` (so defending against
    the leader is rewarded more than defending against a trailer).
    """
    cur_owner = int(target_planet.owner)
    if cur_owner != my_id:
        return 0
    prod = int(target_planet.production)
    return 3 * prod * rem
