"""Hand-crafted opponent archetypes for depth-2 min-regret / maximin search.

The existing `choose_depth2` enumerates the opponent's drop-one candidate
set from v3.5.1's incumbent. That bakes in the assumption "real opp plays
v3.5.1." Both v7_1 (H11) and v7_2 (depth-2 over v3.5.1 drop-ones) failed
in scalar A/B against v7_0_drop_one — suggesting that opp-model
assumption is biased.

This module emits opp candidates that are **policy-agnostic**: hand-crafted
moves that reflect distinct *threat archetypes* an opponent might play.
Each archetype is a small, deterministic function of the world state plus
(optionally) our planned intents, so we can score "the worst opp response
we plausibly face" rather than "the worst opp response v3.5.1 would emit."

Archetypes (each returns an env-format `list[[src, angle, ships], ...]`):

1. `archetype_no_launch`  — opp plays defensively (passes turn).
2. `archetype_v351`       — opp plays the v3.5.1 aggressive pipeline.
3. `archetype_counter_reinforce(our_intents)` — for each launch we made,
   opp sends `our_ships + 1` from their nearest planet to that target.
4. `archetype_counter_snipe` — opp's biggest source fires 70 % of its
   garrison at our most-ships planet.
5. `archetype_cross_attack` — opp's biggest source fires 70 % of its
   garrison at our highest-production planet (different target than #4
   on most boards).

Together: 5 candidates per turn. Combined with our drop-one set of N ≈ 8,
the payoff matrix is 8 × 5 = 40 cells — same order of magnitude as the
current depth-2 (8 × 4).

The functions return env-format actions (already through `realize` so
aim/sun-safety mechanisms are applied). Callers can use them directly
inside the `score_depth2_payoff_matrix` payoff loop or any maximin /
min-regret aggregator.
"""

from __future__ import annotations

import math
from typing import Any

from lib.intent import Intent, World, realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.reinforce import propose_reinforce_missions
from lib.missions.snipe import propose_snipe_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel


# ---------------------------------------------------------------------------
# POV helpers
# ---------------------------------------------------------------------------


def opp_pov_obs(obs: Any, opp_id: int) -> dict:
    """Return a copy of `obs` with `player = opp_id`, suitable for
    `World.from_obs` to produce an opp-POV `World`.

    Same technique as `lib.v7_search._opp_incumbent_action`; factored
    out so archetype builders can be tested in isolation.
    """
    if isinstance(obs, dict):
        obs2 = dict(obs)
        obs2["player"] = opp_id
        return obs2
    keys = (
        "player", "planets", "fleets", "angular_velocity",
        "initial_planets", "comet_planet_ids", "comets",
        "step", "next_fleet_id",
    )
    obs2: dict = {}
    for k in keys:
        v = getattr(obs, k, None)
        if v is not None:
            obs2[k] = v
    obs2["player"] = opp_id
    return obs2


def _largest_source(world: World) -> object | None:
    """Opp's planet with the most ships (excluding comets)."""
    owned = [
        p for p in world.planets_by_id.values()
        if p.owner == world.my_id and p.id not in world.comet_ids
    ]
    if not owned:
        return None
    return max(owned, key=lambda p: p.ships)


def _our_largest_by_ships(world: World) -> object | None:
    """The target — our (= non-opp non-neutral) planet with most ships."""
    candidates = [
        p for p in world.planets_by_id.values()
        if p.owner != world.my_id and p.owner != -1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.ships)


def _our_largest_by_production(world: World) -> object | None:
    """The target — our planet with highest production."""
    candidates = [
        p for p in world.planets_by_id.values()
        if p.owner != world.my_id and p.owner != -1
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.production, p.ships))


# ---------------------------------------------------------------------------
# Archetypes (opp's POV)
# ---------------------------------------------------------------------------


def archetype_no_launch() -> list[list]:
    """Opp passes the turn — pure defensive baseline."""
    return []


def archetype_v351(opp_world: World, opp_model: WorldModel, opp_obs: dict) -> list[list]:
    """Opp plays the v3.5.1 aggressive pipeline (snipe + reinforce)."""
    if not opp_world.planets_by_id:
        return []
    missions = (
        propose_snipe_missions(opp_world, opp_model, aggressive=True)
        + propose_reinforce_missions(opp_world, opp_model)
    )
    intents = settle_plan(missions, opp_world, opp_model)
    return realize(intents, opp_obs, mechanisms=DEFAULT_MECHANISMS, model=opp_model)


def archetype_counter_reinforce(
    opp_world: World, opp_obs: dict, our_intents: list[Intent],
) -> list[list]:
    """For each of our intents, opp sends `our_ships + 1` to that target
    from opp's nearest planet that can fund it. If multiple of our
    launches go to the same target, only the first (strongest) one is
    countered. Empty if we have no intents or opp has no fundable
    source within reach.
    """
    if not our_intents or not opp_world.planets_by_id:
        return []
    seen_targets: set[int] = set()
    counter: list[Intent] = []
    used_sources: set[int] = set()
    for our_intent in our_intents:
        tgt_id = int(our_intent.target_id)
        if tgt_id in seen_targets:
            continue
        target = opp_world.planets_by_id.get(tgt_id)
        if target is None:
            continue
        needed = int(our_intent.ships) + 1
        candidates = [
            (math.hypot(p.x - target.x, p.y - target.y), p)
            for p in opp_world.planets_by_id.values()
            if (
                p.owner == opp_world.my_id
                and p.id not in opp_world.comet_ids
                and p.id not in used_sources
                and int(p.ships) >= needed
            )
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, src = candidates[0]
        seen_targets.add(tgt_id)
        used_sources.add(src.id)
        counter.append(Intent(src_id=src.id, target_id=tgt_id, ships=needed))
    return realize(counter, opp_obs, mechanisms=DEFAULT_MECHANISMS)


def archetype_counter_snipe(opp_world: World, opp_obs: dict) -> list[list]:
    """Opp's biggest source fires 70 % of its garrison at our top-ships
    planet. Single-launch — captures the "concentrated attack" archetype
    yijue1 / bowwowforeach use in top-10 replays.
    """
    src = _largest_source(opp_world)
    tgt = _our_largest_by_ships(opp_world)
    if src is None or tgt is None:
        return []
    if int(src.ships) < 5:
        return []
    ships = max(1, int(src.ships * 0.7))
    return realize(
        [Intent(src_id=src.id, target_id=tgt.id, ships=ships)],
        opp_obs, mechanisms=DEFAULT_MECHANISMS,
    )


def archetype_cross_attack(opp_world: World, opp_obs: dict) -> list[list]:
    """Opp's biggest source fires 70 % of its garrison at our highest-
    production planet. Different from `archetype_counter_snipe` on
    most boards — top-ships ≠ top-prod when an enemy is loading up a
    cheap planet vs holding a high-prod home.
    """
    src = _largest_source(opp_world)
    tgt = _our_largest_by_production(opp_world)
    if src is None or tgt is None:
        return []
    if int(src.ships) < 5:
        return []
    ships = max(1, int(src.ships * 0.7))
    return realize(
        [Intent(src_id=src.id, target_id=tgt.id, ships=ships)],
        opp_obs, mechanisms=DEFAULT_MECHANISMS,
    )


# ---------------------------------------------------------------------------
# Top-level: build the archetype response set
# ---------------------------------------------------------------------------


def build_opp_archetypes(
    opp_obs: dict, our_intents: list[Intent],
) -> list[list[list]]:
    """Return a list of distinct env-format opp actions covering the
    archetype set. Deduplicates exact matches so the payoff matrix
    isn't padded with identical rows.

    `opp_obs` must already have `player = opp_id` (see `opp_pov_obs`).
    `our_intents` is the launch list we plan to emit this turn — used
    by the counter-reinforce archetype.
    """
    opp_world = World.from_obs(opp_obs)
    if not opp_world.planets_by_id:
        return [archetype_no_launch()]
    opp_model = WorldModel.from_world(opp_world)

    archetypes: list[list[list]] = [
        archetype_no_launch(),
        archetype_v351(opp_world, opp_model, opp_obs),
        archetype_counter_reinforce(opp_world, opp_obs, our_intents),
        archetype_counter_snipe(opp_world, opp_obs),
        archetype_cross_attack(opp_world, opp_obs),
    ]

    # Deduplicate by exact equality; preserve order so row 0 (no-launch)
    # stays first — useful for tie-break debugging.
    seen: list[list[list]] = []
    for a in archetypes:
        if a not in seen:
            seen.append(a)
    return seen
