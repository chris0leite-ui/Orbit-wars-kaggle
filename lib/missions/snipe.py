"""Snipe mission builder — capture enemy / neutral planets via cost-aware ROI.

For every (our-planet, non-our-planet) pair, produce one Mission candidate.
**2026-05-11 ROI upgrade**: the score now trades off VALUE against COST
in ships (and travel time), addressing the gap the doc flagged
(`docs/strategies/simple-roi.md` "Where ROI can lose" lines 64-69):

    value = production × max(1, 500 - step - eta)
    score = priority × value / (ships_to_send + distance + 1)

Additive (not multiplicative) cost in the denominator: pure value/cost
over-corrects toward 1-ship 1-prod targets, which is a different bug.
Keeping distance in the denominator preserves the travel-time discount.

**2026-05-11 PM games-analysis upgrade**: two multiplicative priority
modifiers address weaknesses surfaced in
`audit/2026-05-11-v3-snipe-games-analysis.md`:

1. **Neutral / comet bonus** (NEUTRAL_BONUS, COMET_BONUS). 78.6% of
   comet-steps in v3_snipe's live replays sat neutral; we captured
   only 4.9%. Score function under-priced low-production unclaimed
   targets even though they're essentially free (no garrison growth
   while neutral, no opponent claim required).
2. **4P spoiler** (LEADER_MULTIPLIER). When we're ranked 3rd or 4th
   in ship-totals, attack the leader's planets preferentially. In
   v3_snipe wins, median 58% of our 4P captures came from the leader;
   in losses, 45%. v3 has no explicit leader detection.

Filter: drop pairs where the WorldModel predicts the target will already
be ours with surplus garrison at our fleet's arrival step.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel, comet_remaining_lifetime

# Total game length in steps (Configuration table, data/README.md).
EPISODE_STEPS = 500

# Priority multipliers (calibrated from games analysis).
# NEUTRAL_BONUS and COMET_BONUS were attempted at 1.5 / 1.3 but regressed in
# 32-seed 2P A/B (28.1% Wilson [18.6%, 40.1%]); they tipped the scorer toward
# easy neutrals when contested enemy planets were the binding constraint.
# Disabled (= 1.0) pending a more selective heuristic (opening-only, or
# distance-conditioned). See audit/2026-05-11-v3-snipe-games-analysis.md.
NEUTRAL_BONUS = 1.0
COMET_BONUS = 1.0
# LEADER_MULTIPLIER only fires when our_rank >= 2 (4P/larger games where we
# are below 2nd place). 2P games are unaffected. Pending 4P FFA validation.
LEADER_MULTIPLIER = 1.5


def _player_totals(world: World) -> dict[int, float]:
    """Aggregate ships across planets + in-flight fleets for each player.

    Used by the 4P spoiler logic to identify the current leader.
    """
    totals: dict[int, float] = {}
    for p in world.planets_by_id.values():
        if p.owner == -1:
            continue
        totals[p.owner] = totals.get(p.owner, 0) + p.ships
    raw = world.obs_raw
    fleets_raw = (
        raw.get("fleets", []) if isinstance(raw, dict) else getattr(raw, "fleets", [])
    )
    for f in fleets_raw:
        # Fleet schema: [id, owner, x, y, angle, from_planet_id, ships].
        owner = f[1]
        ships = f[6]
        if owner == -1:
            continue
        totals[owner] = totals.get(owner, 0) + ships
    return totals


def _leader_pid(world: World) -> tuple[int | None, int | None]:
    """Return (leader_pid, our_rank) for 4P spoiler scoring.

    Rank is 0-indexed (0 = leader). If we're alone or only-vs-one
    other player, returns (None, None) — no spoiler applies in 2P.
    """
    totals = _player_totals(world)
    if len(totals) < 3:
        return None, None  # 2P or solo — no spoiler
    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    leader_pid = ordered[0][0]
    our_rank = None
    for i, (pid, _ships) in enumerate(ordered):
        if pid == world.my_id:
            our_rank = i
            break
    return leader_pid, our_rank


def propose_snipe_missions(world: World, model: WorldModel) -> list[Mission]:
    """Build one snipe Mission per (our source, non-our target) pair."""
    if not world.planets_by_id:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if not my_planets:
        return []
    targets = [
        p for p in world.planets_by_id.values() if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    leader_pid, our_rank = _leader_pid(world)
    spoiler_on = leader_pid is not None and our_rank is not None and our_rank >= 2

    missions: list[Mission] = []
    for src in my_planets:
        for t in targets:
            d = math.hypot(t.x - src.x, t.y - src.y)
            base_ships = max(1, int(t.ships) + 1)
            v = fleet_speed(base_ships)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            pred_owner = model.owner_at(t.id, eta)
            pred_ships = model.ships_at(t.id, eta) or 0.0
            if pred_owner == world.my_id and pred_ships >= base_ships:
                # Target will be ours with surplus garrison; redundant.
                continue
            # Comet-lifetime correction: comets leave the board at
            # `len(path) - path_index` steps from now; capping time_to_hold
            # by remaining lifetime stops us scoring "long-run yield" on a
            # comet that's about to depart.
            is_comet = t.id in world.comet_ids
            if is_comet:
                rem = comet_remaining_lifetime(t.id, world)
                time_to_hold = max(0, (rem or 0) - eta)
            else:
                time_to_hold = max(1, EPISODE_STEPS - step_now - eta)
            value = t.production * time_to_hold

            # Cost-aware ROI baseline + priority modifiers.
            priority = 1.0
            if t.owner == -1:
                # Unclaimed: no garrison growth during flight, no opponent
                # competition. Bonus reflects the easier capture.
                priority *= COMET_BONUS if is_comet else NEUTRAL_BONUS
            if spoiler_on and t.owner == leader_pid:
                priority *= LEADER_MULTIPLIER
            score = priority * value / (base_ships + d + 1.0)

            missions.append(Mission(
                mission_class="snipe",
                src_id=src.id,
                target_id=t.id,
                ships=base_ships,
                score=score,
                eta=eta,
            ))
    return missions
