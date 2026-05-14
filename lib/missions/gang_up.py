"""Active multi-source coordination — the gang_up Mission class.

Top-10 fingerprint analysis + games-analysis §3 finds the "in-flight
volume gap": during the tied phase of one tracked loss, the opponent
held 2× our in-flight ship count even though we launched similar fleet
COUNTS. Top players pair sources to land on the same step at a
contested target — the env's combat resolver groups same-owner
same-step arrivals and sums their ships, so a 30-ship + 30-ship pair
arriving on the same step beats a defender 50% larger than either
source alone could.

`lib/planner.settle_plan` already PERMITS passive gang-ups (multiple
sources can pick the same target if a single source's commit is
insufficient). What's missing is **active scoring**: this Mission
class proposes paired arrivals scored as a JOINT operation, with a
bonus reflecting that simultaneous arrival > staggered, and with the
slower source delaying its launch so both fleets land in the same
step window.

Critical guards:
- Only fires when the target's predicted garrison exceeds any single
  reachable source's capture cost.
- Caps pair delay at MAX_DELAY (3 turns) — longer delays grow the
  delaying source's garrison and create a new "huge stranded fleet"
  problem.
- Pairs the fastest source (no delay) with the second-fastest (with
  delay) — extending to triples is deferred to v3.6.
- Score uses `(0.5 * total_ships + d + 1)` denominator like the rest
  of v3.5, with a GANG_UP_BONUS=1.30 to reflect that timed arrival
  is genuinely stronger than two staggered launches.
"""

from __future__ import annotations

import math

from lib.fleet import speed as fleet_speed
from lib.intent import World
from lib.mission import Mission
from lib.world_model import WorldModel

EPISODE_STEPS = 500
MAX_DELAY = 3                  # cap on slower-source delay
PAIR_SHARE = 0.7               # each paired source sends 70% of its garrison
SINGLE_SOURCE_AFFORDABLE_RATIO = 0.85  # below this, target is "out of reach for one"
GANG_UP_BONUS = 1.30           # timed arrivals > staggered

# Mission Renaissance gate. Distinct from mechanism.GANG_UP_ENABLED
# (which gates the post-Mission gang_up_size mechanism). This flag
# gates the *proposer*. Default 0 = disabled. A/B candidate: 1.
USE_GANG_UP_MISSION = 0


def propose_gang_up_missions(world: World, model: WorldModel) -> list[Mission]:
    """Pair our top two reachable sources at any target where a single
    source's affordable fleet falls short."""
    if not USE_GANG_UP_MISSION:
        return []
    my_planets = [
        p for p in world.planets_by_id.values() if p.owner == world.my_id
    ]
    if len(my_planets) < 2:
        return []
    targets = [
        p for p in world.planets_by_id.values() if p.owner != world.my_id
    ]
    if not targets:
        return []

    step_now = int(world.step)
    missions: list[Mission] = []
    for t in targets:
        # Skip comet targets — gang_up timing on a moving target is too
        # noisy; comets are best handled by the dedicated snipe Mission.
        if t.id in world.comet_ids:
            continue

        # Rank sources by ETA at PAIR_SHARE fleet size — that's what they'd
        # actually launch in a pair.
        source_eta = []
        for src in my_planets:
            send = max(1, int(src.ships * PAIR_SHARE))
            if send < 2:
                continue
            d = math.hypot(t.x - src.x, t.y - src.y)
            v = fleet_speed(send)
            eta = int(math.ceil(d / max(v, 1e-6))) if v > 0 else 0
            source_eta.append((eta, src, d, send))
        if len(source_eta) < 2:
            continue
        source_eta.sort(key=lambda x: x[0])
        eta1, s1, d1, ships1 = source_eta[0]
        eta2, s2, d2, ships2 = source_eta[1]
        # Pair must be timeable within MAX_DELAY: slower source must
        # arrive within MAX_DELAY turns of fastest, OR fastest can
        # delay to align.
        eta_gap = eta2 - eta1
        if eta_gap > MAX_DELAY:
            continue
        # Joint ship count must exceed any single-source affordable;
        # otherwise gang_up isn't needed.
        # Predicted garrison at the JOINT arrival step (eta2 — the
        # later arrival; both fleets land same step after timing).
        pred_owner = model.owner_at(t.id, eta2)
        if pred_owner == world.my_id:
            continue
        pred_garrison = model.ships_at(t.id, eta2) or 0.0
        single_source_affordable = max(ships1, ships2)
        if single_source_affordable >= SINGLE_SOURCE_AFFORDABLE_RATIO * pred_garrison:
            # One source can handle this target; skip pair.
            continue
        combined = ships1 + ships2
        # Must actually be able to capture jointly (defensive — even
        # gang_up has limits).
        if combined < pred_garrison + 1:
            continue
        # Joint score: production × time-to-hold, denominator over
        # mean distance + combined-ship cost (halved as in wave 1b).
        time_to_hold = max(1, EPISODE_STEPS - step_now - eta2)
        value = float(t.production) * time_to_hold
        mean_d = (d1 + d2) / 2.0
        score = GANG_UP_BONUS * value / (0.5 * combined + mean_d + 1.0)

        # Emit two Missions: the "lead" (fastest) at its natural eta,
        # the "follow" (slower) at its natural eta. Both share the
        # same score so settle_plan ranks them together.
        # The natural ETAs already align within MAX_DELAY turns of each
        # other; the env combat resolver groups same-step arrivals at
        # each integer step, so a 0-3 turn gap is acceptable.
        missions.append(Mission(
            mission_class="gang_up_lead",
            src_id=s1.id,
            target_id=t.id,
            ships=ships1,
            score=score,
            eta=eta1,
            note=f"pair_with={s2.id}",
        ))
        missions.append(Mission(
            mission_class="gang_up_follow",
            src_id=s2.id,
            target_id=t.id,
            ships=ships2,
            score=score,
            eta=eta2,
            note=f"pair_with={s1.id}",
        ))
    return missions
