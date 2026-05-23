"""Per-(src, tgt, launch_tick) candidate enumeration.

Uses the precision-physics substrate already on this branch:

  aim_and_eta(src, tgt, ships, omega, wait_N=)   from agents.baseline.proposer
  predict_fleet_fate(src, tgt, angle, ships, world, wait_N=)   from lib.trajectory
  predict_garrison_at(tgt, eta, arrivals)        from lib.world_model
  _target_holdable_after_capture(...)            from agents.baseline.proposer
                                                  (the B1 fix gated by
                                                  BASELINE_ORBITAL_SAFETY=1
                                                  → +63 μ on the ladder)

A candidate survives only if predict_fleet_fate confirms the fleet hits
the target (not sun, not OOB, not another planet, not timeout), the
delivered ships beat the predicted garrison at arrival, and the capture
is holdable against the nearest opp's counter-launch.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass

from agents.baseline.chooser_trajectory import (
    merge_ledgers,
    predict_opp_responses,
)
from agents.baseline.proposer import (
    _target_holdable_after_capture,
    aim_and_eta,
)
from lib.combat import resolve_arrivals
from lib.trajectory import predict_fleet_fate
from lib.world_model import predict_garrison_at


EPISODE_STEPS = 500
MAX_LAUNCH_TICK = 15     # search wait_N ∈ {0..15} (wide enough to let
                         #   low-ship sources accrue production before firing)
SHIP_REFINE_ITERS = 2    # 2-pass ships ↔ eta fixed-point
MIN_FLEET = 1

# Partial enumeration is opt-in via the same env var as dogpile. When off,
# the per-turn cost of the extra aim_and_eta + predict_fleet_fate calls
# (≤ |src|·|tgt|·MAX_LAUNCH_TICK additional physics evaluations) is
# avoided — empirically this matters: an unconditional partial pass added
# ~65 s wallclock to an n=16 gate run and pushed enough turns past kaggle's
# actTimeout to drop ELIM rate from 13/16 to 6/16.
_DOGPILE_ENABLED = os.environ.get(
    "LAGRANGE_SIMPLE_DOGPILE", "0",
).strip().lower() in ("1", "true", "on", "yes")


@dataclass
class Candidate:
    src_id: int
    tgt_id: int
    launch_tick: int
    angle: float
    ships: int
    eta: int
    arrival_step: int
    value: float            # capture NPV (own production stream gained)
    is_partial: bool = False        # True: this fleet alone cannot capture;
                                    # only useful as part of a multi-source
                                    # dogpile coalition for the (tgt,
                                    # arrival_step) bucket.
    defense_at_arrival: int = 0     # predicted opp garrison at arrival_step;
                                    # dogpile sums ships until > this.


def _refine_ships(src, tgt, launch_tick, omega, base_arrivals):
    """Two-pass: estimate ships needed to capture at arrival tick.

    Returns (ships, angle, eta, arrival_step) or None on failure.
    """
    ships = max(MIN_FLEET, int(tgt.ships) + 1)
    angle = 0.0
    eta = 0
    arrival_step = 0
    for _ in range(SHIP_REFINE_ITERS):
        res = aim_and_eta(src, tgt, ships, omega, wait_N=launch_tick)
        if res is None:
            return None
        angle, eta = res
        arrival_step = int(launch_tick) + int(eta)
        _owner_at_arr, gar_at_arr = predict_garrison_at(
            tgt, arrival_step, base_arrivals,
        )
        needed = int(math.ceil(float(gar_at_arr))) + 1
        if needed == ships:
            break
        ships = max(MIN_FLEET, needed)
    return ships, float(angle), int(eta), int(arrival_step)


def _capture_value(tgt, arrival_step: int) -> float:
    """NPV of capturing `tgt` at `arrival_step`: own-side production stream
    from arrival to episode end."""
    remaining = max(0, EPISODE_STEPS - int(arrival_step))
    return float(tgt.production) * float(remaining)


DEFENSE_HORIZON = 16    # ticks to look ahead for opp counter-arrivals


def _source_defensive_ok(src, launch_ships: int, launch_tick: int,
                          src_arrivals: list, *,
                          horizon: int = DEFENSE_HORIZON) -> bool:
    """True iff `src` retains ownership at every step in
    `[1, launch_tick + horizon]` AFTER subtracting `launch_ships` from
    its garrison at `launch_tick`.

    Walks tick-by-tick (same loop pattern as
    `lib.world_model.predict_garrison_at`):
      - each tick: add production (if owner != -1)
      - at launch_tick: subtract launch_ships
      - resolve any opp arrivals at this tick
      - if owner flips away from src.owner at ANY checked tick → False
    `src_arrivals` is the enriched-ledger entry for src.id: list of
    `(eta, owner, ships)` — opp's projected counters are already merged.
    """
    src_owner = int(src.owner)
    owner = src_owner
    ships = float(src.ships)
    prod = int(src.production)

    if int(launch_tick) == 0:
        ships -= float(launch_ships)
        if ships < 0.0:
            return False

    max_t = int(launch_tick) + int(horizon)
    by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for arrival_eta, arrival_owner, arrival_ships in src_arrivals:
        if int(arrival_ships) <= 0:
            continue
        eta = max(1, int(math.ceil(float(arrival_eta))))
        if eta > max_t:
            continue
        by_turn[eta].append((int(arrival_owner), int(arrival_ships)))

    for t in range(1, max_t + 1):
        if owner != -1:
            ships += float(prod)
        if t == int(launch_tick):
            ships -= float(launch_ships)
            if ships < 0.0:
                return False
        group = by_turn.get(t, [])
        if group:
            owner, ships = resolve_arrivals(owner, ships, group)
            if owner != src_owner:
                return False
    return True


def enumerate_candidates(world, model, my_id: int, omega: float,
                         comet_ids: set | None = None) -> list[Candidate]:
    """Enumerate every viable (src, tgt, launch_tick) capture candidate.

    Filters applied (in order):
      1. ships needed ≤ source's current garrison (time-indexed).
      2. predict_fleet_fate.outcome == "target" (not sun / OOB / wrong planet).
      3. delivered ships > predicted garrison at arrival (capture succeeds).
      4. target NOT already ours at arrival (skip pure reinforce in v1).
      5. _target_holdable_after_capture (B1 — opp can't recapture cheaply).
         RELAXED in dominant-endgame: when we own ≥3× as many planets as
         the opponent, the recapture-risk model over-rejects (opp's few
         remaining planets WILL counter, but our wide base lets us
         re-take cheaply). The PI elim-gate failure on seed 14514 was the
         hold filter blocking every approach to a 3-planet opp pocket.
      6. value > 0 (skip end-of-game captures with zero remaining production).
    """
    comet_ids = comet_ids or set()
    my_planets = [p for p in world.planets_by_id.values()
                  if int(p.owner) == int(my_id)]
    targets = [p for p in world.planets_by_id.values()
               if int(p.owner) != int(my_id)
               and int(p.id) not in comet_ids]
    if not my_planets or not targets:
        return []

    opp_planets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != int(my_id) and int(p.owner) >= 0]
    dominant_endgame = (
        len(opp_planets) > 0 and len(my_planets) >= 3 * len(opp_planets)
    )

    # Opp projection (Phase A): predict opp's 1-turn counter-launches and
    # merge them into the per-planet ledger. After this, every
    # predict_garrison_at call sees opp's expected reinforcements at
    # targets AND opp's expected attacks on our sources — the latter is
    # what Phase B's rear-defense check consumes.
    projected_opp = predict_opp_responses(world, int(my_id), num_seats=2)
    enriched_ledger = merge_ledgers(model.ledger, projected_opp)

    candidates: list[Candidate] = []
    for src in my_planets:
        ships_now = int(src.ships)
        prod = int(src.production)
        if ships_now < MIN_FLEET and prod <= 0:
            continue
        for tgt in targets:
            if int(tgt.id) == int(src.id):
                continue
            base_arrivals = list(enriched_ledger.get(int(tgt.id), []))
            for launch_tick in range(MAX_LAUNCH_TICK + 1):
                # Time-indexed budget: by tick `launch_tick`, src will have
                # ships_now + prod*launch_tick (production accrues each turn).
                budget_at_launch = ships_now + prod * int(launch_tick)
                if budget_at_launch < MIN_FLEET:
                    continue
                # ─── Solo path: 2-pass refinement to ships-that-capture ───
                solo_emitted = False
                refined = _refine_ships(
                    src, tgt, launch_tick, omega, base_arrivals,
                )
                if refined is not None:
                    ships, angle, eta, arrival_step = refined
                    if ships <= budget_at_launch:
                        fate = predict_fleet_fate(
                            src, tgt, angle, ships, world, wait_N=launch_tick,
                        )
                        if (fate.outcome == "target"
                                and fate.hit_planet_id is not None
                                and int(fate.hit_planet_id) == int(tgt.id)):
                            owner_at_arr, gar_at_arr = predict_garrison_at(
                                tgt, arrival_step, base_arrivals,
                            )
                            if (int(owner_at_arr) != int(my_id)
                                    and float(ships) > float(gar_at_arr)):
                                if dominant_endgame or _target_holdable_after_capture(
                                    src, tgt, ships, launch_tick, eta,
                                    world, model, my_id,
                                ):
                                    value = _capture_value(tgt, arrival_step)
                                    if value > 0.0 and _source_defensive_ok(
                                        src, int(ships), int(launch_tick),
                                        list(enriched_ledger.get(int(src.id), [])),
                                    ):
                                        candidates.append(Candidate(
                                            src_id=int(src.id),
                                            tgt_id=int(tgt.id),
                                            launch_tick=int(launch_tick),
                                            angle=float(angle),
                                            ships=int(ships),
                                            eta=int(eta),
                                            arrival_step=int(arrival_step),
                                            value=float(value),
                                            is_partial=False,
                                            defense_at_arrival=int(gar_at_arr),
                                        ))
                                        solo_emitted = True
                # ─── Partial path: only if solo didn't fire AND we are in
                #     dominant endgame. Mid-game partials caused regressions
                #     because they bypass the B1 hold filter (the dogpile
                #     commits to UN-HOLDABLE captures that opp immediately
                #     reclaims, churning ships). In dominant endgame the
                #     B1 filter is already relaxed for solos; partials there
                #     are the same trade-off, and the dogpile is the only
                #     way to break opp's fortified pocket.
                if not solo_emitted and dominant_endgame and _DOGPILE_ENABLED:
                    ships_p = int(budget_at_launch)
                    if ships_p < MIN_FLEET:
                        continue
                    res_p = aim_and_eta(
                        src, tgt, ships_p, omega, wait_N=launch_tick,
                    )
                    if res_p is None:
                        continue
                    angle_p, eta_p = res_p
                    arrival_p = int(launch_tick) + int(eta_p)
                    owner_p, gar_p = predict_garrison_at(
                        tgt, arrival_p, base_arrivals,
                    )
                    if int(owner_p) == int(my_id):
                        continue
                    fate_p = predict_fleet_fate(
                        src, tgt, angle_p, ships_p, world, wait_N=launch_tick,
                    )
                    if fate_p.outcome != "target":
                        continue
                    if (fate_p.hit_planet_id is None
                            or int(fate_p.hit_planet_id) != int(tgt.id)):
                        continue
                    value_p = _capture_value(tgt, arrival_p)
                    if value_p <= 0.0:
                        continue
                    if float(ships_p) > float(gar_p):
                        # Partial-ships count actually captures (different eta
                        # from solo's refined estimate gave a lower defense).
                        # Promote to non-partial solo — but still gate on B1.
                        if not dominant_endgame and not _target_holdable_after_capture(
                            src, tgt, ships_p, launch_tick, eta_p,
                            world, model, my_id,
                        ):
                            continue
                        if not _source_defensive_ok(
                            src, int(ships_p), int(launch_tick),
                            list(enriched_ledger.get(int(src.id), [])),
                        ):
                            continue
                        candidates.append(Candidate(
                            src_id=int(src.id),
                            tgt_id=int(tgt.id),
                            launch_tick=int(launch_tick),
                            angle=float(angle_p),
                            ships=int(ships_p),
                            eta=int(eta_p),
                            arrival_step=int(arrival_p),
                            value=float(value_p),
                            is_partial=False,
                            defense_at_arrival=int(gar_p),
                        ))
                    else:
                        # True partial: alone < defense. Useful only when
                        # the dogpile combines multiple sources at the same
                        # (tgt, arrival_step) bucket. Skip the per-source B1
                        # hold filter — hold-after-capture is a coalition
                        # property; dogpile gates on reduced cost > 0.
                        if not _source_defensive_ok(
                            src, int(ships_p), int(launch_tick),
                            list(enriched_ledger.get(int(src.id), [])),
                        ):
                            continue
                        candidates.append(Candidate(
                            src_id=int(src.id),
                            tgt_id=int(tgt.id),
                            launch_tick=int(launch_tick),
                            angle=float(angle_p),
                            ships=int(ships_p),
                            eta=int(eta_p),
                            arrival_step=int(arrival_p),
                            value=float(value_p),
                            is_partial=True,
                            defense_at_arrival=int(gar_p),
                        ))
    return candidates
