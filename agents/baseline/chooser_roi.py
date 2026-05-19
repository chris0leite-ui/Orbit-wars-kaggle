"""chooser_roi — ROI-prior + opp-modifier chooser.

Architectural pivot (2026-05-19, PI-directed): replace the trajectory rollout
foundation with a closed-form ROI prior + thin opp-vulnerability posterior.
Dispatched via BASELINE_CHOOSER=roi from agents/baseline/main.py.

Pipeline per turn:
  1. solo_roi(src, tgt, ships, eta, wait_N) computes a closed-form ROI per
     proposer candidate using lib/scoring + lib/world_model primitives.
  2. coalition_roi enumerates N-way joint launches per target via merged
     arrival-ledger walk; emit if joint > max(solo) + slack.
  3. (Phase 4, pending) opp_modifier_check scans emit set for exposed
     sources and downsizes/drops candidates the opp would profitably
     counter-target.

Current implementation: Phase 3 (solo_roi + 2..4-leg coalitions).

See /root/.claude/plans/okay-we-can-do-elegant-lampson.md.
"""

from __future__ import annotations

import math
import os
from itertools import combinations

from lib.fleet import speed as fleet_speed
from lib.scoring import (
    T_TOTAL_DEFAULT,
    expected_hold,
    margin_multiplier,
    pv_horizon,
)


# Cost coefficients — to be calibrated via bench in Phase 6. Initial
# values are deliberately conservative so the ROI prior favours
# small-cost moves over speculative captures while we tune.
SHIP_COST_COEF: float = float(os.environ.get("ROI_SHIP_COST", "0.05"))
WAIT_COST_COEF: float = float(os.environ.get("ROI_WAIT_COST", "0.5"))

# Coalition-only knobs.
COALITION_MAX_SOURCES: int = int(os.environ.get("ROI_COALITION_MAX", "4"))
# Coalition wins only if its ROI exceeds the best constituent solo by
# at least this slack. Prevents 3-source over-firing on targets a
# 1-source solo could handle.
COALITION_SLACK: float = float(os.environ.get("ROI_COALITION_SLACK", "1.0"))
# Per-leg residue reserve when re-enumerating coalition seeds (solo
# candidates that the proposer dropped because cap > budget). The
# leg sends src.ships - MIN_COALITION_RESIDUE; smaller residues are
# caught by _source_survives_launch downstream.
MIN_COALITION_RESIDUE: int = int(os.environ.get("ROI_MIN_RESIDUE", "5"))
MIN_LEG_SHIPS: int = 2

# Source-vulnerability knobs (Phase 4). Folded into solo/coalition
# ROI so the comparison sees the true cost of exposing a source.
# Disable with ROI_OPP_MODIFIER=off to fall back to Phase 3 behaviour.
OPP_MODIFIER_ENABLED: bool = (
    os.environ.get("ROI_OPP_MODIFIER", "on").strip().lower() != "off"
)
# Don't consider opp planets with fewer than this many ships as a
# counter-attack threat. Matches the proposer's hold-feasibility
# MIN_COUNTER_SHIPS (proposer.py:446) for symmetry.
VULN_MIN_OPP_SHIPS: int = 20

# Endgame elimination bonus: when capturing this target removes the
# LAST planet owned by a non-me, non-neutral player, the strategic
# value is "uncontested production for the rest of the game" — far
# beyond the per-tick PV math. Bonus = our_total_prod × pv_full,
# representing the unopposed advantage we get after elimination.
ENDGAME_BONUS_ENABLED: bool = (
    os.environ.get("ROI_ENDGAME_BONUS", "on").strip().lower() != "off"
)


def _endgame_finish_bonus(
    target,
    capture_step: int,
    world,
    me: int,
    step: int,
    gamma: float,
) -> float:
    """Bonus PV when `target` is the last planet owned by some opp
    player. After capture, that player is eliminated and we play
    uncontested for the rest of the game.

    Bonus = sum(my_planets.production) × pv_horizon(step, capture_step).
    For two-opp scenarios this only fires if BOTH conditions hold:
    target.owner has no other planets. The bonus reflects the
    margin lead we'd accrue once that player is gone.
    """
    if not ENDGAME_BONUS_ENABLED:
        return 0.0
    if int(target.owner) == me or int(target.owner) == -1:
        return 0.0
    target_owner = int(target.owner)
    for p in world.planets_by_id.values():
        if int(p.id) == int(target.id):
            continue
        if int(p.owner) == target_owner:
            return 0.0  # target_owner has another planet → not last
    our_total_prod = sum(
        float(p.production) for p in world.planets_by_id.values()
        if int(p.owner) == me
    )
    if our_total_prod <= 0:
        return 0.0
    pv_full = pv_horizon(int(step), int(capture_step), gamma=gamma,
                         t_total=T_TOTAL_DEFAULT)
    return our_total_prod * pv_full


def _source_vulnerability_loss(
    src,
    residue: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float = 0.99,
) -> float:
    """Expected production loss if the cheapest opp planet recaptures
    `src` after it's left with `residue` ships.

    Symmetric to _target_holdable_after_capture (proposer.py:407) but
    applied to the SOURCE: after our launch drains src, can any nearby
    opp planet profitably counter-attack? If yes, this returns the
    PV-discounted production stream src would have produced for us over
    the remaining game. Returns 0 if the source is safe (residue
    ≥ safe_garrison or no plausible opp threat).

    No fast_sim. Considers each opp planet's straight-line counter:
    opp arrives with opp.ships + opp.production · opp_eta; we defend
    with residue + src.production · opp_eta. If opp_force > defense + 1
    AND opp_eta ≤ max_horizon, the recapture is feasible.
    """
    if not OPP_MODIFIER_ENABLED:
        return 0.0
    if int(src.production) <= 0:
        return 0.0
    # No early "residue ≥ safe_garrison" out — that threshold ignores
    # the magnitude of nearby opp threats. The per-opp scan below
    # computes opp_force vs our_defense exactly, so it's both correct
    # and cheap (O(opps)).

    worst = 0.0
    for opp in world.planets_by_id.values():
        if int(opp.owner) == me or int(opp.owner) == -1:
            continue
        if int(opp.ships) < VULN_MIN_OPP_SHIPS:
            continue
        d = math.hypot(float(opp.x) - float(src.x),
                       float(opp.y) - float(src.y))
        flight = d - float(opp.radius) - float(src.radius) - 0.1
        if flight <= 0:
            continue
        spd = fleet_speed(int(opp.ships))
        if spd <= 0:
            continue
        opp_eta = int(math.ceil(flight / spd))
        if opp_eta > int(max_horizon):
            continue
        opp_force = int(opp.ships) + int(opp.production) * opp_eta
        our_defense = max(0, int(residue)) + int(src.production) * opp_eta
        if opp_force <= our_defense + 1:
            continue  # we hold
        # Recapture is feasible. Cost to us = margin loss when src
        # flips from ours to opp's: -1 (we lose its prod) + -1 (opp
        # gains its prod) → mult=2 against margin. PV-discount the
        # production stream from opp_eta to game-end.
        loss_pv = pv_horizon(int(step), opp_eta, gamma=gamma,
                             t_total=T_TOTAL_DEFAULT)
        loss = 2.0 * float(src.production) * loss_pv
        if loss > worst:
            worst = loss
    return worst


def solo_roi(
    src,
    tgt,
    ships: int,
    eta: int,
    wait_N: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float = 0.99,
) -> float:
    """Closed-form ROI score for one (src, tgt, ships, eta, wait_N).

    Returns -inf for refused candidates (past horizon, bounced combat).
    Captures bounce with a small negative ship-cost penalty so the
    chooser's "do nothing" baseline (ROI=0) outranks them. Reinforce
    moves (target already ours) return 0 — no margin gained.
    """
    arrival = int(wait_N) + int(eta)
    if arrival > max_horizon:
        return float("-inf")

    arrivals = list(model.ledger.get(int(tgt.id), []))
    arrivals.append((arrival, int(me), int(ships)))

    from lib.world_model import predict_garrison_at  # local — keeps lib/ env-free
    owner_arr, _garrison_arr = predict_garrison_at(tgt, arrival, arrivals)

    if owner_arr != me:
        return -SHIP_COST_COEF * int(ships)

    if int(tgt.owner) == me:
        return 0.0  # reinforce — proposer-side threat reinforce handles defense

    hold = expected_hold(int(tgt.id), arrival, world, model, t_total=T_TOTAL_DEFAULT)
    if hold <= 0:
        return -SHIP_COST_COEF * int(ships)

    pv_held = pv_horizon(int(step), arrival, gamma=gamma,
                         t_total=int(step) + arrival + hold)
    mult = margin_multiplier(tgt, me)

    gross = mult * float(tgt.production) * pv_held
    endgame_bonus = _endgame_finish_bonus(
        tgt, arrival, world, me, step, gamma,
    )
    ship_cost = SHIP_COST_COEF * int(ships)
    wait_cost = WAIT_COST_COEF * int(wait_N) * float(src.production)

    residue = int(src.ships) - int(ships)
    vuln_loss = _source_vulnerability_loss(
        src, residue, world, model, me, step, int(max_horizon), gamma=gamma,
    )

    return gross + endgame_bonus - ship_cost - wait_cost - vuln_loss


def _coalition_legs_for_target(target, my_planets, world, model, me: int,
                               max_horizon: int):
    """Build coalition leg seeds: one per my-planet that could fire at
    `target` within `max_horizon`, sized to keep residue ≥
    MIN_COALITION_RESIDUE.

    Returns [(src, ships, eta, angle), ...]. Empty if no my-planet
    can plausibly reach.
    """
    from agents.baseline.proposer import aim_and_eta, _source_survives_launch
    from lib.trajectory import predict_fleet_fate

    legs = []
    for src in my_planets:
        if int(src.ships) < MIN_LEG_SHIPS + MIN_COALITION_RESIDUE:
            continue
        if int(src.id) == int(target.id):
            continue
        leg_ships = int(src.ships) - MIN_COALITION_RESIDUE
        if leg_ships < MIN_LEG_SHIPS:
            continue
        angle, eta = aim_and_eta(src, target, leg_ships, world.omega, wait_N=0)
        if int(eta) > max_horizon:
            continue
        if not _source_survives_launch(src, leg_ships, 0, world, model, me):
            continue
        # Trajectory admissibility (fire-now only).
        fate = predict_fleet_fate(src, target, float(angle), int(leg_ships), world)
        if fate.outcome != "target":
            continue
        legs.append((src, int(leg_ships), int(eta), float(angle)))
    return legs


def coalition_roi(target, legs, world, model, me: int, step: int,
                  max_horizon: int, gamma: float = 0.99):
    """Closed-form ROI for a multi-leg coalition launch.

    Builds the merged arrival ledger (existing in-flight arrivals at
    target + each leg) and walks arrival ticks until the target flips
    to me. PV-discounts production over the resulting hold horizon.

    Returns (roi_score, capture_step). roi_score is -inf if the
    coalition fails to capture; capture_step is None in that case.
    """
    if not legs:
        return float("-inf"), None

    base_arrivals = list(model.ledger.get(int(target.id), []))
    leg_arrivals = [(int(eta), int(me), int(ships))
                    for (_src, ships, eta, _angle) in legs]
    merged = base_arrivals + leg_arrivals

    arrival_ticks = sorted({a[0] for a in leg_arrivals})
    if not arrival_ticks or arrival_ticks[-1] > max_horizon:
        return float("-inf"), None

    from lib.world_model import predict_garrison_at
    capture_step = None
    for tick in arrival_ticks:
        owner_t, _ = predict_garrison_at(target, tick, merged)
        if owner_t == me:
            capture_step = tick
            break
    if capture_step is None:
        return float("-inf"), None

    hold = expected_hold(int(target.id), capture_step, world, model,
                         t_total=T_TOTAL_DEFAULT)
    if hold <= 0:
        return float("-inf"), None

    pv_held = pv_horizon(int(step), capture_step, gamma=gamma,
                         t_total=int(step) + capture_step + hold)
    mult = margin_multiplier(target, me)
    total_ships = sum(int(ships) for (_src, ships, _eta, _angle) in legs)
    gross = mult * float(target.production) * pv_held
    endgame_bonus = _endgame_finish_bonus(
        target, capture_step, world, me, step, gamma,
    )
    ship_cost = SHIP_COST_COEF * total_ships

    # Coalition vulnerability is capped by opp's counter-capacity.
    # Each opp planet can launch at most one counter-attack per turn.
    # Count plausible opp counter-sources; cap loss at top-K per-leg
    # vulnerabilities, K = num_opp_planets_above_threshold. With one
    # opp planet (the common 2P case), opp can only recapture ONE of
    # our drained sources — taking MAX, not SUM.
    per_leg_vuln: list[float] = []
    for src, ships, _eta, _angle in legs:
        residue = int(src.ships) - int(ships)
        per_leg_vuln.append(_source_vulnerability_loss(
            src, residue, world, model, me, step, int(max_horizon), gamma=gamma,
        ))
    opp_counter_capacity = sum(
        1 for p in world.planets_by_id.values()
        if int(p.owner) != me and int(p.owner) != -1
        and int(p.ships) >= VULN_MIN_OPP_SHIPS
    )
    per_leg_vuln.sort(reverse=True)
    vuln_loss = sum(per_leg_vuln[:max(1, opp_counter_capacity)])

    return gross + endgame_bonus - ship_cost - vuln_loss, capture_step


def _best_coalition_for_target(target, my_planets, world, model, me: int,
                               step: int, max_horizon: int, gamma: float):
    """Enumerate 2..COALITION_MAX_SOURCES leg subsets; return the
    (roi, legs) of the best-scoring coalition for `target`, or
    (-inf, []) if no coalition captures.

    Caps enumeration at COALITION_MAX_SOURCES seeds total. With 4 seeds
    that's C(4,2)+C(4,3)+C(4,4) = 6+4+1 = 11 subsets per target —
    bounded constant overhead.
    """
    seeds = _coalition_legs_for_target(
        target, my_planets, world, model, me, max_horizon,
    )
    if len(seeds) < 2:
        return float("-inf"), []
    seeds.sort(key=lambda leg: leg[2])  # by ETA ascending
    seeds = seeds[:COALITION_MAX_SOURCES]

    best_roi = float("-inf")
    best_legs: list = []
    for r in range(2, len(seeds) + 1):
        for combo in combinations(seeds, r):
            roi, _cap = coalition_roi(
                target, combo, world, model, me, step, max_horizon, gamma=gamma,
            )
            if roi > best_roi:
                best_roi = roi
                best_legs = list(combo)
    return best_roi, best_legs


def choose_roi(
    snap_base,
    prerank,
    me: int,
    num_seats: int,
    wallclock_ms: float,
    min_horizon: int,
    max_horizon: int,
    gamma: float,
    world,
    model,
    step: int,
) -> list:
    """ROI-prior chooser: solo scoring + N-way coalition, greedy emit.

    1. Score every prerank candidate with solo_roi; keep positives.
    2. For each opp/neutral target, enumerate 2..N-way coalitions of
       my-planet seeds; if best coalition ROI > best constituent
       solo ROI + COALITION_SLACK, swap the coalition in.
    3. Greedy commit: sort by ROI desc, one emit per (src, tgt) pair.

    Coalition legs are emitted as independent fire-now launches at
    the same target — the engine handles concurrent arrivals via
    its arrival resolver. wait_N>0 candidates remain solo-only at
    this phase (mixed wait coalition geometry isn't validated by
    predict_fleet_fate).
    """
    # --- Pass 1: solo scoring ---
    solo_scored: list = []  # (score, src, tgt, ships, angle, wait_N)
    solo_by_target: dict[int, list] = {}
    for entry in prerank:
        _cheap, src, tgt, ships, angle, eta, _horizon, wait_N = entry
        score = solo_roi(
            src, tgt, int(ships), int(eta), int(wait_N),
            world, model, int(me), int(step), int(max_horizon),
            gamma=gamma,
        )
        if score == float("-inf") or score <= 0.0:
            continue
        rec = (score, src, tgt, int(ships), float(angle), int(wait_N))
        solo_scored.append(rec)
        solo_by_target.setdefault(int(tgt.id), []).append(rec)

    # --- Pass 2: coalition enumeration per opp/neutral target ---
    my_planets = [p for p in world.planets_by_id.values() if int(p.owner) == me]
    opp_targets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != me]

    coalitions: list = []  # (score, target, legs)
    for target in opp_targets:
        c_roi, c_legs = _best_coalition_for_target(
            target, my_planets, world, model, me, step, max_horizon, gamma,
        )
        if c_roi <= 0.0 or not c_legs:
            continue
        # Coalition beats best solo on this target only if strictly
        # better by the slack margin. Otherwise the solo path is
        # preferred (smaller ship commitment).
        best_solo_on_tgt = max(
            (s[0] for s in solo_by_target.get(int(target.id), [])),
            default=0.0,
        )
        if c_roi <= best_solo_on_tgt + COALITION_SLACK:
            continue
        coalitions.append((c_roi, target, c_legs))

    # --- Pass 3: greedy emit ---
    # Sort all candidates (solo + coalition) by score desc.
    # Coalitions claim ALL their legs' sources atomically.
    combined: list = []
    for rec in solo_scored:
        combined.append(("solo", rec[0], rec))
    for coal in coalitions:
        combined.append(("coalition", coal[0], coal))
    combined.sort(key=lambda c: -c[1])

    used_srcs: set[int] = set()
    used_tgts: set[int] = set()
    moves: list[list] = []
    for kind, _score, payload in combined:
        if kind == "coalition":
            _c_roi, target, legs = payload
            tid = int(target.id)
            if tid in used_tgts:
                continue
            if any(int(leg[0].id) in used_srcs for leg in legs):
                continue
            used_tgts.add(tid)
            for src, ships, _eta, angle in legs:
                sid = int(src.id)
                used_srcs.add(sid)
                moves.append([sid, float(angle), int(ships)])
            continue
        # solo
        _score, src, tgt, ships, angle, wait_N = payload
        sid, tid = int(src.id), int(tgt.id)
        if sid in used_srcs or tid in used_tgts:
            continue
        used_srcs.add(sid)
        used_tgts.add(tid)
        if wait_N == 0:
            moves.append([sid, float(angle), int(ships)])
    return moves
