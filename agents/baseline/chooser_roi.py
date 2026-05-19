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
import time
from itertools import combinations

from lib.fleet import speed as fleet_speed
# Single-line imports — the bundler's line-by-line regex chokes on
# multi-line parenthesised imports (friction
# `bundler-modular-agent-namespace-access-breaks-bundle`, see
# agents/baseline/main.py:71-76).
from lib.scoring import T_TOTAL_DEFAULT
from lib.scoring import expected_hold
from lib.scoring import margin_multiplier
from lib.scoring import pv_horizon


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
# Minimum residue on solo emits — prevents draining a source to zero.
# Mirror of MIN_COALITION_RESIDUE for the solo path. Proposer dedup
# typically picks the budget-sized variant, which would drain the
# source; choose_roi downsizes to honor this floor.
MIN_SOLO_RESIDUE: int = int(os.environ.get("ROI_MIN_SOLO_RESIDUE", "5"))

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

# Defensive-coalition post-pass: after solos and attack-coalitions
# emit, for each exposed source (vuln_loss > 0) try adding an ally
# reinforce leg. If the reinforcement neutralises opp's counter AND
# the joint ROI (attack + reinforce) beats the bare attack, emit the
# reinforce as an additional move. Disable with
# ROI_DEFENSIVE_COALITION=off.
DEFENSIVE_COALITION_ENABLED: bool = (
    os.environ.get("ROI_DEFENSIVE_COALITION", "on").strip().lower() != "off"
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


def _cheapest_opp_counter(
    src,
    residue: int,
    world,
    me: int,
    max_horizon: int,
    extra_defense_ships: int = 0,
    extra_defense_eta: int | None = None,
):
    """Find the worst-case opp counter-attack against `src` left with
    `residue` ships. Returns `(opp, opp_eta, opp_force, our_defense)`
    for the opp planet whose recapture would cost us the most, or
    `None` if no opp can profitably counter.

    `extra_defense_ships` + `extra_defense_eta` model a hypothetical
    reinforcement leg arriving at `src` at tick `extra_defense_eta`.
    If opp's counter ETA is later than the reinforcement, those ships
    add to `src`'s defense at counter-arrival time. Used by the
    defensive-coalition pass to test "does B's reinforcement save A?"
    """
    if int(src.production) <= 0:
        return None

    worst = None
    worst_loss_proxy = -1  # use opp_force-our_defense gap as the tiebreaker
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
        if (extra_defense_ships > 0
                and extra_defense_eta is not None
                and int(extra_defense_eta) <= opp_eta):
            our_defense += int(extra_defense_ships)
        if opp_force <= our_defense + 1:
            continue  # we hold
        gap = opp_force - our_defense
        if gap > worst_loss_proxy:
            worst_loss_proxy = gap
            worst = (opp, opp_eta, opp_force, our_defense)
    return worst


def _source_vulnerability_loss(
    src,
    residue: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float = 0.99,
    extra_defense_ships: int = 0,
    extra_defense_eta: int | None = None,
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

    `extra_defense_ships` + `extra_defense_eta` simulate a planned ally
    reinforcement of `src`: if the reinforcement arrives ≤ opp's
    counter ETA, those ships add to defense. Used by the defensive-
    coalition pass to test whether a reinforcement neutralises the
    vulnerability.
    """
    if not OPP_MODIFIER_ENABLED:
        return 0.0
    threat = _cheapest_opp_counter(
        src, int(residue), world, me, int(max_horizon),
        extra_defense_ships=int(extra_defense_ships),
        extra_defense_eta=extra_defense_eta,
    )
    if threat is None:
        return 0.0
    _opp, opp_eta, _opp_force, _our_defense = threat
    # Vulnerability is a TRANSIENT cost. Opp holds src for roughly
    # `opp_eta` ticks before we counter-counter-attack (symmetric
    # round-trip). Use a finite loss window, NOT the full remaining
    # game, otherwise the closed-form math says every drained capture
    # is a permanent loss and the chooser refuses to emit.
    # Default loss window = opp_eta (one round-trip). 2× margin
    # because src flipping from ours → opp's shifts margin by 2×
    # tgt.production during the loss window.
    loss_end = int(step) + 2 * int(opp_eta)
    loss_pv = pv_horizon(int(step), int(opp_eta), gamma=gamma,
                         t_total=loss_end)
    return 2.0 * float(src.production) * loss_pv


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
        # Reinforce. The proposer emits these only when tgt is
        # threatened (`time_to_enemy_threat` + positive shortfall via
        # `capture_size`). Compute the value of HOLDING the planet:
        # if we reinforce, we keep tgt's production for the rest of
        # the game AND deny opp the gain (2× margin).
        #
        # Sanity check: does the reinforcement actually defend? The
        # proposer's capture_size sized this candidate to cover the
        # shortfall, but rare edge cases (cap > budget, ship truncation)
        # can leave a candidate that arrives but doesn't hold. We use
        # `_cheapest_opp_counter` with the reinforcement folded in via
        # `extra_defense_*` to confirm. If still threatened post-
        # reinforcement, this is a band-aid that won't actually save
        # the planet — score it as wasted ships.
        post_residue = int(tgt.ships)  # tgt's current ships pre-arrival
        threat_post = _cheapest_opp_counter(
            tgt, post_residue, world, int(me), int(max_horizon),
            extra_defense_ships=int(ships),
            extra_defense_eta=int(arrival),
        )
        if threat_post is not None:
            # Reinforcement insufficient — tgt still falls. Wasted.
            return -SHIP_COST_COEF * int(ships)
        # Confirm tgt WAS threatened before reinforcement; otherwise
        # we're wasting ships on a safe planet.
        threat_pre = _cheapest_opp_counter(
            tgt, post_residue, world, int(me), int(max_horizon),
        )
        if threat_pre is None:
            return -SHIP_COST_COEF * int(ships)
        # We save the planet. Value = production stream we preserve.
        # 2× margin (we keep + opp denied) over the rest of the game,
        # discounted from now (we incur ship cost now, save the
        # margin stream from the threat ETA onward).
        _opp_pre, opp_eta_pre, _opp_force, _defense = threat_pre
        save_pv = pv_horizon(int(step), int(opp_eta_pre), gamma=gamma,
                             t_total=T_TOTAL_DEFAULT)
        gross_reinforce = 2.0 * float(tgt.production) * save_pv
        ship_cost = SHIP_COST_COEF * int(ships)
        wait_cost = WAIT_COST_COEF * int(wait_N) * float(src.production)
        residue_src = int(src.ships) - int(ships)
        vuln_loss = _source_vulnerability_loss(
            src, residue_src, world, model, me, step, int(max_horizon),
            gamma=gamma,
        )
        return gross_reinforce - ship_cost - wait_cost - vuln_loss

    hold = expected_hold(int(tgt.id), arrival, world, model, t_total=T_TOTAL_DEFAULT)
    if hold <= 0:
        return -SHIP_COST_COEF * int(ships)

    pv_held = pv_horizon(int(step), arrival, gamma=gamma,
                         t_total=int(step) + arrival + hold)
    # 2P-contested-neutral assumption: in an adversarial 2P game,
    # neutrals are competed for. If we don't capture, opp likely will.
    # Treat neutrals the same as enemy planets for margin purposes
    # (gain + deny opp's potential gain). This symmetrises gross vs
    # the 1× vuln-loss multiplier so the chooser actually emits
    # against contested boards. margin_multiplier(tgt, me) returns
    # 1 for neutrals; we promote to 2 here.
    mult = margin_multiplier(tgt, me)
    if mult == 1 and int(tgt.owner) == -1:
        mult = 2

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


def _best_reinforcement_for(
    src,
    ships_emitted: int,
    world,
    model,
    me: int,
    step: int,
    max_horizon: int,
    gamma: float,
    used_allies: set,
):
    """Find the best ally that, by reinforcing `src` post-launch, neutralises
    opp's counter-attack profitably.

    Returns `(ally, ally_ships, eta_ally, angle_ally, joint_delta)` for the
    winning reinforcement, or `None` if no ally helps.

    `joint_delta` is the ROI improvement vs. emitting just the attack leg:
    `joint_delta = base_vuln_loss − ally_ship_cost − ally_vuln_loss`. Caller
    accepts when `joint_delta > 0`.
    """
    from agents.baseline.proposer import aim_and_eta, _source_survives_launch
    from lib.trajectory import predict_fleet_fate

    residue = int(src.ships) - int(ships_emitted)
    base_vuln = _source_vulnerability_loss(
        src, residue, world, model, me, step, int(max_horizon), gamma=gamma,
    )
    if base_vuln <= 0.0:
        return None  # source already safe

    # Find the threatening opp counter so we can gate reinforce ETA.
    threat = _cheapest_opp_counter(
        src, residue, world, me, int(max_horizon),
    )
    if threat is None:
        return None
    _opp, opp_eta, _opp_force, _our_defense = threat

    best = None
    best_joint_delta = 0.0
    for ally in world.planets_by_id.values():
        if int(ally.owner) != me:
            continue
        if int(ally.id) == int(src.id):
            continue
        if int(ally.id) in used_allies:
            continue
        if int(ally.ships) < MIN_LEG_SHIPS + MIN_COALITION_RESIDUE:
            continue
        ally_ships = int(ally.ships) - MIN_COALITION_RESIDUE
        if ally_ships < MIN_LEG_SHIPS:
            continue
        angle_ally, eta_ally = aim_and_eta(
            ally, src, ally_ships, world.omega, wait_N=0,
        )
        if int(eta_ally) > int(opp_eta):
            continue  # reinforcement arrives too late
        if not _source_survives_launch(ally, ally_ships, 0, world, model, me):
            continue
        fate = predict_fleet_fate(
            ally, src, float(angle_ally), int(ally_ships), world,
        )
        if fate.outcome != "target":
            continue

        # Recompute vuln with the reinforcement folded in. If it
        # neutralises the threat, new_vuln drops to 0; if it merely
        # reduces (e.g. another opp planet still threatens), new_vuln
        # is the residual.
        new_vuln = _source_vulnerability_loss(
            src, residue, world, model, me, step, int(max_horizon),
            gamma=gamma,
            extra_defense_ships=ally_ships,
            extra_defense_eta=int(eta_ally),
        )
        ally_residue = int(ally.ships) - ally_ships
        ally_vuln = _source_vulnerability_loss(
            ally, ally_residue, world, model, me, step, int(max_horizon),
            gamma=gamma,
        )
        joint_delta = base_vuln - new_vuln - SHIP_COST_COEF * ally_ships - ally_vuln
        if joint_delta > best_joint_delta:
            best_joint_delta = joint_delta
            best = (ally, int(ally_ships), int(eta_ally),
                    float(angle_ally), float(joint_delta))
    return best


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
    # Wallclock budget — insurance against dense-board coalition
    # enumeration. Bench at G2 showed max=687ms on 2P self-play, well
    # under the 1000ms cap, so this is structurally cheap. Solo scoring
    # and the defensive post-pass are O(prerank × opps) and bounded;
    # the deadline check sits at the head of coalition enumeration
    # where the cost lives.
    deadline = time.perf_counter() + max(50.0, float(wallclock_ms)) / 1000.0

    # --- Pass 1: solo scoring with ship-count variants ---
    # For each prerank entry, enumerate candidate ship counts and let
    # solo_roi pick the best. Variants:
    #   (a) original (proposer's pick — usually cap or full budget).
    #   (b) at-fire-time max_safe (src.ships + wait_N × src.production
    #       − MIN_SOLO_RESIDUE) when it differs from (a) and still
    #       captures (≥ MIN_LEG_SHIPS).
    # CRITICAL: max_safe must use AT-FIRE-TIME ships for wait_N>0
    # candidates. With wait_N=11 and src.ships=10, effective fire-time
    # ships = 21; max_safe = 16. Computing max_safe off the current
    # src.ships alone clamps the launch to 5 ships, which always
    # bounces against the wait_N=11 cap. That bug made ROI emit nothing
    # for half a game (G3 vs v7_0 lost 0/32).
    # No hard rejection on (a) — vuln_loss is the principled cost
    # of draining the source, and is enforced inside solo_roi.
    solo_scored: list = []  # (score, src, tgt, ships, angle, wait_N)
    solo_by_target: dict[int, list] = {}
    for entry in prerank:
        _cheap, src, tgt, ships_orig, angle, eta, _horizon, wait_N = entry
        src_ships_at_fire = int(src.ships) + int(wait_N) * int(src.production)
        max_safe_at_fire = max(0, src_ships_at_fire - MIN_SOLO_RESIDUE)

        ship_variants: list[int] = []
        if int(ships_orig) >= MIN_LEG_SHIPS:
            ship_variants.append(int(ships_orig))
        if (max_safe_at_fire >= MIN_LEG_SHIPS
                and max_safe_at_fire != int(ships_orig)):
            ship_variants.append(max_safe_at_fire)
        if not ship_variants:
            continue

        best_score = float("-inf")
        best_ships = ship_variants[0]
        for s in ship_variants:
            score = solo_roi(
                src, tgt, s, int(eta), int(wait_N),
                world, model, int(me), int(step), int(max_horizon),
                gamma=gamma,
            )
            if score > best_score:
                best_score = score
                best_ships = s
        if best_score == float("-inf") or best_score <= 0.0:
            continue
        rec = (best_score, src, tgt, int(best_ships), float(angle), int(wait_N))
        solo_scored.append(rec)
        solo_by_target.setdefault(int(tgt.id), []).append(rec)

    # --- Pass 2: coalition enumeration per opp/neutral target ---
    my_planets = [p for p in world.planets_by_id.values() if int(p.owner) == me]
    opp_targets = [p for p in world.planets_by_id.values()
                   if int(p.owner) != me]

    coalitions: list = []  # (score, target, legs)
    for target in opp_targets:
        if time.perf_counter() > deadline:
            break  # wallclock budget exhausted; emit what we have
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

    # --- Pass 4: defensive-coalition post-pass ---
    # For each emitted move whose source ends up vulnerable, see if an
    # idle ally can reinforce before opp's counter arrives. The reinforce
    # is emitted as an additional move.
    if DEFENSIVE_COALITION_ENABLED and moves:
        used_allies: set[int] = set(used_srcs)  # already-emitting srcs can't reinforce
        # Iterate a snapshot so we don't reinforce-reinforce.
        for src_id, _angle, ships in [tuple(m) for m in moves]:
            src = world.planets_by_id.get(int(src_id))
            if src is None:
                continue
            reinforce = _best_reinforcement_for(
                src, int(ships), world, model, me, step,
                int(max_horizon), gamma, used_allies,
            )
            if reinforce is None:
                continue
            ally, ally_ships, _eta_ally, angle_ally, _delta = reinforce
            used_allies.add(int(ally.id))
            moves.append([int(ally.id), float(angle_ally), int(ally_ships)])

    return moves
